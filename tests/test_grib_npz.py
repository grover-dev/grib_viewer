"""Tests for the npz writer, and in particular for cutting one field into parts.

The split is the part with something to get wrong: a boundary in the wrong
place, a part that disagrees with its neighbours about the quantiser, or a
manifest that describes a layout the files do not have would all still produce
files that load. So most of what is below compares a split run against the
single-file run of the same selection -- the two have to be indistinguishable
once reassembled, which is exactly what boatforge::NpzField relies on.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from grib_npz import chunk_path, plan_chunks

ROOT = Path(__file__).resolve().parent.parent

MIB = 1 << 20


def run(*args) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "grib_npz.py"), *map(str, args)],
        capture_output=True, text=True, cwd=ROOT,
    )


def parts_of(directory: Path) -> list[Path]:
    """The parts of a split field, in the order their names sort."""
    return sorted(directory.glob("*.npz"))


# --------------------------------------------------------------------------
# plan_chunks: the boundaries themselves, without the cost of writing anything
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "nt, frame_bytes, limit, want",
    [
        # Room for everything: one part, no overlap to pay for.
        (10, MIB, 100 * MIB, [(0, 10)]),
        (1, MIB, 100 * MIB, [(0, 1)]),
        # 4 frames per part, each starting on the previous part's last frame.
        (10, MIB, 4 * MIB, [(0, 4), (3, 7), (6, 10)]),
        # A tail shorter than a full part is still written.
        (5, MIB, 4 * MIB, [(0, 4), (3, 5)]),
        # Exactly one part's worth.
        (4, MIB, 4 * MIB, [(0, 4)]),
        # One frame over the limit: emitted alone and over budget, since the
        # grid is never cut. No overlap is possible at one frame per part.
        (3, 10 * MIB, 4 * MIB, [(0, 1), (1, 2), (2, 3)]),
    ],
)
def test_plan_chunks(nt, frame_bytes, limit, want):
    assert plan_chunks(nt, frame_bytes, limit) == want


@pytest.mark.parametrize("nt", [1, 2, 3, 7, 8, 9, 64, 100])
@pytest.mark.parametrize("per_part", [1, 2, 3, 5, 16])
def test_plan_chunks_covers_every_frame(nt, per_part):
    """No frame is dropped and no part is empty, at any size or alignment."""
    chunks = plan_chunks(nt, MIB, per_part * MIB)

    covered = set()
    for lo, hi in chunks:
        assert lo < hi
        covered |= set(range(lo, hi))
    assert covered == set(range(nt))

    # Ascending, and each part within the limit.
    assert chunks == sorted(chunks)
    assert all(hi - lo <= max(per_part, 1) for lo, hi in chunks)


@pytest.mark.parametrize("nt", [2, 3, 7, 8, 64])
@pytest.mark.parametrize("per_part", [2, 3, 5, 16])
def test_plan_chunks_overlaps_by_exactly_one_frame(nt, per_part):
    """Consecutive parts share their boundary frame, so no interval is orphaned.

    This is what lets a sampler interpolate across a boundary: the frames either
    side of any instant always live in the same file.
    """
    chunks = plan_chunks(nt, MIB, per_part * MIB)
    for (_, hi), (lo, _) in zip(chunks, chunks[1:]):
        assert lo == hi - 1

    # Stated as the property that matters: every interval between consecutive
    # frames lies wholly inside one part.
    for frame in range(nt - 1):
        assert any(lo <= frame and frame + 1 < hi for lo, hi in chunks)


def test_plan_chunks_single_frame_parts_cannot_overlap():
    """A frame over the limit degenerates to one frame per part -- and then the
    intervals between them belong to no part, which is a real hole rather than
    something this function can fix."""
    chunks = plan_chunks(4, 10 * MIB, MIB)
    assert chunks == [(0, 1), (1, 2), (2, 3), (3, 4)]
    assert not any(lo <= 0 and 1 < hi for lo, hi in chunks)


def test_chunk_path():
    dst = Path("a/waves.npz")
    assert chunk_path(dst, 0, 1) == dst                     # alone: name kept
    assert chunk_path(dst, 0, 12) == Path("a/waves.000.npz")
    assert chunk_path(dst, 3, 12) == Path("a/waves.003.npz")
    # Zero-padded so a glob sorts in time order past ten parts.
    assert sorted(chunk_path(dst, i, 12).name for i in range(12))[:3] == [
        "waves.000.npz", "waves.001.npz", "waves.002.npz",
    ]


# --------------------------------------------------------------------------
# End to end: a split run against the single-file run of the same selection
# --------------------------------------------------------------------------

@pytest.fixture
def whole_and_split(global_grib, tmp_path):
    """The same field written twice: as one cube, and cut into parts.

    The fixture grid is 19x36 uint16, so a frame is 1368 bytes; the limit below
    is chosen to put two frames in a part and force a boundary.
    """
    whole = tmp_path / "whole.npz"
    assert run(global_grib, "u10", whole, "--max-mib", 0).returncode == 0

    split_dir = tmp_path / "split"
    split_dir.mkdir()
    r = run(global_grib, "u10", split_dir / "f.npz", "--max-mib", 2 * 1368 / MIB)
    assert r.returncode == 0, r.stderr

    return np.load(whole), parts_of(split_dir)


def test_split_writes_several_parts(whole_and_split):
    whole, parts = whole_and_split
    assert len(parts) > 1
    assert [p.name for p in parts] == sorted(p.name for p in parts)


def test_split_reassembles_to_the_whole_cube(whole_and_split):
    """The point of the whole exercise: cutting the field changes nothing about
    it. Bit-identical, not merely close -- the parts share one quantiser."""
    whole, parts = whole_and_split

    times, payload = [], []
    for path in parts:
        part = np.load(path)
        # Drop the frame shared with the previous part.
        first = 1 if times and int(part["time"][0]) == times[-1] else 0
        times += [int(t) for t in part["time"][first:]]
        payload.append(part["data"][first:])

    np.testing.assert_array_equal(np.concatenate(payload), whole["data"])
    np.testing.assert_array_equal(np.array(times, dtype="int64"), whole["time"])


def test_parts_share_the_grid_and_the_quantiser(whole_and_split):
    """A part that disagreed would still load, and would silently mean something
    different from its neighbours."""
    whole, parts = whole_and_split
    for path in parts:
        part = np.load(path)
        for key in ("version", "dt", "lat0", "dlat", "nlat", "lon0", "dlon",
                    "nlon", "lon_wrap", "scale", "offset", "fill"):
            assert part[key] == whole[key], f"{path.name}: {key}"
        np.testing.assert_array_equal(part["lat"], whole["lat"])
        np.testing.assert_array_equal(part["lon"], whole["lon"])


def test_parts_are_self_consistent(whole_and_split):
    """Each part's scalars have to describe its own payload, since that is all
    the C++ sampler reads."""
    whole, parts = whole_and_split
    for path in parts:
        part = np.load(path)
        nt = int(part["nt"])
        assert part["data"].shape == (nt, int(part["nlat"]), int(part["nlon"]))
        assert part["time"].size == nt
        assert int(part["t0"]) == int(part["time"][0])
        # The time axis is the arithmetic one the format promises.
        np.testing.assert_array_equal(
            part["time"], int(part["t0"]) + int(part["dt"]) * np.arange(nt))


def test_boundary_frame_is_duplicated_verbatim(whole_and_split):
    """The overlap is a copy, not a re-derivation: sampling exactly at a shared
    stamp gives the same answer whichever part happens to be loaded."""
    whole, parts = whole_and_split
    for earlier, later in zip(parts, parts[1:]):
        a, b = np.load(earlier), np.load(later)
        assert int(a["time"][-1]) == int(b["time"][0])
        np.testing.assert_array_equal(a["data"][-1], b["data"][0])


def test_parts_tile_the_time_axis_without_a_gap(whole_and_split):
    """Every instant the whole cube covers is covered by some part."""
    whole, parts = whole_and_split
    spans = []
    for path in parts:
        part = np.load(path)
        spans.append((int(part["time"][0]), int(part["time"][-1])))

    for stamp in whole["time"]:
        assert any(lo <= stamp <= hi for lo, hi in spans), stamp
    assert spans[0][0] == int(whole["time"][0])
    assert spans[-1][1] == int(whole["time"][-1])


# --------------------------------------------------------------------------
# The embedded manifest
# --------------------------------------------------------------------------

def test_manifest_describes_every_part_from_any_part(whole_and_split):
    """Opening one part has to answer which file covers a given instant -- that
    is the whole reason the manifest is embedded rather than kept alongside."""
    whole, parts = whole_and_split
    loaded = [np.load(p) for p in parts]

    for index, part in enumerate(loaded):
        assert int(part["part"]) == index
        assert int(part["nparts"]) == len(parts)
        # Identical in every part, so no part can go stale against the others.
        np.testing.assert_array_equal(part["part_t0"], loaded[0]["part_t0"])
        np.testing.assert_array_equal(part["part_nt"], loaded[0]["part_nt"])
        assert int(part["overlap"]) == 1

    # ...and it describes the files that are actually there.
    np.testing.assert_array_equal(
        loaded[0]["part_t0"], [int(p["t0"]) for p in loaded])
    np.testing.assert_array_equal(
        loaded[0]["part_nt"], [int(p["nt"]) for p in loaded])


def test_single_file_output_carries_a_manifest_too(global_grib, tmp_path):
    """One code path for consumers: an unsplit field is a one-part field."""
    dst = tmp_path / "one.npz"
    assert run(global_grib, "u10", dst).returncode == 0

    npz = np.load(dst)
    assert int(npz["nparts"]) == 1
    assert int(npz["part"]) == 0
    assert int(npz["overlap"]) == 0
    np.testing.assert_array_equal(npz["part_nt"], [int(npz["nt"])])


def test_manifest_does_not_break_the_v1_format(global_grib, tmp_path):
    """The manifest is additive: version stays 1, so a reader built before the
    split still loads a part as the ordinary field it also is."""
    dst = tmp_path / "one.npz"
    assert run(global_grib, "u10", dst).returncode == 0
    npz = np.load(dst)
    assert int(npz["version"]) == 1
    assert {"t0", "dt", "nt", "lat0", "dlat", "nlat", "lon0", "dlon", "nlon",
            "lon_wrap", "scale", "offset", "fill", "data"} <= set(npz.files)


# --------------------------------------------------------------------------
# --max-mib
# --------------------------------------------------------------------------

def test_max_mib_zero_disables_splitting(global_grib, tmp_path):
    dst = tmp_path / "nosplit.npz"
    r = run(global_grib, "u10", dst, "--max-mib", 0)
    assert r.returncode == 0, r.stderr
    assert dst.exists()
    assert not list(tmp_path.glob("nosplit.0*.npz"))


def test_max_mib_below_one_frame_writes_a_part_per_frame(global_grib, tmp_path):
    """Over budget rather than refused: the grid is never cut."""
    out = tmp_path / "tiny"
    out.mkdir()
    r = run(global_grib, "u10", out / "f.npz", "--max-mib", 1 / MIB)
    assert r.returncode == 0, r.stderr

    parts = parts_of(out)
    assert all(int(np.load(p)["nt"]) == 1 for p in parts)
    # Without room for two frames there is no overlap to report.
    assert all(int(np.load(p)["overlap"]) == 0 for p in parts)


def test_negative_max_mib_is_rejected(global_grib, tmp_path):
    r = run(global_grib, "u10", tmp_path / "x.npz", "--max-mib", -1)
    assert r.returncode != 0
    assert "max-mib" in (r.stderr + r.stdout)


# --------------------------------------------------------------------------
# The split has to compose with the conversions the writer already did
# --------------------------------------------------------------------------

def test_split_accumulated_field_matches_the_whole_one(stepped_grib, tmp_path):
    """ssrd is divided by the step length and re-stamped to the middle of its
    accumulation window. Both are decided from the *source* spacing, so cutting
    the selection must not shift them."""
    whole = tmp_path / "whole.npz"
    assert run(stepped_grib, "ssrd", whole).returncode == 0

    out = tmp_path / "split"
    out.mkdir()
    # 11x21 uint16 -> 462 bytes a frame; two frames to a part.
    r = run(stepped_grib, "ssrd", out / "f.npz", "--max-mib", 2 * 462 / MIB)
    assert r.returncode == 0, r.stderr

    reference = np.load(whole)
    parts = [np.load(p) for p in parts_of(out)]
    assert len(parts) > 1

    for part in parts:
        assert part["scale"] == reference["scale"]
        assert part["offset"] == reference["offset"]

    times, payload = [], []
    for part in parts:
        first = 1 if times and int(part["time"][0]) == times[-1] else 0
        times += [int(t) for t in part["time"][first:]]
        payload.append(part["data"][first:])

    np.testing.assert_array_equal(np.concatenate(payload), reference["data"])
    np.testing.assert_array_equal(np.array(times, dtype="int64"), reference["time"])


def test_split_float32_payload_matches_the_whole_one(global_grib, tmp_path):
    whole = tmp_path / "whole.npz"
    assert run(global_grib, "u10", whole, "--dtype", "f32").returncode == 0

    out = tmp_path / "split"
    out.mkdir()
    r = run(global_grib, "u10", out / "f.npz", "--dtype", "f32",
            "--max-mib", 2 * 19 * 36 * 4 / MIB)
    assert r.returncode == 0, r.stderr

    reference = np.load(whole)
    parts = [np.load(p) for p in parts_of(out)]
    assert len(parts) > 1
    assert all(p["data"].dtype == np.float32 for p in parts)

    times, payload = [], []
    for part in parts:
        first = 1 if times and int(part["time"][0]) == times[-1] else 0
        times += [int(t) for t in part["time"][first:]]
        payload.append(part["data"][first:])

    np.testing.assert_array_equal(np.concatenate(payload), reference["data"])
