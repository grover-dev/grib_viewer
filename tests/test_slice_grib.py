"""Tests for the slicer's CLI logic and its end-to-end behaviour."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import grib_utils as gu
from slice_grib import parse_frames

ROOT = Path(__file__).resolve().parent.parent


def run(*args) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(ROOT / "slice_grib.py"), *map(str, args)],
        capture_output=True, text=True, cwd=ROOT,
    )


@pytest.mark.parametrize(
    "spec, n, want",
    [
        ("0:24", 100, (0, 24)),
        ("100", 200, (100, 101)),
        (":10", 100, (0, 10)),
        ("90:", 100, (90, 100)),
        ("0:999", 50, (0, 50)),   # clamped to what exists
    ],
)
def test_parse_frames(spec, n, want):
    assert parse_frames(spec, n) == want


@pytest.mark.parametrize("spec", ["10:10", "50:20"])
def test_parse_frames_rejects_empty_ranges(spec):
    with pytest.raises(SystemExit):
        parse_frames(spec, 100)


def test_slice_by_frames(global_grib, tmp_path):
    dst = tmp_path / "f.grib"
    r = run(global_grib, dst, "--frames", "1:3")
    assert r.returncode == 0, r.stderr

    src_times = gu.scan_times(str(global_grib))
    got = gu.scan_times(str(dst))
    np.testing.assert_array_equal(got, src_times[1:3])


def test_slice_by_timestamp(global_grib, tmp_path):
    dst = tmp_path / "t.grib"
    r = run(global_grib, dst, "--start", "2025-01-01T06:00", "--end", "2025-01-01T12:00")
    assert r.returncode == 0, r.stderr

    got = gu.scan_times(str(dst))
    assert len(got) == 2
    assert got[0].astype("datetime64[m]") == np.datetime64("2025-01-01T06:00")
    assert got[-1].astype("datetime64[m]") == np.datetime64("2025-01-01T12:00")


def test_frames_and_timestamps_are_mutually_exclusive(global_grib, tmp_path):
    r = run(global_grib, tmp_path / "x.grib", "--frames", "0:2", "--start", "2025-01-01")
    assert r.returncode != 0
    assert "not both" in r.stdout + r.stderr


def test_stride(global_grib, tmp_path):
    dst = tmp_path / "s.grib"
    assert run(global_grib, dst, "--stride", "2").returncode == 0
    src = gu.scan_times(str(global_grib))
    np.testing.assert_array_equal(gu.scan_times(str(dst)), src[::2])


def test_bbox_rejects_bad_latitudes(global_grib, tmp_path):
    r = run(global_grib, tmp_path / "x.grib", "--bbox", "0", "10", "50", "20")  # S > N
    assert r.returncode != 0
    assert "latitudes" in r.stdout + r.stderr


def test_time_and_area_compose(global_grib, tmp_path):
    """The two selections are independent and must apply together."""
    dst = tmp_path / "both.grib"
    assert run(global_grib, dst, "--frames", "0:2", "--bbox", "-20", "20", "-10", "10").returncode == 0

    fields = gu.open_grib(str(dst))
    u10 = fields["u10"]
    assert u10.sizes["time"] == 2
    assert float(u10.latitude.max()) <= 10
    assert float(u10.longitude.min()) >= -20

    # smaller than the source on both axes
    src = gu.open_grib(str(global_grib))["u10"]
    assert u10.sizes["longitude"] < src.sizes["longitude"]


def test_list_does_not_write(global_grib, tmp_path):
    r = run(global_grib, "--list")
    assert r.returncode == 0
    assert "time steps" in r.stdout
