"""The parts of the .npz field format that do not care where the data came from.

`grib_npz.py` and `netcdf4_npz.py` read completely different files -- indexed
GRIB messages through ecCodes, HDF5 datasets through h5py -- but they write the
same thing, and the C++ reader has exactly one parser for it. Anything that
would have to be changed in both writers at once lives here instead: the layout
of the members, the arithmetic the layout promises, and the rules for splitting
a long selection across files.

The format itself is documented in `grib_npz.py`; this module is its
implementation, not its explanation.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np

# uint16 payload: 65535 is reserved for "no data", leaving 0..65534 for values.
U16_FILL = 65535
U16_MAX = 65534


def parse_frames(spec: str, n: int) -> tuple[int, int]:
    """'0:24' -> (0, 24); '100' -> (100, 101). End-exclusive, like python."""
    if ":" in spec:
        lo, _, hi = spec.partition(":")
        start = int(lo) if lo.strip() else 0
        stop = int(hi) if hi.strip() else n
    else:
        start = int(spec)
        stop = start + 1
    start, stop = max(0, start), min(n, stop)
    if start >= stop:
        raise SystemExit(f"empty frame range {spec!r} against {n} frames")
    return start, stop


def uniform_step(values: np.ndarray, what: str, tol: float) -> float:
    """The constant spacing of an axis, or a hard error explaining why there isn't one.

    O(1) sampling is only sound if the axis really is uniform, so this is a
    precondition of the format rather than a diagnostic.
    """
    if values.size < 2:
        raise SystemExit(f"{what} axis has {values.size} point(s); need at least 2")
    steps = np.diff(values.astype("float64"))
    spread = float(steps.max() - steps.min())
    if spread > tol:
        raise SystemExit(
            f"{what} axis is not uniform (steps span {steps.min():g}..{steps.max():g}); "
            "the npz format addresses it arithmetically and cannot represent that"
        )
    return float(steps.mean())


def plan_chunks(nt: int, frame_bytes: int, limit: int) -> list[tuple[int, int]]:
    """Time ranges, end-exclusive, each at most `limit` bytes of payload.

    Consecutive parts overlap by one frame. A sampler at time t needs the frames
    on either side of it, so without the overlap the interval between the last
    frame of one part and the first of the next would belong to no file at all --
    a hole at every boundary. Duplicating one frame costs a fraction of a percent
    and makes each part independently samplable across its whole span.

    A single frame over the limit cannot be split further -- the grid is not cut --
    so it is emitted alone and over budget rather than refused.
    """
    per_chunk = max(1, limit // max(frame_bytes, 1))
    if per_chunk >= nt:
        return [(0, nt)]

    chunks, start = [], 0
    while start < nt:
        stop = min(nt, start + per_chunk)
        chunks.append((start, stop))
        if stop >= nt:
            break
        # Step back one so this part's last frame is the next part's first.
        start = stop - 1 if per_chunk > 1 else stop
    return chunks


def chunk_path(dst: Path, index: int, total: int) -> Path:
    """'waves.npz' -> 'waves.npz' alone, or 'waves.000.npz', 'waves.001.npz', ...

    Zero-padded so the parts sort in time order in a shell glob, and the suffix
    is kept last so they still look like npz files to everything downstream.
    """
    if total == 1:
        return dst
    return dst.with_suffix(f".{index:03d}{dst.suffix}")


def field_members(
    *,
    payload: np.ndarray,
    times: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    dt: float,
    dlat: float,
    dlon: float,
    lon_wrap: bool,
    scale: float,
    offset: float,
    fill: int,
    part: int,
    nparts: int,
    part_t0: np.ndarray,
    part_nt: np.ndarray,
) -> dict:
    """The members of one output file. `times` is this part's slice of the axis.

    Scalars are int64/float64 0-d arrays; the C++ reader pulls them by name and
    never has to parse a string, so no text metadata is written at all.

    The manifest follows that rule instead of sitting in a sidecar file: every
    part carries the time span of *every* part, so opening any one of them
    reveals the whole layout -- which part covers a given instant, and whether
    more exist -- without a second format to parse or a file that can go
    missing. Part filenames are not stored; they follow from `chunk_path`.
    These members are additive and `version` stays 1, so a reader built before
    the split still loads a part as the plain field it also is.
    """
    return dict(
        version=np.int32(1),
        t0=np.int64(times[0]),
        dt=np.int64(round(dt)),
        nt=np.int64(times.size),
        lat0=np.float64(lat[0]),
        dlat=np.float64(dlat),
        nlat=np.int64(lat.size),
        lon0=np.float64(lon[0]),
        dlon=np.float64(dlon),
        nlon=np.int64(lon.size),
        lon_wrap=np.int32(1 if lon_wrap else 0),
        scale=np.float64(scale),
        offset=np.float64(offset),
        fill=np.int64(fill),
        # Embedded manifest: which part this is, and the span of each.
        part=np.int32(part),
        nparts=np.int32(nparts),
        part_t0=part_t0,
        part_nt=part_nt,
        overlap=np.int32(1 if nparts > 1 and part_nt[0] > 1 else 0),
        data=payload,
        # For python-side use and for verifying the arithmetic above; the C++
        # sampler derives its indices from the scalars and ignores these.
        time=times,
        lat=lat,
        lon=lon,
    )


def save_npz(path: Path, level: int, **members) -> None:
    """`np.savez_compressed` with the deflate level exposed. Level 0 stores.

    numpy hardcodes zlib's default (6), which is the wrong end of the curve for
    this payload: on quantised ERA5 fields levels 1 and 6 both land within half a
    percent of 2x compression, and 6 spends 20% longer getting there. That time
    is not free -- deflate is single-threaded and runs at ~70 MiB/s, so writing a
    512 MiB part takes several seconds, and it is the whole of the pause a caller
    sees between parts. Writers should also keep this off the critical path
    entirely by running it on a background thread; zlib releases the GIL.
    """
    kind = zipfile.ZIP_STORED if level == 0 else zipfile.ZIP_DEFLATED
    with zipfile.ZipFile(path, "w", kind, allowZip64=True,
                         compresslevel=None if level == 0 else level) as z:
        for name, value in members.items():
            # force_zip64: a part may legitimately exceed 4 GiB with --max-mib
            with z.open(f"{name}.npy", "w", force_zip64=True) as fh:
                np.lib.format.write_array(fh, np.asanyarray(value), allow_pickle=False)
