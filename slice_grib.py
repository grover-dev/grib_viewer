"""Slice a GRIB down to the times you care about, writing a new GRIB.

Standalone counterpart to the viewer's area export: this one only cuts the time
axis. Select either by timestamp or by frame index — a frame is a position in
the file's sorted list of distinct valid times, i.e. the same numbering the
viewer's slider shows.

Works on the raw GRIB messages, so nothing is decoded into memory and all the
original metadata survives into the output.

    # what's in the file?
    uv run slice_grib.py data/data.grib --list

    # by timestamp (inclusive)
    uv run slice_grib.py data/data.grib data/jan.grib --start 2025-01-05 --end 2025-01-12

    # by frame index: 0-based, end-exclusive, like a python slice
    uv run slice_grib.py data/data.grib data/first_day.grib --frames 0:24
    uv run slice_grib.py data/data.grib data/one.grib      --frames 100

    # thin a run: every 6th step
    uv run slice_grib.py data/data.grib data/thin.grib --stride 6

    # crop to an area too (W E S N, in -180..180); combines with any of the above
    uv run slice_grib.py data/data.grib data/uk.grib --frames 0:24 --bbox -12 5 48 62
"""

from __future__ import annotations

import argparse
from pathlib import Path

import eccodes as ec
import numpy as np


def _valid_time(h) -> np.datetime64:
    """Valid time of a message.

    Uses validity*, not dataDate/dataTime: accumulated ERA5 fields (tp, ssrd,
    tsr, cdir) carry a step offset from their reference time, so their reference
    time is not the hour the data describes.
    """
    vd = ec.codes_get(h, "validityDate")  # YYYYMMDD
    vt = ec.codes_get(h, "validityTime")  # HHMM
    return np.datetime64(
        f"{vd // 10000:04d}-{vd // 100 % 100:02d}-{vd % 100:02d}"
        f"T{vt // 100:02d}:{vt % 100:02d}"
    )


def scan_times(path: str) -> np.ndarray:
    """Every distinct valid time in the file, sorted. Cheap — headers only."""
    times = set()
    with open(path, "rb") as fh:
        while (h := ec.codes_grib_new_from_file(fh)) is not None:
            try:
                times.add(_valid_time(h))
            finally:
                ec.codes_release(h)
    if not times:
        raise SystemExit(f"no GRIB messages found in {path}")
    return np.array(sorted(times))


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


def _fmt(t) -> str:
    return np.datetime_as_string(t, unit="m") + "Z"


def _crop_message(h, bbox: tuple) -> bool:
    """Crop one message's grid to bbox = (W, E, S, N) in -180..180 degrees.

    Returns False if the message doesn't overlap the box (caller should skip it).

    Longitude is the fiddly part. ERA5 ships on a 0..360 grid while the box is
    given in -180..180, and a box may straddle either seam (Greenwich on a 0..360
    grid, or the dateline). Measuring every longitude as an offset east of the
    box's west edge -- (lon - W) % 360 -- makes the wanted columns one contiguous
    ascending run in every one of those cases, so the same code path handles all
    of them; the columns are then gathered in that order.
    """
    west, east, south, north = bbox
    ni, nj = ec.codes_get(h, "Ni"), ec.codes_get(h, "Nj")
    lat_first = ec.codes_get(h, "latitudeOfFirstGridPointInDegrees")
    lat_last = ec.codes_get(h, "latitudeOfLastGridPointInDegrees")
    lon_first = ec.codes_get(h, "longitudeOfFirstGridPointInDegrees")
    lon_last = ec.codes_get(h, "longitudeOfLastGridPointInDegrees")

    lats = np.linspace(lat_first, lat_last, nj)
    lons = np.linspace(lon_first, lon_last, ni)
    values = ec.codes_get_values(h).reshape(nj, ni)

    jm = (lats >= south) & (lats <= north)          # lat keeps the source scan order
    width = (east - west) % 360 or 360.0            # W==E means "all longitudes"
    offset = (lons - west) % 360                    # degrees east of the box's west edge
    im = offset <= width
    if not jm.any() or not im.any():
        return False

    cols = np.flatnonzero(im)
    cols = cols[np.argsort(offset[cols])]           # contiguous, ascending from west
    rows = np.flatnonzero(jm)

    values = values[np.ix_(rows, cols)]
    out_lats = lats[rows]
    out_lons = west + offset[cols]                  # monotonic; may run past 180
    nj, ni = values.shape

    ec.codes_set(h, "Ni", ni)
    ec.codes_set(h, "Nj", nj)
    ec.codes_set(h, "latitudeOfFirstGridPointInDegrees", float(out_lats[0]))
    ec.codes_set(h, "latitudeOfLastGridPointInDegrees", float(out_lats[-1]))
    ec.codes_set(h, "longitudeOfFirstGridPointInDegrees", float(out_lons[0]))
    ec.codes_set(h, "longitudeOfLastGridPointInDegrees", float(out_lons[-1]))
    if ni > 1:
        ec.codes_set(h, "iDirectionIncrementInDegrees",
                     float(abs(out_lons[1] - out_lons[0])))
    if nj > 1:
        ec.codes_set(h, "jDirectionIncrementInDegrees",
                     float(abs(out_lats[1] - out_lats[0])))
    ec.codes_set_values(h, values.ravel())
    return True


def slice_grib(src: str, dst: str, keep: set[np.datetime64], bbox: tuple | None = None) -> int:
    """Copy messages whose valid time is in `keep`, cropped to bbox if given."""
    written = 0
    with open(src, "rb") as fin, open(dst, "wb") as fout:
        while (h := ec.codes_grib_new_from_file(fin)) is not None:
            try:
                if _valid_time(h) not in keep:
                    continue
                if bbox is not None:
                    grid = ec.codes_get(h, "gridType")
                    if grid != "regular_ll":
                        raise SystemExit(
                            f"--bbox only supports regular lat/lon grids, got {grid!r}"
                        )
                    if not _crop_message(h, bbox):
                        continue
                ec.codes_write(h, fout)
                written += 1
            finally:
                ec.codes_release(h)
    if not written:
        raise SystemExit("no messages matched the requested times/area")
    return written


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("src", help="source GRIB")
    p.add_argument("dst", nargs="?", help="output GRIB (omit with --list)")
    p.add_argument("--list", action="store_true", help="show the time axis and exit")

    sel = p.add_argument_group("selection (timestamp and frame are mutually exclusive)")
    sel.add_argument("--start", help="first time to keep, YYYY-MM-DD[THH:MM] (inclusive)")
    sel.add_argument("--end", help="last time to keep (inclusive)")
    sel.add_argument("--frames", help="frame index or range, 0-based end-exclusive: '0:24', '100'")
    sel.add_argument("--stride", type=int, default=1, help="keep every Nth step of the selection")
    sel.add_argument("--bbox", type=float, nargs=4, metavar=("W", "E", "S", "N"),
                     help="crop to this area, degrees in -180..180 (may cross the dateline)")
    args = p.parse_args()

    if args.frames and (args.start or args.end):
        raise SystemExit("choose either --frames or --start/--end, not both")

    times = scan_times(args.src)
    n = len(times)

    if args.list or not args.dst:
        step = np.median(np.diff(times)) if n > 1 else None
        print(f"{args.src}: {n} time steps")
        print(f"  first  [0]      {_fmt(times[0])}")
        print(f"  last   [{n - 1}]{'':>{max(0, 6 - len(str(n - 1)))}} {_fmt(times[-1])}")
        if step is not None:
            print(f"  spacing         {step / np.timedelta64(1, 'h'):g} h (median)")
        if not args.dst and not args.list:
            raise SystemExit("\ngive an output path to write a slice")
        return

    # --- pick the times to keep ---------------------------------------
    if args.frames:
        lo, hi = parse_frames(args.frames, n)
        chosen = times[lo:hi]
        how = f"frames {lo}:{hi}"
    else:
        start = np.datetime64(args.start) if args.start else times[0]
        end = np.datetime64(args.end) if args.end else times[-1]
        if end < start:
            raise SystemExit(f"end {_fmt(end)} is before start {_fmt(start)}")
        chosen = times[(times >= start) & (times <= end)]
        how = f"{_fmt(start)} .. {_fmt(end)}"

    if args.stride > 1:
        chosen = chosen[:: args.stride]
    if not len(chosen):
        raise SystemExit(f"no time steps matched {how}")

    bbox = tuple(args.bbox) if args.bbox else None
    if bbox and not (-90 <= bbox[2] < bbox[3] <= 90):
        raise SystemExit(f"bad latitudes: need -90 <= S < N <= 90, got S={bbox[2]} N={bbox[3]}")

    written = slice_grib(args.src, args.dst, set(chosen.tolist()), bbox)

    src_mb = Path(args.src).stat().st_size / 1e6
    dst_mb = Path(args.dst).stat().st_size / 1e6
    print(f"selected {how}, stride {args.stride}" + (f", bbox {bbox}" if bbox else ""))
    print(f"kept {len(chosen)}/{n} steps: {_fmt(chosen[0])} .. {_fmt(chosen[-1])}")
    print(f"wrote {args.dst}: {written} messages, {dst_mb:.1f} MB (from {src_mb:.1f} MB)")


if __name__ == "__main__":
    main()
