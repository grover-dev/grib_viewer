"""Slice a GRIB down to the times and area you care about, writing a new GRIB.

Select the time axis either by timestamp or by frame index — a frame is a
position in the file's sorted list of distinct valid times, i.e. the same
numbering the viewer's slider shows. Add --bbox to crop the area too.

Works on the raw GRIB messages (see grib_utils), so nothing is decoded into
memory and all the original metadata survives into the output.

    # what's in the file?  (grib_info.py tells you much more)
    uv run slice_grib.py data/data.grib --list

    # by timestamp (inclusive)
    uv run slice_grib.py data/data.grib data/jan.grib --start 2025-01-05 --end 2025-01-12

    # by frame index: 0-based, end-exclusive, like a python slice
    uv run slice_grib.py data/data.grib data/first_day.grib --frames 0:24
    uv run slice_grib.py data/data.grib data/one.grib      --frames 100

    # thin a run: every 6th step
    uv run slice_grib.py data/data.grib data/thin.grib --stride 6

    # crop to an area (W E S N, in -180..180); combines with any of the above.
    # W > E crosses the dateline.
    uv run slice_grib.py data/data.grib data/uk.grib  --frames 0:24 --bbox -12 5 48 62
    uv run slice_grib.py data/data.grib data/pac.grib --frames 0:24 --bbox 170 -170 -20 20
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import grib_utils as gu


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
                     help="crop to this area, degrees in -180..180 (W > E crosses the dateline)")
    args = p.parse_args()

    if args.frames and (args.start or args.end):
        raise SystemExit("choose either --frames or --start/--end, not both")

    times = gu.scan_times(args.src)
    n = len(times)

    if args.list or not args.dst:
        step = np.median(np.diff(times)) if n > 1 else None
        print(f"{args.src}: {n} time steps")
        print(f"  first  [0]      {gu.fmt_time(times[0])}")
        print(f"  last   [{n - 1}]{'':>{max(0, 6 - len(str(n - 1)))}} {gu.fmt_time(times[-1])}")
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
            raise SystemExit(f"end {gu.fmt_time(end)} is before start {gu.fmt_time(start)}")
        chosen = times[(times >= start) & (times <= end)]
        how = f"{gu.fmt_time(start)} .. {gu.fmt_time(end)}"

    if args.stride > 1:
        chosen = chosen[:: args.stride]
    if not len(chosen):
        raise SystemExit(f"no time steps matched {how}")

    bbox = tuple(args.bbox) if args.bbox else None
    if bbox and not (-90 <= bbox[2] < bbox[3] <= 90):
        raise SystemExit(f"bad latitudes: need -90 <= S < N <= 90, got S={bbox[2]} N={bbox[3]}")

    written = gu.write_subset(args.src, args.dst, set(chosen.tolist()), bbox)

    src_mb = Path(args.src).stat().st_size / 1e6
    dst_mb = Path(args.dst).stat().st_size / 1e6
    print(f"selected {how}, stride {args.stride}" + (f", bbox {bbox}" if bbox else ""))
    print(f"kept {len(chosen)}/{n} steps: {gu.fmt_time(chosen[0])} .. {gu.fmt_time(chosen[-1])}")
    print(f"wrote {args.dst}: {written} messages, {dst_mb:.1f} MB (from {src_mb:.1f} MB)")


if __name__ == "__main__":
    main()
