"""Report what's inside a GRIB: fields, time axis, grids, and what it'd cost to load.

Reads headers and coordinates only, so it stays fast on a multi-GB file.

    uv run grib_info.py data/data.grib
    uv run grib_info.py data/data.grib --times      # also list every valid time
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import grib_utils as gu


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return f"{n:.1f} {unit}"
        n /= 1024


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("grib", help="path to a GRIB file")
    p.add_argument("--times", action="store_true", help="list every valid time")
    args = p.parse_args()

    path = Path(args.grib)
    if not path.exists():
        raise SystemExit(f"no such file: {path}")

    # open_grib only — NOT derive_fields. Deriving wind speed runs np.hypot over
    # the full cubes, which materializes gigabytes just to describe the file.
    fields = gu.open_grib(str(path))
    times = gu.available_times(fields)
    n = len(times)

    # --- file ---------------------------------------------------------
    print(f"\n{path}")
    print(f"  on disk        {human(path.stat().st_size)}")

    # --- time axis ----------------------------------------------------
    print(f"\ntime: {n} frames")
    print(f"  first  [0]     {gu.fmt_time(times[0])}")
    print(f"  last   [{n - 1}]{' ' * max(0, 5 - len(str(n - 1)))}  {gu.fmt_time(times[-1])}")
    if n > 1:
        gaps = np.diff(times) / np.timedelta64(1, "h")
        step = float(np.median(gaps))
        print(f"  spacing        {step:g} h"
              + ("" if gaps.min() == gaps.max() else
                 f"  (uneven: {gaps.min():g}..{gaps.max():g} h)"))
        span = (times[-1] - times[0]) / np.timedelta64(1, "h")
        print(f"  span           {span:g} h ({span / 24:.1f} days)")

    if args.times:
        for i, t in enumerate(times):
            print(f"    [{i:>4}] {gu.fmt_time(t)}")

    # --- fields -------------------------------------------------------
    print(f"\nfields: {len(fields)}")
    print(f"  {'name':<10} {'long name':<42} {'units':>10}  {'grid':>11} {'frames':>6} {'full load':>10}")
    total = 0
    for k, da in sorted(fields.items()):
        ydim, xdim = gu.ydim_of(da), gu.xdim_of(da)
        ny, nx = da.sizes[ydim], da.sizes[xdim]
        nt = da.sizes["time"]
        cost = gu.estimate_bytes({k: da}, nt)
        total += cost
        label = (da.attrs.get("long_name") or "?")[:40]
        units = (da.attrs.get("units") or "-")[:10]
        print(f"  {k:<10} {label:<42} {units:>10}  {ny:>5}x{nx:<5} {nt:>6} {human(cost):>10}")

    # --- grids --------------------------------------------------------
    grids = {}
    for k, da in fields.items():
        ydim, xdim = gu.ydim_of(da), gu.xdim_of(da)
        lat, lon = da[ydim].values, da[xdim].values
        key = (da.sizes[ydim], da.sizes[xdim],
               round(float(lat.max()), 2), round(float(lat.min()), 2),
               round(float(lon.min()), 2), round(float(lon.max()), 2))
        grids.setdefault(key, []).append(k)

    print(f"\ngrids: {len(grids)}")
    for (ny, nx, north, south, west, east), members in grids.items():
        res = abs(north - south) / (ny - 1) if ny > 1 else 0
        print(f"  {ny}x{nx} @ {res:g}°   lat {south}..{north}   lon {west}..{east}")
        print(f"    {', '.join(sorted(members))}")

    print(f"\nloading everything costs roughly {human(total)} of RAM")
    if total > 4e9:
        print("  -> slice it down first:  uv run slice_grib.py "
              f"{path} out.grib --frames 0:24 --bbox W E S N")
    print()


if __name__ == "__main__":
    main()
