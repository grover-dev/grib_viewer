"""Extract solar radiation from a GRIB into an .npz the C++ side can sample in O(1).

    uv run solar_npz.py data/18fdfe.../data.grib solar.npz --frames 0:48
    uv run solar_npz.py data/18fdfe.../data.grib atl.npz --bbox -40 5 20 60
    uv run solar_npz.py data/18fdfe.../data.grib solar.npz --list

The point of the output format is that *no search is needed to sample it*. ERA5
ships a regular lat/lon grid on a uniform hourly axis, so a (time, lat, lon)
query resolves by arithmetic:

    i = (t - t0) / dt        j = (lat - lat0) / dlat        k = (lon - lon0) / dlon

No k-d tree, no binary search, no index -- three divides and a strided load.
So the file stores those six constants as scalars, and the payload is one dense
C-order cube. The explicit `time`/`lat`/`lon` axes are also written, but only for
Python-side convenience and verification; boatforge::SolarField ignores them.

Two conversions happen on the way out, both because ERA5 stores radiation as an
accumulation rather than a rate:

* **J/m^2 -> W/m^2.** Each value is the energy accumulated over the preceding
  model step. Dividing by that step's length gives the mean irradiance, which is
  what a panel model actually wants.

* **The time axis is shifted back half a step.** A value stamped 12:00 covers
  11:00..12:00, so the mean irradiance it represents is centred on 11:30.
  Interpolating against the raw stamps biases every sample half an hour late.
  `--no-center` keeps the original stamps.

Values are quantised to uint16 with a scale/offset by default -- irradiance tops
out near 1400 W/m^2, so ~0.02 W/m^2 of resolution is far below the accuracy of
the underlying reanalysis, and it halves the file. `--dtype f32` stores the
floats verbatim.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import grib_utils as gu

# Preference order when --var isn't given. ssrd is what reaches a horizontal
# surface at sea level and so is the one a panel sees; the others are fallbacks
# for files that don't carry it (the 16 GB January request, for instance, has
# cdir and tsr but no ssrd).
SOLAR_VARS = ("ssrd", "cdir", "tsr")

ACCUMULATED_UNITS = ("j m**-2", "j/m2", "j m-2")

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


def pick_var(fields: dict, requested: str | None) -> str:
    if requested:
        if requested not in fields:
            raise SystemExit(
                f"{requested!r} not in this GRIB; it has: {', '.join(sorted(fields))}"
            )
        return requested
    for name in SOLAR_VARS:
        if name in fields:
            return name
    raise SystemExit(
        "no solar field found (looked for "
        f"{', '.join(SOLAR_VARS)}); this GRIB has: {', '.join(sorted(fields))}\n"
        "pass --var to choose one explicitly"
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("grib", help="source GRIB")
    p.add_argument("npz", nargs="?", help="output .npz (omit with --list)")
    p.add_argument("--list", action="store_true",
                   help="show the solar fields available and exit")
    p.add_argument("--var", help=f"field to extract (default: first of {'/'.join(SOLAR_VARS)})")

    sel = p.add_argument_group("selection (--frames and --start/--end are mutually exclusive)")
    sel.add_argument("--start", help="first time to keep, YYYY-MM-DD[THH:MM] (inclusive)")
    sel.add_argument("--end", help="last time to keep (inclusive)")
    sel.add_argument("--frames", help="frame index or range, 0-based end-exclusive: '0:24', '100'")
    sel.add_argument("--stride", type=int, default=1, help="keep every Nth step of the selection")
    sel.add_argument("--bbox", type=float, nargs=4, metavar=("W", "E", "S", "N"),
                     help="crop to this area, degrees in -180..180")
    sel.add_argument("--thin", type=int, default=1, metavar="N",
                     help="keep every Nth grid point in lat and lon (coarsens the grid)")

    out = p.add_argument_group("output")
    out.add_argument("--dtype", choices=("u16", "f32"), default="u16",
                     help="payload type: quantised uint16 (default) or raw float32")
    out.add_argument("--no-compress", action="store_true",
                     help="store members uncompressed (bigger, loads faster)")
    out.add_argument("--no-center", action="store_true",
                     help="keep ERA5's end-of-accumulation stamps instead of centring them")
    out.add_argument("--keep-units", action="store_true",
                     help="do not convert J/m^2 to W/m^2")
    args = p.parse_args()

    if args.frames and (args.start or args.end):
        raise SystemExit("choose either --frames or --start/--end, not both")
    if not args.npz and not args.list:
        raise SystemExit("give an output path, or --list to see what's available")

    src = Path(args.grib)
    if not src.exists():
        raise SystemExit(f"no such file: {src}")

    fields = gu.open_grib(str(src))

    if args.list:
        print(f"\n{src}\nfields carrying radiation:")
        for k in sorted(fields):
            a = fields[k].attrs
            mark = "*" if k in SOLAR_VARS else " "
            units = a.get("units", "?")
            if "j m" in units.lower() or "w m" in units.lower():
                print(f" {mark} {k:<8} {a.get('long_name', '?')[:46]:<46} {units}")
        print("\n  * = picked automatically, in the order listed\n")
        return

    da = fields[pick_var(fields, args.var)]
    var = da.name
    units = (da.attrs.get("units") or "").strip().lower()

    # --- time selection ---------------------------------------------------
    # Accumulation length comes from the *source* spacing, before any striding:
    # each message still covers one model step no matter how many we keep.
    all_times = da["time"].values
    step_s = uniform_step(all_times.astype("datetime64[s]").astype("int64"),
                          "time", tol=0.0)

    if args.frames:
        lo, hi = parse_frames(args.frames, all_times.size)
        da = da.isel(time=slice(lo, hi))
        how = f"frames {lo}:{hi}"
    else:
        start = np.datetime64(args.start) if args.start else all_times[0]
        end = np.datetime64(args.end) if args.end else all_times[-1]
        if end < start:
            raise SystemExit(f"end {gu.fmt_time(end)} is before start {gu.fmt_time(start)}")
        da = da.sel(time=slice(start, end))
        how = f"{gu.fmt_time(start)} .. {gu.fmt_time(end)}"
    if args.stride > 1:
        da = da.isel(time=slice(None, None, args.stride))
    if not da.sizes["time"]:
        raise SystemExit(f"no time steps matched {how}")

    # --- area selection ---------------------------------------------------
    if args.bbox:
        west, east, south, north = args.bbox
        if not (-90 <= south < north <= 90):
            raise SystemExit(f"bad latitudes: need -90 <= S < N <= 90, got S={south} N={north}")
        if west >= east:
            raise SystemExit(
                f"bad longitudes: need W < E, got W={west} E={east}. "
                "This tool does not cross the dateline; slice_grib.py does."
            )
        (da,) = gu.subset_area({var: da}, tuple(args.bbox)).values()

    ydim, xdim = gu.ydim_of(da), gu.xdim_of(da)

    # Decimation, not averaging: it keeps the axes exactly uniform, which the
    # format requires. A thinned global axis only stays wrappable when the
    # stride divides the column count -- 1440/8 does, 1440/7 does not -- and the
    # wrap flag below is computed from the result, so an awkward stride simply
    # yields a non-wrapping grid rather than a subtly wrong seam.
    if args.thin > 1:
        da = da.isel({ydim: slice(None, None, args.thin),
                      xdim: slice(None, None, args.thin)})
        if da.sizes[ydim] < 2 or da.sizes[xdim] < 2:
            raise SystemExit(f"--thin {args.thin} leaves fewer than 2 points per axis")

    # Latitude is stored ascending so both spatial axes have a positive step and
    # the C++ sampler needs one sign convention, not two. ERA5 scans north->south.
    lat = da[ydim].values.astype("float64")
    flip_lat = lat.size > 1 and lat[0] > lat[-1]
    if flip_lat:
        da = da.isel({ydim: slice(None, None, -1)})
        lat = lat[::-1]
    lon = da[xdim].values.astype("float64")

    dlat = uniform_step(lat, "latitude", tol=1e-6)
    dlon = uniform_step(lon, "longitude", tol=1e-6)

    times = da["time"].values.astype("datetime64[s]").astype("int64")
    dt = uniform_step(times, "time", tol=0.0)

    convert = not args.keep_units and units in ACCUMULATED_UNITS
    center = not args.no_center and convert
    if center:
        times = times - int(round(step_s / 2))

    nt, nlat, nlon = da.sizes["time"], lat.size, lon.size

    # A global axis wraps: the cell after the last one is the first one again,
    # so a query at 179.9 interpolates across the seam instead of falling off it.
    lon_wrap = abs(nlon * dlon - 360.0) < 1e-6

    print(f"\n{src}")
    print(f"  field       {var}  ({da.attrs.get('long_name', '?')}) [{units or '?'}]")
    print(f"  selected    {how}, stride {args.stride}"
          + (f", bbox {tuple(args.bbox)}" if args.bbox else "")
          + (f", thin {args.thin}" if args.thin > 1 else ""))
    print(f"  time        {nt} frames, {gu.fmt_time(np.datetime64(int(times[0]), 's'))}"
          f" .. {gu.fmt_time(np.datetime64(int(times[-1]), 's'))}, step {dt / 3600:g} h")
    if center:
        print(f"              (centred: stamps moved back {step_s / 2 / 60:g} min to the "
              "middle of each accumulation window)")
    print(f"  grid        {nlat}x{nlon} @ {abs(dlat):g}deg x {abs(dlon):g}deg   "
          f"lat {lat[0]:g}..{lat[-1]:g}   lon {lon[0]:g}..{lon[-1]:g}"
          + ("   (wraps)" if lon_wrap else ""))

    # --- read, convert, quantise -----------------------------------------
    # One frame at a time: the source is dask-backed and a global cube is ~1 GB
    # as float32, so materialising it whole just to rescale it is avoidable.
    payload = np.empty((nt, nlat, nlon),
                       dtype="uint16" if args.dtype == "u16" else "float32")
    scale, offset = 1.0, 0.0

    if args.dtype == "u16":
        # Two passes. The quantiser needs the range up front, and holding the
        # float cube to avoid the second read is the very cost being dodged.
        lo_v, hi_v = np.inf, -np.inf
        for i in range(nt):
            frame = np.asarray(da.isel(time=i).values, dtype="float32")
            finite = np.isfinite(frame)
            if finite.any():
                lo_v = min(lo_v, float(frame[finite].min()))
                hi_v = max(hi_v, float(frame[finite].max()))
            print(f"\r  scanning    {i + 1}/{nt}", end="", flush=True)
        print()
        if not np.isfinite(lo_v):
            raise SystemExit("every value is NaN; nothing to write")
        if convert:
            lo_v, hi_v = lo_v / step_s, hi_v / step_s
        # Irradiance is floored at zero physically; anchoring there keeps the
        # quantisation grid aligned to a meaningful origin.
        offset = min(0.0, lo_v)
        scale = max(hi_v - offset, 1e-6) / U16_MAX

    for i in range(nt):
        frame = np.asarray(da.isel(time=i).values, dtype="float32")
        if convert:
            frame = frame / np.float32(step_s)
        if args.dtype == "u16":
            q = np.rint((frame - offset) / scale)
            np.clip(q, 0, U16_MAX, out=q)
            payload[i] = np.where(np.isfinite(frame), q, U16_FILL).astype("uint16")
        else:
            payload[i] = frame
        print(f"\r  converting  {i + 1}/{nt}", end="", flush=True)
    print()

    # --- write ------------------------------------------------------------
    # Scalars are int64/float64 0-d arrays; the C++ reader pulls them by name and
    # never has to parse a string, so no text metadata is written at all.
    members = dict(
        version=np.int32(1),
        t0=np.int64(times[0]),
        dt=np.int64(round(dt)),
        nt=np.int64(nt),
        lat0=np.float64(lat[0]),
        dlat=np.float64(dlat),
        nlat=np.int64(nlat),
        lon0=np.float64(lon[0]),
        dlon=np.float64(dlon),
        nlon=np.int64(nlon),
        lon_wrap=np.int32(1 if lon_wrap else 0),
        scale=np.float64(scale),
        offset=np.float64(offset),
        fill=np.int64(U16_FILL if args.dtype == "u16" else -1),
        data=payload,
        # For python-side use and for verifying the arithmetic above; the C++
        # sampler derives its indices from the scalars and ignores these.
        time=times,
        lat=lat,
        lon=lon,
    )

    save = np.savez if args.no_compress else np.savez_compressed
    dst = Path(args.npz)
    save(dst, **members)

    raw = payload.nbytes
    disk = dst.stat().st_size
    print(f"\nwrote {dst}: {disk / 1e6:.1f} MB on disk"
          f"  ({raw / 1e6:.1f} MB in memory as {payload.dtype})")
    if args.dtype == "u16":
        print(f"  quantised: value = raw * {scale:.6g} + {offset:.6g}"
              f"  (step {scale:.4g} W/m^2)")
    print()


if __name__ == "__main__":
    main()
