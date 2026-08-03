"""Extract one GRIB field into an .npz the C++ side can sample in O(1).

    uv run grib_npz.py data/18fdfe.../data.grib --list
    uv run grib_npz.py data/18fdfe.../data.grib ssrd solar.npz --frames 0:48
    uv run grib_npz.py data/18fdfe.../data.grib swh  waves.npz --bbox -40 5 20 60

The point of the output format is that *no search is needed to sample it*. ERA5
ships a regular lat/lon grid on a uniform time axis, so a (time, lat, lon)
query resolves by arithmetic:

    i = (t - t0) / dt        j = (lat - lat0) / dlat        k = (lon - lon0) / dlon

No k-d tree, no binary search, no index -- three divides and a strided load.
So the file stores those six constants as scalars, and the payload is one dense
C-order cube. The explicit `time`/`lat`/`lon` axes are also written, but only for
python-side convenience and verification; boatforge::NpzField ignores them.

Any field in the file works: radiation, wind components, wave height, pressure,
temperature. Two things are decided per field rather than hardcoded:

* **Accumulations become rates.** ERA5 stores radiation as energy accumulated
  over the model step (J/m^2), not as a flux. Those are divided by the step
  length to give W/m^2, which is what a panel model wants. Fields already stored
  as instantaneous values (u10, swh, t2m, ...) are passed through untouched.

* **Accumulation stamps are re-centred.** A value stamped 12:00 covers
  11:00..12:00, so the mean it represents belongs at 11:30. Interpolating
  against the raw stamps biases every sample half a step late. Instantaneous
  fields are already stamped at the instant they describe and are left alone.

`--accumulated` overrides the guess, which matters for fields that accumulate
without saying so in their units -- total precipitation is metres, not a rate.

Values are quantised to uint16 with a scale/offset by default, which halves the
file for a resolution loss far below the accuracy of the underlying reanalysis.
`--dtype f32` stores the floats verbatim.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import grib_utils as gu

# Units that mean "accumulated over the step" for the purposes of the two
# conversions above. ERB5 writes radiation this way; see --accumulated for the
# fields that accumulate without advertising it.
ACCUMULATED_UNITS = ("j m**-2", "j/m2", "j m-2")

# What an accumulated field becomes once divided by its step length.
RATE_UNITS = {"j m**-2": "W m**-2", "j/m2": "W m**-2", "j m-2": "W m**-2"}

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


def list_fields(path: Path, fields: dict) -> None:
    """Every field in the file, with what it would cost to extract."""
    print(f"\n{path}")
    times = gu.available_times(fields)
    print(f"  {len(times)} frames, {gu.fmt_time(times[0])} .. {gu.fmt_time(times[-1])}\n")
    print(f"  {'name':<8} {'long name':<44} {'units':>10}  {'grid':>11} {'frames':>7}")
    for name, da in sorted(fields.items()):
        ydim, xdim = gu.ydim_of(da), gu.xdim_of(da)
        units = (da.attrs.get("units") or "-")[:10]
        mark = "*" if units.strip().lower() in ACCUMULATED_UNITS else " "
        print(f" {mark}{name:<8} {(da.attrs.get('long_name') or '?')[:44]:<44} "
              f"{units:>10}  {da.sizes[ydim]:>5}x{da.sizes[xdim]:<5} {da.sizes['time']:>7}")
    print("\n  * accumulated over the step: converted to a rate and re-centred in time\n")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("grib", help="source GRIB")
    p.add_argument("var", nargs="?", help="field to extract; see --list")
    p.add_argument("npz", nargs="?", help="output .npz")
    p.add_argument("--list", action="store_true",
                   help="show every field in the file and exit")

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
    out.add_argument("--accumulated", choices=("auto", "yes", "no"), default="auto",
                     help="treat the field as accumulated over the step. 'auto' (default) "
                          "decides from the units; 'yes' for accumulations that do not "
                          "advertise it, such as total precipitation")
    out.add_argument("--no-compress", action="store_true",
                     help="store members uncompressed (bigger, loads faster)")
    out.add_argument("--no-center", action="store_true",
                     help="keep the source stamps instead of centring accumulations")
    out.add_argument("--keep-units", action="store_true",
                     help="do not divide accumulations by the step length")
    args = p.parse_args()

    if args.frames and (args.start or args.end):
        raise SystemExit("choose either --frames or --start/--end, not both")

    src = Path(args.grib)
    if not src.exists():
        raise SystemExit(f"no such file: {src}")

    fields = gu.open_grib(str(src))

    if args.list:
        list_fields(src, fields)
        return
    if not args.var or not args.npz:
        raise SystemExit(
            "give a field and an output path, or --list to see what is available\n"
            f"  fields in this file: {', '.join(sorted(fields))}"
        )
    if args.var not in fields:
        raise SystemExit(
            f"{args.var!r} is not in this GRIB.\n"
            f"  available: {', '.join(sorted(fields))}"
        )

    da = fields[args.var]
    units = (da.attrs.get("units") or "").strip().lower()

    # Whether the two accumulation conversions apply. Deciding once, here, keeps
    # the unit change and the time shift from drifting apart -- they describe the
    # same property of the field and must agree.
    accumulated = units in ACCUMULATED_UNITS if args.accumulated == "auto" \
        else args.accumulated == "yes"

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
        (da,) = gu.subset_area({args.var: da}, tuple(args.bbox)).values()

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

    convert = accumulated and not args.keep_units
    center = accumulated and not args.no_center
    if center:
        times = times - int(round(step_s / 2))
    out_units = RATE_UNITS.get(units, units) if convert else units

    nt, nlat, nlon = da.sizes["time"], lat.size, lon.size

    # A global axis wraps: the cell after the last one is the first one again,
    # so a query at 179.9 interpolates across the seam instead of falling off it.
    lon_wrap = abs(nlon * dlon - 360.0) < 1e-6

    print(f"\n{src}")
    print(f"  field       {args.var}  ({da.attrs.get('long_name', '?')}) [{units or '?'}]"
          + (f" -> [{out_units}]" if convert else ""))
    print(f"  treated as  {'accumulated over the step' if accumulated else 'instantaneous'}"
          + ("" if args.accumulated == "auto" else f"  (--accumulated {args.accumulated})"))
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
        # Anchored on the field's own minimum rather than on zero, so a narrow
        # band far from the origin -- sea surface temperature in kelvin, say --
        # spends its 16 bits on the range it actually occupies.
        offset = lo_v
        scale = max(hi_v - lo_v, 1e-12) / U16_MAX

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
              f"  (step {scale:.4g} {out_units or '?'})")
    print()


if __name__ == "__main__":
    main()
