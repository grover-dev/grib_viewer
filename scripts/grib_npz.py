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

A long selection is split across several outputs rather than written as one
cube, so neither this script nor its reader ever holds more than `--max-mib` of
payload at once. The split is along time -- the grid is never cut, so every part
is a complete field over a shorter window, addressed by the same arithmetic. See
`npz_out.plan_chunks` for how the boundaries are placed.

The manifest is embedded rather than written alongside: each part carries
`part`, `nparts`, `part_t0` and `part_nt`, the last two being the first stamp
and frame count of *every* part. So opening any single part answers which file
covers a given instant, and parts cannot drift out of sync with an index that
lives somewhere else. Filenames are not stored -- they follow the
`npz_out.chunk_path` pattern. All four members are additive, and `version` stays
1, so a reader that predates the split still loads a part as the ordinary field
it also is.

Everything above describes the *format*, not this script, and none of it is
implemented here: `npz_out` holds the members, the split, the quantiser and the
writer, so `netcdf4_npz.py` produces the identical thing from a NetCDF4 source.
What is left below is GRIB: opening the file, finding a field in it, narrowing
it, and handing back the frames.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import grib_utils as gu
from npz_out import (
    Grid,
    add_output_args,
    add_selection_args,
    describe,
    parse_frames,
    plan_layout,
    uniform_step,
    write_field,
)

# Units that mean "accumulated over the step" for the purposes of the two
# conversions above. ERB5 writes radiation this way; see --accumulated for the
# fields that accumulate without advertising it.
ACCUMULATED_UNITS = ("j m**-2", "j/m2", "j m-2")

# What an accumulated field becomes once divided by its step length.
RATE_UNITS = {"j m**-2": "W m**-2", "j/m2": "W m**-2", "j m-2": "W m**-2"}


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

    add_selection_args(p)
    add_output_args(
        p,
        accumulated_help="treat the field as accumulated over the step. 'auto' (default) "
                         "decides from the units; 'yes' for accumulations that do not "
                         "advertise it, such as total precipitation",
        batch_help="how much of the time axis to decode per read (default 128). "
                   "Frames are read in batches of this size and dask decodes "
                   "the batch in parallel, so this is both the peak read "
                   "buffer and the unit of parallelism -- lower it on a "
                   "memory-tight machine, raise it on a fast disk",
    )
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

    # A global axis wraps: the cell after the last one is the first one again,
    # so a query at 179.9 interpolates across the seam instead of falling off it.
    lon_wrap = abs(lon.size * dlon - 360.0) < 1e-6

    grid = Grid(times=times, lat=lat, lon=lon,
                dt=dt, dlat=dlat, dlon=dlon, lon_wrap=lon_wrap)
    layout = plan_layout(grid, dtype=args.dtype,
                         max_mib=args.max_mib, batch_mib=args.batch_mib)
    level = 0 if args.no_compress else args.compress_level

    print(f"\n{src}")
    print(f"  field       {args.var}  ({da.attrs.get('long_name', '?')}) [{units or '?'}]"
          + (f" -> [{out_units}]" if convert else ""))
    print(f"  treated as  {'accumulated over the step' if accumulated else 'instantaneous'}"
          + ("" if args.accumulated == "auto" else f"  (--accumulated {args.accumulated})"))
    print(f"  selected    {how}, stride {args.stride}"
          + (f", bbox {tuple(args.bbox)}" if args.bbox else "")
          + (f", thin {args.thin}" if args.thin > 1 else ""))
    describe(grid, layout, max_mib=args.max_mib, writers=args.writers, level=level,
             note=(f"(centred: stamps moved back {step_s / 2 / 60:g} min to the "
                   "middle of each accumulation window)") if center else "",
             reading_note=", decoded in parallel")

    # A batch at a time, never the whole cube: a year of global hourly frames is
    # ~36 GB as float32, so materialising it just to rescale it is out of the
    # question. The batch is what makes this fast as well as small -- one dask
    # compute over a batch decodes its frames in parallel and pays the scheduler
    # once, where a frame-at-a-time walk pays it 8760 times over a graph 8760
    # tasks long. See grib_utils.frame_block.
    def read(lo: int, hi: int) -> np.ndarray:
        """Frames [lo, hi) as a writable float32 cube, freshly allocated."""
        return np.asarray(gu.frame_block(da, lo, hi).compute(), dtype="float32")

    write_field(Path(args.npz), read, grid, layout,
                level=level, writers=args.writers,
                divide_by=step_s if convert else None, units=out_units)


if __name__ == "__main__":
    main()
