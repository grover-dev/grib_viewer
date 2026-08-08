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
`plan_chunks` for how the boundaries are placed.

The manifest is embedded rather than written alongside: each part carries
`part`, `nparts`, `part_t0` and `part_nt`, the last two being the first stamp
and frame count of *every* part. So opening any single part answers which file
covers a given instant, and parts cannot drift out of sync with an index that
lives somewhere else. Filenames are not stored -- they follow the `chunk_path`
pattern. All four members are additive, and `version` stays 1, so a reader that
predates the split still loads a part as the ordinary field it also is.
"""

from __future__ import annotations

import argparse
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

import grib_utils as gu
from npz_out import (
    U16_FILL,
    U16_MAX,
    chunk_path,
    field_members,
    parse_frames,
    plan_chunks,
    save_npz,
    uniform_step,
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
    out.add_argument("--max-mib", type=float, default=512.0, metavar="MiB",
                     help="largest payload to put in one file (default 512). A longer "
                          "selection is split along time into <name>.000.npz, "
                          "<name>.001.npz, ... which overlap by one frame; 0 disables "
                          "splitting")
    out.add_argument("--batch-mib", type=float, default=128.0, metavar="MiB",
                     help="how much of the time axis to decode per read (default 128). "
                          "Frames are read in batches of this size and dask decodes "
                          "the batch in parallel, so this is both the peak read "
                          "buffer and the unit of parallelism -- lower it on a "
                          "memory-tight machine, raise it on a fast disk")
    out.add_argument("--writers", type=int, default=2, metavar="N",
                     help="parts compressed in the background at once (default 2). "
                          "zlib runs at ~70 MiB/s on one core, so writing a 512 MiB "
                          "part takes several seconds during which nothing is read; "
                          "overlapping the writes with the next part's reads hides "
                          "that, at the cost of N payload buffers in flight rather "
                          "than one. 1 restores the old serial behaviour")
    out.add_argument("--compress-level", type=int, default=1, metavar="1-9",
                     choices=range(1, 10),
                     help="zlib level for the payload (default 1). Levels above 1 "
                          "are not worth their time on quantised fields: on ERA5 "
                          "radiation, 1 and 6 both compress ~2x")
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

    # How the selection is cut up. The grid is never split, so the unit of
    # division is one frame and the plan is fixed before anything is read --
    # which is what keeps the peak allocation below to one chunk, not one cube.
    itemsize = 2 if args.dtype == "u16" else 4
    frame_bytes = nlat * nlon * itemsize
    if args.max_mib < 0:
        raise SystemExit(f"--max-mib must be >= 0, got {args.max_mib}")
    limit = int(args.max_mib * (1 << 20))
    chunks = plan_chunks(nt, frame_bytes, limit) if limit else [(0, nt)]

    if args.batch_mib <= 0:
        raise SystemExit(f"--batch-mib must be > 0, got {args.batch_mib}")
    # Frames per read. Decoded frames are float32 whatever the payload type is,
    # so the budget is measured against that and not against `itemsize`.
    per_batch = max(1, int(args.batch_mib * (1 << 20)) // max(nlat * nlon * 4, 1))

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
    print(f"  reading     {per_batch} frame(s) per batch "
          f"({per_batch * nlat * nlon * 4 / (1 << 20):.0f} MiB, decoded in parallel)")
    if len(chunks) > 1:
        print(f"  writing     zlib level {args.compress_level if not args.no_compress else 0}"
              f", up to {min(args.writers, len(chunks))} part(s) compressed in the "
              f"background ({min(args.writers, len(chunks)) * frame_bytes * chunks[0][1] / (1 << 20):.0f}"
              " MiB of payload buffers)")
    total_bytes = nt * frame_bytes
    if len(chunks) > 1:
        print(f"  split       {len(chunks)} files of <= {args.max_mib:g} MiB "
              f"({total_bytes / (1 << 20):.1f} MiB total, {frame_bytes / (1 << 20):.2f} MiB "
              f"per frame, 1 frame of overlap between parts)")
    elif limit and frame_bytes > limit:
        print(f"  split       none possible: one frame is {frame_bytes / (1 << 20):.1f} MiB, "
              f"over the {args.max_mib:g} MiB limit")

    # --- read, convert, quantise -----------------------------------------
    # A batch at a time, never the whole cube: a year of global hourly frames is
    # ~36 GB as float32, so materialising it just to rescale it is out of the
    # question. The batch is what makes this fast as well as small -- one dask
    # compute over `per_batch` frames decodes them in parallel and pays the
    # scheduler once, where a frame-at-a-time walk pays it 8760 times over a
    # graph 8760 tasks long. See grib_utils.frame_block.
    scale, offset = 1.0, 0.0

    def batches(lo: int, hi: int):
        for start in range(lo, hi, per_batch):
            yield start, min(hi, start + per_batch)

    def read(lo: int, hi: int) -> np.ndarray:
        """Frames [lo, hi) as a writable float32 cube, freshly allocated."""
        return np.asarray(gu.frame_block(da, lo, hi).compute(), dtype="float32")

    if args.dtype == "u16":
        # Two passes. The quantiser needs the range up front, and holding the
        # float cube to avoid the second read is the very cost being dodged.
        lo_v, hi_v = np.inf, -np.inf
        for lo, hi in batches(0, nt):
            cube = read(lo, hi)
            finite = np.isfinite(cube)
            if finite.all():
                # The usual case, and worth its own branch: the masked gather
                # below copies the batch, min/max over the cube itself doesn't.
                lo_v, hi_v = min(lo_v, float(cube.min())), max(hi_v, float(cube.max()))
            elif finite.any():
                vals = cube[finite]
                lo_v, hi_v = min(lo_v, float(vals.min())), max(hi_v, float(vals.max()))
            print(f"\r  scanning    {hi}/{nt}", end="", flush=True)
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

    # --- write ------------------------------------------------------------
    # The member layout, and why the manifest is embedded rather than written
    # alongside, is in `npz_out.field_members`. Every part carries the span of
    # every part, so both spans are computed once here, up front.
    part_t0 = np.array([times[lo] for lo, _ in chunks], dtype="int64")
    part_nt = np.array([hi - lo for lo, hi in chunks], dtype="int64")

    dst = Path(args.npz)
    level = 0 if args.no_compress else args.compress_level
    if args.writers < 1:
        raise SystemExit(f"--writers must be >= 1, got {args.writers}")

    # Compression is serial within a part but the parts are independent files, so
    # it belongs on background threads -- zlib releases the GIL, and the reads
    # that would otherwise be stalled behind it are what fill the next buffer.
    # The semaphore is the memory bound: a part cannot start converting until a
    # writer has finished with one of the N buffers, so peak payload memory is N
    # parts and not however many the reader can run ahead by.
    slots = threading.Semaphore(args.writers)
    pool = ThreadPoolExecutor(max_workers=args.writers, thread_name_prefix="npz-write")
    pending = []
    done = 0

    def write_part(out_path: Path, members: dict) -> tuple[Path, int, int]:
        try:
            save_npz(out_path, level, **members)
            return out_path, members["data"].nbytes, out_path.stat().st_size
        finally:
            slots.release()

    for index, (lo, hi) in enumerate(chunks):
        span = hi - lo
        # Allocated per part, not per selection: this buffer is what the byte
        # limit actually bounds. Ownership passes to the writer thread below, so
        # the next iteration allocates a fresh one rather than reusing this.
        slots.acquire()
        payload = np.empty((span, nlat, nlon),
                           dtype="uint16" if args.dtype == "u16" else "float32")

        # The conversion is vectorised over the batch and done in place, so the
        # only arrays alive here are the batch, its finite mask and the payload.
        for a, b in batches(lo, hi):
            cube = read(a, b)
            if convert:
                cube /= np.float32(step_s)
            if args.dtype == "u16":
                finite = np.isfinite(cube)      # taken before cube is overwritten
                cube -= np.float32(offset)
                cube /= np.float32(scale)
                np.rint(cube, out=cube)
                np.clip(cube, 0, U16_MAX, out=cube)
                # `where=` on the negated mask, negated into itself, and a cast
                # straight into the payload: no copy of the batch is made at any
                # step, which is the difference between one batch of headroom
                # and three.
                np.copyto(cube, np.float32(U16_FILL),
                          where=np.logical_not(finite, out=finite))
            payload[a - lo:b - lo] = cube
            done += b - a
            print(f"\r  converting  {done}/{nt + len(chunks) - 1}", end="", flush=True)

        members = field_members(
            payload=payload, times=times[lo:hi], lat=lat, lon=lon,
            dt=dt, dlat=dlat, dlon=dlon, lon_wrap=lon_wrap,
            scale=scale, offset=offset,
            fill=U16_FILL if args.dtype == "u16" else -1,
            part=index, nparts=len(chunks), part_t0=part_t0, part_nt=part_nt,
        )

        out_path = chunk_path(dst, index, len(chunks))
        pending.append(pool.submit(write_part, out_path, members))
        del payload, members  # the writer owns the buffer now

    # Futures resolve in chunk order because they were submitted in it, so the
    # report below still lines up with `chunks`. A writer that raised does so
    # here, once, rather than being swallowed in its thread.
    if len(pending) > 1:
        print(f"\r  converting  {done}/{nt + len(chunks) - 1}, "
              f"finishing {min(args.writers, len(pending))} write(s)", end="", flush=True)
    written: list[tuple[Path, int, int]] = [f.result() for f in pending]
    pool.shutdown()
    print()

    raw_total = sum(raw for _, raw, _ in written)
    disk_total = sum(disk for _, _, disk in written)
    dtype_name = "uint16" if args.dtype == "u16" else "float32"
    print()
    for (path, raw, disk), (lo, hi) in zip(written, chunks):
        print(f"wrote {path}: {disk / 1e6:.1f} MB on disk"
              f"  ({raw / 1e6:.1f} MB in memory as {dtype_name})")
        print(f"      frames {lo}:{hi}  "
              f"{gu.fmt_time(np.datetime64(int(times[lo]), 's'))} .. "
              f"{gu.fmt_time(np.datetime64(int(times[hi - 1]), 's'))}")
    if len(written) > 1:
        print(f"\n{len(written)} parts, {disk_total / 1e6:.1f} MB on disk total"
              f"  ({raw_total / 1e6:.1f} MB in memory, largest part "
              f"{max(raw for _, raw, _ in written) / (1 << 20):.1f} MiB)")
        print("  each part also carries the span of every part (part, nparts, "
              "part_t0, part_nt)")
    if args.dtype == "u16":
        print(f"  quantised: value = raw * {scale:.6g} + {offset:.6g}"
              f"  (step {scale:.4g} {out_units or '?'})")
    print()


if __name__ == "__main__":
    main()
