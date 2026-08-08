"""The parts of the .npz field format that do not care where the data came from.

`grib_npz.py` and `netcdf4_npz.py` read completely different files -- indexed
GRIB messages through ecCodes, HDF5 datasets through h5py -- but they write the
same thing, and the C++ reader has exactly one parser for it. Anything that
would have to be changed in both writers at once lives here instead: the layout
of the members, the arithmetic the layout promises, the rules for splitting a
long selection across files, and the flags that control all three.

What is left in each script is exactly what is particular to its format: how the
file is opened, how a field is found in it, how a selection turns into a read,
and how that read is decoded. That last piece is handed back here as a `read(lo,
hi)` callable returning float32 frames, and everything downstream of it --
quantising, splitting, compressing, reporting -- happens once, here, for both.

The format itself is documented in `grib_npz.py`; this module is its
implementation, not its explanation.
"""

from __future__ import annotations

import threading
import zipfile
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
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


def fmt_time(t) -> str:
    """A valid time, rendered the one way every tool in this project shows it.

    Same one line as `grib_utils.fmt_time`, repeated rather than imported: that
    module pulls in ecCodes and dask, which a NetCDF reader has no use for.
    """
    return np.datetime_as_string(t, unit="m") + "Z"


# ---------------------------------------------------------------------------
# The command line these tools share
# ---------------------------------------------------------------------------

def add_selection_args(p) -> None:
    """--start/--end/--frames/--stride/--bbox/--thin, identical in both tools.

    Both writers narrow a source the same way and had better keep meaning the
    same thing by it: a flag that quietly differed between them would be found
    by a caller comparing two outputs, which is the worst place to find it.
    """
    sel = p.add_argument_group("selection (--frames and --start/--end are mutually exclusive)")
    sel.add_argument("--start", help="first time to keep, YYYY-MM-DD[THH:MM] (inclusive)")
    sel.add_argument("--end", help="last time to keep (inclusive)")
    sel.add_argument("--frames", help="frame index or range, 0-based end-exclusive: '0:24', '100'")
    sel.add_argument("--stride", type=int, default=1, help="keep every Nth step of the selection")
    sel.add_argument("--bbox", type=float, nargs=4, metavar=("W", "E", "S", "N"),
                     help="crop to this area, degrees in -180..180")
    sel.add_argument("--thin", type=int, default=1, metavar="N",
                     help="keep every Nth grid point in lat and lon (coarsens the grid)")
    return sel


def add_output_args(p, *, batch_help: str, accumulated_help: str):
    """Everything about the payload and the files it goes into.

    Only two helps are passed in, and both describe something genuinely
    different between the tools: what a batch costs to read, and where the
    "this is an accumulation" guess comes from.
    """
    out = p.add_argument_group("output")
    out.add_argument("--dtype", choices=("u16", "f32"), default="u16",
                     help="payload type: quantised uint16 (default) or raw float32")
    out.add_argument("--accumulated", choices=("auto", "yes", "no"), default="auto",
                     help=accumulated_help)
    out.add_argument("--max-mib", type=float, default=512.0, metavar="MiB",
                     help="largest payload to put in one file (default 512). A longer "
                          "selection is split along time into <name>.000.npz, "
                          "<name>.001.npz, ... which overlap by one frame; 0 disables "
                          "splitting")
    out.add_argument("--batch-mib", type=float, default=128.0, metavar="MiB", help=batch_help)
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
    return out


# ---------------------------------------------------------------------------
# The selection, once both tools have finished deciding what it is
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Grid:
    """The six constants the format addresses a sample with, and the axes.

    `times` is int64 epoch seconds and already carries whatever shift the field
    needed; `lat` ascends. Both writers reduce their source to this before
    anything is read, and nothing downstream needs to know which one built it.
    """
    times: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    dt: float
    dlat: float
    dlon: float
    lon_wrap: bool

    @property
    def nt(self) -> int:
        return int(self.times.size)

    @property
    def nlat(self) -> int:
        return int(self.lat.size)

    @property
    def nlon(self) -> int:
        return int(self.lon.size)


@dataclass(frozen=True)
class Layout:
    """How the selection is cut into files and how much is read at a time.

    Fixed before anything is read, which is what keeps peak memory to a part
    and a batch rather than to the whole cube.
    """
    dtype: str                      # 'u16' or 'f32'
    chunks: list[tuple[int, int]]
    per_batch: int
    frame_bytes: int
    limit: int

    @property
    def payload_dtype(self) -> str:
        return "uint16" if self.dtype == "u16" else "float32"


def plan_layout(grid: Grid, *, dtype: str, max_mib: float, batch_mib: float) -> Layout:
    """Where the file boundaries fall and how many frames a read covers."""
    if max_mib < 0:
        raise SystemExit(f"--max-mib must be >= 0, got {max_mib}")
    if batch_mib <= 0:
        raise SystemExit(f"--batch-mib must be > 0, got {batch_mib}")

    # The grid is never split, so the unit of division is one frame.
    frame_bytes = grid.nlat * grid.nlon * (2 if dtype == "u16" else 4)
    limit = int(max_mib * (1 << 20))
    chunks = plan_chunks(grid.nt, frame_bytes, limit) if limit else [(0, grid.nt)]

    # Frames per read. Decoded frames are float32 whatever the payload type is,
    # so the budget is measured against that and not against the payload's.
    per_batch = max(1, int(batch_mib * (1 << 20)) // max(grid.nlat * grid.nlon * 4, 1))
    return Layout(dtype=dtype, chunks=chunks, per_batch=per_batch,
                  frame_bytes=frame_bytes, limit=limit)


def describe(grid: Grid, layout: Layout, *, max_mib: float, writers: int, level: int,
             note: str = "", grid_note: str = "", reading_note: str = "") -> None:
    """The half of the banner that is about the output rather than the source.

    Each tool prints its own first few lines -- which field, from what, treated
    how -- and then hands over here, so the geometry, the read plan and the
    split are reported in one voice whichever tool produced them.
    """
    nt, nlat, nlon = grid.nt, grid.nlat, grid.nlon
    chunks, frame_bytes = layout.chunks, layout.frame_bytes

    print(f"  time        {nt} frames, {fmt_time(np.datetime64(int(grid.times[0]), 's'))}"
          f" .. {fmt_time(np.datetime64(int(grid.times[-1]), 's'))}, "
          f"step {grid.dt / 3600:g} h")
    if note:
        print(f"              {note}")
    print(f"  grid        {nlat}x{nlon} @ {abs(grid.dlat):g}deg x {abs(grid.dlon):g}deg   "
          f"lat {grid.lat[0]:g}..{grid.lat[-1]:g}   lon {grid.lon[0]:g}..{grid.lon[-1]:g}"
          + ("   (wraps)" if grid.lon_wrap else "") + grid_note)
    print(f"  reading     {layout.per_batch} frame(s) per batch "
          f"({layout.per_batch * nlat * nlon * 4 / (1 << 20):.0f} MiB{reading_note})")
    if len(chunks) > 1:
        print(f"  writing     zlib level {level}, up to {min(writers, len(chunks))} part(s) "
              "compressed in the background "
              f"({min(writers, len(chunks)) * frame_bytes * chunks[0][1] / (1 << 20):.0f}"
              " MiB of payload buffers)")
        print(f"  split       {len(chunks)} files of <= {max_mib:g} MiB "
              f"({nt * frame_bytes / (1 << 20):.1f} MiB total, "
              f"{frame_bytes / (1 << 20):.2f} MiB per frame, 1 frame of overlap "
              "between parts)")
    elif layout.limit and frame_bytes > layout.limit:
        print(f"  split       none possible: one frame is {frame_bytes / (1 << 20):.1f} MiB, "
              f"over the {max_mib:g} MiB limit")


# ---------------------------------------------------------------------------
# Reading, quantising and writing -- the same for both sources
# ---------------------------------------------------------------------------

# Frames [lo, hi) of the selection as a writable float32 cube, freshly
# allocated, latitude ascending, missing values NaN. The one thing each tool
# still owns, because it is the only step that knows what the source is.
Reader = Callable[[int, int], np.ndarray]


def _quantiser(read: Reader, grid: Grid, layout: Layout,
               divide_by: float | None) -> tuple[float, float]:
    """The scale and offset that fit this field's own range into 0..65534.

    Two passes over the source. The quantiser needs the range up front, and
    holding the float cube to avoid the second read is the very cost being
    dodged.
    """
    lo_v, hi_v = np.inf, -np.inf
    for lo, hi in _batches(0, grid.nt, layout.per_batch):
        cube = read(lo, hi)
        finite = np.isfinite(cube)
        if finite.all():
            # The usual case, and worth its own branch: the masked gather below
            # copies the batch, min/max over the cube itself doesn't.
            lo_v, hi_v = min(lo_v, float(cube.min())), max(hi_v, float(cube.max()))
        elif finite.any():
            vals = cube[finite]
            lo_v, hi_v = min(lo_v, float(vals.min())), max(hi_v, float(vals.max()))
        print(f"\r  scanning    {hi}/{grid.nt}", end="", flush=True)
    print()
    if not np.isfinite(lo_v):
        raise SystemExit("every value is NaN; nothing to write")
    if divide_by:
        lo_v, hi_v = lo_v / divide_by, hi_v / divide_by
    # Anchored on the field's own minimum rather than on zero, so a narrow band
    # far from the origin -- sea surface temperature in kelvin, say -- spends
    # its 16 bits on the range it actually occupies.
    return max(hi_v - lo_v, 1e-12) / U16_MAX, lo_v


def _batches(lo: int, hi: int, per_batch: int):
    for start in range(lo, hi, per_batch):
        yield start, min(hi, start + per_batch)


def write_field(dst: Path, read: Reader, grid: Grid, layout: Layout, *,
                level: int, writers: int, divide_by: float | None = None,
                units: str = "") -> list[tuple[Path, int, int]]:
    """Read the selection through `read`, and write it as one file or several.

    `divide_by` is the step length in seconds when the field is an accumulation
    being turned into a rate, and None when the values are already what they
    should be. It reaches the payload in two places -- the range the quantiser
    is built from and the values it quantises -- which is why it is one
    argument rather than a flag and a number that could disagree.

    Returns (path, payload bytes, file bytes) per part, in chunk order.
    """
    if writers < 1:
        raise SystemExit(f"--writers must be >= 1, got {writers}")

    nt, nlat, nlon = grid.nt, grid.nlat, grid.nlon
    chunks = layout.chunks
    quantise = layout.dtype == "u16"
    scale, offset = _quantiser(read, grid, layout, divide_by) if quantise else (1.0, 0.0)

    # Every part carries the span of every part, so both spans are computed once
    # here, up front. See `field_members` for why they are in the file at all.
    part_t0 = np.array([grid.times[lo] for lo, _ in chunks], dtype="int64")
    part_nt = np.array([hi - lo for lo, hi in chunks], dtype="int64")

    # Compression is serial within a part but the parts are independent files, so
    # it belongs on background threads -- zlib releases the GIL, and the reads
    # that would otherwise be stalled behind it are what fill the next buffer.
    # The semaphore is the memory bound: a part cannot start converting until a
    # writer has finished with one of the N buffers, so peak payload memory is N
    # parts and not however many the reader can run ahead by.
    slots = threading.Semaphore(writers)
    pool = ThreadPoolExecutor(max_workers=writers, thread_name_prefix="npz-write")
    pending = []
    done = 0

    def write_part(out_path: Path, members: dict) -> tuple[Path, int, int]:
        try:
            save_npz(out_path, level, **members)
            return out_path, members["data"].nbytes, out_path.stat().st_size
        finally:
            slots.release()

    try:
        for index, (lo, hi) in enumerate(chunks):
            # Allocated per part, not per selection: this buffer is what the byte
            # limit actually bounds. Ownership passes to the writer thread below,
            # so the next iteration allocates a fresh one rather than reusing this.
            slots.acquire()
            payload = np.empty((hi - lo, nlat, nlon), dtype=layout.payload_dtype)

            # The conversion is vectorised over the batch and done in place, so
            # the only arrays alive here are the batch, its finite mask and the
            # payload.
            for a, b in _batches(lo, hi, layout.per_batch):
                cube = read(a, b)
                if divide_by:
                    cube /= np.float32(divide_by)
                if quantise:
                    finite = np.isfinite(cube)   # taken before cube is overwritten
                    cube -= np.float32(offset)
                    cube /= np.float32(scale)
                    np.rint(cube, out=cube)
                    np.clip(cube, 0, U16_MAX, out=cube)
                    # `where=` on the negated mask, negated into itself, and a
                    # cast straight into the payload: no copy of the batch is
                    # made at any step, which is the difference between one
                    # batch of headroom and three.
                    np.copyto(cube, np.float32(U16_FILL),
                              where=np.logical_not(finite, out=finite))
                payload[a - lo:b - lo] = cube
                done += b - a
                print(f"\r  converting  {done}/{nt + len(chunks) - 1}", end="", flush=True)

            members = field_members(
                payload=payload, times=grid.times[lo:hi], lat=grid.lat, lon=grid.lon,
                dt=grid.dt, dlat=grid.dlat, dlon=grid.dlon, lon_wrap=grid.lon_wrap,
                scale=scale, offset=offset, fill=U16_FILL if quantise else -1,
                part=index, nparts=len(chunks), part_t0=part_t0, part_nt=part_nt,
            )
            pending.append(pool.submit(write_part, chunk_path(dst, index, len(chunks)),
                                       members))
            del payload, members    # the writer owns the buffer now

        # Futures resolve in chunk order because they were submitted in it, so
        # the report below still lines up with `chunks`. A writer that raised
        # does so here, once, rather than being swallowed in its thread.
        if len(pending) > 1:
            print(f"\r  converting  {done}/{nt + len(chunks) - 1}, "
                  f"finishing {min(writers, len(pending))} write(s)", end="", flush=True)
        written = [fut.result() for fut in pending]
    finally:
        pool.shutdown()
    print()

    _report(written, chunks, grid, layout, scale, offset, units)
    return written


def _report(written: list[tuple[Path, int, int]], chunks: list[tuple[int, int]],
            grid: Grid, layout: Layout, scale: float, offset: float, units: str) -> None:
    """What was written, where, and how to read the numbers back out of it."""
    print()
    for (path, raw, disk), (lo, hi) in zip(written, chunks):
        print(f"wrote {path}: {disk / 1e6:.1f} MB on disk"
              f"  ({raw / 1e6:.1f} MB in memory as {layout.payload_dtype})")
        print(f"      frames {lo}:{hi}  "
              f"{fmt_time(np.datetime64(int(grid.times[lo]), 's'))} .. "
              f"{fmt_time(np.datetime64(int(grid.times[hi - 1]), 's'))}")
    if len(written) > 1:
        disk_total = sum(disk for _, _, disk in written)
        raw_total = sum(raw for _, raw, _ in written)
        print(f"\n{len(written)} parts, {disk_total / 1e6:.1f} MB on disk total"
              f"  ({raw_total / 1e6:.1f} MB in memory, largest part "
              f"{max(raw for _, raw, _ in written) / (1 << 20):.1f} MiB)")
        print("  each part also carries the span of every part (part, nparts, "
              "part_t0, part_nt)")
    if layout.dtype == "u16":
        print(f"  quantised: value = raw * {scale:.6g} + {offset:.6g}"
              f"  (step {scale:.4g} {units or '?'})")
    print()
