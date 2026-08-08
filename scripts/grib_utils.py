"""Shared GRIB handling for the viewer, the slicer and the info tool.

Everything here works on raw GRIB messages via ecCodes. There is no cfgrib in the
read path, and that is the point:

* **Opening** builds a message index -- one header pass recording what each
  message is, when it is valid for, and its byte offset -- then caches it. On a
  16 GB file that is ~5s cold and a fraction of a second warm. cfgrib's
  open_datasets reconstructs the entire cube geometry up front and takes ~140s on
  the same file, every single run.

* **Reading a frame** is a seek to that offset plus one message decode, wrapped
  in dask so the xarray API downstream (.sel, subsetting, unit conversion) still
  works and stays lazy. Frames are decoded only when something asks for them.
  A tool that walks the whole time axis should pull batches through
  `frame_block` rather than slicing frame by frame; see its docstring for why.

* **Subsetting** (write_subset) copies messages verbatim. Going through xarray
  would be lossy -- cfgrib's writer needs the original GRIB_* attrs intact, and
  unit conversion destroys them.
"""

from __future__ import annotations

import hashlib
import os
import pickle
import warnings
from pathlib import Path

import dask
import dask.array
import eccodes as ec
import numpy as np
import xarray as xr

Y_NAMES = ("latitude", "lat", "y")
X_NAMES = ("longitude", "lon", "x")

# where cfgrib's message indexes are kept (override with BOATFORGE_CACHE)
CACHE_DIR = Path(os.environ.get("BOATFORGE_CACHE", Path(__file__).parent / ".cache"))


def fmt_time(t) -> str:
    """A valid time, rendered the one way every tool in this project shows it."""
    return np.datetime_as_string(t, unit="m") + "Z"


def ydim_of(da: xr.DataArray) -> str:
    return next(d for d in da.dims if d in Y_NAMES)


def xdim_of(da: xr.DataArray) -> str:
    return next(d for d in da.dims if d in X_NAMES)


# ---------------------------------------------------------------------------
# Message layer — raw GRIB, nothing decoded
# ---------------------------------------------------------------------------


def message_valid_time(h) -> np.datetime64:
    """Valid time of a message: the hour it actually describes.

    Uses validity*, not dataDate/dataTime. ERA5 accumulations (tp, ssrd, tsr,
    cdir) are stored against a reference time plus a forecast step, so their
    reference time is a different hour from the one the data is about.
    """
    vd = ec.codes_get(h, "validityDate")  # YYYYMMDD
    vt = ec.codes_get(h, "validityTime")  # HHMM
    return np.datetime64(
        f"{vd // 10000:04d}-{vd // 100 % 100:02d}-{vd % 100:02d}"
        f"T{vt // 100:02d}:{vt % 100:02d}"
    )


def scan_times(path: str) -> np.ndarray:
    """Every distinct valid time in the file, sorted. Reads headers only."""
    times = set()
    with open(path, "rb") as fh:
        while (h := ec.codes_grib_new_from_file(fh)) is not None:
            try:
                times.add(message_valid_time(h))
            finally:
                ec.codes_release(h)
    if not times:
        raise SystemExit(f"no GRIB messages found in {path}")
    return np.array(sorted(times))


def crop_message(h, bbox: tuple) -> bool:
    """Crop one message's grid to bbox = (W, E, S, N), degrees in -180..180.

    Returns False if the message doesn't overlap the box.

    Longitude is the fiddly part. ERA5 ships on a 0..360 grid while the box is
    given in -180..180, and a box may straddle either seam (Greenwich on a 0..360
    grid, or the dateline when W > E). Measuring each longitude as an offset east
    of the box's west edge -- (lon - W) % 360 -- makes the wanted columns one
    contiguous ascending run in all of those cases, so a single code path covers
    them; the columns are then gathered in that order.
    """
    west, east, south, north = bbox
    ni, nj = ec.codes_get(h, "Ni"), ec.codes_get(h, "Nj")
    lats = np.linspace(
        ec.codes_get(h, "latitudeOfFirstGridPointInDegrees"),
        ec.codes_get(h, "latitudeOfLastGridPointInDegrees"), nj,
    )
    lons = np.linspace(
        ec.codes_get(h, "longitudeOfFirstGridPointInDegrees"),
        ec.codes_get(h, "longitudeOfLastGridPointInDegrees"), ni,
    )
    values = ec.codes_get_values(h).reshape(nj, ni)

    jm = (lats >= south) & (lats <= north)   # lat keeps the source scan order
    width = (east - west) % 360 or 360.0     # W == E means "every longitude"
    offset = (lons - west) % 360             # degrees east of the box's west edge
    im = offset <= width
    if not jm.any() or not im.any():
        return False

    rows = np.flatnonzero(jm)
    cols = np.flatnonzero(im)
    cols = cols[np.argsort(offset[cols])]    # contiguous, ascending from west

    values = values[np.ix_(rows, cols)]
    out_lats, out_lons = lats[rows], west + offset[cols]  # monotonic; may pass 180
    nj, ni = values.shape

    ec.codes_set(h, "Ni", ni)
    ec.codes_set(h, "Nj", nj)
    ec.codes_set(h, "latitudeOfFirstGridPointInDegrees", float(out_lats[0]))
    ec.codes_set(h, "latitudeOfLastGridPointInDegrees", float(out_lats[-1]))
    ec.codes_set(h, "longitudeOfFirstGridPointInDegrees", float(out_lons[0]))
    ec.codes_set(h, "longitudeOfLastGridPointInDegrees", float(out_lons[-1]))
    if ni > 1:
        ec.codes_set(h, "iDirectionIncrementInDegrees", float(abs(out_lons[1] - out_lons[0])))
    if nj > 1:
        ec.codes_set(h, "jDirectionIncrementInDegrees", float(abs(out_lats[1] - out_lats[0])))
    ec.codes_set_values(h, values.ravel())
    return True


def write_subset(src: str, dst: str, keep=None, bbox: tuple | None = None) -> int:
    """Copy messages to a new GRIB, filtered by time and cropped to an area.

    `keep` is a set of valid times (None keeps every time). `bbox` is
    (W, E, S, N) or None. Returns the number of messages written.
    """
    written = 0
    with open(src, "rb") as fin, open(dst, "wb") as fout:
        while (h := ec.codes_grib_new_from_file(fin)) is not None:
            try:
                if keep is not None and message_valid_time(h) not in keep:
                    continue
                if bbox is not None:
                    grid = ec.codes_get(h, "gridType")
                    if grid != "regular_ll":
                        raise SystemExit(
                            f"area subsetting only supports regular lat/lon grids, got {grid!r}"
                        )
                    if not crop_message(h, bbox):
                        continue
                ec.codes_write(h, fout)
                written += 1
            finally:
                ec.codes_release(h)
    if not written:
        raise SystemExit("no messages matched the requested times/area")
    return written


# ---------------------------------------------------------------------------
# xarray layer — decoded fields, lazily indexed
# ---------------------------------------------------------------------------


def coord_tag(dim: str, value) -> str:
    """Readable suffix for a layer split out along an extra dim (e.g. a level).

    Timedeltas become hours, not raw nanoseconds: a coordinate printed as
    '10800000000000 nanoseconds' is unusable as a variable name.
    """
    if np.issubdtype(np.asarray(value).dtype, np.timedelta64):
        return f"{dim}{int(np.asarray(value) / np.timedelta64(1, 'h'))}h"
    v = np.asarray(value).item()
    return f"{dim}{v:g}" if isinstance(v, float) else f"{dim}{v}"


def normalize_lons(da: xr.DataArray) -> xr.DataArray:
    """Put a *global* 0..360 grid onto -180..180, and leave everything else alone.

    ERA5 ships global fields on 0..360, which renders with the Atlantic torn down
    the middle unless rolled.

    A regional grid must NOT be touched. A Pacific window runs e.g. 170..190;
    wrapping that into -180..180 and sorting scatters it into two clumps with a
    hole between them, and imshow -- which assumes uniform spacing -- would smear
    it across the map. Such a grid is already contiguous in its own frame, so it
    is left alone and the viewer re-centres the projection instead.
    """
    xdim = next((d for d in da.dims if d in X_NAMES), None)
    if xdim is None:
        return da
    lons = da[xdim].values
    if lons.max() <= 180.0:
        return da

    step = float(np.median(np.diff(lons))) if lons.size > 1 else 0.0
    if lons.size > 1 and (lons.max() - lons.min() + abs(step)) < 359.9:
        return da  # regional window: already contiguous in its own frame

    da = da.assign_coords({xdim: ((lons + 180.0) % 360.0) - 180.0})
    return da.sortby(xdim)


def _index_key(path: str) -> str:
    """Cache key for a file: path + size + mtime.

    Deliberately not cfgrib's scheme. cfgrib compares the index's mtime against
    the GRIB's and rebuilds if the index looks older -- which never succeeds on a
    file whose mtime is in the future (a zip extracted across a timezone will do
    it, as this project's own sample does), so it silently re-indexes every run.
    Hashing the stat instead is immune: a file that has not changed produces the
    same key no matter what the clock says.
    """
    st = os.stat(path)
    stamp = f"{os.path.abspath(path)}|{st.st_size}|{st.st_mtime_ns}"
    return hashlib.sha1(stamp.encode()).hexdigest()[:16]


def _time_key(t) -> int:
    """A valid time as whole seconds, so a lookup can't miss on datetime64 units.

    The index keys frames by datetime64[m]; xarray hands back datetime64[ns].
    Both round-trip through seconds exactly, and an int is an unambiguous key.
    """
    return int(np.asarray(t).astype("datetime64[s]").astype("int64"))


def _as_index(idx: np.ndarray):
    """`idx` as a slice if it is an arithmetic progression, else unchanged.

    Every spatial selection in this project is a stride -- a bbox crop, a thin,
    the latitude flip -- so the gather below is nearly always a view rather than
    a fancy-index copy. The exception is a rolled longitude axis, which is a
    genuine permutation and stays an array.
    """
    idx = np.asarray(idx, dtype="int64")
    if idx.size == 0:
        return idx
    step = int(idx[1] - idx[0]) if idx.size > 1 else 1
    if step == 0 or not np.array_equal(idx, idx[0] + step * np.arange(idx.size)):
        return idx
    stop = int(idx[-1]) + step
    # stop < 0 means a descending run that ends at column 0; slice() reads a
    # negative stop as "from the end", so it has to be None instead.
    return slice(int(idx[0]), stop if stop >= 0 else None, step)


def _read_frame(path: str, offset: int, shape: tuple[int, int]) -> np.ndarray:
    """Decode one message's values by seeking straight to it.

    This is the whole data path: a frame costs one seek plus one message decode,
    no matter how big the file is.
    """
    with open(path, "rb") as fh:
        fh.seek(offset)
        h = ec.codes_grib_new_from_file(fh)
        if h is None:
            raise IOError(f"no GRIB message at offset {offset} in {path}")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ec.codes_set_double(h, "missingValue", float("nan"))
            return ec.codes_get_values(h).reshape(shape).astype("float32")
        finally:
            ec.codes_release(h)


def _read_frame_sel(path: str, offset: int, shape: tuple[int, int], rows, cols) -> np.ndarray:
    """One frame, cropped to `rows`/`cols` of the source grid, as a fresh array.

    The crop is applied here rather than downstream so a cropped batch costs the
    cropped size: the copy is what lets the full decoded message be freed
    immediately instead of being pinned alive by a view for as long as the batch
    is held.
    """
    frame = _read_frame(path, offset, shape)
    if isinstance(rows, slice) and isinstance(cols, slice):
        return np.ascontiguousarray(frame[rows, cols])
    r = np.arange(shape[0])[rows] if isinstance(rows, slice) else rows
    c = np.arange(shape[1])[cols] if isinstance(cols, slice) else cols
    return frame[np.ix_(r, c)]


def index_grib(path: str) -> dict:
    """One header pass: what is in the file, and where.

    Records, per message, the variable it belongs to, the hour it is valid for,
    its grid, and its byte offset. That is everything needed to serve any frame
    on demand -- and it costs a single sweep of the headers (~5s on 16 GB),
    because the packed values are skipped entirely.

    This replaces cfgrib's open_datasets, which reconstructs the whole cube
    geometry up front and takes ~140s on the same file.
    """
    fields: dict[str, dict] = {}

    with open(path, "rb") as fh:
        while True:
            offset = fh.tell()
            h = ec.codes_grib_new_from_file(fh)
            if h is None:
                break
            try:
                # cfVarName is the CF-style name (u10), shortName is the GRIB one
                # (10u). cfgrib exposes the former, so use it too -- otherwise
                # every downstream reference to u10/t2m breaks.
                short = ec.codes_get(h, "shortName")
                try:
                    name = ec.codes_get(h, "cfVarName") or short
                except ec.KeyValueNotFoundError:
                    name = short
                if name in ("unknown", ""):
                    name = short

                level_type = ec.codes_get(h, "typeOfLevel")
                level = ec.codes_get(h, "level")

                # one field per level, so each is independently renderable
                key = name if level_type in ("surface", "meanSea", "entireAtmosphere") \
                    or level in (0, 1) else f"{name}_{level_type}{level}"

                nj, ni = ec.codes_get(h, "Nj"), ec.codes_get(h, "Ni")
                grid = (
                    nj, ni,
                    ec.codes_get(h, "latitudeOfFirstGridPointInDegrees"),
                    ec.codes_get(h, "latitudeOfLastGridPointInDegrees"),
                    ec.codes_get(h, "longitudeOfFirstGridPointInDegrees"),
                    ec.codes_get(h, "longitudeOfLastGridPointInDegrees"),
                )

                field = fields.setdefault(key, {
                    "grid": grid,
                    "attrs": {
                        "long_name": ec.codes_get(h, "name"),
                        "units": ec.codes_get(h, "units"),
                        "paramId": ec.codes_get(h, "paramId"),
                        "shortName": short,
                    },
                    "frames": {},   # valid time -> offset
                })
                # validityDate/Time already accounts for the forecast step, so
                # accumulations land on the hour they describe with no folding,
                # no rectangular (time x step) grid, and no phantom frames.
                field["frames"].setdefault(message_valid_time(h), offset)
            finally:
                ec.codes_release(h)

    if not fields:
        raise SystemExit(f"no GRIB messages found in {path}")
    return fields


def _load_index(path: str) -> dict:
    """The index, from cache if the file hasn't changed since we built it."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"{_index_key(path)}.index"
    if cache.exists():
        try:
            with open(cache, "rb") as fh:
                return pickle.load(fh)
        except Exception:
            cache.unlink(missing_ok=True)  # corrupt or stale format: rebuild

    index = index_grib(path)
    tmp = cache.with_suffix(".tmp")
    with open(tmp, "wb") as fh:
        pickle.dump(index, fh, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(cache)  # atomic: a killed run can't leave a half-written index
    return index


def open_grib(path: str, chunks: dict | None = None) -> dict[str, xr.DataArray]:
    """Flatten a GRIB into {name: DataArray(time, y, x)}, keeping every field.

    Backed by the message index rather than cfgrib: opening reads headers only
    (and on a second open, not even those -- the index is cached), and each frame
    is decoded from its file offset when something actually asks for it. The
    arrays are dask-backed, one task per frame, so all the xarray downstream
    (.sel, subsetting, unit conversion) keeps working and stays lazy.
    """
    index = _load_index(path)
    fields: dict[str, xr.DataArray] = {}

    for key, field in index.items():
        times = np.array(sorted(field["frames"]))
        nj, ni, lat0, lat1, lon0, lon1 = field["grid"]
        lats = np.linspace(lat0, lat1, nj)
        lons = np.linspace(lon0, lon1, ni)

        # one dask task per frame: nothing is read until a frame is asked for
        blocks = [
            dask.array.from_delayed(
                dask.delayed(_read_frame)(path, field["frames"][t], (nj, ni)),
                shape=(nj, ni),
                dtype="float32",
            )
            for t in times
        ]
        data = dask.array.stack(blocks, axis=0)  # -> (time, y, x), one chunk per frame

        da = xr.DataArray(
            data,
            dims=("time", "latitude", "longitude"),
            coords={
                "time": times, "latitude": lats, "longitude": lons,
                # Where each point sits in the *source message*. These ride along
                # through every selection xarray applies -- a bbox crop, a thin,
                # the latitude flip, the longitude roll below -- so `frame_block`
                # can reproduce the same selection as a plain numpy gather
                # without re-deriving it from coordinate values.
                "_row": ("latitude", np.arange(nj)),
                "_col": ("longitude", np.arange(ni)),
            },
            # Everything frame_block needs to read a frame itself. Copied, not
            # aliased: the index it comes from is cached and shared.
            attrs=dict(field["attrs"], _source={
                "path": path,
                "shape": (nj, ni),
                "offsets": {_time_key(t): o for t, o in field["frames"].items()},
            }),
            name=key,
        )
        fields[key] = normalize_lons(da)

    return fields


def frame_block(da: xr.DataArray, lo: int = 0, hi: int | None = None):
    """Frames [lo, hi) of `da` as a dask array whose graph is only that long.

    Slicing the array returned by `open_grib` does *not* give you this. That one
    carries a task per frame in the file, and dask culls the whole graph on every
    compute -- so pulling frames one at a time costs O(nt) of scheduling per
    frame and O(nt^2) overall. On a year of hourly ERA5 (8760 frames) that is
    ~840 ms of bookkeeping to decode a 6 ms frame, and it is why extracting a
    long run appeared to be I/O bound when almost none of it was I/O.

    Rebuilding a graph over just the wanted frames removes the scheduling from
    the inner loop entirely, and the batch is still read lazily and in parallel
    by dask's threaded scheduler -- peak memory is one batch, set by the caller.

    The selection already applied to `da` (bbox, thin, flip, roll) is carried by
    the `_row`/`_col` coordinates and re-applied inside each worker. A DataArray
    without them -- a derived field, say -- falls back to slicing its own graph.
    """
    hi = da.sizes["time"] if hi is None else hi
    src = da.attrs.get("_source")
    if src is None or "_row" not in da.coords or "_col" not in da.coords:
        return da.isel(time=slice(lo, hi)).data

    rows, cols = _as_index(da["_row"].values), _as_index(da["_col"].values)
    shape = tuple(src["shape"])
    ny, nx = da.sizes[ydim_of(da)], da.sizes[xdim_of(da)]

    blocks = [
        dask.array.from_delayed(
            dask.delayed(_read_frame_sel)(
                src["path"], src["offsets"][_time_key(t)], shape, rows, cols),
            shape=(ny, nx),
            dtype="float32",
        )
        for t in da["time"].values[lo:hi]
    ]
    if not blocks:  # an empty range is a legal slice; dask.stack won't take one
        return dask.array.zeros((0, ny, nx), dtype="float32")
    return dask.array.stack(blocks, axis=0)
def derive_fields(fields: dict[str, xr.DataArray]) -> dict[str, xr.DataArray]:
    """Add wind speed for any u/v pair present (u10/v10, u100/v100, u/v, ...).

    Run this *after* subsetting: np.hypot forces the arrays into memory, so doing
    it on the full cube would defeat the point of subsetting at all.
    """
    for uname in list(fields):
        vname = "v" + uname[1:]
        if not uname.startswith("u") or vname not in fields:
            continue
        ws = np.hypot(fields[uname], fields[vname])
        ws.attrs = {
            "long_name": f"wind speed{uname[1:] and f' ({uname[1:]} m)' or ''}",
            "units": fields[uname].attrs.get("units", "m s**-1"),
        }
        fields[f"ws{uname[1:]}"] = ws
    return fields


# ---------------------------------------------------------------------------
# Subsetting decoded fields (the viewer's in-memory equivalent of write_subset)
# ---------------------------------------------------------------------------


def available_times(fields: dict[str, xr.DataArray]) -> np.ndarray:
    """Every valid time present, across all fields."""
    return np.unique(np.concatenate([f["time"].values for f in fields.values()]))


def estimate_bytes(fields: dict[str, xr.DataArray], n_times: int) -> int:
    """Rough RAM cost of loading n_times steps of every field."""
    return sum(
        int(np.prod([s for d, s in f.sizes.items() if d != "time"])) * n_times * f.dtype.itemsize
        for f in fields.values()
    )


def subset_times(fields, start, end, stride: int = 1) -> dict[str, xr.DataArray]:
    """Restrict every field to [start, end] inclusive, with an optional stride.

    Each field is sliced on its own time coordinate, so a field whose axis is
    offset from the others still lands in the requested window.
    """
    out = {}
    for k, f in fields.items():
        sel = f.sel(time=slice(start, end))
        if stride > 1:
            sel = sel.isel(time=slice(None, None, stride))
        if sel.sizes.get("time", 0):
            out[k] = sel
    if not out:
        raise SystemExit(f"No data in the window {start} .. {end}")
    return out


def subset_area(fields, bbox: tuple | None) -> dict[str, xr.DataArray]:
    """Crop every field to bbox = (W, E, S, N), in degrees.

    Latitude is sliced in whatever direction the grid scans (ERA5 runs north ->
    south), since xarray's slice() follows coordinate order, not value order:
    slicing (south, north) against a descending axis returns nothing.
    """
    if bbox is None:
        return fields
    west, east, south, north = bbox
    out = {}
    for k, f in fields.items():
        ydim, xdim = ydim_of(f), xdim_of(f)
        lat = f[ydim].values
        ysel = slice(north, south) if lat[0] > lat[-1] else slice(south, north)
        sel = f.sel({ydim: ysel, xdim: slice(west, east)})
        if sel.sizes[ydim] and sel.sizes[xdim]:
            out[k] = sel
    if not out:
        raise SystemExit(f"No data inside bbox {bbox}")
    return out
