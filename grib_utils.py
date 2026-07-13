"""Shared GRIB handling for the viewer, the slicer and the info tool.

Two layers live here, and they exist for different reasons:

* an **xarray layer** (open_grib and friends) that decodes a GRIB into lazily
  indexed (time, y, x) fields for rendering, and

* a **message layer** (scan_times, write_subset) that manipulates raw GRIB
  messages without decoding them. Subsetting has to live down here: cfgrib's
  writer needs the original GRIB_* attrs intact, which the unit conversions and
  derived fields of the xarray layer destroy. Working on the raw messages keeps
  every bit of the source metadata and stays cheap on multi-GB files.
"""

from __future__ import annotations

import warnings

import cfgrib
import eccodes as ec
import numpy as np
import xarray as xr

Y_NAMES = ("latitude", "lat", "y")
X_NAMES = ("longitude", "lon", "x")


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


def collapse_step(da: xr.DataArray) -> xr.DataArray:
    """Fold an accumulation's (time, step) axes into a single valid-time axis."""
    stacked = [d for d in ("time", "step") if d in da.dims]
    valid = da["valid_time"] if "valid_time" in da.coords else da["time"] + da["step"]

    da = da.stack(_vt=stacked)
    times = valid.stack(_vt=stacked).values if set(stacked) <= set(valid.dims) else valid.values

    # drop the MultiIndex the stack created, then relabel with the real times
    da = da.reset_index("_vt", drop=True).drop_vars(
        [c for c in ("time", "step", "valid_time") if c in da.coords], errors="ignore"
    )
    da = da.assign_coords(_vt=("_vt", times)).rename(_vt="time")

    # a step ladder overlaps at the boundaries (18Z+6h == 00Z+0h); keep one
    da = da.drop_duplicates("time").sortby("time")
    da = da.transpose("time", ydim_of(da), xdim_of(da))
    da.attrs["_folded_step"] = True  # tells open_grib to vet these times
    return da


def use_valid_time(da: xr.DataArray) -> xr.DataArray:
    """Put a field on its *valid* time axis — the hour it actually describes.

    Must run BEFORE squeeze(): a field with a single step carries step/valid_time
    as scalar coords, and squeeze would drop them, silently leaving the field on
    its reference time. That is a different hour, so it would land on a phantom
    timestamp of its own, separate from the instantaneous fields.
    """
    if "step" in da.dims:
        return collapse_step(da)
    if "valid_time" not in da.coords:
        return da

    vt = da["valid_time"]
    drop = [c for c in ("time", "step", "valid_time") if c in da.coords]
    if vt.ndim == 0:  # single step: scalar valid_time
        return da.drop_vars(drop).expand_dims(time=[vt.values])
    if "time" in da.dims and vt.dims == ("time",):
        return da.drop_vars(drop).assign_coords(time=vt.values)
    return da


def open_grib(path: str, chunks: dict | None = None) -> dict[str, xr.DataArray]:
    """Flatten a GRIB into {name: DataArray(time, y, x)}, keeping every field.

    Backed by dask (one chunk per time step by default), so the fields are a
    recipe rather than data: nothing is read until a caller asks for a specific
    frame. That is not just an optimisation. Putting accumulations onto their
    valid-time axis needs stack/sortby/drop_duplicates, and on a plain
    numpy-backed array xarray must materialize the *whole cube* to reorder it --
    ~3 GB per accumulated field on a month of global ERA5, which OOMs the machine
    before the file is even described. Under dask those become graph nodes.

    A single GRIB can hold incompatible hypercubes (different level types or step
    conventions), so cfgrib returns a *list* of datasets; we take the variables
    from all of them. A variable on multiple levels is split into one field per
    level so each is independently renderable.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)  # cfgrib's xr.merge compat warning
        datasets = cfgrib.open_datasets(
            path,
            backend_kwargs={"indexpath": ""},
            chunks=chunks if chunks is not None else {"time": 1},
        )
    fields: dict[str, xr.DataArray] = {}

    for ds in datasets:
        for name, da in ds.data_vars.items():
            if "valid_time" in da.dims:
                da = da.rename(valid_time="time")
            da = use_valid_time(da)
            # squeeze only the nuisance dims (number, surface, ...): a blanket
            # squeeze would also eat a length-1 time axis, e.g. a 1-frame file
            junk = [d for d in da.dims
                    if da.sizes[d] == 1 and d not in ("time",) + Y_NAMES + X_NAMES]
            da = da.squeeze(junk, drop=True)
            da = normalize_lons(da)
            if "time" not in da.dims:
                da = da.expand_dims("time")

            spatial = [d for d in da.dims if d in Y_NAMES + X_NAMES]
            if len(spatial) != 2:
                continue

            extra = [d for d in da.dims if d not in spatial + ["time"]]
            if not extra:
                fields[str(name)] = da
            else:
                for idx in np.ndindex(*(da.sizes[d] for d in extra)):
                    sub = da.isel({d: int(i) for d, i in zip(extra, idx)})
                    tag = "_".join(coord_tag(d, sub[d].values) for d in extra)
                    sub.attrs = dict(da.attrs)
                    fields[f"{name}_{tag}"] = sub

    if not fields:
        raise SystemExit(f"No (time, y, x) fields found in {path}")

    return _drop_phantom_times(path, fields)


def _drop_phantom_times(path: str, fields: dict[str, xr.DataArray]) -> dict[str, xr.DataArray]:
    """Remove valid times that no GRIB message actually backs.

    cfgrib presents an accumulation as a rectangular (time x step) grid, but the
    file only holds the combinations that were actually produced. Folding that
    rectangle onto valid time therefore invents slots at the edges -- a reference
    time of 18:00Z with a 1 h step yields 19:00Z even when no such message exists
    -- and they surface as all-NaN frames, and as a time axis longer than the
    file's own. The message headers are ground truth, and scanning them is cheap
    (headers only), so anything they don't vouch for is dropped.
    """
    stepped = [k for k, f in fields.items() if f.attrs.get("_folded_step")]
    if not stepped:
        return fields

    # compare as datetime64[ns] arrays: .item() on a datetime64[ns] returns an
    # int (ns since epoch), while .tolist() on a coarser dtype returns datetime
    # objects, so a set-membership test across the two never matches
    real = scan_times(path).astype("datetime64[ns]")
    for k in stepped:
        f = fields[k]
        mask = np.isin(f["time"].values.astype("datetime64[ns]"), real)
        if not mask.all():
            fields[k] = f.isel(time=np.flatnonzero(mask))
    return fields


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
