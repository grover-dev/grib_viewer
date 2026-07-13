"""Animated viewer for ERA5 GRIB data.

Everything renderable is discovered from the file itself: any variable with a
(time, y, x) shape becomes a selectable layer. Colormap, unit conversion and
scale limits are inferred from the GRIB/CF metadata (units, standard name,
accumulation step) plus the data's own distribution — no per-variable whitelist.

Usage:
    uv run era5_viewer.py data.grib --list          # show what was found
    uv run era5_viewer.py data.grib                 # interactive player
    uv run era5_viewer.py data.grib --save out.mp4
"""

from __future__ import annotations

import argparse
import re
import sys
import warnings
from dataclasses import dataclass, field

from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cfgrib
import eccodes as ec
import matplotlib
import numpy as np
import xarray as xr


def use_interactive_backend() -> None:
    """Pick a GUI backend for the live player.

    matplotlib defaults to the headless Agg canvas when no GUI toolkit is
    importable, and then plt.show() just warns and returns. Fail loudly instead.
    """
    if matplotlib.get_backend().lower() not in ("agg", "template"):
        return  # a backend was already forced (e.g. via MPLBACKEND)
    for backend in ("QtAgg", "TkAgg", "GTK4Agg"):
        try:
            matplotlib.use(backend, force=True)
            return
        except ImportError:
            continue
    raise SystemExit(
        "No interactive matplotlib backend available.\n"
        "Install one (e.g. `uv add pyqt6`), or use --save to render a file instead."
    )


import matplotlib.pyplot as plt  # noqa: E402  (after backend selection helper)
from matplotlib.widgets import (  # noqa: E402
    Button,
    CheckButtons,
    RadioButtons,
    RectangleSelector,
    Slider,
)

Y_NAMES = ("latitude", "lat", "y")
X_NAMES = ("longitude", "lon", "x")


# ---------------------------------------------------------------------------
# Layer + metadata-driven styling
# ---------------------------------------------------------------------------


@dataclass
class Layer:
    key: str
    data: xr.DataArray = field(repr=False)
    label: str
    units: str
    cmap: str
    vmin: float
    vmax: float

    @property
    def ydim(self) -> str:
        return next(d for d in self.data.dims if d in Y_NAMES)

    @property
    def xdim(self) -> str:
        return next(d for d in self.data.dims if d in X_NAMES)


def _norm_units(u: str) -> str:
    """GRIB writes units like 'm s**-1', 'J m**-2', '(0 - 1)'."""
    return re.sub(r"\s+", " ", (u or "").strip().lower())


def _accumulation_seconds(da: xr.DataArray) -> float | None:
    """Seconds an accumulated field covers, so J/m² can be shown as W/m².

    ERA5 radiation/flux fields accumulate over the model step. Prefer the GRIB
    step metadata; fall back to the spacing of the time axis.
    """
    step = da.coords.get("step")
    if step is not None and np.issubdtype(step.dtype, np.timedelta64):
        secs = float(np.asarray(step.values).ravel()[0] / np.timedelta64(1, "s"))
        if secs > 0:
            return secs
    t = da["time"].values
    if len(t) > 1 and np.issubdtype(t.dtype, np.datetime64):
        secs = float(np.median(np.diff(t)) / np.timedelta64(1, "s"))
        if secs > 0:
            return secs
    return None


def _limits(da: xr.DataArray, diverging: bool) -> tuple[float, float]:
    """Robust limits from the data itself (2nd/98th percentile, ignoring NaNs).

    Sampled, not exhaustive: a global 0.25° month is ~1 GB per variable, and
    reading every one in full just to pick colour limits blows up both startup
    time and RSS. A stride over time and space gives the same percentiles to
    well within a colourbar tick.
    """
    nt = da.sizes.get("time", 1)
    tstride = max(1, nt // 8)  # at most ~8 time slices
    sample = da.isel(time=slice(None, None, tstride))

    ny, nx = (sample.sizes[d] for d in sample.dims if d in Y_NAMES + X_NAMES)
    sstride = max(1, int(np.sqrt(ny * nx / 40_000)))  # cap ~40k points per slice
    ydim = next(d for d in sample.dims if d in Y_NAMES)
    xdim = next(d for d in sample.dims if d in X_NAMES)
    sample = sample.isel({ydim: slice(None, None, sstride), xdim: slice(None, None, sstride)})

    values = np.asarray(sample.values)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    lo, hi = (float(x) for x in np.percentile(finite, [2, 98]))
    if diverging:
        m = max(abs(lo), abs(hi)) or 1.0
        return -m, m
    if hi - lo < 1e-9:
        hi = lo + 1.0
    return lo, hi


def build_layer(key: str, da: xr.DataArray) -> Layer:
    """Infer presentation entirely from the variable's own attributes + values."""
    attrs = da.attrs
    label = attrs.get("long_name") or attrs.get("GRIB_name") or key
    units = _norm_units(attrs.get("units", ""))
    name = f"{key} {label}".lower()

    # A field is diverging if it can meaningfully be negative about zero:
    # vector components, anomalies, net fluxes.
    diverging = bool(
        re.search(r"\b(u|v)-?component|eastward|northward|anomaly|tendency", name)
    ) or re.fullmatch(r"[uv]\d*", key.lower()) is not None

    # --- unit-driven conversions -------------------------------------------
    if units in ("k", "kelvin"):
        da = da - 273.15
        units = "°C"
    elif units == "pa":
        da = da / 100.0
        units = "hPa"
    elif units in ("j m**-2", "j/m2", "j m-2"):
        secs = _accumulation_seconds(da)
        if secs:
            da = da / secs
            units = "W m**-2"

    # --- colormap from what the quantity *is* -------------------------------
    u = units.lower()  # units may have been reassigned above with capitals
    if diverging:
        cmap = "RdBu_r"
    elif u in ("w m**-2", "w/m2", "w m-2"):
        cmap = "inferno"  # radiation / irradiance
    elif u in ("(0 - 1)", "1", "fraction", "%") or "cover" in name:
        cmap = "Blues_r"  # cloud / land fractions
    elif u in ("m s**-1", "m/s", "m s-1"):
        cmap = "turbo"  # wind speed, gusts
    elif u == "°c":
        cmap = "coolwarm"
    elif u == "hpa":
        cmap = "cividis"
    else:
        cmap = "viridis"

    vmin, vmax = _limits(da, diverging)
    if u in ("(0 - 1)", "1", "fraction"):
        vmin, vmax = 0.0, 1.0
    elif not diverging and vmin > 0 and re.search(r"radiation|irradiance|speed|gust|cover", name):
        vmin = 0.0  # these are physically floored at zero; don't clip the dark end

    da.attrs["display_units"] = units
    return Layer(key=key, data=da, label=label, units=units, cmap=cmap, vmin=vmin, vmax=vmax)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _coord_tag(dim: str, value) -> str:
    """Readable suffix for a layer split out along an extra dim (e.g. a level).

    Timedeltas get formatted as hours rather than raw nanoseconds — a coordinate
    printed as '10800000000000 nanoseconds' is unusable as a variable name.
    """
    if np.issubdtype(np.asarray(value).dtype, np.timedelta64):
        return f"{dim}{int(np.asarray(value) / np.timedelta64(1, 'h'))}h"
    v = np.asarray(value).item()
    return f"{dim}{v:g}" if isinstance(v, float) else f"{dim}{v}"


def normalize_lons(da: xr.DataArray) -> xr.DataArray:
    """Put a global 0..360 grid onto -180..180, and leave everything else alone.

    ERA5 ships *global* fields on 0..360, which renders with the Atlantic torn
    down the middle unless it is rolled onto -180..180.

    A regional grid must NOT be touched. A Pacific window written by slice_grib
    runs e.g. 170..190: wrapping that into -180..180 and sorting scatters it into
    two clumps (-180..-170 and 170..180) with a hole in between, and imshow --
    which assumes uniform spacing -- would smear it across the whole map. Such a
    grid is already contiguous in its own frame, so it is left as it is; the
    Player re-centres the projection instead.
    """
    xdim = next((d for d in da.dims if d in X_NAMES), None)
    if xdim is None:
        return da

    lons = da[xdim].values
    if lons.max() <= 180.0:
        return da

    step = float(np.median(np.diff(lons))) if lons.size > 1 else 0.0
    spans_globe = lons.size > 1 and (lons.max() - lons.min() + abs(step)) >= 359.9
    if not spans_globe:
        return da  # regional window: already contiguous in its own frame

    da = da.assign_coords({xdim: ((lons + 180.0) % 360.0) - 180.0})
    return da.sortby(xdim)


def use_valid_time(da: xr.DataArray) -> xr.DataArray:
    """Put the field on its *valid* time axis — the hour it actually describes.

    ERA5 accumulations (tp, ssrd, tsr, cdir) are stored against a reference time
    plus a forecast step; valid time is time + step. Must run BEFORE squeeze():
    a field with a single step has step/valid_time as scalar coords, and squeeze
    would drop them, silently leaving the field on its reference time — which is
    a different hour, so it lands on a phantom timestamp of its own.
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

    ydim = next(d for d in da.dims if d in Y_NAMES)
    xdim = next(d for d in da.dims if d in X_NAMES)
    return da.transpose("time", ydim, xdim)


def open_grib(path: str) -> dict[str, xr.DataArray]:
    """Flatten a GRIB into {name: DataArray(time, y, x)}, keeping every field.

    Nothing is read here beyond the GRIB index and coordinates: the arrays come
    back lazily-indexed, so this stays cheap even on a multi-GB file. That is
    what lets us show the user the time axis *before* committing to loading it.

    A single GRIB can hold incompatible hypercubes (different level types or
    step conventions), so cfgrib returns a *list* of datasets; we take the
    variables from all of them. Variables on multiple levels are split into one
    layer per level so each one is independently renderable.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)  # cfgrib's xr.merge compat warning
        datasets = cfgrib.open_datasets(path, backend_kwargs={"indexpath": ""})
    fields: dict[str, xr.DataArray] = {}

    for ds in datasets:
        for name, da in ds.data_vars.items():
            if "valid_time" in da.dims:
                da = da.rename(valid_time="time")
            da = use_valid_time(da)  # BEFORE squeeze: it would drop scalar step coords
            # squeeze only the nuisance dims (number, surface, ...): a blanket
            # squeeze would also eat a length-1 time axis, e.g. a 1-frame file
            junk = [d for d in da.dims
                    if da.sizes[d] == 1 and d not in ("time",) + Y_NAMES + X_NAMES]
            da = da.squeeze(junk, drop=True)
            da = normalize_lons(da)
            if "time" not in da.dims:
                da = da.expand_dims("time")  # single snapshot still plays

            spatial = [d for d in da.dims if d in Y_NAMES + X_NAMES]
            if len(spatial) != 2:
                continue

            extra = [d for d in da.dims if d not in spatial + ["time"]]
            if not extra:
                fields[str(name)] = da
            else:
                # e.g. isobaricInhPa / number -> one layer per coordinate value
                for idx in np.ndindex(*(da.sizes[d] for d in extra)):
                    sel = {d: int(i) for d, i in zip(extra, idx)}
                    sub = da.isel(sel)
                    tag = "_".join(_coord_tag(d, sub[d].values) for d in extra)
                    sub.attrs = dict(da.attrs)
                    fields[f"{name}_{tag}"] = sub

    if not fields:
        raise SystemExit(f"No (time, y, x) fields found in {path}")
    return fields


def derive_fields(fields: dict[str, xr.DataArray]) -> dict[str, xr.DataArray]:
    """Add wind speed for any u/v pair present (u10/v10, u100/v100, u/v, ...).

    Runs *after* time subsetting: np.hypot forces the arrays into memory, so
    doing it on the full cube would defeat the point of asking the user what to
    load.
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
# Time selection — what to load, decided before anything is read
# ---------------------------------------------------------------------------


def available_times(fields: dict[str, xr.DataArray]) -> np.ndarray:
    """Every valid time present in the file, across all fields."""
    return np.unique(np.concatenate([f["time"].values for f in fields.values()]))


def estimate_bytes(fields: dict[str, xr.DataArray], n_times: int) -> int:
    """Rough RAM cost of loading n_times steps of every field."""
    return sum(
        int(np.prod([s for d, s in f.sizes.items() if d != "time"])) * n_times * f.dtype.itemsize
        for f in fields.values()
    )


def subset_times(
    fields: dict[str, xr.DataArray], start, end, stride: int = 1
) -> dict[str, xr.DataArray]:
    """Restrict every field to [start, end] (inclusive) with an optional stride.

    Fields whose axes differ are each sliced on their own time coordinate, so a
    field offset from the others still lands in the requested window.
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


def subset_area(fields: dict[str, xr.DataArray], bbox: tuple | None) -> dict[str, xr.DataArray]:
    """Crop every field to bbox = (west, east, south, north), in degrees.

    Latitude is sliced in whatever direction the grid actually scans (ERA5 runs
    north -> south), since xarray's slice() follows coordinate order, not value
    order — slicing (south, north) against a descending axis returns nothing.
    """
    if bbox is None:
        return fields
    west, east, south, north = bbox
    out = {}
    for k, f in fields.items():
        ydim = next(d for d in f.dims if d in Y_NAMES)
        xdim = next(d for d in f.dims if d in X_NAMES)
        lat = f[ydim].values
        ysel = slice(north, south) if lat[0] > lat[-1] else slice(south, north)
        sel = f.sel({ydim: ysel, xdim: slice(west, east)})
        if sel.sizes[ydim] and sel.sizes[xdim]:
            out[k] = sel
    if not out:
        raise SystemExit(f"No data inside bbox {bbox}")
    return out


def _fmt(t) -> str:
    return np.datetime_as_string(t, unit="m") + "Z"


def choose_window(fields: dict[str, xr.DataArray], times: np.ndarray) -> tuple:
    """Interactive prompt: how much of the time axis should we actually load?"""
    full = estimate_bytes(fields, len(times)) / 1e9
    print(f"\n  {len(fields)} fields: {', '.join(sorted(fields))}")
    print(f"  {len(times)} time steps: {_fmt(times[0])} .. {_fmt(times[-1])}")
    print(f"  loading all of it costs roughly {full:.2f} GB of RAM\n")
    print("  [a] load all")
    print("  [r] load a range")
    print("  [n] load every Nth step (thin the whole run)")

    choice = input("\n  choose [a/r/n] (default a): ").strip().lower() or "a"

    if choice == "n":
        stride = max(1, int(input("  keep every Nth step, N = ").strip() or 1))
        return times[0], times[-1], stride

    if choice == "r":
        print(f"\n  enter times as YYYY-MM-DD[THH:MM]; blank = keep the end of the range")
        s = input(f"  start [{_fmt(times[0])}]: ").strip()
        e = input(f"  end   [{_fmt(times[-1])}]: ").strip()
        start = np.datetime64(s) if s else times[0]
        end = np.datetime64(e) if e else times[-1]
        if end < start:
            raise SystemExit("end is before start")
        kept = int(((times >= start) & (times <= end)).sum())
        if not kept:
            raise SystemExit(f"no steps between {_fmt(start)} and {_fmt(end)}")
        cost = estimate_bytes(fields, kept) / 1e9
        print(f"\n  -> {kept} steps, roughly {cost:.2f} GB")
        return start, end, 1

    return times[0], times[-1], 1


# ---------------------------------------------------------------------------
# Export — carve a smaller GRIB out of the source
# ---------------------------------------------------------------------------


def export_grib(src: str, dst: str, bbox: tuple | None, start=None, end=None, stride: int = 1) -> int:
    """Write a new GRIB containing only the requested area and time window.

    Works on the raw messages rather than going back through xarray: cfgrib's
    writer needs the original GRIB_* attrs intact, which our unit conversions
    and derived fields have already destroyed. Cropping the message values and
    rewriting the grid headers keeps the output a valid GRIB that any tool
    (including this viewer) can read.

    Returns the number of messages written.
    """
    kept_times: dict = {}
    written = 0

    with open(src, "rb") as fin, open(dst, "wb") as fout:
        while (h := ec.codes_grib_new_from_file(fin)) is not None:
            try:
                grid = ec.codes_get(h, "gridType")
                if grid != "regular_ll":
                    raise SystemExit(
                        f"can only subset regular lat/lon grids, got {grid!r}. "
                        "Re-download with a regular_ll grid, or export without --bbox."
                    )

                # valid time of this message (validity*, not dataDate: accumulated
                # fields carry a step offset from their reference time)
                vd = ec.codes_get(h, "validityDate")  # YYYYMMDD
                vt = ec.codes_get(h, "validityTime")  # HHMM
                when = np.datetime64(
                    f"{vd // 10000:04d}-{vd // 100 % 100:02d}-{vd % 100:02d}"
                    f"T{vt // 100:02d}:{vt % 100:02d}"
                )
                if (start is not None and when < start) or (end is not None and when > end):
                    continue

                if stride > 1:
                    # index each distinct time we see, keep every Nth
                    idx = kept_times.setdefault(when, len(kept_times))
                    if idx % stride:
                        continue

                ni = ec.codes_get(h, "Ni")
                nj = ec.codes_get(h, "Nj")
                lat_first = ec.codes_get(h, "latitudeOfFirstGridPointInDegrees")
                lat_last = ec.codes_get(h, "latitudeOfLastGridPointInDegrees")
                lon_first = ec.codes_get(h, "longitudeOfFirstGridPointInDegrees")
                lon_last = ec.codes_get(h, "longitudeOfLastGridPointInDegrees")

                lats = np.linspace(lat_first, lat_last, nj)
                lons = np.linspace(lon_first, lon_last, ni)
                values = ec.codes_get_values(h).reshape(nj, ni)

                if bbox is not None:
                    west, east, south, north = bbox
                    # handle a 0..360 source against a -180..180 request
                    req_lons = lons if lons.max() <= 180 else np.where(lons > 180, lons - 360, lons)
                    jm = (lats >= south) & (lats <= north)
                    im = (req_lons >= west) & (req_lons <= east)
                    if not jm.any() or not im.any():
                        continue
                    j0, j1 = np.flatnonzero(jm)[[0, -1]]
                    i0, i1 = np.flatnonzero(im)[[0, -1]]

                    values = values[j0:j1 + 1, i0:i1 + 1]
                    lats, lons = lats[j0:j1 + 1], lons[i0:i1 + 1]
                    nj, ni = values.shape

                    ec.codes_set(h, "Ni", ni)
                    ec.codes_set(h, "Nj", nj)
                    ec.codes_set(h, "latitudeOfFirstGridPointInDegrees", float(lats[0]))
                    ec.codes_set(h, "latitudeOfLastGridPointInDegrees", float(lats[-1]))
                    ec.codes_set(h, "longitudeOfFirstGridPointInDegrees", float(lons[0]))
                    ec.codes_set(h, "longitudeOfLastGridPointInDegrees", float(lons[-1]))
                    ec.codes_set_values(h, values.ravel())

                ec.codes_write(h, fout)
                written += 1
            finally:
                ec.codes_release(h)

    if not written:
        raise SystemExit("nothing matched the requested area/time window")
    size = Path(dst).stat().st_size / 1e6
    print(f"wrote {dst}: {written} messages, {size:.1f} MB")
    return written


# ---------------------------------------------------------------------------
# Viewer
# ---------------------------------------------------------------------------


class Player:
    def __init__(
        self,
        fields: dict[str, xr.DataArray],
        start_var: str | None,
        quiver: bool,
        basemap: bool = True,
        global_colors: bool = False,
        chrome: bool = True,
        src: str | None = None,
    ):
        # source GRIB, so the export button can re-carve it at message level
        self.src = src
        # chrome=False builds an export-only figure: map, title, colorbar and
        # nothing else. The widgets would otherwise be baked into every frame of
        # a saved video, frozen and useless.
        self.chrome = chrome
        # Keep the uncropped fields: every zoom re-crops from these, so repeated
        # zooms don't compound, and zooming back out is lossless.
        self.source = dict(sorted(fields.items()))
        self.global_colors = global_colors
        self.bbox = None

        self.layers = {k: build_layer(k, v) for k, v in self.source.items()}
        # Colour limits over the whole domain, kept for --global-colors so the
        # scale doesn't shift under you when you zoom.
        self.full_limits = {k: (lay.vmin, lay.vmax) for k, lay in self.layers.items()}
        self.keys = list(self.layers)
        self.key = start_var if start_var in self.layers else self.keys[0]
        self.frame = 0
        self.playing = False

        # any u/v pair in the file can drive the vector overlay
        self.uv = next(
            ((f"u{s}", f"v{s}") for s in ("10", "100", "")
             if f"u{s}" in fields and f"v{s}" in fields),
            None,
        )
        self.quiver_on = quiver and self.uv is not None

        # Master timeline = union of every layer's valid times. Layers are then
        # selected *by timestamp*, not by position: ERA5 accumulated fields
        # (ssrd) can sit on a different / offset time axis from instantaneous
        # ones (u10), so positional indexing would show mismatched hours.
        self.times = np.unique(
            np.concatenate([lay.data["time"].values for lay in self.layers.values()])
        )
        self.n = len(self.times)

        cur = self.current
        lat = cur.data[cur.ydim].values
        lon = cur.data[cur.xdim].values
        self.extent = (float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max()))
        self.origin = "upper" if lat[0] > lat[-1] else "lower"

        self.fig = plt.figure(figsize=(13, 7))
        self.fig.canvas.manager.set_window_title("ERA5 viewer")

        # no widgets to make room for -> give the map the whole frame
        rect = (0.26, 0.16, 0.70, 0.76) if chrome else (0.06, 0.05, 0.86, 0.88)

        # The data is always plain lat/lon, so that is what we transform *from*.
        # The map we draw *onto* is re-centred on the data's own longitudes: a
        # Pacific window runs past 180, and a 0-centred map would split it.
        self.data_crs = ccrs.PlateCarree() if basemap else None
        self.lon0 = round((self.extent[0] + self.extent[1]) / 2 / 10) * 10 if basemap else 0
        self.crs = ccrs.PlateCarree(central_longitude=self.lon0) if basemap else None
        self.ax = self.fig.add_axes(rect, projection=self.crs) if basemap \
            else self.fig.add_axes(rect)

        geo = {"transform": self.data_crs} if basemap else {}

        self.im = self.ax.imshow(
            self._frame_data(), extent=self.extent, origin=self.origin,
            cmap=cur.cmap, vmin=cur.vmin, vmax=cur.vmax,
            interpolation="bilinear", aspect="auto", **geo,
        )
        if basemap:
            self._draw_basemap()
        self.cbar = self.fig.colorbar(self.im, ax=self.ax, pad=0.02)
        self.qv = None
        self._draw_quiver()
        self._relabel()

        # --- widgets -----------------------------------------------------
        if not chrome:
            return

        h = min(0.62, 0.035 * len(self.keys) + 0.04)
        ax_radio = self.fig.add_axes((0.01, 0.92 - h, 0.22, h))
        ax_radio.set_title("variable", fontsize=9)
        self.radio = RadioButtons(ax_radio, self.keys, active=self.keys.index(self.key))
        for lbl in self.radio.labels:
            lbl.set_fontsize(8)
        self.radio.on_clicked(self.set_var)

        # The slider indexes frames under the hood, but reads as a clock: the
        # value text is the valid time, not "37 of 720".
        # Slider stops well short of the button: its value text is drawn just
        # right of the track, and a full timestamp needs the room.
        ax_slider = self.fig.add_axes((0.34, 0.07, 0.34, 0.03))
        self.slider = Slider(ax_slider, "time", 0, max(self.n - 1, 0), valinit=0, valstep=1)
        self.slider.on_changed(self.set_frame)
        self.slider.valtext.set_fontsize(9)
        self._retick()

        self.btn = Button(self.fig.add_axes((0.88, 0.06, 0.08, 0.05)), "▶ play")
        self.btn.on_clicked(self.toggle)

        self.btn_reset = Button(self.fig.add_axes((0.88, 0.005, 0.08, 0.045)), "⤢ reset zoom")
        self.btn_reset.label.set_fontsize(7)
        self.btn_reset.on_clicked(lambda _e: self.set_bbox(None))

        # Colour limits: re-fit to the zoomed area (default), or pin to the
        # full domain so the scale stays comparable across zooms.
        ax_check = self.fig.add_axes((0.01, 0.16, 0.22, 0.06))
        ax_check.set_frame_on(False)
        self.check = CheckButtons(ax_check, ["global colors"], [self.global_colors])
        for lbl in self.check.labels:
            lbl.set_fontsize(8)
        self.check.on_clicked(self.toggle_global_colors)

        if self.src:
            self.btn_export = Button(
                self.fig.add_axes((0.01, 0.09, 0.22, 0.05)), "⬇ export selection to GRIB"
            )
            self.btn_export.label.set_fontsize(7)
            self.btn_export.on_clicked(self.export_selection)

        # Drag a box on the map to crop the data to it.
        self.selector = RectangleSelector(
            self.ax, self._on_select, useblit=False, button=[1],
            minspanx=0.5, minspany=0.5, spancoords="data", interactive=False,
            props=dict(facecolor="none", edgecolor="yellow", linewidth=1.4),
        )

        self.timer = self.fig.canvas.new_timer(interval=120)
        self.timer.add_callback(self.tick)

    # -- export -----------------------------------------------------------
    def export_selection(self, _event):
        """Carve the visible area + loaded time window out into a new GRIB."""
        stem = Path(self.src).stem
        w, e, s, n = (round(v, 3) for v in (self.extent[0], self.extent[1],
                                            self.extent[2], self.extent[3]))
        dst = Path(self.src).parent / (
            f"{stem}_{self._stamp(self.times[0])[:13]}_{w}_{e}_{s}_{n}.grib"
            .replace(":", "").replace("-", "")
        )
        self.ax.set_title(f"exporting to {dst.name} ...", fontsize=11, y=1.02)
        self.fig.canvas.draw()
        try:
            export_grib(self.src, str(dst), (w, e, s, n), self.times[0], self.times[-1])
            msg = f"exported {dst.name}"
        except SystemExit as exc:  # export_grib refuses non-regular_ll grids
            msg = f"export failed: {exc}"
        print(msg)
        self.ax.set_title(msg, fontsize=11, y=1.02)
        self.fig.canvas.draw()

    # -- zoom -------------------------------------------------------------
    def toggle_global_colors(self, _label):
        """Flip between full-domain and visible-area colour limits, in place."""
        self.global_colors = not self.global_colors
        self.set_bbox(self.bbox)  # rebuild layers under the new limit policy

    def _on_select(self, eclick, erelease):
        # Mouse coords come back in *projection* space. With a re-centred map
        # that is offset from true longitude by lon0, so undo it before cropping.
        west, east = sorted((eclick.xdata, erelease.xdata))
        south, north = sorted((eclick.ydata, erelease.ydata))
        if self.crs is not None:
            west, east = west + self.lon0, east + self.lon0
        self.set_bbox((west, east, south, north))

    def set_bbox(self, bbox: tuple | None):
        """Re-crop the data to bbox (None = full domain) and redraw.

        The crop is applied to the *source* arrays and the layers rebuilt, so
        this genuinely narrows what gets rendered — not just the axis limits.
        """
        self.bbox = bbox
        cropped = subset_area(self.source, bbox)
        self.layers = {k: build_layer(k, v) for k, v in cropped.items()}
        if self.global_colors:  # keep the full-domain scale
            for k, lay in self.layers.items():
                lay.vmin, lay.vmax = self.full_limits[k]

        if self.key not in self.layers:
            self.key = next(iter(self.layers))
        cur = self.current
        lat, lon = cur.data[cur.ydim].values, cur.data[cur.xdim].values
        self.extent = (float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max()))

        self.im.set_extent(self.extent)
        self.im.set_clim(cur.vmin, cur.vmax)
        if self.crs is not None:
            self.ax.set_extent(self.extent, crs=self.data_crs)
        else:
            self.ax.set_xlim(self.extent[:2])
            self.ax.set_ylim(self.extent[2:])
        self.render()

    # -- state ----------------------------------------------------------
    @property
    def current(self) -> Layer:
        return self.layers[self.key]

    @property
    def now(self) -> np.datetime64:
        """The valid time currently being displayed."""
        return self.times[self.frame]

    def _at(self, layer: Layer) -> xr.DataArray:
        """This layer at the current wall-clock time (nearest step if it lags)."""
        return layer.data.sel(time=self.now, method="nearest")

    def _frame_data(self) -> np.ndarray:
        return self._at(self.current).values

    def _stamp(self, when=None) -> str:
        return np.datetime_as_string(when if when is not None else self.now, unit="m") + "Z"

    def _draw_basemap(self):
        """Coastlines/borders clipped to the rendered area.

        Natural Earth resolution is chosen from the span of the domain: a
        basin-scale window gets 10 m detail, a hemisphere would drown in it.
        """
        lon0, lon1, lat0, lat1 = self.extent
        span = max(lon1 - lon0, lat1 - lat0)
        res = "10m" if span <= 15 else "50m" if span <= 60 else "110m"

        self.ax.set_extent(self.extent, crs=self.data_crs)  # extent is in data lons
        self.ax.add_feature(
            cfeature.LAND.with_scale(res), facecolor="none", edgecolor="none", zorder=1
        )
        self.ax.add_feature(
            cfeature.COASTLINE.with_scale(res), linewidth=0.9, edgecolor="black", zorder=3
        )
        self.ax.add_feature(
            cfeature.BORDERS.with_scale(res), linewidth=0.5,
            edgecolor="black", alpha=0.45, zorder=3,
        )
        gl = self.ax.gridlines(
            draw_labels=True, linewidth=0.4, color="gray", alpha=0.4, linestyle=":", zorder=4
        )
        gl.top_labels = gl.right_labels = False

    def _relabel(self):
        c = self.current
        actual = self._at(c)["time"].values  # what we actually got, post-nearest
        title = f"{c.label}  —  {self._stamp()}"
        if actual != self.now:
            # this layer has no step at the master time; say so rather than lie
            title += f"  (layer valid {self._stamp(actual)})"
        # y is pinned deliberately: cartopy auto-places the title above the
        # gridline labels, and with draw_labels=True that calculation can come
        # back as inf, putting the title off-canvas entirely.
        self.ax.set_title(title, fontsize=13, y=1.02)
        self.cbar.set_label(c.units)
        if self.crs is None:  # cartopy gridlines already label the axes
            self.ax.set_xlabel("longitude")
            self.ax.set_ylabel("latitude")

    def _draw_quiver(self):
        if self.qv is not None:
            self.qv.remove()
            self.qv = None
        if not self.quiver_on:
            return
        u = self.layers[self.uv[0]]
        v = self.layers[self.uv[1]]
        us, vs = self._at(u), self._at(v)
        ydim, xdim = u.ydim, u.xdim
        step = max(1, min(us.sizes[ydim], us.sizes[xdim]) // 20)
        sl = {ydim: slice(None, None, step), xdim: slice(None, None, step)}
        us, vs = us.isel(sl), vs.isel(sl)
        self.qv = self.ax.quiver(
            us[xdim].values, us[ydim].values, us.values, vs.values,
            color="white", alpha=0.7, scale=400, width=0.0018, zorder=2,
            **({"transform": self.data_crs} if self.data_crs else {}),
        )

    # -- callbacks --------------------------------------------------------
    def set_var(self, key: str):
        self.key = key
        c = self.current
        self.im.set_cmap(c.cmap)
        self.im.set_clim(c.vmin, c.vmax)
        self.render()

    def _retick(self):
        """Slider reads as a wall clock, with the span of the run alongside it."""
        if not self.chrome:
            return
        span = f"{self._stamp(self.times[0])} → {self._stamp(self.times[-1])}"
        self.slider.valtext.set_text(f"{self._stamp()}   [{self.frame + 1}/{self.n}]")
        self.slider.label.set_text(f"time\n{span}")
        self.slider.label.set_fontsize(7)

    def set_frame(self, val):
        self.frame = int(val) % self.n
        self.render()

    def render(self):
        self.im.set_data(self._frame_data())
        self._draw_quiver()
        self._relabel()
        self._retick()
        self.fig.canvas.draw_idle()

    def toggle(self, _event):
        self.playing = not self.playing
        self.btn.label.set_text("⏸ pause" if self.playing else "▶ play")
        self.timer.start() if self.playing else self.timer.stop()

    def tick(self):
        if self.playing:
            self.slider.set_val((self.frame + 1) % self.n)  # triggers set_frame

    def show(self):
        plt.show()

    def save(self, out: str, fps: int = 8, dpi: int = 110):
        """Render one frame per loaded time step to a video/GIF.

        Expects a chrome-free figure (Player(..., chrome=False)); saving the
        interactive one bakes the frozen widgets into every frame.
        """
        from matplotlib.animation import FuncAnimation

        def update(i):
            self.frame = i
            self.render()
            return (self.im,)

        anim = FuncAnimation(self.fig, update, frames=self.n, blit=False)
        anim.save(out, fps=fps, dpi=dpi)
        print(f"wrote {out}: {self.n} frames @ {fps} fps "
              f"({self._stamp(self.times[0])} .. {self._stamp(self.times[-1])})")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("grib", help="path to the ERA5 GRIB file")
    p.add_argument("--list", action="store_true", help="print discovered fields and exit")
    p.add_argument("--var", help="variable to show first (default: first discovered)")
    p.add_argument("--quiver", action="store_true", help="overlay wind vectors if u/v present")
    p.add_argument("--no-map", dest="basemap", action="store_false",
                   help="disable the coastline/border basemap")
    p.add_argument("--save", metavar="OUT.mp4", help="render an animation instead of opening the UI")
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--dpi", type=int, default=110, help="resolution of the saved animation")
    p.add_argument("--start", help="load from this time (YYYY-MM-DD[THH:MM])")
    p.add_argument("--end", help="load up to this time (inclusive)")
    p.add_argument("--stride", type=int, default=1, help="keep every Nth step")
    p.add_argument("--all", action="store_true", help="load the whole run without prompting")
    p.add_argument("--bbox", type=float, nargs=4, metavar=("W", "E", "S", "N"),
                   help="only load data inside this area, in degrees")
    p.add_argument("--export", metavar="OUT.grib",
                   help="write the selected area/time window to a new GRIB and exit")
    args = p.parse_args()

    # Stage 1: open lazily — index + coords only, no field data read yet.
    fields = open_grib(args.grib)
    times = available_times(fields)

    if args.list:
        for k, da in fields.items():
            layer = build_layer(k, da)
            print(
                f"{k:22s} {layer.label[:40]:42s} [{layer.units or '-':>8s}]  "
                f"cmap={layer.cmap:9s} range={layer.vmin:.3g}..{layer.vmax:.3g}  "
                f"t={da.sizes['time']} grid={da.sizes[layer.ydim]}x{da.sizes[layer.xdim]}"
            )
        return

    # Stage 2: decide how much of the time axis to actually load.
    start, end, stride = times[0], times[-1], args.stride
    if args.start or args.end:
        start = np.datetime64(args.start) if args.start else start
        end = np.datetime64(args.end) if args.end else end
    elif not args.all and sys.stdin.isatty():
        start, end, stride = choose_window(fields, times)

    # Export works on the raw messages, so it never loads the fields at all.
    if args.export:
        export_grib(args.grib, args.export,
                    tuple(args.bbox) if args.bbox else None, start, end, stride)
        return

    # Stage 3: subset time+area first, *then* derive/convert — those materialize.
    fields = subset_area(subset_times(fields, start, end, stride), tuple(args.bbox) if args.bbox else None)
    fields = derive_fields(fields)
    kept = len(available_times(fields))
    print(f"loading {kept} of {len(times)} steps "
          f"({_fmt(start)} .. {_fmt(end)}, stride {stride})")

    if not args.save:
        use_interactive_backend()  # before the figure is created

    player = Player(fields, args.var, args.quiver, basemap=args.basemap,
                    chrome=not args.save, src=args.grib)
    if args.save:
        player.save(args.save, args.fps, args.dpi)
    else:
        player.show()


if __name__ == "__main__":
    main()
