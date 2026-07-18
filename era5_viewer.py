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
from dataclasses import dataclass, field
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib
import numpy as np
import xarray as xr

from grib_utils import (
    X_NAMES,
    Y_NAMES,
    available_times,
    derive_fields,
    estimate_bytes,
    fmt_time,
    open_grib,
    scan_times,
    subset_area,
    subset_times,
    write_subset,
    xdim_of,
    ydim_of,
)


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
        return ydim_of(self.data)

    @property
    def xdim(self) -> str:
        return xdim_of(self.data)


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

    ydim, xdim = ydim_of(sample), xdim_of(sample)
    ny, nx = sample.sizes[ydim], sample.sizes[xdim]
    sstride = max(1, int(np.sqrt(ny * nx / 40_000)))  # cap ~40k points per slice
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


class FrameCache:
    """LRU cache of decoded frames, bounded by *bytes* rather than count.

    Count-bounding is a trap here: a frame of a regional crop is ~30 KB, but a
    global 0.25° frame is 8 MB, so "keep 64 frames" would silently hold half a
    gigabyte and undo the point of loading lazily in the first place.
    """

    def __init__(self, budget_bytes: int = 256 << 20):  # 256 MB
        self.budget = budget_bytes
        self.used = 0
        self._items: dict = {}  # insertion-ordered; oldest first

    def get(self, key):
        if key not in self._items:
            return None
        value = self._items.pop(key)  # reinsert to mark as most recently used
        self._items[key] = value
        return value

    def put(self, key, value) -> None:
        size = int(value.nbytes)
        if size > self.budget:  # a single frame larger than the budget: don't cache
            return
        if key in self._items:
            self.used -= int(self._items.pop(key).nbytes)
        self._items[key] = value
        self.used += size
        while self.used > self.budget and len(self._items) > 1:
            oldest = next(iter(self._items))  # dicts keep insertion order
            self.used -= int(self._items.pop(oldest).nbytes)

    def clear(self) -> None:
        self._items.clear()
        self.used = 0


def build_layer(key: str, da: xr.DataArray) -> Layer:
    """Infer presentation entirely from the variable's own attributes + values."""
    attrs = da.attrs
    label = attrs.get("long_name") or attrs.get("GRIB_name") or key
    units = _norm_units(attrs.get("units", ""))
    name = f"{key} {label}".lower()

    # A field is diverging if it can meaningfully be negative about zero: vector
    # components, anomalies, tendencies. Match the long name as well as the key
    # -- ERA5 writes "10 metre U wind component", and keying off the short name
    # alone misses anything not literally u10/v10 (GRIB also uses 10u/10v), which
    # would put a signed field on a sequential colormap and hide the sign.
    diverging = bool(
        re.search(
            r"\b[uv][- ]?(component|wind)\b"      # "U wind component", "u-component"
            r"|\bcomponent of wind\b"
            r"|eastward|northward|anomaly|tendency",
            name,
        )
    ) or re.fullmatch(r"(10|100)?[uv](10|100|n)?", key.lower()) is not None

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
# Time selection — what to load, decided before anything is read
# ---------------------------------------------------------------------------


def choose_window(fields: dict[str, xr.DataArray], times: np.ndarray) -> tuple:
    """Interactive prompt: how much of the time axis should we actually load?"""
    full = estimate_bytes(fields, len(times)) / 1e9
    print(f"\n  {len(fields)} fields: {', '.join(sorted(fields))}")
    print(f"  {len(times)} time steps: {fmt_time(times[0])} .. {fmt_time(times[-1])}")
    print(f"  loading all of it costs roughly {full:.2f} GB of RAM\n")
    print("  [a] load all")
    print("  [r] load a range")
    print("  [n] load every Nth step (thin the whole run)")

    choice = input("\n  choose [a/r/n] (default a): ").strip().lower() or "a"

    if choice == "n":
        stride = max(1, int(input("  keep every Nth step, N = ").strip() or 1))
        return times[0], times[-1], stride

    if choice == "r":
        print("\n  enter times as YYYY-MM-DD[THH:MM]; blank = keep the end of the range")
        s = input(f"  start [{fmt_time(times[0])}]: ").strip()
        e = input(f"  end   [{fmt_time(times[-1])}]: ").strip()
        start = np.datetime64(s) if s else times[0]
        end = np.datetime64(e) if e else times[-1]
        if end < start:
            raise SystemExit("end is before start")
        kept = int(((times >= start) & (times <= end)).sum())
        if not kept:
            raise SystemExit(f"no steps between {fmt_time(start)} and {fmt_time(end)}")
        cost = estimate_bytes(fields, kept) / 1e9
        print(f"\n  -> {kept} steps, roughly {cost:.2f} GB")
        return start, end, 1

    return times[0], times[-1], 1


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

        self.cache = FrameCache()
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
        self._scalebar_artists: list = []
        self._draw_scalebar()
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
            # Export exactly the frames on the timeline (which already reflect
            # any --start/--end/--stride) and the visible area. self.times holds
            # them, so there is no need to rescan the source.
            keep = set(self.times.tolist())
            written = write_subset(self.src, str(dst), keep, (w, e, s, n))
            msg = f"exported {dst.name} ({written} messages)"
        except SystemExit as exc:  # refuses non-regular_ll grids
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
        self.cache.clear()  # cached frames are the previous crop's shape
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
        self._draw_scalebar()  # both km/degree and a sensible length changed
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
        """This layer at the current wall-clock time (nearest step if it lags).

        Decoded frames are cached, because the fields are lazy: without this,
        every pass of the play loop re-reads and re-decompresses the same frames
        off disk. The dask graph is only evaluated on a miss.
        """
        key = (layer.key, self.now)
        hit = self.cache.get(key)
        if hit is None:
            hit = layer.data.sel(time=self.now, method="nearest").load()
            self.cache.put(key, hit)
        return hit

    def _frame_data(self) -> np.ndarray:
        return self._at(self.current).values

    def _stamp(self, when=None) -> str:
        return np.datetime_as_string(when if when is not None else self.now, unit="m") + "Z"

    def _draw_scalebar(self):
        """Distance scale in the lower-left corner, sized for the current view.

        A degree of longitude is 111.32 km only at the equator; at the latitude
        the bar is drawn it shrinks by cos(lat). Sized there, not at the equator
        -- at 60N the difference is a factor of two. Redrawn on every zoom, since
        both the km/degree ratio and a sensible round length change with the
        view.
        """
        for artist in self._scalebar_artists:
            artist.remove()
        self._scalebar_artists = []

        west, east, south, north = self.extent
        # anchor: 5% in from the lower-left corner of the visible area
        y = south + 0.06 * (north - south)
        x0 = west + 0.05 * (east - west)

        km_per_deg = 111.32 * max(np.cos(np.deg2rad(y)), 0.02)  # floor near poles
        view_km = (east - west) * km_per_deg

        # a round 1-2-5 number close to a quarter of the view width
        target = view_km / 4
        mag = 10 ** np.floor(np.log10(target))
        length_km = max(1, int(min((n for n in (1, 2, 5, 10)), key=lambda n: abs(n * mag - target)) * mag))
        dx = length_km / km_per_deg

        geo = {"transform": self.data_crs} if self.data_crs else {}
        tick = 0.012 * (north - south)
        line, = self.ax.plot([x0, x0 + dx], [y, y], color="black", lw=2.5,
                             solid_capstyle="butt", zorder=6, **geo)
        ends = self.ax.vlines([x0, x0 + dx], y - tick, y + tick,
                              color="black", lw=1.5, zorder=6, **geo)
        label = self.ax.text(
            x0 + dx / 2, y + tick * 1.4, f"{length_km:,} km",
            ha="center", va="bottom", fontsize=9, zorder=6,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.65, pad=1.5), **geo,
        )
        self._scalebar_artists = [line, ends, label]

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

    # Export works on the raw messages, so it never decodes the fields at all.
    if args.export:
        keep = times[(times >= start) & (times <= end)][::stride]
        written = write_subset(args.grib, args.export, set(keep.tolist()),
                               tuple(args.bbox) if args.bbox else None)
        print(f"wrote {args.export}: {written} messages, {len(keep)} steps "
              f"({fmt_time(keep[0])} .. {fmt_time(keep[-1])})")
        return

    # Stage 3: subset time+area first, *then* derive/convert — those materialize.
    bbox = tuple(args.bbox) if args.bbox else None
    fields = derive_fields(subset_area(subset_times(fields, start, end, stride), bbox))
    kept = len(available_times(fields))
    print(f"loading {kept} of {len(times)} steps "
          f"({fmt_time(start)} .. {fmt_time(end)}, stride {stride})")

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
