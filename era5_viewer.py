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
import warnings
from dataclasses import dataclass, field

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cfgrib
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
from matplotlib.widgets import Button, RadioButtons, Slider  # noqa: E402

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


def load_grib(path: str) -> dict[str, xr.DataArray]:
    """Flatten a GRIB into {name: DataArray(time, y, x)}, keeping every field.

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
            da = da.squeeze(drop=True)
            if "valid_time" in da.dims:
                da = da.rename(valid_time="time")
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
                    tag = "_".join(f"{d}{sub[d].values}" for d in extra)
                    sub.attrs = dict(da.attrs)
                    fields[f"{name}_{tag}"] = sub

    # Derived wind speed for any u/v pair present (u10/v10, u100/v100, u/v, ...)
    for uname in list(fields):
        if not uname.startswith("u"):
            continue
        vname = "v" + uname[1:]
        if vname not in fields:
            continue
        ws = np.hypot(fields[uname], fields[vname])
        ws.attrs = {
            "long_name": f"wind speed{uname[1:] and f' ({uname[1:]} m)' or ''}",
            "units": fields[uname].attrs.get("units", "m s**-1"),
        }
        fields[f"ws{uname[1:]}"] = ws

    if not fields:
        raise SystemExit(f"No (time, y, x) fields found in {path}")
    return fields


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
    ):
        self.layers = {k: build_layer(k, v) for k, v in sorted(fields.items())}
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

        rect = (0.26, 0.16, 0.70, 0.76)
        self.crs = ccrs.PlateCarree() if basemap else None
        self.ax = self.fig.add_axes(rect, projection=self.crs) if basemap \
            else self.fig.add_axes(rect)

        # data is on a plain lat/lon grid, so everything is drawn in PlateCarree
        geo = {"transform": self.crs} if basemap else {}

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

        self.timer = self.fig.canvas.new_timer(interval=120)
        self.timer.add_callback(self.tick)

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

        self.ax.set_extent(self.extent, crs=self.crs)
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
        self.ax.set_title(title, fontsize=13)
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
            **({"transform": self.crs} if self.crs else {}),
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

    def save(self, out: str, fps: int = 8):
        from matplotlib.animation import FuncAnimation

        def update(i):
            self.frame = i
            self.render()
            return (self.im,)

        FuncAnimation(self.fig, update, frames=self.n, blit=False).save(out, fps=fps, dpi=110)
        print(f"wrote {out}")


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
    args = p.parse_args()

    fields = load_grib(args.grib)

    if args.list:
        for k, da in fields.items():
            layer = build_layer(k, da)
            print(
                f"{k:22s} {layer.label[:40]:42s} [{layer.units or '-':>8s}]  "
                f"cmap={layer.cmap:9s} range={layer.vmin:.3g}..{layer.vmax:.3g}  "
                f"t={da.sizes['time']} grid={da.sizes[layer.ydim]}x{da.sizes[layer.xdim]}"
            )
        return

    if not args.save:
        use_interactive_backend()  # before the figure is created

    player = Player(fields, args.var, args.quiver, basemap=args.basemap)
    if args.save:
        player.save(args.save, args.fps)
    else:
        player.show()


if __name__ == "__main__":
    main()
