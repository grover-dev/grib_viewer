"""Regression tests for the viewer: styling inference, time handling, rendering."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # no display in CI; must precede the pyplot import

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import xarray as xr  # noqa: E402

import era5_viewer as ev  # noqa: E402
import grib_utils as gu  # noqa: E402


def _fields(path):
    return gu.derive_fields(gu.open_grib(str(path)))


# ---------------------------------------------------------------------------
# Style inference from metadata
# ---------------------------------------------------------------------------


def _bare(units, name="thing", values=None, n=4):
    da = xr.DataArray(
        np.linspace(0, 1, n * 6).reshape(n, 2, 3) if values is None else values,
        dims=("time", "latitude", "longitude"),
        coords={"time": np.array([np.datetime64("2025-01-01T00") + np.timedelta64(h, "h")
                                  for h in range(n)]),
                "latitude": [10.0, 0.0], "longitude": [0.0, 1.0, 2.0]},
        attrs={"units": units, "long_name": name},
    )
    return da


@pytest.mark.parametrize(
    "units, name, want_cmap",
    [
        ("K", "2 metre temperature", "coolwarm"),
        ("Pa", "surface pressure", "cividis"),
        ("(0 - 1)", "total cloud cover", "Blues_r"),
        ("m s**-1", "wind speed", "turbo"),
        ("m s**-1", "10 metre U wind component", "RdBu_r"),  # signed component
    ],
)
def test_colormap_inferred_from_units(units, name, want_cmap):
    """BUG: converted units were compared case-sensitively.

    build_layer rewrites units to '°C' / 'W m**-2' (capitals) and then matched
    them against lowercase literals, so temperature and irradiance silently fell
    through to the default colormap.
    """
    layer = ev.build_layer("x", _bare(units, name))
    assert layer.cmap == want_cmap


def test_kelvin_is_converted_to_celsius():
    layer = ev.build_layer("t2m", _bare("K", "2 metre temperature",
                                        values=np.full((4, 2, 3), 300.0)))
    assert layer.units == "°C"
    assert float(layer.data.max()) == pytest.approx(26.85, abs=0.01)


def test_accumulated_joules_become_watts():
    """ERA5 radiation is J/m2 accumulated over the step; show it as W/m2."""
    vals = np.full((4, 2, 3), 3.6e6)  # 1 kWh/m2 per hour == 1000 W/m2
    layer = ev.build_layer("ssrd", _bare("J m**-2", "surface solar radiation", values=vals))
    assert layer.units == "W m**-2"
    assert float(layer.data.max()) == pytest.approx(1000.0)
    assert layer.cmap == "inferno"


def test_limits_are_sampled_not_exhaustive():
    """Colour limits must not read the whole cube (that was 2.5 GB on a global file)."""
    big = _bare("m s**-1", "wind speed", values=np.random.default_rng(0).random((4, 2, 3)))
    layer = ev.build_layer("ws10", big)
    assert layer.vmin <= layer.vmax
    assert np.isfinite([layer.vmin, layer.vmax]).all()


# ---------------------------------------------------------------------------
# Player
# ---------------------------------------------------------------------------


def test_player_selects_layers_by_timestamp_not_position():
    """BUG: layers were indexed positionally with a modulo.

    Two fields on different axes (one hourly, one 3-hourly and offset) would be
    paired up at mismatched hours, and a short axis would wrap and replay.
    """
    t_fast = np.array([np.datetime64("2025-01-01T00") + np.timedelta64(h, "h") for h in range(6)])
    t_slow = np.array([np.datetime64("2025-01-01T00"), np.datetime64("2025-01-01T03")])

    def mk(times, base):
        return xr.DataArray(
            np.stack([np.full((2, 3), base + i) for i in range(len(times))]),
            dims=("time", "latitude", "longitude"),
            coords={"time": times, "latitude": [10.0, 0.0], "longitude": [0.0, 1.0, 2.0]},
            attrs={"units": "m s**-1", "long_name": "x"},
        )

    p = ev.Player({"fast": mk(t_fast, 0), "slow": mk(t_slow, 100)}, "slow", False, basemap=False)

    assert p.n == 6  # union of both axes
    p.frame = 4      # 04:00Z -> slow's nearest step is 03:00Z (its index 1)
    assert float(p._frame_data()[0, 0]) == 101.0

    p.key = "fast"   # fast has an exact step at 04:00Z
    assert float(p._frame_data()[0, 0]) == 4.0


def test_title_is_on_canvas_with_a_basemap(single_frame_grib):
    """BUG: cartopy auto-placed the title at y=inf, so it vanished off-canvas.

    It rendered blank in both the window and every frame of a saved video.
    """
    p = ev.Player(_fields(single_frame_grib), "t2m", False, basemap=True)
    p.fig.canvas.draw()
    y = p.ax.title.get_position()[1]
    assert np.isfinite(y), "title position must be finite"
    assert p.ax.get_title(), "title must not be empty"
    assert "2025-01-02T03:00Z" in p.ax.get_title()


def test_export_figure_has_no_widgets(single_frame_grib):
    """A saved video must not bake in a frozen play button and slider."""
    p = ev.Player(_fields(single_frame_grib), "t2m", False, basemap=False, chrome=False)
    for widget in ("slider", "btn", "radio", "check"):
        assert not hasattr(p, widget), f"{widget} should not exist on an export figure"
    p.render()  # must not blow up reaching for the absent slider


def test_pacific_window_recentres_the_projection():
    """A window past 180 needs a re-centred map, or cartopy splits it."""
    lons = np.arange(170.0, 190.1, 5.0)
    da = xr.DataArray(
        np.zeros((1, 3, len(lons))),
        dims=("time", "latitude", "longitude"),
        coords={"time": [np.datetime64("2025-01-01T00")],
                "latitude": [10.0, 0.0, -10.0], "longitude": lons},
        attrs={"units": "m s**-1", "long_name": "x"},
    )
    p = ev.Player({"x": da}, "x", False, basemap=True)
    assert p.lon0 == 180
    assert p.extent[0] >= 170 and p.extent[1] <= 190


# ---------------------------------------------------------------------------
# Loading order: subset before the eager steps
# ---------------------------------------------------------------------------


def test_frame_cache_is_bounded_by_bytes_not_count():
    """A global frame is 8 MB; bounding by count would quietly hold 100s of MB."""
    cache = ev.FrameCache(budget_bytes=1000)

    class Fake:
        def __init__(self, n):
            self.nbytes = n

    for i in range(10):
        cache.put(("v", i), Fake(300))

    assert cache.used <= 1000, "cache must respect its byte budget"
    assert cache.get(("v", 0)) is None, "oldest frames must be evicted"
    assert cache.get(("v", 9)) is not None, "newest frame must survive"


def test_frame_cache_evicts_least_recently_used():
    cache = ev.FrameCache(budget_bytes=1000)

    class Fake:
        def __init__(self, n):
            self.nbytes = n

    for i in range(3):
        cache.put(("v", i), Fake(300))
    cache.get(("v", 0))          # touch the oldest, making it most recent
    cache.put(("v", 3), Fake(300))  # forces an eviction

    assert cache.get(("v", 0)) is not None, "a recently used frame must not be evicted"
    assert cache.get(("v", 1)) is None, "the least recently used one should go"


def test_zoom_invalidates_the_frame_cache(global_grib):
    """Cached frames belong to the previous crop; reusing them would be a shape error."""
    p = ev.Player(_fields(global_grib), "u10", False, basemap=False)
    p.set_frame(0)
    full_shape = p._frame_data().shape
    assert p.cache.used > 0

    p.set_bbox((0, 30, 0, 30))

    # set_bbox clears the cache and then renders, so it legitimately holds one
    # frame again -- what matters is that nothing of the OLD shape survived
    cropped_shape = p.current.data.isel(time=0).shape
    assert cropped_shape != full_shape
    assert all(v.shape == cropped_shape for v in p.cache._items.values()), \
        "no frame from the previous crop may survive the zoom"
    assert p._frame_data().shape == cropped_shape


def test_scalebar_accounts_for_latitude(global_grib):
    """A degree of longitude is 111.32 km only at the equator.

    At 60N it is half that; a bar sized with the equatorial figure would claim
    double the true distance. The bar's spanned degrees, converted at the
    latitude it is drawn at, must equal its label.
    """
    p = ev.Player(_fields(global_grib), "u10", False, basemap=False)
    p.set_bbox((-12, 5, 48, 62))  # UK-ish: cos(lat) ~ 0.6

    line, _, label = p._scalebar_artists
    x0, x1 = line.get_xdata()
    lat = line.get_ydata()[0]
    km = (x1 - x0) * 111.32 * np.cos(np.deg2rad(lat))
    stated = float(label.get_text().replace(" km", "").replace(",", ""))

    assert km == pytest.approx(stated, rel=1e-6)
    assert stated in {s * 10 ** e for s in (1, 2, 5) for e in range(6)}, \
        "length should come from the 1-2-5 ladder"


def test_scalebar_redraws_on_zoom(global_grib):
    p = ev.Player(_fields(global_grib), "u10", False, basemap=False)
    before = p._scalebar_artists[2].get_text()
    p.set_bbox((-30, 10, 40, 70))  # comfortably wider than the fixture's 10 deg grid
    after = p._scalebar_artists[2].get_text()
    assert before != after, "zooming in must shrink the scale bar"


def test_subset_then_derive_keeps_it_lazy(global_grib):
    """derive_fields materializes (np.hypot), so it must run AFTER subsetting."""
    fields = gu.open_grib(str(global_grib))
    times = gu.available_times(fields)

    small = gu.derive_fields(gu.subset_times(fields, times[0], times[1]))
    assert small["u10"].sizes["time"] == 2
    assert small["ws10"].sizes["time"] == 2
    # the whole point: the derived field covers only what we asked for
    assert gu.estimate_bytes(small, 2) < gu.estimate_bytes(fields, len(times))
