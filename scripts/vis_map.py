"""vis_map.py -- render a generated ocean map on an interactive 3D globe.

Shows which water the (K, W) rules kept. Navigable cells are drawn as hexagons on
a sphere, shaded by clearance margin; excluded water and land recede into
neutrals. Because the map is a pyramid, the hexagons come out at mixed sizes --
huge in open ocean, small along coastlines -- which is the adaptive resolution
made visible rather than an artifact.

Built on PyVista/VTK rather than matplotlib. mplot3d has no depth buffer, so a
track on the far side of the globe would draw over the planet instead of behind
it; VTK depth-tests properly and takes the polygon counts without decimation.

Extending it
------------

`GlobeView` is a layer stack. Every `add_*` returns its VTK actor and registers it
in `view.actors` by name, so layers can be toggled, restyled or removed later.
Layers sit at increasing radii so they never z-fight:

    RADII = {planet, cells, coast, graticule, track, marker}

To add a data layer, convert lat/lon to points with `lonlat_to_xyz(lat, lng, r)`
and hand the result to PyVista. The two hooks a boat course needs already exist:

    view.add_track(lat, lng, values=battery)   # polyline, optionally colour-mapped
    view.add_markers(lat, lng, labels=[...])   # waypoints, start/end, failures

`add_track` takes an optional scalar per point, so plotting speed, battery or
collected energy along the course is a matter of passing a different array. For a
time series, call it once per leg, or rebuild the actor per frame from
`view.actors["track"]`.

Running it
----------

    uv run python map_gen.py -K 5 -W 10 --save med.npz    # build a map first
    uv run python vis_map.py med.npz                      # then look at it

    uv run python vis_map.py med.npz --show-excluded      # include rejected water
    uv run python vis_map.py med.npz --demo-track         # example course overlay
    uv run python vis_map.py med.npz --save globe.png     # off-screen render
    uv run python vis_map.py med.npz --view 36 -5         # camera at lat/lon

Options:

    --show-excluded   draw water the (K, W) rules rejected, in neutral grey
    --demo-track      a great-circle course through the strait, to show the hook
    --coastlines {10m,50m,110m,none}   coastline detail (default 50m)
    --graticule DEG   meridian/parallel spacing, 0 to disable (default 15)
    --view LAT LON    camera target (default: centre of the map's bbox)
    --save PNG        render off-screen to a file instead of opening a window
    --size W H        window size in pixels (default 1400 1000)

Controls
--------

    drag            rotate; the speed scales with how close you are, so zooming
                    in does not send the view skidding
    scroll          zoom, bounded so you can neither enter the planet nor lose it
    n               put north back at the top without moving the camera
    b               follow the boat -- toggles, so it keeps up during playback
    space           play / pause
    left / right    step back / forward
    [ ]             slower / faster
    r               back to the start
    q               quit
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pyvista as pv
from h3.api import basic_int as h3
from matplotlib.colors import LinearSegmentedColormap

from map_utils import NavMap, lonlat_to_xyz, xyz_to_lonlat

# Palette: one sequential blue hue for magnitude, neutrals for everything that is
# not data. Checked pairwise in OKLab under normal vision and simulated
# protan/deutan/tritan -- worst pair 17.0 normal, 15.8 CVD.
SURFACE = "#0d0d0d"
LAND = "#262623"
EXCLUDED = "#898781"
COAST = "#c3c2b7"
GRATICULE = "#2c2c2a"
TRACK = "#eb6834"  # the one warm accent, reserved for the boat
INK = "#ffffff"

# blue steps 400 -> 100: low margin is deepest, open ocean palest
MARGIN_CMAP = LinearSegmentedColormap.from_list(
    "margin",
    ["#3987e5", "#5598e7", "#6da7ec", "#86b6ef", "#9ec5f4", "#b7d3f6", "#cde2fb"],
)

# A second sequential context takes the next categorical hue, as its own one-hue
# ramp -- orange, matching the track accent. Used for battery on the course.
BATTERY_CMAP = LinearSegmentedColormap.from_list(
    "battery",
    ["#7d2f11", "#a8401a", "#d05523", "#eb6834", "#f28a5f", "#f7ac8b", "#fbcdb8"],
)

# Layer altitudes in earth radii. These are deliberately tiny and tightly packed:
# any radial gap between two layers is parallax under a perspective camera, so a
# track floating well above the cells appears to slide across them as the globe
# turns, and near the limb it looks like the course cuts through hexes it does
# not. A gap of 0.006 (~38 km) was very visible; 0.0006 (~4 km) is not.
#
# Packing them this tightly is only safe because cells_to_mesh lifts each cell to
# compensate for its own sagitta -- see there.
# How near and far the camera may get, in earth radii from the centre. The near
# limit sits just off the surface; the far one keeps the globe filling enough of
# the frame to still be a map.
ZOOM_LIMITS = (1.03, 12.0)

RADII = {
    "planet": 0.9994,
    "cells": 1.0,
    "coast": 1.0004,
    "graticule": 1.0002,
    "track": 1.0008,
    "marker": 1.0012,
}


def cells_to_mesh(cells, radius: float) -> pv.PolyData:
    """One PolyData holding every H3 cell as a face, so it draws in a single pass.

    Faces are variable length: hexagons give 6 vertices, the 12 pentagons give 5.

    Each cell is lifted by 1/cos(its angular radius) so that the *middle* of the
    flat facet lands on `radius` rather than its corners. A polygon inscribed in a
    sphere sags below it, and a coarse H3 cell is not small: res 2 spans ~200 km
    and dips ~3 km at the centre. Without this the overlay layers would have to
    float kilometres clear of the cells to avoid being swallowed by them, and that
    gap is exactly what produces visible parallax as the globe rotates.
    """
    verts, faces = [], []
    offset = 0
    for c in cells.tolist():
        b = h3.cell_to_boundary(c)
        n = len(b)
        pts = lonlat_to_xyz([p[0] for p in b], [p[1] for p in b], 1.0)
        centre = pts.mean(axis=0)
        centre /= np.linalg.norm(centre)
        # smallest cos over the corners == largest angle from the centre
        cos_max = float(np.min(pts @ centre))
        verts.append(pts * (radius / max(cos_max, 1e-9)))
        faces.append(np.r_[n, np.arange(offset, offset + n)])
        offset += n
    if not verts:
        return pv.PolyData()
    return pv.PolyData(np.vstack(verts), np.concatenate(faces))


# --------------------------------------------------------------------------
# time
# --------------------------------------------------------------------------


class TimeLayer:
    """A layer whose appearance depends on time.

    Subclasses own their actor and mutate it in `update`. That is the whole
    contract, so a new time-varying phenomenon -- a radiation frame, a current
    field, a fleet of boats -- only has to answer "what do you look like at t".
    """

    name = "layer"

    def span(self) -> tuple[float, float]:
        """The closed interval this layer has data for."""
        raise NotImplementedError

    def update(self, t: float) -> None:
        raise NotImplementedError


class Timeline:
    """The clock. Holds time layers and drives them together.

    Playback is driven by a VTK timer event rather than a loop, which is what
    keeps the camera live: the interactor goes on handling drag and scroll between
    ticks, so the globe can be rotated mid-flight.
    """

    def __init__(self):
        self.layers: list[TimeLayer] = []
        self.t = 0.0
        self.playing = True
        self.speed = 1.0  # units of layer-time per second of wall clock

    def add(self, layer: TimeLayer) -> TimeLayer:
        self.layers.append(layer)
        return layer

    @property
    def span(self) -> tuple[float, float]:
        spans = [lyr.span() for lyr in self.layers]
        if not spans:
            return 0.0, 1.0
        return min(s[0] for s in spans), max(s[1] for s in spans)

    def seek(self, t: float) -> None:
        t0, t1 = self.span
        self.t = min(max(t, t0), t1)
        for lyr in self.layers:
            lyr.update(self.t)

    def advance(self, dt: float, loop: bool = True) -> None:
        t0, t1 = self.span
        t = self.t + dt * self.speed
        if t > t1:
            t = t0 if loop else t1
        self.seek(t)


class TrackLayer(TimeLayer):
    """A course drawn as it is sailed, with the boat at the head of it.

    The full geometry is uploaded once; each update only rewrites the line
    connectivity to expose the sailed prefix, and moves a one-point mesh to the
    interpolated position. Nothing is rebuilt per frame.
    """

    name = "track"

    def __init__(self, plotter, lat, lng, times, values=None, cmap=None,
                 color=TRACK, width=5.0, boat_size=13.0, radius=None):
        self.plotter = plotter
        self.times = np.asarray(times, dtype=float)
        self.pts = lonlat_to_xyz(lat, lng, radius or RADII["track"])
        self.mesh = pv.PolyData(self.pts)
        # Drop the vertex cells PolyData creates for a bare point cloud. Left in,
        # they draw every point of the course from the first frame onward, so the
        # whole route appears sailed no matter where the boat is -- and the
        # polyline underneath truncates correctly, which makes it look like the
        # reveal is broken when it is not.
        self.mesh.verts = np.empty(0, dtype=np.int64)
        self.mesh.lines = np.empty(0, dtype=np.int64)
        kw = dict(line_width=width, lighting=False, show_scalar_bar=False)
        if values is not None:
            self.mesh.point_data["value"] = np.asarray(values, dtype=float)
            kw.update(scalars="value", cmap=cmap or BATTERY_CMAP, clim=(0.0, 1.0))
        else:
            kw["color"] = color
        self.actor = plotter.add_mesh(self.mesh, **kw)

        self.boat = pv.PolyData(self.pts[:1].copy())
        self.boat_actor = plotter.add_points(
            self.boat, color=INK, point_size=boat_size, render_points_as_spheres=True
        )

    def span(self) -> tuple[float, float]:
        return float(self.times[0]), float(self.times[-1])

    def update(self, t: float) -> None:
        k = int(np.searchsorted(self.times, t, side="right"))
        if k >= 2:
            self.mesh.lines = np.concatenate([[k], np.arange(k)]).astype(np.int64)
        else:
            self.mesh.lines = np.empty(0, dtype=np.int64)
        # place the boat between the two bracketing samples
        i = min(max(k - 1, 0), len(self.times) - 1)
        j = min(i + 1, len(self.times) - 1)
        dt = self.times[j] - self.times[i]
        f = 0.0 if dt <= 0 else float(np.clip((t - self.times[i]) / dt, 0.0, 1.0))
        p = self.pts[i] * (1 - f) + self.pts[j] * f
        self.boat.points = (p / np.linalg.norm(p) * RADII["marker"])[None, :]


class CellFieldLayer(TimeLayer):
    """A scalar over fixed cells that changes with time -- radiation, ice, sea state.

    `frames` is (n_times, n_cells). Values are interpolated linearly between the
    two bracketing frames, which is the same treatment Rad(t, p) gives a GRIB
    field, so what is drawn matches what the simulator would sample.
    """

    name = "field"

    def __init__(self, plotter, cells, frames, times, *, cmap=MARGIN_CMAP,
                 clim=None, radius=None, opacity=1.0):
        self.times = np.asarray(times, dtype=float)
        self.frames = np.asarray(frames, dtype=float)
        self.mesh = cells_to_mesh(cells, radius or RADII["cells"])
        self.mesh.cell_data["value"] = self.frames[0].copy()
        self.actor = plotter.add_mesh(
            self.mesh, scalars="value", cmap=cmap,
            clim=clim or (float(self.frames.min()), float(self.frames.max())),
            lighting=False, opacity=opacity, show_scalar_bar=False,
        )

    def span(self) -> tuple[float, float]:
        return float(self.times[0]), float(self.times[-1])

    def update(self, t: float) -> None:
        j = int(np.clip(np.searchsorted(self.times, t), 1, len(self.times) - 1))
        i = j - 1
        dt = self.times[j] - self.times[i]
        f = 0.0 if dt <= 0 else float(np.clip((t - self.times[i]) / dt, 0.0, 1.0))
        self.mesh.cell_data["value"] = self.frames[i] * (1 - f) + self.frames[j] * f


class ClockLayer(TimeLayer):
    """A heads-up readout of the current time."""

    name = "clock"

    # index of the lower-left corner in vtkCornerAnnotation
    _LOWER_LEFT = 0

    def __init__(self, plotter, fmt=None, span=(0.0, 1.0)):
        self.plotter = plotter
        self.fmt = fmt or (lambda t: f"t = {t:.1f}")
        self._span = span
        # Built once and then written in place. Calling add_text per frame would
        # tear down and re-add an actor 30 times a second, and that churn is not
        # free -- it dirties the renderer's actor list on every tick, which is
        # exactly when the interactor is also trying to redraw.
        self.actor = plotter.add_text(
            self.fmt(span[0]), position="lower_left", font_size=11, color=INK, name="clock"
        )

    def span(self) -> tuple[float, float]:
        return self._span

    def update(self, t: float) -> None:
        self.actor.SetText(self._LOWER_LEFT, self.fmt(t))


# --------------------------------------------------------------------------
# framing
# --------------------------------------------------------------------------


def camera_distance(bbox) -> float:
    """Pull back far enough to hold the built area, and to leave dark margin
    around the globe for the title and scalar bar to sit on."""
    if bbox is None:
        return 4.6
    lat0, lon0, lat1, lon1 = bbox
    return 2.6 + 2.0 * min(max(lat1 - lat0, lon1 - lon0), 180.0) / 180.0


def load_track(path: str) -> dict[str, np.ndarray]:
    """A track written by demo_run.py, or anything with the same keys.

    `lat`, `lng` and `time` are required. Every other 1-D array of the same
    length is offered as a plottable scalar, so a new channel is a new key.
    """
    z = np.load(path)
    track = {k: z[k] for k in z.files}
    missing = {"lat", "lng", "time"} - set(track)
    if missing:
        raise SystemExit(f"{path}: track is missing {sorted(missing)}")
    return track


# --------------------------------------------------------------------------
# the globe
# --------------------------------------------------------------------------


class GlobeView:
    """A 3D globe you add layers to. Every add_* returns its actor."""

    def __init__(self, size=(1400, 1000), off_screen: bool = False):
        self.plotter = pv.Plotter(window_size=list(size), off_screen=off_screen)
        self.plotter.set_background(SURFACE)
        self.actors: dict[str, object] = {}
        self.timeline = Timeline()
        self._slider = None
        self.following = False
        self._interacting = False

    # -- base layers -------------------------------------------------------

    def add_planet(self, color: str = LAND):
        """The sphere under everything. Land is whatever no cell covers."""
        globe = pv.Sphere(radius=RADII["planet"], theta_resolution=180, phi_resolution=180)
        a = self.plotter.add_mesh(globe, color=color, smooth_shading=True, specular=0.0)
        self.actors["planet"] = a
        return a

    def add_coastlines(self, ne_res: str = "50m", color: str = COAST, width: float = 1.0):
        import cartopy.io.shapereader as shpreader

        path = shpreader.natural_earth(resolution=ne_res, category="physical", name="coastline")
        blocks = []
        for geom in shpreader.Reader(path).geometries():
            parts = geom.geoms if hasattr(geom, "geoms") else [geom]
            for part in parts:
                xy = np.asarray(part.coords)
                if len(xy) < 2:
                    continue
                pts = lonlat_to_xyz(xy[:, 1], xy[:, 0], RADII["coast"])
                blocks.append(pv.lines_from_points(pts))
        if not blocks:
            return None
        a = self.plotter.add_mesh(
            pv.merge(blocks), color=color, line_width=width, lighting=False
        )
        self.actors["coast"] = a
        return a

    def add_graticule(self, step: int = 15, color: str = GRATICULE, opacity: float = 0.3):
        blocks = []
        for lon in range(-180, 180, step):
            lat = np.linspace(-90, 90, 91)
            blocks.append(
                pv.lines_from_points(lonlat_to_xyz(lat, np.full_like(lat, lon), RADII["graticule"]))
            )
        for lat in range(-90 + step, 90, step):
            lon = np.linspace(-180, 180, 181)
            blocks.append(
                pv.lines_from_points(lonlat_to_xyz(np.full_like(lon, lat), lon, RADII["graticule"]))
            )
        a = self.plotter.add_mesh(
            pv.merge(blocks), color=color, line_width=1, lighting=False, opacity=opacity
        )
        self.actors["graticule"] = a
        return a

    # -- data layers -------------------------------------------------------

    def add_cells(
        self,
        cells,
        values=None,
        *,
        name: str = "cells",
        color: str | None = None,
        cmap=MARGIN_CMAP,
        clim=None,
        scalar_bar: str | None = None,
        radius: float | None = None,
        opacity: float = 1.0,
    ):
        """Draw H3 cells, flat-shaded by `values` or in a single colour."""
        mesh = cells_to_mesh(cells, radius or RADII["cells"])
        if mesh.n_points == 0:
            return None
        kw = dict(lighting=False, opacity=opacity, show_scalar_bar=False)
        if values is not None:
            mesh.cell_data["value"] = np.asarray(values, float)
            kw.update(scalars="value", cmap=cmap, clim=clim)
        else:
            kw["color"] = color
        a = self.plotter.add_mesh(mesh, **kw)
        if scalar_bar and values is not None:
            self.plotter.add_scalar_bar(
                title=scalar_bar, color=INK, title_font_size=15, label_font_size=12,
                n_labels=5, width=0.28, height=0.05, position_x=0.36, position_y=0.06,
            )
        self.actors[name] = a
        return a

    def add_track(
        self,
        lat,
        lng,
        values=None,
        *,
        times=None,
        name: str = "track",
        color: str = TRACK,
        width: float = 4.0,
        cmap=None,
    ):
        """A course over the globe.

        `values` colours it by any per-point scalar. Passing `times` makes it a
        time layer instead of a static line: the course is then drawn as it is
        sailed, with the boat at the head, and it registers with the timeline.
        """
        if times is not None:
            layer = TrackLayer(
                self.plotter, lat, lng, times, values=values, cmap=cmap,
                color=color, width=width,
            )
            self.timeline.add(layer)
            self.actors[name] = layer.actor
            return layer
        pts = lonlat_to_xyz(lat, lng, RADII["track"])
        line = pv.lines_from_points(pts)
        kw = dict(line_width=width, lighting=False, show_scalar_bar=False)
        if values is not None:
            line.point_data["value"] = np.asarray(values, float)
            kw.update(scalars="value", cmap=cmap or MARGIN_CMAP)
        else:
            kw["color"] = color
        a = self.plotter.add_mesh(line, **kw)
        self.actors[name] = a
        return a

    def add_cell_field(self, cells, frames, times, **kw):
        """A time-varying scalar over cells -- radiation, ice, sea state."""
        layer = CellFieldLayer(self.plotter, cells, frames, times, **kw)
        self.timeline.add(layer)
        self.actors[layer.name] = layer.actor
        return layer

    def add_clock(self, fmt=None):
        span = self.timeline.span
        return self.timeline.add(ClockLayer(self.plotter, fmt=fmt, span=span))

    def add_markers(self, lat, lng, labels=None, *, name="markers", color=TRACK, size=9.0):
        pts = lonlat_to_xyz(lat, lng, RADII["marker"])
        a = self.plotter.add_points(
            pts, color=color, point_size=size, render_points_as_spheres=True
        )
        self.actors[name] = a
        if labels:
            self.plotter.add_point_labels(
                pts, labels, text_color=INK, font_size=11, shape=None,
                always_visible=False, show_points=False,
            )
        return a

    # -- chrome ------------------------------------------------------------

    def add_title(self, text: str, subtitle: str = ""):
        self.plotter.add_text(text, position="upper_left", font_size=13, color=INK)
        if subtitle:
            self.plotter.add_text(
                subtitle, position=(0.02, 0.90), viewport=True, font_size=9, color=COAST
            )

    def look_at(self, lat: float, lng: float, distance: float = 3.0):
        eye = lonlat_to_xyz([lat], [lng], np.clip(distance, *ZOOM_LIMITS))[0]
        self.plotter.camera_position = [tuple(eye), (0, 0, 0), (0, 0, 1)]
        self.align_north()
        self._scale_rotation()

    def align_north(self, *_args) -> None:
        """Put north back at the top of the screen without moving the camera.

        Only the roll changes: the up vector becomes whatever is left of the pole
        axis once the component along the view direction is removed. Looking
        straight down a pole there is nothing left, and any roll is as good as
        another, so it is left alone rather than snapped to an arbitrary choice.
        """
        cam = self.plotter.camera
        pos = np.asarray(cam.position, dtype=float)
        forward = np.asarray(cam.focal_point, dtype=float) - pos
        n = np.linalg.norm(forward)
        if n < 1e-12:
            return
        forward /= n
        up = np.array([0.0, 0.0, 1.0]) - forward * np.dot([0.0, 0.0, 1.0], forward)
        if np.linalg.norm(up) < 1e-6:
            return  # camera is over a pole; roll is undefined
        cam.up = tuple(up / np.linalg.norm(up))
        self.plotter.render()

    def centre_on(self, lat: float, lng: float) -> None:
        """Swing to a lat/lon, keeping the current zoom and putting north up."""
        distance = float(np.linalg.norm(self.plotter.camera.position))
        self.look_at(lat, lng, distance)

    def boat_position(self) -> tuple[float, float] | None:
        """Where the boat is right now, or None if no track is loaded."""
        for layer in self.timeline.layers:
            if isinstance(layer, TrackLayer):
                return xyz_to_lonlat(np.asarray(layer.boat.points[0], dtype=float))
        return None

    def centre_on_boat(self, *_args) -> None:
        where = self.boat_position()
        if where is None:
            print("  no track loaded -- nothing to centre on")
            return
        self.centre_on(*where)

    def toggle_follow(self, *_args) -> None:
        """Keep the camera on the boat as it moves, rather than once."""
        if self.boat_position() is None:
            print("  no track loaded -- nothing to follow")
            return
        self.following = not self.following
        print(f"  follow boat: {'on' if self.following else 'off'}")
        if self.following:
            self.centre_on_boat()

    def _clamp_zoom(self, *_args) -> None:
        """Keep the camera between skimming the surface and losing the globe.

        VTK's dolly is unbounded, so a couple of extra scroll clicks either put
        the camera inside the planet or leave it so far out the map is a speck,
        and getting back is fiddly.
        """
        cam = self.plotter.camera
        pos = np.asarray(cam.position, dtype=float)
        r = float(np.linalg.norm(pos))
        clamped = float(np.clip(r, *ZOOM_LIMITS))
        if r > 1e-12 and clamped != r:
            cam.position = tuple(pos / r * clamped)

    def _scale_rotation(self, *_args) -> None:
        """Slow the drag-to-rotate as the camera closes on the surface.

        VTK's trackball turns the camera by a fixed *angle* per pixel dragged, so
        the speed that feels right looking at the whole globe sends the view
        skidding once zoomed into a coastline. Scaling the motion factor by height
        above the surface keeps the ground speed under the cursor roughly constant
        instead of the angular speed.
        """
        try:
            style = self.plotter.iren.interactor.GetInteractorStyle()
        except AttributeError:
            return  # off-screen: no interactor to tune
        if not hasattr(style, "SetMotionFactor"):
            return
        altitude = max(float(np.linalg.norm(self.plotter.camera.position)) - 1.0, 1e-3)
        style.SetMotionFactor(float(np.clip(10.0 * altitude / 2.0, 0.35, 10.0)))

    def _watch_zoom(self) -> None:
        """Re-tune the rotation speed whenever the camera might have moved."""
        try:
            iren = self.plotter.iren
        except AttributeError:
            return
        # Deliberately not InteractionEvent: that fires on every mouse motion
        # during a drag, and doing work there competes with the redraw it is
        # trying to service. The end of an interaction is soon enough.
        for event in (
            "MouseWheelForwardEvent",
            "MouseWheelBackwardEvent",
            "EndInteractionEvent",
        ):
            try:
                iren.add_observer(event, self._on_camera_change)
            except Exception:  # noqa: BLE001 - observer set is best-effort
                pass

    def _on_camera_change(self, *_args) -> None:
        self._clamp_zoom()
        self._scale_rotation()

    def bind_keys(self) -> None:
        """Camera keys, available whether or not anything is animating."""
        # Wrapped in no-arg lambdas: add_key_event rejects any parameter without
        # a default, and *args counts as one, so binding these methods directly
        # raises even though they take nothing required.
        self.plotter.add_key_event("n", lambda: self.align_north())
        self.plotter.add_key_event("b", lambda: self.toggle_follow())

    def show(self, title: str = "boatforge - ocean map"):
        self._watch_zoom()
        self.bind_keys()
        print("  n north-up   b follow boat   q quit")
        self.plotter.show(title=title)

    def save(self, path: str):
        self.plotter.screenshot(path)
        print(f"wrote {path}")

    # -- playback ----------------------------------------------------------

    def play(self, fps: int = 30, speed: float = 1.0, loop: bool = True,
             title: str = "boatforge - ocean map"):
        """Open the window and animate, with the camera live throughout.

        Playback runs off a VTK timer event, not a loop. That is the part that
        matters: a `for t in frames` loop owns the thread and the window freezes,
        whereas the interactor keeps servicing drag and scroll between ticks, so
        the globe can be rotated and zoomed while the boat is moving.
        """
        tl = self.timeline
        if not tl.layers:
            return self.show(title=title)

        t0, t1 = tl.span
        tl.speed = speed
        tl.seek(t0)
        step = 1.0 / fps

        def on_slider(value):
            tl.playing = False  # scrubbing takes over from playback
            tl.seek(value)

        self._slider = self.plotter.add_slider_widget(
            on_slider, (t0, t1), value=t0, title="", color=COAST,
            pointa=(0.30, 0.06), pointb=(0.70, 0.06), interaction_event="always",
            slider_width=0.02, tube_width=0.004,
        )

        def sync_slider():
            if self._slider is not None:
                self._slider.GetRepresentation().SetValue(tl.t)

        # While the mouse is driving the camera, the interactor is already
        # redrawing as fast as it can. Forcing another full render from the
        # animation timer on top of that queues work faster than it completes,
        # and the view visibly falls behind the cursor -- which is why pausing
        # made zooming feel fine. The clock keeps advancing; only our own extra
        # render is dropped, and the interactor's redraw shows the new frame
        # anyway.
        self._interacting = False

        def interaction(flag):
            def go(*_a):
                self._interacting = flag
            return go

        def toggle():
            tl.playing = not tl.playing

        def restart():
            tl.seek(t0)
            sync_slider()

        def nudge(sign):
            def go():
                tl.playing = False
                tl.seek(tl.t + sign * (t1 - t0) / 200.0)
                sync_slider()
            return go

        def rate(factor):
            def go():
                tl.speed = float(np.clip(tl.speed * factor, 0.05, 200.0))
            return go

        self.bind_keys()
        self.plotter.add_key_event("space", toggle)
        self.plotter.add_key_event("r", restart)
        self.plotter.add_key_event("Left", nudge(-1))
        self.plotter.add_key_event("Right", nudge(+1))
        self.plotter.add_key_event("bracketleft", rate(0.5))
        self.plotter.add_key_event("bracketright", rate(2.0))

        # Adaptive throttle. A timer at `fps` will happily ask for another render
        # before the last one finished, and the queue only grows -- the view then
        # trails the mouse by seconds and never catches up. Measuring how long a
        # render actually takes and refusing to start one more often than that
        # keeps the loop honest: the animation drops frames instead of falling
        # behind, and zooming stays responsive because the interactor gets a share.
        self._render_s = 1.0 / fps
        self._last_render = 0.0

        def tick(_step):
            if not tl.playing:
                return
            tl.advance(step, loop=loop)
            if self._interacting:
                return  # the interactor's own redraw will pick this up
            now = time.perf_counter()
            if now - self._last_render < max(step, self._render_s):
                return
            sync_slider()
            if self.following:
                where = self.boat_position()
                if where is not None:
                    self.centre_on(*where)
            self._last_render = now
            self.plotter.render()
            self._render_s = 0.8 * self._render_s + 0.2 * (time.perf_counter() - now)

        self.plotter.add_timer_event(max_steps=10**9, duration=int(1000 / fps), callback=tick)
        try:
            iren = self.plotter.iren
            iren.add_observer("StartInteractionEvent", interaction(True))
            iren.add_observer("EndInteractionEvent", interaction(False))
        except (AttributeError, RuntimeError):
            pass
        self._watch_zoom()
        print(
            "  space play/pause   left/right step   [ ] speed   r restart\n"
            "  n north-up        b follow boat                     q quit"
        )
        self.plotter.show(title=title)

    def record(self, path: str, n_frames: int = 240, fps: int = 30):
        """Write the animation out. Interactive playback is `play`; this is export."""
        t0, t1 = self.timeline.span
        movie = path.endswith((".mp4", ".gif"))
        if movie:
            (self.plotter.open_gif if path.endswith(".gif") else self.plotter.open_movie)(path)
        else:
            Path(path).mkdir(parents=True, exist_ok=True)
        self.plotter.show(auto_close=False)
        for i, t in enumerate(np.linspace(t0, t1, n_frames)):
            self.timeline.seek(t)
            # Force the frame before grabbing it. Changing a mesh's topology
            # happens to trigger a redraw on its own, but moving a point or
            # rewriting a text actor does not, so without this the capture shows
            # the previous frame's boat and clock against the current track.
            self.plotter.render()
            if movie:
                self.plotter.write_frame()
            else:
                self.plotter.screenshot(str(Path(path) / f"frame_{i:05d}.png"))
        self.plotter.close()
        print(f"wrote {n_frames} frames to {path}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("npz", help="artifact from  map_gen.py --save")
    p.add_argument("--show-excluded", action="store_true", help="draw rejected water too")
    p.add_argument(
        "--color-by",
        default="margin",
        choices=("margin", "budget"),
        help="margin: how legal. budget: how far before the next map lookup.",
    )
    p.add_argument("--track", metavar="NPZ", help="a track from demo_run.py, animated")
    p.add_argument(
        "--track-scalar",
        default="battery",
        help="which channel of the track to shade it by (default battery)",
    )
    p.add_argument("--coastlines", default="50m", choices=("10m", "50m", "110m", "none"))
    p.add_argument("--graticule", type=int, default=15, metavar="DEG", help="0 to disable")
    p.add_argument("--view", type=float, nargs=2, metavar=("LAT", "LON"))
    p.add_argument(
        "--distance", type=float, help="camera distance in earth radii (default: fits the bbox)"
    )
    p.add_argument("--size", type=int, nargs=2, default=(1400, 1000), metavar=("W", "H"))
    p.add_argument("--save", metavar="PNG", help="render a still off-screen to a file")
    p.add_argument("--record", metavar="PATH", help="write the animation (.mp4, .gif, or a dir)")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--frames", type=int, default=240, help="frames to write with --record")
    p.add_argument(
        "--speed", type=float, default=6.0, help="hours of course per second of playback"
    )
    args = p.parse_args()

    art = NavMap.load(args.npz)
    legal, margin, budget, illegal = art.leaves()
    values = budget if args.color_by == "budget" else margin
    bar = {"margin": "clearance margin (km)", "budget": "travel budget (km)"}[args.color_by]
    print(f"K={art.K} km, W={art.W} km, res {art.res_min}..{art.res_base}")
    print(f"{len(legal):,} navigable cells, {len(illegal):,} excluded")
    print(f"margin {margin.min():.1f}..{margin.max():.1f} km")
    print(f"budget {budget.min():.1f}..{budget.max():.1f} km")

    view = GlobeView(size=tuple(args.size), off_screen=bool(args.save))
    view.add_planet()
    if args.show_excluded and len(illegal):
        # Held translucent on purpose: a coarse cell is "wholly illegal" on the
        # evidence of the water sampled inside it, but the hexagon also spans any
        # land in there, which was never sampled. Letting the coastline read
        # through keeps the claim honest.
        view.add_cells(illegal, name="excluded", color=EXCLUDED, opacity=0.55)
    view.add_cells(
        legal, values, name="navigable", clim=(0, float(np.percentile(values, 98))),
        scalar_bar=bar,
    )
    if args.coastlines != "none":
        view.add_coastlines(args.coastlines)
    if args.graticule:
        view.add_graticule(args.graticule)

    target = art.centre
    if args.track:
        tr = load_track(args.track)
        lat, lng, hours = tr["lat"], tr["lng"], tr["time"]
        # Centre on the course, not the bbox: the bbox centre is often over water
        # the (K, W) rules excluded, which points the camera at nothing.
        target = (float(np.mean(lat)), float(np.mean(lng)))
        scalar = tr.get(args.track_scalar)
        if scalar is not None and scalar.shape != lat.shape:
            scalar = None  # e.g. speed_kmh, a header value rather than a channel
        view.add_track(lat, lng, values=scalar, times=hours, width=5)
        view.add_markers([lat[0], lat[-1]], [lng[0], lng[-1]], labels=["start", "goal"])
        view.add_clock(lambda t: f"T+{t:5.1f} h   day {int(t // 24) + 1}")
        channels = [k for k, v in tr.items() if getattr(v, "shape", None) == lat.shape]
        print(f"track: {len(lat)} points over {hours[-1]:.1f} h, channels {channels}")

    view.add_title(
        f"navigable ocean   K={art.K:g} km   W={art.W:g} km",
        f"{len(legal):,} cells, res {art.res_min}-{art.res_base}   |   shaded by {args.color_by}"
        + ("   |   excluded water in grey" if args.show_excluded else ""),
    )
    view.look_at(
        *(args.view if args.view else target),
        distance=args.distance or camera_distance(art.bbox),
    )

    if args.record:
        view.record(args.record, n_frames=args.frames, fps=args.fps)
    elif args.save:
        view.save(args.save)
    elif view.timeline.layers:
        view.play(fps=args.fps, speed=args.speed)
    else:
        view.show()


if __name__ == "__main__":
    main()
