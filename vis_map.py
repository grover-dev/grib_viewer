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

Drag to rotate, scroll to zoom, `q` to quit.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import h3
import numpy as np
import pyvista as pv
from matplotlib.colors import LinearSegmentedColormap

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

# layer altitudes, in earth radii -- each layer clears the one beneath it
RADII = {
    "planet": 1.0,
    "cells": 1.0015,
    "coast": 1.0035,
    "graticule": 1.0045,
    "track": 1.0075,
    "marker": 1.0095,
}


def lonlat_to_xyz(lat, lng, r: float = 1.0) -> np.ndarray:
    """(n, 3) points on a sphere of radius r. Degrees in, cartesian out."""
    p, t = np.radians(np.asarray(lat, float)), np.radians(np.asarray(lng, float))
    return np.column_stack([r * np.cos(p) * np.cos(t), r * np.cos(p) * np.sin(t), r * np.sin(p)])


def great_circle(a: tuple[float, float], b: tuple[float, float], n: int = 64) -> np.ndarray:
    """(n, 2) lat/lng along the great circle from a to b, by slerp."""
    v1, v2 = lonlat_to_xyz(*[[x] for x in a])[0], lonlat_to_xyz(*[[x] for x in b])[0]
    omega = np.arccos(np.clip(np.dot(v1, v2), -1, 1))
    if omega < 1e-9:
        return np.array([a, b])
    s = np.linspace(0, 1, n)[:, None]
    v = (np.sin((1 - s) * omega) * v1 + np.sin(s * omega) * v2) / np.sin(omega)
    return np.column_stack(
        [np.degrees(np.arcsin(v[:, 2])), np.degrees(np.arctan2(v[:, 1], v[:, 0]))]
    )


def cells_to_mesh(cells, radius: float) -> pv.PolyData:
    """One PolyData holding every H3 cell as a face, so it draws in a single pass.

    Faces are variable length: hexagons give 6 vertices, the 12 pentagons give 5.
    """
    verts, faces = [], []
    offset = 0
    for c in cells:
        b = h3.cell_to_boundary(h3.int_to_str(int(c)))
        n = len(b)
        verts.append(lonlat_to_xyz([p[0] for p in b], [p[1] for p in b], radius))
        faces.append(np.r_[n, np.arange(offset, offset + n)])
        offset += n
    if not verts:
        return pv.PolyData()
    return pv.PolyData(np.vstack(verts), np.concatenate(faces))


# --------------------------------------------------------------------------
# the artifact written by map_gen.py --save
# --------------------------------------------------------------------------


@dataclass
class Artifact:
    K: float
    W: float
    res_min: int
    res_base: int
    bbox: tuple[float, float, float, float] | None = None
    levels: dict[int, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)

    @property
    def centre(self) -> tuple[float, float]:
        if self.bbox is None:
            return 0.0, 0.0
        lat0, lon0, lat1, lon1 = self.bbox
        return (lat0 + lat1) / 2, (lon0 + lon1) / 2

    @property
    def camera_distance(self) -> float:
        """Pull back far enough to hold the built area, and to leave dark margin
        around the globe for the title and scalar bar to sit on."""
        if self.bbox is None:
            return 4.6
        lat0, lon0, lat1, lon1 = self.bbox
        span = max(lat1 - lat0, lon1 - lon0)
        return 2.6 + 2.0 * min(span, 180.0) / 180.0

    @classmethod
    def load(cls, path: str) -> Artifact:
        z = np.load(path)
        art = cls(
            K=float(z["K"]), W=float(z["W"]),
            res_min=int(z["res_min"]), res_base=int(z["res_base"]),
            bbox=tuple(z["bbox"]) if "bbox" in z else None,
        )
        for r in range(art.res_min, art.res_base + 1):
            if f"r{r}_id" in z:
                art.levels[r] = (z[f"r{r}_id"], z[f"r{r}_mm"])
        return art

    def leaves(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """(legal ids, their margin, their travel budget, illegal ids).

        A cell is a leaf when it decided -- wholly legal or wholly illegal. Cells
        that were undecided have children stored beneath them, so drawing only the
        leaves tiles the area once. H3 does not nest exactly, so the tiling has
        hairline seams and slivers of overlap; harmless for a picture.

        Margin and budget differ wherever a basin's entrance is narrower than its
        interior: inside the Mediterranean the margin is pinned by Gibraltar while
        the budget follows the local coastline. `--color-by budget` shows it.
        """
        legal, margin, budget, illegal = [], [], [], []
        for _r, (ids, mm) in sorted(self.levels.items()):
            lo, hi = mm[:, 0], mm[:, 1]
            ok = lo >= 0
            legal.append(ids[ok])
            margin.append(lo[ok])
            budget.append(mm[ok, 2] if mm.shape[1] > 2 else lo[ok])
            illegal.append(ids[hi < 0])
        cat = lambda xs, dt=float: np.concatenate(xs) if xs else np.array([], dt)  # noqa: E731
        return cat(legal, np.uint64), cat(margin), cat(budget), cat(illegal, np.uint64)


# --------------------------------------------------------------------------
# the globe
# --------------------------------------------------------------------------


class GlobeView:
    """A 3D globe you add layers to. Every add_* returns its actor."""

    def __init__(self, size=(1400, 1000), off_screen: bool = False):
        self.plotter = pv.Plotter(window_size=list(size), off_screen=off_screen)
        self.plotter.set_background(SURFACE)
        self.actors: dict[str, object] = {}

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
        name: str = "track",
        color: str = TRACK,
        width: float = 4.0,
        cmap=None,
    ):
        """A course over the globe. `values` colours it by any per-point scalar."""
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
        eye = lonlat_to_xyz([lat], [lng], distance)[0]
        self.plotter.camera_position = [tuple(eye), (0, 0, 0), (0, 0, 1)]

    def show(self, title: str = "boatforge - ocean map"):
        self.plotter.show(title=title)

    def save(self, path: str):
        self.plotter.screenshot(path)
        print(f"wrote {path}")


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
    p.add_argument("--demo-track", action="store_true", help="overlay an example course")
    p.add_argument("--coastlines", default="50m", choices=("10m", "50m", "110m", "none"))
    p.add_argument("--graticule", type=int, default=15, metavar="DEG", help="0 to disable")
    p.add_argument("--view", type=float, nargs=2, metavar=("LAT", "LON"))
    p.add_argument(
        "--distance", type=float, help="camera distance in earth radii (default: fits the bbox)"
    )
    p.add_argument("--size", type=int, nargs=2, default=(1400, 1000), metavar=("W", "H"))
    p.add_argument("--save", metavar="PNG", help="render off-screen to a file")
    args = p.parse_args()

    art = Artifact.load(args.npz)
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

    if args.demo_track:
        # An Atlantic course, kept clear of the strait so it stays in navigable
        # water whatever (K, W) the map was built with. It is a placeholder for a
        # real solver track: the scalar is arbitrary, and stands in for whatever
        # gets plotted later -- battery, speed, collected energy.
        legs = [(42.0, -10.5), (38.5, -11.0), (35.0, -9.5), (31.5, -11.5)]
        pts = np.vstack([great_circle(a, b, 48) for a, b in zip(legs, legs[1:])])
        view.add_track(pts[:, 0], pts[:, 1], width=5)
        view.add_markers(
            [legs[0][0], legs[-1][0]], [legs[0][1], legs[-1][1]], labels=["start", "goal"]
        )

    view.add_title(
        f"navigable ocean   K={art.K:g} km   W={art.W:g} km",
        f"{len(legal):,} cells, res {art.res_min}-{art.res_base}   |   shaded by {args.color_by}"
        + ("   |   excluded water in grey" if args.show_excluded else ""),
    )
    view.look_at(
        *(args.view if args.view else art.centre),
        distance=args.distance or art.camera_distance,
    )

    view.save(args.save) if args.save else view.show()


if __name__ == "__main__":
    main()
