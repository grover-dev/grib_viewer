"""Shared pieces of the ocean-map pipeline: geodesy, the map artifact, and the
clearance budget.

Three scripts sit on top of this and none of them talk to each other:

    map_gen.py    builds a map from coastline data and writes the .npz
    demo_run.py   reads the .npz, produces a track
    vis_map.py    reads both and draws them

The artifact format is defined once, here, because two separate readers of one
format drift apart.

The .npz layout
---------------

Header: `K`, `W` (km, baked in at build time), `res_min`, `res_base`, and `bbox`
as (lat_min, lon_min, lat_max, lon_max).

Then per resolution r in res_min..res_base:

    r{r}_id   uint64 (n,)     H3 cell ids as integers, ascending
    r{r}_mm   float32 (n, 3)  [min_margin, max_margin, min_budget], km

Row i of `r{r}_mm` describes cell `r{r}_id[i]`. A level may be absent if it stored
nothing. Only cells that a query might still reach are stored: a decided cell
keeps no children, so the count is far below the number of cells measured.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Integer cell ids throughout: as Python str they cost ~130 bytes each in a list
# and ~250 in a dict; as uint64 in a numpy array, 8.
from h3.api import basic_int as h3

EARTH_R_KM = 6371.0088
GLOBAL_BBOX = (-90.0, -180.0, 90.0, 180.0)


# --------------------------------------------------------------------------
# geodesy
# --------------------------------------------------------------------------


def great_circle_km(lat1, lng1, lat2, lng2) -> np.ndarray:
    """Haversine distance in km. Vectorised over any broadcastable inputs."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = (
        np.sin((p2 - p1) / 2) ** 2
        + np.cos(p1) * np.cos(p2) * np.sin((np.radians(lng2) - np.radians(lng1)) / 2) ** 2
    )
    return 2 * EARTH_R_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def leg_lengths_km(lat, lng) -> np.ndarray:
    """Distance between consecutive points of a track, in km. Length n-1."""
    return great_circle_km(lat[:-1], lng[:-1], lat[1:], lng[1:])


def lonlat_to_xyz(lat, lng, r: float = 1.0) -> np.ndarray:
    """(n, 3) points on a sphere of radius r. Degrees in, cartesian out."""
    p, t = np.radians(np.asarray(lat, float)), np.radians(np.asarray(lng, float))
    return np.column_stack([r * np.cos(p) * np.cos(t), r * np.cos(p) * np.sin(t), r * np.sin(p)])


def xyz_to_lonlat(v: np.ndarray) -> tuple[float, float]:
    """Inverse of lonlat_to_xyz for a single vector, in degrees."""
    v = v / np.linalg.norm(v)
    return float(np.degrees(np.arcsin(v[2]))), float(np.degrees(np.arctan2(v[1], v[0])))


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


def initial_bearing(lat1, lng1, lat2, lng2) -> float:
    """Great-circle bearing at the start point, degrees clockwise from north."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(lng2 - lng1)
    y = np.sin(dl) * np.cos(p2)
    x = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl)
    return float(np.degrees(np.arctan2(y, x)) % 360.0)


def step_along(lat, lng, bearing_deg: float, distance_km: float) -> tuple[float, float]:
    """Move `distance_km` from a point along a bearing, on the sphere."""
    d = distance_km / EARTH_R_KM
    p, b = np.radians(lat), np.radians(bearing_deg)
    p2 = np.arcsin(np.sin(p) * np.cos(d) + np.cos(p) * np.sin(d) * np.cos(b))
    t2 = np.radians(lng) + np.arctan2(
        np.sin(b) * np.sin(d) * np.cos(p), np.cos(d) - np.sin(p) * np.sin(p2)
    )
    return float(np.degrees(p2)), float((np.degrees(t2) + 540) % 360 - 180)


# --------------------------------------------------------------------------
# the map artifact
# --------------------------------------------------------------------------


@dataclass
class NavMap:
    """Signed clearance over a pyramid of H3 resolutions, at a fixed K and W.

    Each cell stores the min and max margin over its footprint. A query walks
    coarse to fine and stops at the first level that decides, so open water costs
    one lookup and only coastlines need the descent.
    """

    K: float
    W: float
    res_min: int
    res_base: int
    levels: dict[int, dict[int, tuple[float, float, float]]]
    bbox: tuple[float, float, float, float] | None = None

    @property
    def n_entries(self) -> int:
        return sum(len(v) for v in self.levels.values())

    @property
    def centre(self) -> tuple[float, float]:
        if self.bbox is None:
            return 0.0, 0.0
        lat0, lon0, lat1, lon1 = self.bbox
        return (lat0 + lat1) / 2, (lon0 + lon1) / 2

    def decide(self, lat: float, lng: float) -> tuple[float, float, int]:
        """(margin, travel budget, resolution that decided).

        Both numbers are the deciding cell's bound rather than the point's exact
        value -- a wholly-legal cell reports its minimum, so each is a guarantee
        for every point in it.

        Margin and budget are different quantities and it matters. Margin answers
        "is this legal", and its `access_width` half is a property of the whole
        basin: inside the Mediterranean it is fixed by the width of Gibraltar no
        matter where you stand. Spending that as a travel budget would force a
        lookup every couple of km across the entire basin. Once a point is known
        legal, what limits movement is only the local shore distance, because
        staying further than K + W/2 off the beach keeps the corridor behind you
        at least W + 2K wide:

            budget = dist_to_shore - (K + W/2)

        Descent re-indexes the position at each level rather than following
        parent/child links: H3 hexagons do not nest.
        """
        lo = budget = 0.0
        for r in range(self.res_min, self.res_base + 1):
            entry = self.levels[r].get(h3.latlng_to_cell(lat, lng, r))
            if entry is None:
                return float("-inf"), 0.0, r  # outside the built area
            lo, hi, budget = entry
            if lo >= 0.0:
                return lo, budget, r
            if hi < 0.0:
                return hi, 0.0, r
        return lo, budget, self.res_base

    def clearance(self, lat: float, lng: float) -> float:
        """Signed km of margin: >= 0 is navigable."""
        return self.decide(lat, lng)[0]

    def budget(self, lat: float, lng: float) -> float:
        """How far the boat may travel in any direction before re-checking."""
        return self.decide(lat, lng)[1]

    def legal(self, lat: float, lng: float) -> bool:
        return self.clearance(lat, lng) >= 0.0

    def shore_distance(self, lat: float, lng: float) -> float:
        """Distance to the nearest coast, implied by the stored budget.

        `budget = dist_to_shore - (K + W/2)`, so this inverts it. Two caveats,
        both from the fact that the budget is a *cell* minimum rather than a point
        measurement:

        * It is a LOWER bound. Far offshore a coarse cell decides the query, and
          its minimum understates the true distance badly -- a res-2 cell mid
          Atlantic might report 60 km where the truth is 400.
        * Near a coast, where cells resolve fine, it is tight. That is the only
          region where it carries any weight, which is what makes it usable for
          coastal steering despite the above.

        Returns inf outside the built area and nan on non-navigable water, so a
        caller cannot silently treat "no answer" as "close to shore".

        Note this does NOT stop at the deciding level the way `decide` does.
        Legality is answered by the coarsest cell that can answer it, which in open
        water is res 2 -- and a res-2 budget is one number across a cell hundreds
        of km wide, so every candidate heading a controller tries scores
        identically. Here the descent runs to the deepest cell actually stored, so
        the answer is as sharp as the artifact allows. That depth is what
        map_gen's --floor-res buys; without a floor this still flattens out
        offshore.
        """
        margin, _, _ = self.decide(lat, lng)
        if margin == float("-inf"):
            return float("inf")
        if margin < 0.0:
            return float("nan")
        deepest = None
        for r in range(self.res_min, self.res_base + 1):
            entry = self.levels[r].get(h3.latlng_to_cell(lat, lng, r))
            if entry is None:
                break  # nothing finer was stored here
            deepest = entry[2]
        if deepest is None:
            return float("inf")
        return deepest + self.K + self.W / 2.0

    # -- persistence -------------------------------------------------------

    def save(self, path: str) -> None:
        out: dict[str, object] = {
            "K": self.K,
            "W": self.W,
            "res_min": self.res_min,
            "res_base": self.res_base,
            "bbox": np.array(self.bbox if self.bbox else GLOBAL_BBOX),
        }
        for r, lvl in self.levels.items():
            out[f"r{r}_id"] = np.fromiter(lvl.keys(), dtype=np.uint64, count=len(lvl))
            out[f"r{r}_mm"] = np.array(list(lvl.values()), dtype=np.float32)
        np.savez_compressed(path, **out)

    @classmethod
    def load(cls, path: str) -> NavMap:
        z = np.load(path)
        res_min, res_base = int(z["res_min"]), int(z["res_base"])
        levels = {}
        for r in range(res_min, res_base + 1):
            if f"r{r}_id" not in z:
                continue
            ids, mm = z[f"r{r}_id"], z[f"r{r}_mm"]
            levels[r] = {int(i): (float(a), float(b), float(c)) for i, (a, b, c) in zip(ids, mm)}
        return cls(
            K=float(z["K"]),
            W=float(z["W"]),
            res_min=res_min,
            res_base=res_base,
            levels=levels,
            bbox=tuple(z["bbox"]) if "bbox" in z else None,
        )

    def leaves(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """(legal ids, their margin, their travel budget, illegal ids).

        A cell is a leaf when it decided. Undecided cells have children stored
        beneath them, so drawing only the leaves tiles the area once. H3 does not
        nest exactly, so the tiling has hairline seams and slivers of overlap.
        """
        legal, margin, budget, illegal = [], [], [], []
        for _r, lvl in sorted(self.levels.items()):
            ids = np.fromiter(lvl.keys(), dtype=np.uint64, count=len(lvl))
            mm = np.array(list(lvl.values()), dtype=np.float32)
            if not len(ids):
                continue
            ok = mm[:, 0] >= 0
            legal.append(ids[ok])
            margin.append(mm[ok, 0])
            budget.append(mm[ok, 2])
            illegal.append(ids[mm[:, 1] < 0])

        def cat(xs, dt=float):
            return np.concatenate(xs) if xs else np.array([], dt)

        return cat(legal, np.uint64), cat(margin), cat(budget), cat(illegal, np.uint64)


class ClearanceBudget:
    """Skip the map lookup until the boat could plausibly have reached trouble.

    One lookup certifies a disc around the current position, so nothing needs
    checking until the accumulated travel exhausts it. Mid-ocean a single lookup
    covers hundreds of km; near a coast the budget collapses and it checks almost
    every step.
    """

    def __init__(self, nav: NavMap):
        self.nav = nav
        self.budget = 0.0
        self.lookups = 0
        self.steps = 0

    def step(self, lat: float, lng: float, distance_km: float) -> bool:
        """Advance to (lat, lng) having travelled `distance_km`. False if illegal."""
        self.steps += 1
        self.budget -= distance_km
        if self.budget > 0:
            return True
        self.lookups += 1
        margin, budget, _ = self.nav.decide(lat, lng)
        if margin < 0.0:
            self.budget = 0.0
            return False
        self.budget = budget
        return True
