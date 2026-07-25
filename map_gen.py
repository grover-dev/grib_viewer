"""map_gen.py -- build the navigable-ocean map from docs/routing_proposal.md.

The build takes a keep-out distance K and a minimum channel width W and bakes them
into the artifact. Two raw fields are measured first:

    dist_to_shore   km to the nearest non-ocean thing (continent, island, lake edge)
    access_width    width of the narrowest bottleneck on the widest corridor
                    connecting the cell to open ocean

and then collapse into a single number per cell:

    margin = min( dist_to_shore - K,  (access_width - W - 2K) / 2 )

Legal water is margin >= 0. The `W + 2K` term is the coupled K->W semantics of
Stage 2d: the keep-out eats into every channel from both sides, so a corridor needs
to be W + 2K wide to leave W of usable water.

Margin does double duty. It is signed clearance in km, and it drops by at most one
km per km travelled, so it is also how far the boat may move in any direction and
still be legal -- which is what lets the simulator skip almost every map lookup.

Three things here are easy to get subtly wrong, and are called out in the code:

* Distance is measured on the H3 grid, never on a lat/lon raster. A Euclidean
  distance transform in lat/lon finds the nearest pixel in *pixel* space, and the
  true metric is 2:1 anisotropic at 60 degrees -- no cos-latitude factor applied
  afterwards repairs a wrong nearest neighbour.
* Pyramid levels are aggregated by re-indexing each sample's own position, not by
  h3.cell_to_parent. H3's parent relation is index truncation and does not imply
  geometric containment; hexagons do not nest.
* A cell on the edge of the requested bbox is not a shoreline, and a shoreline cell
  is not at distance zero. Getting either wrong invents geography.


Running it
----------

    uv run python map_gen.py                       # defaults: K=0, W=10, res 7
    uv run python map_gen.py -K 5 -W 30            # 5 km keep-out, 30 km channels
    uv run python map_gen.py --res 6               # coarser and ~5x faster
    uv run python map_gen.py --save med.npz        # write the artifact

Each run builds the map, prints a probe table, and shows how many map lookups the
clearance budget skips on a short run west from Gibraltar.

Options:

    -K KM            keep-out distance from shore (default 0)
    -W KM            minimum navigable channel width (default 10)
    --res N          H3 build resolution (default 7)
    --res-min N      coarsest pyramid level (default 2)
    --bbox LAT0 LON0 LAT1 LON1      area to build (default: western Med + Atlantic)
    --seed LAT LON   a point in open ocean, used to define "connected to the sea"
    --ne-res {10m,50m,110m}         Natural Earth coastline detail (default 10m)
    --save NPZ       write the pyramid to an .npz

The default area covers Gibraltar, so it doubles as the validation case: the strait
measures ~14 km, and the Mediterranean should stay open at W=10 and seal at W=30.

    uv run python map_gen.py -K 0 -W 10            # Alboran Sea legal
    uv run python map_gen.py -K 0 -W 30            # Alboran Sea sealed

Resolution matters more than it looks. W can only adjudicate passages the grid can
resolve, so res 7 (~2.1 km spacing) is the coarsest that measures Gibraltar
correctly -- res 6 reports it as 9 km wide and seals the Mediterranean at every W.
The default bbox at res 7 takes ~25 s and ~1 GB; res 6 takes ~5 s.

The first run downloads the Natural Earth ocean polygon (a few MB) into cartopy's
data directory and caches it there.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import h3
import numpy as np
import shapely
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra

EARTH_R_KM = 6371.0088

# (lat_min, lon_min, lat_max, lon_max): Gibraltar, the western Mediterranean, and
# enough Atlantic to be unambiguously open ocean.
GIBRALTAR_BBOX = (30.0, -12.0, 46.0, 20.0)
ATLANTIC_SEED = (35.0, -10.0)


def _log(msg: str, t0: float | None = None) -> float:
    now = time.time()
    print(f"  {msg}{'' if t0 is None else f'  [{now - t0:.1f}s]'}", flush=True)
    return now


def great_circle_km(lat1, lng1, lat2, lng2) -> np.ndarray:
    """Haversine, vectorised, in km."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = (
        np.sin((p2 - p1) / 2) ** 2
        + np.cos(p1) * np.cos(p2) * np.sin((np.radians(lng2) - np.radians(lng1)) / 2) ** 2
    )
    return 2 * EARTH_R_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


# --------------------------------------------------------------------------
# Stage 2a -- the ocean polygon
# --------------------------------------------------------------------------


def load_ocean(ne_res: str = "10m"):
    """Natural Earth's ocean layer, prepared for repeated point tests.

    Inland water is simply not part of this polygon, so lakes and rivers come out
    non-navigable by construction rather than by heuristic.
    """
    import cartopy.io.shapereader as shpreader

    path = shpreader.natural_earth(resolution=ne_res, category="physical", name="ocean")
    ocean = shapely.union_all(list(shpreader.Reader(path).geometries()))
    shapely.prepare(ocean)
    return ocean


def shore_distance_km(boundary, lat: np.ndarray, lng: np.ndarray) -> np.ndarray:
    """True great-circle distance from each point to the coastline.

    Used only to initialise the narrow band -- the cells that touch land. Doing it
    for every cell would be exact but costs ~0.8 ms a point, which is the whole
    reason the interior is filled by marching over the grid instead.
    """
    lines = shapely.shortest_line(shapely.points(lng, lat), boundary)
    c = shapely.get_coordinates(lines).reshape(-1, 2, 2)
    return great_circle_km(c[:, 0, 1], c[:, 0, 0], c[:, 1, 1], c[:, 1, 0])


# --------------------------------------------------------------------------
# Stage 2b-2d -- cells, distance, width
# --------------------------------------------------------------------------


@dataclass
class BaseMap:
    """Raw measurements at the build resolution. Knows nothing about K or W."""

    res: int
    bbox: tuple[float, float, float, float]
    cells: np.ndarray  # uint64 H3 ids, ocean and connected to the seed
    lat: np.ndarray
    lng: np.ndarray
    dist_to_shore: np.ndarray  # km
    access_width: np.ndarray  # km


def enumerate_cells(bbox, res: int) -> list[str]:
    lat0, lon0, lat1, lon1 = bbox
    poly = h3.LatLngPoly([(lat0, lon0), (lat0, lon1), (lat1, lon1), (lat1, lon0)])
    return list(h3.polygon_to_cells(poly, res))


def _neighbour_pairs(cells: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Index pairs for cells adjacent within the set, plus a truncation flag.

    `truncated[i]` marks a cell with a neighbour outside the enumerated set, i.e.
    one on the bbox edge. Those are excluded from shoreline seeding so the box
    boundary does not masquerade as a coast.
    """
    index = {c: i for i, c in enumerate(cells)}
    src, dst = [], []
    truncated = np.zeros(len(cells), dtype=bool)
    for i, c in enumerate(cells):
        found = 0
        for n in h3.grid_disk(c, 1):
            if n == c:
                continue
            j = index.get(n)
            if j is None:
                truncated[i] = True
            else:
                found += 1
                if j > i:
                    src.append(i)
                    dst.append(j)
        if found < 5:  # pentagons legitimately have 5; fewer means truncation
            truncated[i] = True
    return np.asarray(src, dtype=np.int64), np.asarray(dst, dtype=np.int64), truncated


def build_base(
    bbox=GIBRALTAR_BBOX,
    res: int = 7,
    ne_res: str = "10m",
    seed_latlng: tuple[float, float] = ATLANTIC_SEED,
    verbose: bool = True,
) -> BaseMap:
    log = _log if verbose else (lambda *a, **k: time.time())

    t = log("loading ocean polygon")
    ocean = load_ocean(ne_res)
    t = log(f"enumerating res-{res} cells", t)
    cells = enumerate_cells(bbox, res)
    ll = np.array([h3.cell_to_latlng(c) for c in cells])
    lat, lng = ll[:, 0], ll[:, 1]
    n = len(cells)
    t = log(f"{n:,} cells; classifying", t)

    # Classification has no metric, so a point test in lat/lon is safe. This is the
    # prototype's stand-in for burning a boolean raster: same answer, less code.
    is_ocean = shapely.contains_xy(ocean, lng, lat)
    t = log(f"{is_ocean.sum():,} ocean cells; building adjacency", t)

    src, dst, truncated = _neighbour_pairs(cells)
    t = log(f"{len(src):,} adjacent pairs", t)

    # --- connectivity: keep only water reachable from the open-ocean seed ------
    both_ocean = is_ocean[src] & is_ocean[dst]
    adj = coo_matrix(
        (np.ones(both_ocean.sum()), (src[both_ocean], dst[both_ocean])), shape=(n, n)
    )
    n_comp, labels = connected_components(adj + adj.T, directed=False)
    seed_cell = h3.latlng_to_cell(*seed_latlng, res)
    if seed_cell not in cells:
        raise SystemExit(f"seed {seed_latlng} is outside the bbox")
    seed_idx = cells.index(seed_cell)
    if not is_ocean[seed_idx]:
        raise SystemExit(f"seed {seed_latlng} is not ocean")
    connected = (labels == labels[seed_idx]) & is_ocean
    t = log(f"{connected.sum():,} connected to seed ({n_comp} components total)", t)

    # --- dist_to_shore: multi-source march over the H3 grid --------------------
    # Great-circle weights between cell centres. Equal-area cells with near-uniform
    # spacing make this isotropic at every latitude and across the antimeridian.
    keep = both_ocean & connected[src] & connected[dst]
    w = great_circle_km(lat[src[keep]], lng[src[keep]], lat[dst[keep]], lng[dst[keep]])

    touches_land = np.zeros(n, dtype=bool)
    for a, b in ((src, dst), (dst, src)):
        touches_land[a[connected[a] & ~is_ocean[b]]] = True
    seeds = np.flatnonzero(touches_land & ~truncated)
    if seeds.size == 0:
        raise SystemExit("no shoreline found inside the bbox -- widen it")

    # Initialise the narrow band with true distances. Seeding these at zero would
    # make every cell adjacent to land a zero, so a strait two or three cells wide
    # would have a zero-width bottleneck -- a resolution artifact that looks like
    # geography.
    offset = shore_distance_km(ocean.boundary, lat[seeds], lng[seeds])
    t = log(f"narrow band: {seeds.size:,} cells, median {np.median(offset):.1f} km", t)

    # One virtual node wired to each seed at its own offset turns the per-seed
    # initial values into a single-source problem.
    g = coo_matrix(
        (
            np.r_[w, offset],
            (np.r_[src[keep], seeds], np.r_[dst[keep], np.full(seeds.size, n)]),
        ),
        shape=(n + 1, n + 1),
    )
    d = dijkstra((g + g.T).tocsr(), indices=n)[:n]
    d[~connected] = np.nan
    t = log(f"dist_to_shore from {seeds.size:,} shoreline cells, max {np.nanmax(d):.0f} km", t)

    aw = _access_width(src, dst, keep, d, connected, seed_idx)
    t = log(f"access_width computed, max {np.nanmax(aw):.0f} km", t)

    idx = np.flatnonzero(connected)
    return BaseMap(
        res=res,
        bbox=bbox,
        cells=np.array([h3.str_to_int(cells[i]) for i in idx], dtype=np.uint64),
        lat=lat[idx],
        lng=lng[idx],
        dist_to_shore=d[idx],
        access_width=aw[idx],
    )


def _access_width(src, dst, keep, dist, connected, seed_idx) -> np.ndarray:
    """Widest bottleneck on the best corridor to the open-ocean seed.

        access_width(c) = max over paths P from c to the seed
                             of  2 * min_{x in P} dist_to_shore(x)

    Computed by adding cells in descending dist_to_shore and union-finding them.
    The level at which a cell's component first contains the seed is its
    bottleneck. This flood is also the connectivity pass: an enclosed basin never
    joins the seed and keeps a width of nan.
    """
    n = len(dist)
    order = [int(i) for i in np.argsort(-np.nan_to_num(dist, nan=-1.0)) if connected[i]]

    nbrs: list[list[int]] = [[] for _ in range(n)]
    for a, b, k in zip(src, dst, keep):
        if k:
            nbrs[a].append(int(b))
            nbrs[b].append(int(a))

    parent = np.arange(n, dtype=np.int64)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return int(x)

    aw = np.full(n, np.nan)
    added = np.zeros(n, dtype=bool)
    pending: dict[int, list[int]] = {}  # unlabelled members, per component root

    for i in order:
        added[i] = True
        pending[i] = [i]
        for j in nbrs[i]:
            if not added[j]:
                continue
            ri, rj = find(i), find(j)
            if ri == rj:
                continue
            if len(pending[ri]) > len(pending[rj]):
                ri, rj = rj, ri
            parent[ri] = rj
            pending[rj].extend(pending[ri])
            del pending[ri]
        if added[seed_idx]:
            root = find(i)
            if root == find(seed_idx):
                for m in pending[root]:
                    aw[m] = 2.0 * dist[i]
                pending[root] = []
    return aw


# --------------------------------------------------------------------------
# Stage 3b -- the artifact
# --------------------------------------------------------------------------


@dataclass
class NavMap:
    """The runtime artifact: signed clearance, at as coarse a level as will do.

    Each cell stores the min and max margin over its footprint. A query walks
    coarse to fine and stops at the first level that decides, so open water costs
    one lookup and only coastlines need the descent. Children exist only under
    cells that were undecided, which is what makes the pyramid smaller than a flat
    map rather than larger.
    """

    K: float
    W: float
    res_min: int
    res_base: int
    levels: dict[int, dict[int, tuple[float, float]]]

    @property
    def n_entries(self) -> int:
        return sum(len(v) for v in self.levels.values())

    def decide(self, lat: float, lng: float) -> tuple[float, int]:
        """(margin, resolution that decided).

        The margin is the deciding cell's bound, not the point's exact value: a
        wholly-legal cell reports its minimum, so the number is a guarantee for
        every point in it rather than a measurement at one. That is what makes it
        safe to spend as a travel budget.
        """
        for r in range(self.res_min, self.res_base + 1):
            entry = self.levels[r].get(h3.str_to_int(h3.latlng_to_cell(lat, lng, r)))
            if entry is None:
                return float("-inf"), r  # outside the built area
            lo, hi = entry
            if lo >= 0.0:
                return lo, r
            if hi < 0.0:
                return hi, r
        return lo, self.res_base

    def clearance(self, lat: float, lng: float) -> float:
        return self.decide(lat, lng)[0]

    def legal(self, lat: float, lng: float) -> bool:
        return self.clearance(lat, lng) >= 0.0


def build_map(base: BaseMap, K: float, W: float, res_min: int = 2, verbose: bool = True) -> NavMap:
    log = _log if verbose else (lambda *a, **k: time.time())
    t = log(f"building artifact for K={K} km, W={W} km")

    d = np.nan_to_num(base.dist_to_shore, nan=-1e9)
    a = np.nan_to_num(base.access_width, nan=-1e9)
    margin = np.minimum(d - K, (a - W - 2.0 * K) / 2.0)
    t = log(f"{(margin >= 0).sum():,} of {len(margin):,} samples navigable", t)

    levels: dict[int, dict[int, tuple[float, float]]] = {}
    live = np.ones(len(margin), dtype=bool)

    for r in range(res_min, base.res + 1):
        if r == base.res:
            ids = base.cells
        else:
            # Re-index each sample's own position. NOT h3.cell_to_parent: that is
            # index truncation, and hexagons do not nest, so a parent's footprint
            # is not the union of its children's.
            ids = np.fromiter(
                (h3.str_to_int(h3.latlng_to_cell(la, lo, r)) for la, lo in zip(base.lat, base.lng)),
                dtype=np.uint64,
                count=len(base.lat),
            )
        order = np.argsort(ids, kind="stable")
        sid = ids[order]
        starts = np.flatnonzero(np.r_[True, sid[1:] != sid[:-1]])
        counts = np.diff(np.r_[starts, len(sid)])
        lo = np.minimum.reduceat(margin[order], starts)
        hi = np.maximum.reduceat(margin[order], starts)
        # store only cells something might still descend into
        keep = np.maximum.reduceat(live[order].astype(np.int8), starts) > 0
        levels[r] = {
            int(u): (float(p), float(q))
            for u, p, q, k in zip(sid[starts], lo, hi, keep)
            if k
        }
        # a cell that is wholly legal or wholly illegal settles its samples
        settled = (lo >= 0.0) | (hi < 0.0)
        live[order[np.repeat(settled, counts)]] = False
        t = log(f"res {r}: {len(levels[r]):,} stored, {live.sum():,} samples still live", t)

    return NavMap(K=K, W=W, res_min=res_min, res_base=base.res, levels=levels)


# --------------------------------------------------------------------------
# The clearance budget (Stage 3f)
# --------------------------------------------------------------------------


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
        self.steps += 1
        self.budget -= distance_km
        if self.budget > 0:
            return True
        self.lookups += 1
        self.budget = self.nav.clearance(lat, lng)
        return self.budget >= 0.0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

PROBES = [
    ("Atlantic, off Portugal", 38.0, -11.0),
    ("Strait of Gibraltar", 35.95, -5.6),
    ("Alboran Sea (inside Med)", 36.2, -3.0),
    ("Balearic Sea (inside Med)", 40.5, 4.0),
    ("Bay of Biscay", 43.6, -2.0),
    ("inland Spain", 40.4, -3.7),
]


def report(base: BaseMap, nav: NavMap) -> None:
    flat = len(base.cells)
    print(f"\nmap: res {base.res}, K={nav.K} km, W={nav.W} km")
    print(f"  {nav.n_entries:,} pyramid entries vs {flat:,} flat  ({nav.n_entries / flat:.2f}x)")
    print("  d_shore/width are the res-{} sample; margin is the deciding cell's bound".format(base.res))
    print(f"\n  {'':28} {'legal':>6} {'d_shore':>8} {'width':>8} {'margin':>8} {'@res':>5}")
    for name, la, lo in PROBES:
        cell = h3.str_to_int(h3.latlng_to_cell(la, lo, base.res))
        hit = np.flatnonzero(base.cells == cell)
        ds = f"{base.dist_to_shore[hit[0]]:.1f}" if hit.size else "-"
        aw = f"{base.access_width[hit[0]]:.1f}" if hit.size else "-"
        c, r = nav.decide(la, lo)
        m = "outside" if c == float("-inf") else f"{c:.1f}"
        print(f"  {name:28} {str(c >= 0):>6} {ds:>8} {aw:>8} {m:>8} {r:>5}")


def demo_budget(nav: NavMap, lat=36.0, lng=-6.5, step_km=2.0, n=300) -> None:
    b = ClearanceBudget(nav)
    for _ in range(n):
        lng -= step_km / (111.32 * np.cos(np.radians(lat)))
        if not b.step(lat, lng, step_km):
            break
    print(f"\nclearance budget, {step_km} km steps west from 36N 6.5W:")
    print(f"  {b.steps} steps, {b.lookups} lookups (naive would be {b.steps})")
    print(f"  {100 * (1 - b.lookups / b.steps):.0f}% of lookups skipped")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("-K", type=float, default=0.0, help="keep-out distance from shore, km")
    p.add_argument("-W", type=float, default=10.0, help="minimum channel width, km")
    p.add_argument("--res", type=int, default=7, help="H3 build resolution (default 7)")
    p.add_argument("--res-min", type=int, default=2, help="coarsest pyramid level")
    p.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        default=GIBRALTAR_BBOX,
        metavar=("LAT0", "LON0", "LAT1", "LON1"),
    )
    p.add_argument("--seed", type=float, nargs=2, default=ATLANTIC_SEED, metavar=("LAT", "LON"))
    p.add_argument("--ne-res", default="10m", choices=("10m", "50m", "110m"))
    p.add_argument("--save", metavar="NPZ", help="write the artifact")
    args = p.parse_args()

    print(f"building {tuple(args.bbox)} at res {args.res}")
    base = build_base(
        bbox=tuple(args.bbox), res=args.res, ne_res=args.ne_res, seed_latlng=tuple(args.seed)
    )
    nav = build_map(base, K=args.K, W=args.W, res_min=args.res_min)

    report(base, nav)
    demo_budget(nav)

    if args.save:
        out = {"K": nav.K, "W": nav.W, "res_min": nav.res_min, "res_base": nav.res_base}
        for r, lvl in nav.levels.items():
            out[f"r{r}_id"] = np.fromiter(lvl.keys(), dtype=np.uint64, count=len(lvl))
            out[f"r{r}_mm"] = np.array(list(lvl.values()), dtype=np.float32)
        np.savez_compressed(args.save, **out)
        print(f"\nwrote {args.save}")


if __name__ == "__main__":
    main()
