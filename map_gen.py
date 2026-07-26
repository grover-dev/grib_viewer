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
    uv run python map_gen.py --global --res 5      # the whole planet
    uv run python map_gen.py --save med.npz        # write the artifact

Each run builds the map, prints a probe table, and shows how many map lookups the
clearance budget skips on a short run west from the seed.

Options:

    -K KM            keep-out distance from shore (default 0)
    -W KM            minimum navigable channel width (default 10)
    --res N          H3 build resolution (default 7)
    --res-min N      coarsest pyramid level (default 2)
    --bbox LAT0 LON0 LAT1 LON1      area to build (default: western Med + Atlantic)
    --global         build the whole planet, ignoring --bbox
    --seed LAT LON   a point in open ocean, used to define "connected to the sea"
    --ne-res {10m,50m,110m}         Natural Earth coastline detail (default 10m)
    --save NPZ       write the pyramid to an .npz

Cost scales with cell count, at roughly 280 bytes and 17 us a cell (measured):

    res    global cells    peak RAM    wall clock    furthest-from-land
    4         288,122        0.3 GB         13 s        3541 km
    5       2,016,842        0.7 GB         47 s        3381 km
    6      14,117,882        3.9 GB       3m 56s        2854 km
    7      98,825,162        ~27 GB       ~30 min            --

That last column is the Point Nemo check, and it is the honest way to read the
resolution: the true answer is 2690 km at 48.9S 123.4W. Coarse builds overshoot
because a cell is classified by its centre, so small islands vanish and the
nearest shore ends up further away than it really is. res 6 lands within 6% and
in the right place (47.8S 131.7W).

`--global --res 7` is what W needs to resolve Gibraltar, and it does not fit in a
pure-Python prototype -- it wants a compiled implementation. Global res 6 is the
practical ceiling here, and it is still too coarse to adjudicate a 14 km strait,
so a global build seals the Mediterranean at any W. Build regionally when the
answer near a narrow passage matters.

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
import gc
import resource
import time
from dataclasses import dataclass

import numpy as np
import shapely

# The integer API throughout. Cell ids as Python str cost ~130 bytes each in a
# list and ~250 in a dict; as uint64 they cost 8 in a numpy array. At global
# resolutions that difference is the whole memory budget.
from h3.api import basic_int as h3
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra

from map_utils import GLOBAL_BBOX, ClearanceBudget, NavMap, great_circle_km

CHUNK = 1 << 20  # rows per pass when building adjacency, to cap transient memory

# (lat_min, lon_min, lat_max, lon_max): Gibraltar, the western Mediterranean, and
# enough Atlantic to be unambiguously open ocean.
GIBRALTAR_BBOX = (30.0, -12.0, 46.0, 20.0)
ATLANTIC_SEED = (35.0, -10.0)


def peak_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2


def _log(msg: str, t0: float | None = None) -> float:
    now = time.time()
    tail = "" if t0 is None else f"  [{now - t0:.1f}s, {peak_gb():.2f} GB peak]"
    print(f"  {msg}{tail}", flush=True)
    return now


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


def enumerate_cells(bbox, res: int) -> np.ndarray:
    """Sorted uint64 ids of every cell at `res` inside bbox, or the whole globe.

    The global path descends from the 122 res-0 cells instead of filling a
    polygon, which sidesteps the antimeridian entirely -- there is no boundary to
    cross. It also means no cell is truncated, so the whole ocean is one connected
    body and a single seed reaches all of it.

    Sorted, because every later lookup is a searchsorted into this array. A dict
    keyed by cell id would cost ~30x the memory for the same answer.
    """
    if bbox is None:
        parts = [
            np.fromiter(h3.cell_to_children(c, res), dtype=np.uint64)
            for c in h3.get_res0_cells()
        ]
        cells = np.concatenate(parts)
    else:
        lat0, lon0, lat1, lon1 = bbox
        poly = h3.LatLngPoly([(lat0, lon0), (lat0, lon1), (lat1, lon1), (lat1, lon0)])
        cells = np.fromiter(h3.polygon_to_cells(poly, res), dtype=np.uint64)
    cells.sort()
    return cells


def global_cell_count(res: int) -> int:
    return 2 + 120 * 7**res


def estimate_gb(n_cells: int) -> float:
    """Rough peak working set. ~230 bytes a cell, dominated by the Dijkstra CSR."""
    return n_cells * 230 / 1024**3


def _neighbour_pairs(cells: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Index pairs for cells adjacent within the set, plus a truncation flag.

    `truncated[i]` marks a cell with a neighbour outside the enumerated set, i.e.
    one on the bbox edge. Those are excluded from shoreline seeding so the box
    boundary does not masquerade as a coast. A global build truncates nothing.

    Built in chunks into preallocated int32 arrays. The obvious version -- Python
    lists of ints appended per neighbour -- costs ~28 bytes an entry plus list
    overhead, which is several GB of garbage at global resolutions.
    """
    n = len(cells)
    idx = np.empty(n * 6, dtype=np.int32)
    truncated = np.zeros(n, dtype=bool)

    for lo in range(0, n, CHUNK):
        hi = min(lo + CHUNK, n)
        ring = np.full(((hi - lo), 6), np.uint64(0), dtype=np.uint64)
        for k, c in enumerate(cells[lo:hi].tolist()):
            nb = [x for x in h3.grid_disk(c, 1) if x != c]
            ring[k, : len(nb)] = nb
            if len(nb) < 5:  # pentagons legitimately have 5
                truncated[lo + k] = True
        flat = ring.ravel()
        pos = np.searchsorted(cells, flat)
        np.clip(pos, 0, n - 1, out=pos)
        hit = (cells[pos] == flat) & (flat != 0)
        chunk_idx = np.where(hit, pos, -1).astype(np.int32)
        # a real neighbour that is not in the set means the set has an edge
        missing = (~hit & (flat != 0)).reshape(-1, 6).any(axis=1)
        truncated[lo:hi] |= missing
        idx[lo * 6 : hi * 6] = chunk_idx
        del ring, flat, pos, hit, chunk_idx

    src = np.repeat(np.arange(n, dtype=np.int32), 6)
    dst = idx
    keep = (dst >= 0) & (dst > src)  # each undirected pair once
    return src[keep], dst[keep], truncated


def build_base(
    bbox=GIBRALTAR_BBOX,
    res: int = 7,
    ne_res: str = "10m",
    seed_latlng: tuple[float, float] = ATLANTIC_SEED,
    verbose: bool = True,
) -> BaseMap:
    log = _log if verbose else (lambda *a, **k: time.time())
    if bbox is None:
        log(f"global build: {global_cell_count(res):,} cells at res {res}")

    t = log("loading ocean polygon")
    ocean = load_ocean(ne_res)
    t = log(f"enumerating res-{res} cells", t)
    cells = enumerate_cells(bbox, res)
    n = len(cells)
    lat = np.empty(n, dtype=np.float64)
    lng = np.empty(n, dtype=np.float64)
    for i, c in enumerate(cells.tolist()):  # fill in place; no list of tuples
        lat[i], lng[i] = h3.cell_to_latlng(c)
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
        (np.ones(both_ocean.sum(), dtype=np.int8), (src[both_ocean], dst[both_ocean])),
        shape=(n, n),
    )
    n_comp, labels = connected_components(adj, directed=False, connection="weak")
    del adj
    gc.collect()
    seed_cell = h3.latlng_to_cell(*seed_latlng, res)
    seed_idx = int(np.searchsorted(cells, np.uint64(seed_cell)))
    if seed_idx >= n or cells[seed_idx] != seed_cell:
        raise SystemExit(f"seed {seed_latlng} is outside the bbox")
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
    # initial values into a single-source problem. Both directions are written
    # into one COO rather than built as g + g.T, which would hold three copies of
    # the matrix at once.
    a = np.r_[src[keep], seeds, dst[keep], np.full(seeds.size, n, dtype=np.int32)]
    b = np.r_[dst[keep], np.full(seeds.size, n, dtype=np.int32), src[keep], seeds]
    g = coo_matrix((np.r_[w, offset, w, offset], (a, b)), shape=(n + 1, n + 1)).tocsr()
    del a, b
    gc.collect()
    d = dijkstra(g, indices=n)[:n]
    del g
    gc.collect()
    d[~connected] = np.nan
    far = int(np.nanargmax(d))
    t = log(
        f"dist_to_shore from {seeds.size:,} shoreline cells; furthest {d[far]:.0f} km "
        f"at {lat[far]:.1f} {lng[far]:.1f}",
        t,
    )

    aw = _access_width(src, dst, keep, d, connected, seed_idx)
    t = log(f"access_width computed, max {np.nanmax(aw):.0f} km", t)

    idx = np.flatnonzero(connected)
    return BaseMap(
        res=res,
        bbox=bbox if bbox is not None else GLOBAL_BBOX,
        cells=cells[idx],
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
    o = np.argsort(-np.nan_to_num(dist, nan=-1.0))
    order = o[connected[o]].astype(np.int64).tolist()
    del o

    # CSR adjacency rather than a list of lists: at global res 6 the latter is 14M
    # Python list objects before a single neighbour is stored.
    a = np.r_[src[keep], dst[keep]]
    nbr = np.r_[dst[keep], src[keep]]
    s = np.argsort(a, kind="stable")
    a, nbr = a[s], nbr[s]
    indptr = np.searchsorted(a, np.arange(n + 1)).astype(np.int64)
    del a, s
    gc.collect()

    parent = np.arange(n, dtype=np.int32)
    # Unlabelled members of each component, as an intrusive linked list over three
    # int32 arrays. The readable version -- dict[root] -> list of members -- costs
    # a Python list object per component and boxes every member.
    head = np.full(n, -1, dtype=np.int32)
    tail = np.full(n, -1, dtype=np.int32)
    nxt = np.full(n, -1, dtype=np.int32)
    size = np.ones(n, dtype=np.int32)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return int(x)

    aw = np.full(n, np.nan)
    added = np.zeros(n, dtype=bool)

    for i in order:
        added[i] = True
        head[i] = tail[i] = i
        for j in nbr[indptr[i] : indptr[i + 1]].tolist():
            if not added[j]:
                continue
            ri, rj = find(i), find(j)
            if ri == rj:
                continue
            if size[ri] > size[rj]:
                ri, rj = rj, ri
            parent[ri] = rj
            size[rj] += size[ri]
            if head[ri] != -1:  # splice ri's pending chain onto rj's
                if head[rj] == -1:
                    head[rj], tail[rj] = head[ri], tail[ri]
                else:
                    nxt[tail[rj]] = head[ri]
                    tail[rj] = tail[ri]
                head[ri] = tail[ri] = -1
        if added[seed_idx]:
            root = find(i)
            if root == find(seed_idx):
                level = 2.0 * dist[i]
                m = int(head[root])
                while m != -1:
                    aw[m] = level
                    following = int(nxt[m])
                    nxt[m] = -1
                    m = following
                head[root] = tail[root] = -1
    return aw


# --------------------------------------------------------------------------
# Stage 3b -- the artifact
# --------------------------------------------------------------------------


def build_map(
    base: BaseMap,
    K: float,
    W: float,
    res_min: int = 2,
    floor_res: int = 5,
    verbose: bool = True,
) -> NavMap:
    log = _log if verbose else (lambda *a, **k: time.time())
    t = log(f"building artifact for K={K} km, W={W} km")

    d = np.nan_to_num(base.dist_to_shore, nan=-1e9)
    a = np.nan_to_num(base.access_width, nan=-1e9)
    margin = np.minimum(d - K, (a - W - 2.0 * K) / 2.0)
    budget = d - (K + W / 2.0)  # see NavMap.decide for why this is not `margin`
    t = log(f"{(margin >= 0).sum():,} of {len(margin):,} samples navigable", t)

    levels: dict[int, dict[int, tuple[float, float, float]]] = {}
    live = np.ones(len(margin), dtype=bool)

    for r in range(res_min, base.res + 1):
        if r == base.res:
            ids = base.cells
        else:
            # Re-index each sample's own position. NOT h3.cell_to_parent: that is
            # index truncation, and hexagons do not nest, so a parent's footprint
            # is not the union of its children's.
            ids = np.fromiter(
                (h3.latlng_to_cell(la, lo, r) for la, lo in zip(base.lat, base.lng)),
                dtype=np.uint64,
                count=len(base.lat),
            )
        order = np.argsort(ids, kind="stable")
        sid = ids[order]
        starts = np.flatnonzero(np.r_[True, sid[1:] != sid[:-1]])
        counts = np.diff(np.r_[starts, len(sid)])
        lo = np.minimum.reduceat(margin[order], starts)
        hi = np.maximum.reduceat(margin[order], starts)
        bud = np.minimum.reduceat(budget[order], starts)
        # store only cells something might still descend into
        keep = np.maximum.reduceat(live[order].astype(np.int8), starts) > 0
        levels[r] = {
            int(u): (float(p), float(q), float(b))
            for u, p, q, b, k in zip(sid[starts], lo, hi, bud, keep)
            if k
        }
        # A cell that is wholly legal or wholly illegal settles its samples, so
        # nothing below it needs storing. Dilate the undecided set by one ring
        # first: levels are independent tilings, so a finer cell can straddle a
        # decided cell and an undecided one, and if all its own samples sat on the
        # decided side it would be pruned away -- leaving a descent that reaches
        # for it and finds a hole.
        undecided = sid[starts][(lo < 0.0) & (hi >= 0.0)]
        if undecided.size:
            ring = {
                x
                for c in undecided.tolist()
                for x in h3.grid_disk(c, 1)
            }
            settled = ~np.isin(sid[starts], np.fromiter(ring, np.uint64, len(ring)))
        else:
            settled = np.ones(len(starts), dtype=bool)
        # Nothing settles above the floor. Legality alone would stop at res 2 in
        # open water, which leaves `budget` -- and so shore distance -- constant
        # over cells hundreds of km across. Anything steering by distance to land
        # then has no gradient to follow: a whole fan of candidate headings scores
        # identically. The floor keeps a usable field at the cost of storing cells
        # no legality query would ever need.
        if r < floor_res:
            settled[:] = False
        live[order[np.repeat(settled, counts)]] = False
        t = log(f"res {r}: {len(levels[r]):,} stored, {live.sum():,} samples still live", t)

    return NavMap(
        K=K, W=W, res_min=res_min, res_base=base.res, levels=levels, bbox=base.bbox
    )


# --------------------------------------------------------------------------
# The clearance budget (Stage 3f)
# --------------------------------------------------------------------------


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
    print(f"\n  {'':28} {'legal':>6} {'d_shore':>8} {'width':>8} {'margin':>8} {'budget':>8} {'@res':>5}")
    for name, la, lo in PROBES:
        cell = h3.latlng_to_cell(la, lo, base.res)
        hit = np.flatnonzero(base.cells == cell)
        ds = f"{base.dist_to_shore[hit[0]]:.1f}" if hit.size else "-"
        aw = f"{base.access_width[hit[0]]:.1f}" if hit.size else "-"
        c, b, r = nav.decide(la, lo)
        m = "outside" if c == float("-inf") else f"{c:.1f}"
        bs = f"{b:.1f}" if c >= 0 else "-"
        print(f"  {name:28} {str(c >= 0):>6} {ds:>8} {aw:>8} {m:>8} {bs:>8} {r:>5}")


def demo_budget(nav: NavMap, start: tuple[float, float], step_km=2.0, n=400) -> None:
    """Walk west from a known-good point, counting how often the map is consulted."""
    lat, lng = start
    b = ClearanceBudget(nav)
    for _ in range(n):
        lng -= step_km / (111.32 * np.cos(np.radians(lat)))
        if not b.step(lat, lng, step_km):
            break
    print(f"\nclearance budget, {step_km:g} km steps west from {lat:.1f} {lng:.1f}:")
    print(f"  {b.steps} steps, {b.lookups} lookups (naive would be {b.steps})")
    if b.steps:
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
        "--floor-res",
        type=int,
        default=5,
        help="always store down to this resolution, so shore distance has a "
        "gradient to steer by (0 to disable; smaller artifact, but coastal "
        "steering will not work)",
    )
    p.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        default=GIBRALTAR_BBOX,
        metavar=("LAT0", "LON0", "LAT1", "LON1"),
    )
    p.add_argument(
        "--global",
        dest="whole_world",
        action="store_true",
        help="build the whole planet, ignoring --bbox",
    )
    p.add_argument(
        "--force", action="store_true", help="build even if the memory estimate says no"
    )
    p.add_argument("--seed", type=float, nargs=2, default=ATLANTIC_SEED, metavar=("LAT", "LON"))
    p.add_argument("--ne-res", default="10m", choices=("10m", "50m", "110m"))
    p.add_argument("--save", metavar="NPZ", help="write the artifact")
    args = p.parse_args()

    bbox = None if args.whole_world else tuple(args.bbox)
    if bbox is None:
        n = global_cell_count(args.res)
        gb = estimate_gb(n)
        print(f"building the whole planet at res {args.res}: {n:,} cells, ~{gb:.1f} GB")
        if gb > 8.0 and not args.force:
            raise SystemExit(
                f"  refusing: ~{gb:.0f} GB is beyond this pure-Python prototype.\n"
                f"  res 6 (14.1M cells, 3.9 GB, ~4 min) is the practical global ceiling.\n"
                f"  res 7 is what W needs to resolve Gibraltar and wants a compiled build.\n"
                f"  override with --force if you know your machine can take it."
            )
        if gb > 2.0:
            print(f"  expect several minutes; peak RSS around {gb:.0f} GB")
    else:
        print(f"building {bbox} at res {args.res}")
    base = build_base(bbox=bbox, res=args.res, ne_res=args.ne_res, seed_latlng=tuple(args.seed))
    nav = build_map(
        base, K=args.K, W=args.W, res_min=args.res_min, floor_res=args.floor_res
    )

    report(base, nav)
    demo_budget(nav, start=tuple(args.seed))

    if args.save:
        nav.save(args.save)
        print(f"\nwrote {args.save}")


if __name__ == "__main__":
    main()
