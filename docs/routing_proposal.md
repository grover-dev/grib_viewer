# Proposal: GRIB solar-radiation → traversable graph with landmass keep-out

## Goal

Route a vehicle across the sea using a graph traversal / shortest-path
algorithm, where:

- **surface solar radiation** (from GRIB, varying over space and time) drives the
  cost/objective of a route;
- **landmasses, inland water, and a configurable keep-out buffer** are excluded
  from the traversable space — the focus is coastal / open-ocean operation, so
  lakes and rivers are treated as land;
- **the vehicle's own state** (power/battery, mode) affects how fast it moves,
  and therefore how far it gets — a feedback loop, not a fixed edge cost;
- **future models** (currents, wind, sea state, …) can be added later without
  reworking the core.

This document is language- and library-agnostic: it describes the data model,
the transformations, and the decisions, not the code.

### The central design decision: decouple everything from the GRIB grid

The GRIB cells are coarse (~0.25° ≈ 25 km) and are the *wrong* unit for a route
whose step size is 100s of metres to single-digit km. So the routing graph gets
its **own geometry**, independent of the radiation grid, and the radiation grid
is exposed only as a **sampled function** `Rad(t, lat, lon)` used to price edges.

More importantly: because traversal *speed* depends on the vehicle's internal
state, and that state depends on the whole path taken so far, a plain
node-weighted graph cannot express the problem. The search state is not a node —
it is `(node, time, battery, …)`. That single fact drives the data structures
below.

Three seams keep the pieces independent, and the search only ever touches the
world through them:

```
neighbours(node)                        — geometry (the precomputed nav graph)
field.sample(t, lat, lon)               — environment (radiation, later currents/wind)
advance(state, edge, env, mode)         — dynamics (the boat, later other models)
```

The map geometry (which water is navigable) is **baked once, ahead of time**;
the environment and dynamics stay live at query time.

---

## The pipeline in one picture

```
                     ┌──────────── BUILD ONCE (geometry) ─────────────┐
Ocean polygon ──▶ rasterize ──▶ ocean raster ──▶ distance transform
                                     │                    │
                                     ▼                    ▼
                              is_ocean(p)          dist_to_shore(p)
                                     │                    │
                     flood-fill from open-ocean seed(s)  ─┘
                                     ▼
                   BASE graph: all ocean cells (H3) + per-node
                              dist_to_shore + access_width
                                     │
                              prune(K, W)  ── keep-out + min channel width
                                     ▼
                   RUNTIME artifact: compacted navigable graph
                        nodes = navigable water cells
                        edges = neighbour links (length, bearing only)
                     └────────────────────────────────────────────────┘
                                     │
GRIB messages ─index─▶ coarse fields │ Rad(t,p) sampler (bi/trilinear, W/m²)
                                     ▼            │
   Vehicle dynamics  advance(state, edge, env, mode) -> (new_state, dt, cost)
        boat(time, mode) + internal state (battery); future models compose here
                                     ▼
   State-augmented search  over labels (node, time, battery, …)
        Dijkstra / A* on the augmented state space
```

Each block is independent and testable in isolation. The GRIB indexing already
exists in this codebase; the proposal is the rest.

---

## Stage 1 — Radiation as a field sampler (not node data)

The existing index already yields, per valid time, a radiation frame on a uniform
lat/lon grid. Wrap it as a **sampler** rather than baking values into nodes:

- **Interface:** `Rad(t, lat, lon) -> W/m²`. Internally: **bilinear** in lat/lon,
  **linear** between the two time frames bracketing `t` (trilinear overall). This
  is what decouples the fine route from the coarse grid — a caller gets a smooth
  value at any point and any instant.
- **Units** normalised at this boundary: accumulated J/m² → instantaneous W/m²
  by dividing by the accumulation seconds (the viewer already does this). The
  graph consumes a rate.
- **Cache:** sampling at `t` needs only the two bracketing frames. The existing
  `FrameCache` (byte-bounded LRU of decoded frames) is exactly the right
  structure — keep those frames warm there.
- **One interface, many fields:** radiation is *scalar*; future currents/wind are
  *vector* (`-> (u, v)`). Define `ScalarField.sample` and `VectorField.sample` so
  every runtime environmental input looks the same to the rest of the system. The
  "environment" is then just a **list of fields**.

(The land/ocean mask is *not* one of these runtime fields — it is consumed at
build time to bake the graph geometry, Stage 2.)

---

## Stage 2 — Navigable-water selection (build time)

This stage decides which water is navigable and bakes it into geometry. It never
runs during search.

### 2a. Source of "what is navigable": the ocean polygon

Because inland water is to be treated as land, define the mask as **"is this
ocean?"**, not "is this water?":

- **Build the mask from an ocean polygon** (e.g. Natural Earth's `ocean` layer),
  already the family of data used for the basemap. Lakes and rivers are simply
  *not ocean*, so they are non-navigable **by construction** — no heuristics.
- A generic water test can't do this: rivers and estuaries are topologically
  connected to the sea, so any "any water" flood would flow up them. Inland water
  is excluded by **classification**, which the ocean polygon gives for free.
- **The ERA5 land–sea field is not sufficient here.** It marks a lake as water and
  can't distinguish it from the sea, so it is at best a coarse fallback that would
  need a separate lake mask bolted on. The ocean polygon is the primary source.
- **Marginal seas** (Mediterranean, Baltic, Black Sea) *are* part of the ocean
  polygon and are kept — whether they survive is governed by the `W` parameter
  below, not by this classification.

### 2b. Rasterize once — one artifact serves the land test and the K field

`is_ocean(lat, lon)` and `dist_to_shore(lat, lon)` are queried millions of times
during the build, so precompute them as arrays rather than doing point-in-polygon
per query:

1. **Burn the ocean polygon into a boolean raster** at a resolution *finer* than
   the H3 cells (so islands and narrow inlets aren't lost). `is_ocean` = a raster
   lookup.
2. **Distance transform** over the non-ocean region → every ocean pixel holds its
   distance to the nearest non-ocean thing (continent, island, *or* lake edge).
   Convert pixels → km with the cos-latitude factor (the scale bar's correction).
   `dist_to_shore` = a raster lookup.

One raster + its distance transform powers both the ocean test and the keep-out
field, as pure O(1) array lookups.

### 2c. Enumerate + connectivity: flood-fill from an open-ocean seed

Enumerate the navigable cells by BFS from a known open-ocean cell, expanding
through ocean and refusing to cross non-ocean:

```
seed  = latlng_to_cell(open_ocean_point, R)
queue = [seed];  seen = {seed};  ocean = []
while queue:
    c = queue.pop()
    if not is_ocean(center(c)):   # do NOT expand past non-ocean
        continue
    ocean.append(c); dts[c] = dist_to_shore(center(c))
    for n in grid_disk(c, 1) - {c}:
        if n not in seen: seen.add(n); queue.append(n)
```

Three properties fall out for free: enumeration + mask + connectivity in one pass;
it never materialises land; and `grid_disk` handles the antimeridian and poles
correctly in H3 space. Since the world ocean is one connected body, **one seed
usually suffices**; use several if you operate across genuinely disconnected
basins, and validate that the ocean-cell count matches the expected sea area (a
missing seed drops a whole basin).

### 2d. The two selection parameters: K and W

Two scalars decide the final navigable set. Both are computed from `dist_to_shore`
and applied when the graph is **pruned** (Stage 3a) — the map is built once and
reused, so these are build parameters, not per-query knobs.

- **K — keep-out distance (km).** "No vehicle within K of shore."
  `dist_to_shore(cell) >= K`. `K = 0` (hug the coast) is valid.
- **W — minimum navigable channel width (km).** A body of water counts only if it
  connects to open ocean through a channel at least `W` wide. This is a
  morphological *opening* of the ocean by radius `W/2`, and it is what selectively
  masks marginal seas: the Strait of Gibraltar is ~14 km wide, so `W = 30 km`
  seals it and the whole Mediterranean drops out, while `W = 10 km` keeps it. It
  is a **uniform rule** — `W` closes *every* passage narrower than `W`, trimming
  narrow coastal inlets as well as severing marginal seas.

`W` is realised through a per-cell **`access_width`**: the widest bottleneck on
the best path to open ocean,

```
access_width(c) = max over paths P from c to an open-ocean seed
                     of  2 · min_{x in P} dist_to_shore(x)
```

computed once with a **maximum-bottleneck flood** (sort cells by `dist_to_shore`
descending, union-find them, and record the `dist_to_shore` at which each first
connects to a seed × 2). This same flood *is* the connectivity pass — a truly
enclosed lake never connects to a seed, so its `access_width` is 0 and it is
masked for any `W > 0`, on top of already being excluded by the ocean polygon.

**Coupled K→W semantics.** Because K and W are applied at build time, do it in the
physically correct order: apply K first (remove the near-shore band), recompute
`access_width` on the remaining water, then apply W. Then `W` means the width of
the corridor you can *actually use after the keep-out*, not the raw shore-to-shore
width — a 30 km strait with a 5 km keep-out on each side is effectively 20 km of
navigable water.

---

## Stage 3 — Navigation graph + state-augmented search

Three parts: **geometry** (built + pruned), **state**, and **search**.

### 3a. Graph geometry — H3, built once, pruned to a runtime artifact

The graph's spacing is *yours*, chosen for a 100 m – few-km step, unrelated to the
25 km grid.

**Cell geometry** (Open Question 3):

- **Geodesic hex grid (H3)** — *recommended.* Cells are near-equal-area
  everywhere, so pole distortion and the antimeridian seam disappear, and every
  cell has **6 edge-adjacent neighbours at one uniform distance** — no √2
  bookkeeping, no directional bias. A cell id is a 64-bit integer, giving node
  identity for free (no floats-as-keys trap). Resolution sets the step: res 7 ≈
  1.2 km, res 8 ≈ 460 m, res 9 ≈ 170 m — your whole target range.
- **Fine regular lattice** — simplest, but cos-latitude distortion returns in every
  distance and the square grid biases paths toward the axes/diagonals.
- **Navmesh / visibility graph from coastline polygons** — sparse; a bigger build.

**On H3 and the library question.** H3's hex *math* is simple; what needs the
library is tiling a **sphere** with hexagons globally without seams (icosahedron
projection, cross-face transforms, the 12 unavoidable pentagon cells, exact index
encoding). Those failures are *silent* — a precision slip near a face boundary
mislabels a cell's neighbours and the search returns a subtly wrong path with no
error, which is why the tested library is worth it when the domain is global.
**If the routing domain is bounded** (a basin, a coastline, one crossing), you can
skip the dependency: project the region to a local gnomonic / equal-area plane,
lay a flat hex grid there, and map cell centres back to lat/lon — a few hundred
lines that keep the uniform-hex benefit within the region, at the cost of
distortion far from centre and no poles/dateline.

**Build once, then prune.** The geometry seam is *static* — which cells are
navigable never depends on time, radiation, or boat state. Split it in two so the
expensive coastline work is done once and only the cheap parameter step reruns:

```
BASE build (K/W-agnostic, uses H3):
  1. ocean cells   = flood-fill from seed(s)                 # Stage 2c
  2. per cell:  dist_to_shore, access_width                  # Stage 2b/2d
  3. edges         = grid_disk(c,1) neighbours also ocean    # ≤6 per cell
  4. per edge:  length + bearing from cell-centre great-circle distance
  5. keep H3 id per cell

PRUNE(K, W)  ── coupled K→W (Stage 2d):
  6. keep cells with dist_to_shore ≥ K
  7. recompute access_width on survivors; keep those with access_width ≥ W
  8. drop components no longer connected to a seed
  9. RECOMPACT: renumber survivors 0..N'-1, rebuild adjacency + point index

RUNTIME artifact:  a plain navigable graph, no K/W/mask awareness, smaller
```

Only **geometry** is serialised — never costs, times, or radiation, which stay in
`advance`/`field.sample`. Pruning yields a smaller graph and a hot loop with **no
per-expansion mask predicate**: excluded cells simply aren't there, edges are
pre-severed, and dropping stranded components means the search can't wander into
dead-end pockets. Because H3 is hierarchical, the base build can also go
**adaptive** — fine cells near the coast, coarse in open water — baked into the
same flat integer graph.

**Node ids are not stable across prunes.** Recompaction renumbers everything, so a
node id in the `K=5` map is a different cell in the `K=10` map. Persist anything
durable — saved routes, waypoints, start/goal — by **H3 id or lat/lon**, and
re-resolve against whichever pruned map is loaded. Keep the **base artifact**
around so trying a new `(K, W)` is a cheap re-prune, never a re-rasterize of the
coastline. (A lightweight version stamp on each artifact — inputs it was built
from — lets a stale one rebuild; detailed caching is out of scope for now.)

**Edges carry geometry only** — length and bearing. **No time, no cost stored on
the edge**; both are computed per traversal, because they depend on state.

### 3b. Search state — why nodes aren't enough

Traversal time over an edge = `length ÷ speed(state)`, and speed depends on
battery, which depends on every edge crossed before it. So the thing the search
explores is an **augmented state**:

```
State = (node, time, battery, …future state…)
```

This is the **resource-constrained shortest path** shape. Two ways to keep a
standard algorithm applicable:

- **Bucketed state** — discretize battery into N levels (and time into steps). The
  search space becomes a **product graph** `node × battery × time`, and ordinary
  Dijkstra / A\* runs unchanged. Simple, bounded memory, approximate. *Start
  here.*
- **Continuous state with Pareto labels** (label-setting) — each node keeps a
  **frontier of non-dominated labels** `(arrival_time, battery, cost)`; one label
  dominates another if it arrives no later, no more depleted, and no costlier.
  Exact, but frontiers can grow. *Move here if buckets prove too coarse.*

Data structures for the search:

- a **priority queue of labels** (the frontier), ordered by cost or time;
- a **per-node label set** with a dominance check — a short sorted list per node
  is usually enough;
- the pruned **adjacency** from 3a;
- a **came-from map** keyed by *label*, not node, to reconstruct the path.

### 3c. Vehicle dynamics — the `advance` seam

The boat, and every future model, lives behind one function the search calls per
edge:

```
advance(state, edge, environment, mode) -> (new_state, dt, cost)
```

- samples whatever fields it needs at `state.time` and the edge location
  (radiation now; current/wind later);
- computes **speed** from state + mode (low battery → slower), then
  `dt = edge.length / speed`, and integrates **battery** over the crossing (solar
  in via `Rad`, propulsion out) to produce `new_state`;
- returns the incremental `cost` for whatever objective is chosen (Open
  Question 1).

**Future rate models compose here** as a small **pipeline of speed modifiers** —
each reads the environment and scales or caps speed (a current pushes you along;
heavy seas slow you). The search never learns they exist; it only ever calls
`advance`.

Two modelling notes:

- **Fixed-space stepping fits graph search:** edges are geometric segments and
  `dt` is derived. (The alternative — fixed-time stepping where you land between
  nodes — is a simulation, not a graph traversal, and doesn't fit Dijkstra/A\*.)
- **Admissible A\* heuristic:** `remaining_great_circle_distance ÷ max_speed` is a
  valid lower bound on time, because no model can exceed top speed. It keeps A\*
  correct even with state-dependent edge times.

---

## Stage 4 — Handoff to the traversal algorithm

Keep the algorithm ignorant of GRIB, boats, and coastlines. It sees only:

- `neighbours(node)` — expand the frontier over the pruned adjacency;
- `advance(state, edge, env, mode)` — the true cost/time/state transition of a
  step;
- `sample`-backed fields, reached *through* `advance`, never directly;
- a **coordinate ↔ node** mapping (nearest navigable node to an arbitrary
  lat/lon) so callers give start/goal in real coordinates.

Any uniform-cost (Dijkstra) or heuristic (A\*) search over the augmented state
runs unmodified.

---

## Core data structures, at a glance

| Concern | Structure | Notes |
|---|---|---|
| Radiation / env input | **Field sampler** (`ScalarField` / `VectorField`) over the coarse grid | Trilinear interpolation; backed by `FrameCache`; scalar vs vector |
| Ocean test + keep-out | **Boolean ocean raster + distance transform** | Built once from the ocean polygon; yields `is_ocean` and `dist_to_shore` lookups |
| Basin selection | **`access_width` per cell** (max-bottleneck flood / union-find) | Powers `W`; the same flood is the connectivity pass |
| Route space (base) | **All-ocean H3 graph** + per-node `dist_to_shore`, `access_width`, H3 id | Built once; input to pruning; kept for cheap re-prune |
| Route space (runtime) | **Pruned, recompacted graph** — adjacency over contiguous int ids | K & W baked in; no mask predicate at search time; node ids not stable across prunes |
| Search unit | **Augmented state** `(node, time, battery, …)` | Bucketed → product graph, or continuous → Pareto labels |
| Frontier | **Priority queue of labels** + **per-node non-dominated label set** | Dominance check prunes; `came-from` keyed by label |
| Vehicle + future models | **`advance` transition function** + **pipeline of speed modifiers** | Boat is the first; currents/wind/sea-state compose without touching search |

---

## Recommended phasing

1. **Ocean raster + fields.** Rasterize the ocean polygon; distance transform →
   `dist_to_shore`; `Rad(t,p)` sampler with interpolation and `FrameCache`.
   Visualise the ocean mask and distance field as overlays. *No graph yet.*
2. **Base build + prune + distance-only path.** Flood-fill the H3 ocean graph with
   `dist_to_shore` and `access_width`; prune with `(K, W)`; run shortest path at
   fixed speed (no battery). Validates the build, connectivity, the K/W selection
   (e.g. Gibraltar sealing the Med at `W = 30 km`), and coordinate round-tripping.
3. **State-augmented routing (bucketed).** Introduce `advance`, battery state, and
   the objective function; the low-power→slower feedback loop appears here.
   Product-graph Dijkstra/A\*.
4. **Refinements as needed.** Pareto labels if buckets are too coarse; additional
   speed-modifier models (currents, wind); soft distance-to-shore costing.

Each phase is independently useful, and each reuses the previous one's seams
unchanged.

---

## Decisions so far

- **Graph geometry — settled.** A **geodesic H3 hex grid**, built once and pruned
  into a compacted runtime graph over contiguous integer node ids. Resolution
  picks the step (res 7–9 ≈ 1.2 km – 170 m). If the operating domain is bounded,
  the H3 library is optional — a local hex projection gives the same benefit
  dependency free.
- **What counts as navigable — settled.** Mask from the **ocean polygon**, so
  **inland water (lakes, rivers) is treated as land**; the ERA5 land–sea field is
  not used. Coastal / open-ocean operation is the target.
- **Keep-out & basin selection — settled.** Two build-time parameters applied at
  **prune** time to a reused map: **K** (keep-out distance from shore) and **W**
  (minimum navigable channel width, via `access_width`), with **coupled K→W**
  semantics (apply K, recompute width, apply W). `W` selectively masks marginal
  seas (e.g. `W = 30 km` masks the Mediterranean via the Strait of Gibraltar).
- **Inland-water connectivity — settled/closed.** Handled by the ocean polygon
  plus the `access_width` flood; no separate connectivity question remains.

## Still open (need intent, not code)

1. **Objective.** What does `advance` return as `cost` — maximise solar energy
   collected, minimise travel time, minimise time subject to a battery floor, a
   weighted blend? This defines the edge cost and possibly the label dimensions.
   *(The single decision that most constrains `advance`.)*
2. **Vehicle state model.** What exactly is in the boat's internal state beyond
   battery, and how does `mode` map to power draw and speed? This fixes the
   `advance` transition and how finely battery must be bucketed.
3. **Domain extent.** Global (needs the H3 library and antimeridian handling) or a
   bounded region (unlocks the dependency-free local-hex option)? And is
   start/goal fixed, or arbitrary per query? This picks *H3-library vs local
   projection*, not the grid type.
4. **Time horizon.** Over how long a route does radiation change meaningfully
   (hours? days?) — this sets how much the time axis of `Rad` actually matters and
   how large the augmented state space gets.
