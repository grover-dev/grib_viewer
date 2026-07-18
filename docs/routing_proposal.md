# Proposal: GRIB solar-radiation → traversable graph with landmass keep-out

## Goal

Route a vehicle across the sea using a graph traversal / shortest-path
algorithm, where:

- **surface solar radiation** (from GRIB, varying over space and time) drives the
  cost/objective of a route;
- **landmasses plus a configurable keep-out buffer** are excluded from the
  traversable space;
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
neighbours(node)                        — geometry (the nav graph)
field.sample(t, lat, lon)               — environment (radiation, later currents/wind, the mask)
advance(state, edge, env, mode)         — dynamics (the boat, later other models)
```

---

## The pipeline in one picture

```
GRIB messages ──index──▶ coarse fields  R(t, lat, lon)      (~25 km cells)
                                │
                                ▼
                    Field samplers  (interpolate in space + time)
                    ├─ Rad(t, p)        scalar   ── W/m²
                    ├─ Mask(p)          boolean  ── traversable?  (land + keep-out)
                    └─ …future: Current(t,p), Wind(t,p)  vector
                                │
                                │  queried by
                                ▼
   Navigation graph  (own geometry, YOUR step size, independent of the grid)
        nodes = points in navigable water
        edges = neighbour links, carrying geometry only (length, bearing)
                                │
                                │  priced/advanced per edge by
                                ▼
   Vehicle dynamics  advance(state, edge, env, mode) -> (new_state, dt, cost)
        boat(time, mode) + internal state (battery); future models compose here
                                │
                                ▼
   State-augmented search  over labels (node, time, battery, …)
        Dijkstra / A* on the augmented state space
```

Each block is independent and testable in isolation. The first already exists in
this codebase; the proposal is the rest.

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
  every environmental input — including the keep-out mask — looks the same to the
  rest of the system. The "environment" is then just a **list of fields**.

---

## Stage 2 — Land mask + configurable keep-out (as a field)

The keep-out test becomes just another sampler: `Mask(lat, lon) -> traversable?`.

### 2a. Source of "what is land"

| Option | Source | Pros | Cons |
|---|---|---|---|
| **A. Land polygons** | Natural Earth coastlines (already used for the basemap) | Independent of the GRIB; crisp coastline; resolution selectable; queryable at any point | Needs point-in-polygon / rasterize; vector dependency |
| **B. Land–sea field** | A land–sea mask variable in the GRIB, if present (fraction 0–1) | Already a field on the grid; zero extra data | Only if the dataset includes it; coarse at the coast; threshold picks land vs sea |

**Recommendation: A primary, B automatic fallback.** Both reduce to the same
product — a **boolean land test** — so nothing downstream cares which was used.
Because the nav graph is finer than the grid, polygons (A) are the better fit:
they answer "is this exact point on land?" without resampling.

### 2b. The keep-out buffer (the configurable part)

Keep-out = "no vehicle within K of shore." Expose **K as a single distance
(km)** and grow the land region by it:

- Buffer in **true distance**, not raw cells — a degree of longitude shrinks with
  latitude (× cos φ), so a fixed cell count is wrong near the poles. (Same
  cos-latitude correction the scale bar already applies.)
- With **polygons**, buffer the coastline geometry by K directly, or keep a
  **distance-to-shore** value so the test is `distance_to_shore >= K`. Exposing
  distance-to-shore as its own scalar field also enables a *soft* keep-out later
  (penalise near-shore cells instead of hard-excluding them).
- **K is a graph-build parameter, not an ingest parameter** — changing it is a
  cheap re-query of the mask, no GRIB re-read. `K = 0` (hug the coast) must be
  valid.
- **Enclosed water:** a raw land test leaves lakes/inland seas "navigable." If
  that matters, keep only the water region connected to the start (a connectivity
  fill) and drop the rest.

Output: a `Mask` sampler (hard boolean) and optionally a `DistanceToShore` scalar
field (for soft costing).

---

## Stage 3 — Navigation graph + state-augmented search

This is the heart of the redesign. Three parts: **geometry**, **state**, and
**search**.

### 3a. Graph geometry — decoupled, and precomputed once

The graph's spacing is *yours*, chosen for a 100 m – few-km step, unrelated to the
25 km grid. Two independent choices: **what shape the cells are**, and **whether
the graph is generated on demand or baked ahead of time**.

**Cell geometry** (Open Question 3):

- **Fine regular lattice** at your spacing — simplest; cos-latitude distortion
  returns and must be corrected in every distance (the scale bar's cos φ factor).
  Also carries the square-grid anisotropy: 4 edge-neighbours at `s`, 4 diagonals
  at `s√2`, which biases least-cost paths toward the axes/diagonals.
- **Geodesic hex grid (H3)** — *recommended.* Cells are near-equal-area
  everywhere, so the pole distortion and the antimeridian seam disappear, and
  every cell has **6 edge-adjacent neighbours at one uniform distance** — no √2
  bookkeeping, no directional bias. A cell id is a 64-bit integer, which also
  gives node identity for free (no floats-as-keys trap). Resolution sets the step:
  res 7 ≈ 1.2 km, res 8 ≈ 460 m, res 9 ≈ 170 m — your whole target range.
- **Navmesh / visibility graph from coastline polygons** — sparse; makes step size
  a property of the search rather than a grid. Strong fit for coast-hugging, but a
  bigger build.

**On H3 and the library question.** H3's hex *math* is simple; what needs the
library is tiling a **sphere** with hexagons globally without seams (icosahedron
projection, cross-face transforms, the 12 unavoidable pentagon cells, exact index
encoding). The catch to know: those failures are *silent* — a precision slip near
a face boundary mislabels a cell's neighbours and the search returns a subtly
wrong path with no error, which is why the tested library is worth it when the
domain is global. **If the routing domain is bounded** (a basin, a coastline, one
crossing), you can skip the dependency entirely: project the region to a local
gnomonic / equal-area plane, lay a flat hex grid there, and map cell centres back
to lat/lon — a few hundred lines that keep the uniform-hex benefit within the
region, at the cost of distortion far from centre and no poles/dateline.

**Precomputed navigation graph (the "navmesh").** The geometry seam is *static* —
which cells are water and how they connect never depends on time, radiation, or
boat state. So bake it once and query it forever:

```
Build (offline, uses H3):
  1. cells       = polygon_to_cells(domain, R)
  2. water cells = [c for c in cells if Mask.water(center(c))]     # coastline test, once
  3. edges       = grid_disk(c,1) neighbours that are also water   # ≤6 per cell
  4. per edge: length + bearing from cell-centre great-circle distance
  5. renumber cells to contiguous ints 0..N-1                      # drop 64-bit ids from the hot path
  6. store: CSR adjacency + node→(lat,lon) + node→distance_to_shore

Query (runtime, no coastline geometry, little/no H3):
  - start/goal → node via latlng_to_cell + id map, or a k-d tree over centres
  - search over the CSR graph; advance() prices edges live
```

Only **geometry** is serialised — never costs, times, or radiation, which stay in
`advance`/`field.sample`. The build freezes `neighbours(node)` into a compact
integer-indexed adjacency (tens of MB for a basin); runtime sees a plain graph.

Because H3 is hierarchical, the build can go **adaptive** — fine cells near the
coast, coarse in open water — baked into the *same* flat integer graph, so the
mixed-resolution complexity is paid once and the search never sees it.

**Keep-out K stays a query-time knob (the part you liked).** Do **not** bake K
into the graph. Build the full water graph at **K = 0** and store
**distance-to-shore per node**; at runtime, `distance_to_shore(node) >= K` filters
nodes/edges on the fly. K is then adjustable per query over one artifact — no
rebuild to tighten or loosen the coastal margin — and the same stored field powers
a *soft* keep-out (penalise near-shore) later.

**Version the artifact** on `(coastline data, resolution, domain, K-policy)` so a
stale one rebuilds — the same move as the existing `.cache/*.index` for GRIB
messages (`_index_key` hashing path+size+mtime), generalised from "index the
messages" to "index the navigable geometry."

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
- the implicit **neighbour generator** from 3a;
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

- `neighbours(node)` — expand the frontier;
- `advance(state, edge, env, mode)` — the true cost/time/state transition of a
  step;
- `sample`-backed fields, reached *through* `advance`, never directly;
- a **coordinate ↔ node** mapping (nearest navigable node to an arbitrary
  lat/lon) so callers give start/goal in real coordinates.

Any uniform-cost (Dijkstra) or heuristic (A\*) search over the augmented state
runs unmodified. Optionally serialise the built mask/graph geometry so a heavy
build is reused across many route queries — but note that **costs are never
serialised**, only geometry, since costs are recomputed from state every run.

---

## Core data structures, at a glance

| Concern | Structure | Notes |
|---|---|---|
| Radiation / env input | **Field sampler** (`ScalarField` / `VectorField`) over the coarse grid | Trilinear interpolation; backed by `FrameCache`; scalar vs vector |
| "Is this water?" | **Mask sampler** (+ optional `DistanceToShore` field) | Same interface as any field; K applied at query time |
| Route space | **Precomputed nav graph** — CSR adjacency over contiguous int node ids | Built once via H3 (or local hex projection); edges = length+bearing only; K applied at query time via stored distance-to-shore |
| Search unit | **Augmented state** `(node, time, battery, …)` | Bucketed → product graph, or continuous → Pareto labels |
| Frontier | **Priority queue of labels** + **per-node non-dominated label set** | Dominance check prunes; `came-from` keyed by label |
| Vehicle + future models | **`advance` transition function** + **pipeline of speed modifiers** | Boat is the first; currents/wind/sea-state compose without touching search |

---

## Recommended phasing

1. **Field samplers + mask.** `Rad(t,p)` with interpolation and `FrameCache`;
   `Mask(p)` from polygons (land–sea fallback) with a configurable km keep-out.
   Visualise the mask as an overlay. *No graph yet.*
2. **Precompute the nav graph + distance-only path.** Bake the H3 (or
   local-hex) water graph with per-node distance-to-shore; run shortest path at
   fixed speed (no battery) with K applied as a query-time filter. Validates
   the build, connectivity, K-via-distance-to-shore, and coordinate
   round-tripping.
3. **State-augmented routing (bucketed).** Introduce `advance`, battery state, and
   the objective function; the low-power→slower feedback loop appears here.
   Product-graph Dijkstra/A\*.
4. **Refinements as needed.** Pareto labels if buckets are too coarse; additional
   speed-modifier models (currents, wind); soft distance-to-shore keep-out.

Each phase is independently useful, and each reuses the previous one's seams
unchanged.

---

## Open questions (need intent, not code)

1. **Objective.** What does `advance` return as `cost` — maximise solar energy
   collected, minimise travel time, minimise time subject to a battery floor, a
   weighted blend? This defines the edge cost and possibly the label dimensions.
2. **Vehicle state model.** What exactly is in the boat's internal state beyond
   battery, and how does `mode` map to power draw and speed? This fixes the
   `advance` transition and how finely battery must be bucketed.
3. **Graph geometry & domain.** Fine lattice, geodesic grid, or navmesh? Global
   (needs antimeridian wrap) or a regional crop? Fixed start/goal or arbitrary per
   query?
4. **Keep-out semantics.** Hard exclusion only, or also a soft distance-to-shore
   penalty? Do inland lakes/seas need connectivity filtering?
5. **Time horizon.** Over how long a route does radiation change meaningfully
   (hours? days?) — this sets how much the time axis of `Rad` actually matters and
   how large the augmented state space gets.
