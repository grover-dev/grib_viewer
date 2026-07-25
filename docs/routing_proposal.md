# Proposal: solar-radiation routing over a global ocean map

## Goal

Simulate a solar-powered vessel crossing the sea, where:

- surface solar radiation (from GRIB, varying in space and time) drives the objective;
- landmasses, inland water, and a configurable keep-out buffer are excluded from the
  navigable space — the target is coastal and open-ocean operation, so lakes and
  rivers count as land;
- the vehicle's own state (battery, mode) affects how fast it moves and therefore how
  far it gets;
- further models (currents, wind, sea state) can be added without reworking the core.

This document describes the data model, the transformations, and the decisions, not
the code.

### Why this is not a graph search

The GRIB cells are coarse (~0.25° ≈ 25 km), far too coarse for a route whose step is
hundreds of metres to a few km. So the route lives in continuous position, and the
radiation grid is exposed only as a sampled function `Rad(t, lat, lon)`.

Speed depends on vehicle state, and that state depends on the whole track so far, so
edge costs cannot be precomputed. The problem is a closed loop: a solver picks an
action, the boat's dynamics say what that action achieves, and the ocean carries the
boat somewhere.

Four seams keep the pieces independent:

```
solve(state, env, value_fn)          — the decision
advance(state, request, env)         — dynamics (the boat)
propagate(state, velocity, env, dt)  — kinematics (drift and currents)
field.sample(t, lat, lon)            — environment (radiation, later currents/wind)
```

Plus two map lookups, `legal(p)` and `value_fn(p)`. The map is built once ahead of
time; everything else runs live.

---

## The pipeline in one picture

```
                     ┌──────────── BUILD ONCE (the map) ──────────────┐
Ocean polygon ──▶ rasterize ──▶ ocean raster ──▶ is_ocean(p)
                                                      │
                     fast marching on the H3 grid ────┘
                                     ▼
                              dist_to_shore(p)
                                     ▼
                        access_width(p)  (max-bottleneck flood)
                                     ▼
                   H3 PYRAMID: per cell, min/max of dist_to_shore
                              and access_width over its footprint
                                     ▼
                   RUNTIME artifact:  legal(p)   — K, W applied at query
                                      value_fn(p) — cost-to-go per goal
                     └────────────────────────────────────────────────┘
                                     │
GRIB messages ─index─▶ coarse fields │ Rad(t,p) sampler (bi/trilinear, W/m²)
                                     ▼            │
   Solver            solve(state, env, value_fn) -> request vector
                                     ▼
   Vehicle dynamics  advance(state, request, env) -> velocity through water, draw
        boat(time, mode) + internal state (battery); future models compose here
                                     ▼
   Ocean kinematics  propagate(state, velocity, env, dt) -> end lat/lon, battery
        drift and currents; swept path checked against legal(p)
```

Each block is independent and testable in isolation. The GRIB indexing already exists
in this codebase; the proposal is the rest.

---

## Stage 1 — Radiation as a sampled field

The existing index yields, per valid time, a radiation frame on a uniform lat/lon
grid. Wrap it as a sampler:

- Interface: `Rad(t, lat, lon) -> W/m²`, bilinear in lat/lon and linear between the
  two frames bracketing `t`. A caller gets a value at any point and any instant, so
  nothing downstream inherits the 25 km grid.
- Units are normalised here: accumulated J/m² becomes W/m² by dividing by the
  accumulation seconds, as the viewer already does.
- Timestamp each frame at the midpoint of its accumulation window, not at its valid
  time. Dividing an accumulation by its window gives the mean rate over that window,
  and for ERA5 the window is `(t − 1h, t]`, so the value describes `t − 30 min`.
  Attributing it to `t` shifts the modelled diurnal cycle half a window late, moving
  sunrise by 30 minutes and biasing every energy integral with it.
- Sampling at `t` needs only the two bracketing frames, so the existing `FrameCache`
  (byte-bounded LRU of decoded frames) is the right structure.
- Radiation is scalar; currents and wind will be vector (`-> (u, v)`). Define
  `ScalarField.sample` and `VectorField.sample` so the environment is just a list of
  fields.

Known error, accepted for now: linear interpolation between hourly means does not
reproduce the accumulated daily total, and it softens the sunrise and sunset knees
where the error concentrates. If the objective is energy collected, that lands
directly on the quantity being optimised. Two drop-in fixes if it proves to matter:
interpolate the cumulative curve with a monotone spline (PCHIP) and differentiate it,
which reproduces every frame's total by construction and cannot emit a negative rate;
or interpolate a clear-sky index (the ratio to computed top-of-atmosphere irradiance)
and re-multiply, which gets the diurnal shape closer. Neither changes the interface.

Out of range is a hard stop. `Rad` is undefined past the last frame, before the first,
or outside the GRIB's spatial subset. The sampler signals that rather than
extrapolating or clamping, and the simulation terminates when it runs out of data. A
rollout reaching the edge of the dataset ends there; that is the end of what can be
evaluated, not a failure of the route.

This runs on historical reanalysis. ERA5 describes weather that already happened, so
the tool answers what a boat would have done over a past period — backtesting and
design evaluation, not operational routing. Forecast horizon, forecast uncertainty,
and update cadence are therefore out of scope. Going live would mean swapping the
source for a forecast product (IFS, GFS) behind the same interface.

The land/ocean mask is not one of these runtime fields; it is consumed at build time
(Stage 2).

---

## Stage 2 — The map (build time)

This stage produces the two scalar fields that decide which water is navigable. It
never runs at query time.

### 2a. Ocean, not water

Inland water counts as land, so the mask asks "is this ocean?", not "is this water?".

Build it from an ocean polygon (e.g. Natural Earth's `ocean` layer, already the family
of data used for the basemap). Lakes and rivers are not ocean, so they are
non-navigable by construction, with no heuristics involved. A generic water test
cannot do this: rivers and estuaries connect to the sea, so any flood over "water"
runs up them. Inland water has to be excluded by classification, which the ocean
polygon supplies. The ERA5 land–sea field has the same problem — it marks a lake as
water with no way to distinguish it from sea — so it is at best a coarse fallback
needing a separate lake mask bolted on.

Marginal seas (Mediterranean, Baltic, Black Sea) are part of the ocean polygon and are
kept. Whether they survive is decided by `W` below, not by classification.

### 2b. Classify in lat/lon, measure on H3

Two jobs with different requirements; separating them is what removes a metric error.

Classification has no metric, so lat/lon is fine. Burn the ocean polygon into a
boolean raster fine enough to resolve the narrowest passage that matters (~0.005° ≈
500 m). Longitude wrapping handles the antimeridian, and the degenerate pole rows
carry nothing a marine map needs. `is_ocean` is then a raster lookup.

Distance has a metric, so it cannot be measured in lat/lon. A Euclidean distance
transform on a lat/lon raster finds the nearest pixel in pixel space, but the true
metric is anisotropic — 2:1 at 60°N — and no cos-latitude factor applied afterwards
repairs a wrong nearest neighbour. The error runs between 1× and 1/cos φ depending on
which way the coast lies. Since `K` is denominated in km and `W + 2K` inherits it,
this would corrupt the one parameter the build exists to serve.

Instead compute `dist_to_shore` by multi-source fast marching over the H3 grid,
seeded from every ocean cell adjacent to a non-ocean cell. H3 cells are near-equal-area
with near-uniform centre spacing, so the metric is isotropic at every latitude and
across the antimeridian, with no projection involved.

Build resolution is set by the narrowest passage `W` must adjudicate. Gibraltar at
~14 km needs cells well under that: res 7 (~2.1 km centre spacing) puts about 7 cells
across the strait, res 6 (~5.6 km) puts 2–3 and is marginal. At res 7 the global ocean
is ~7×10⁷ cells, a few hundred MB of float32 as a one-time offline working set. The
runtime artifact is far smaller, since the pyramid stores children only for mixed
cells (3b).

Validation: the maximum of `dist_to_shore` should land near Point Nemo at ~2690 km. A
peak elsewhere, or much higher, means a missing landmass or a broken metric.

### 2c. Connectivity: flood-fill from an open-ocean seed

Enumerate the navigable cells from a known open-ocean cell, expanding through ocean
and refusing to cross non-ocean:

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

This gives enumeration, mask, and connectivity in one pass, never materialises land,
and leaves the antimeridian and poles to `grid_disk`. The world ocean is one connected
body, so one seed usually suffices; use several for genuinely disconnected basins, and
check the ocean-cell count against expected sea area, since a missing seed silently
drops a whole basin.

### 2d. K and W

Two scalars decide the navigable set. Both are thresholds on fields computed here at
build time but applied at query time, so one build serves any `(K, W)`.

K, the keep-out distance in km: no vehicle within K of shore, `dist_to_shore ≥ K`.
`K = 0` (hug the coast) is valid.

W, the minimum navigable channel width in km: a body of water counts only if it
connects to open ocean through a channel at least `W` wide. This is a morphological
opening of the ocean by radius `W/2`, and it is what selectively masks marginal seas —
Gibraltar is ~14 km wide, so `W = 30 km` seals it and the Mediterranean drops out,
while `W = 10 km` keeps it. The rule is uniform: `W` closes every passage narrower
than itself, trimming coastal inlets as well as severing marginal seas.

W is realised through a per-cell `access_width`, the widest bottleneck on the best
path to open ocean:

```
access_width(c) = max over paths P from c to an open-ocean seed
                     of  2 · min_{x in P} dist_to_shore(x)
```

computed with a maximum-bottleneck flood: sort cells by `dist_to_shore` descending,
union-find them, and record twice the `dist_to_shore` at which each first connects to
a seed. This flood is also the connectivity pass — an enclosed lake never connects to
a seed, so its `access_width` is 0 and any `W > 0` masks it, on top of the ocean
polygon already excluding it.

Coupled K→W, in closed form. `W` should mean the width of the corridor usable after
the keep-out, not the raw shore-to-shore width: a 30 km strait with a 5 km keep-out
each side leaves 20 km of navigable water. That reads as a two-pass procedure — erode
the near-shore band, recompute `access_width` on what remains, then threshold — but it
is not necessary. Eroding by `K` shifts every bottleneck in by exactly `K` on each
side, and subtracting a constant preserves the ordering the max-bottleneck flood
depends on, so `access_width_after_keepout(p) = access_width(p) − 2K` exactly. The only
exception is corridors whose raw bottleneck was already below `K`, which the
`dist_to_shore` test excludes anyway. So the coupling reduces to one comparison
against the unmodified field:

```
legal(p)  ≡  dist_to_shore(p) ≥ K   and   access_width(p) ≥ W + 2K
```

There is no second flood, no prune pass, and no per-`(K, W)` artifact, which is what
makes both parameters free to change at query time.

---

## Stage 3 — Runtime map and the simulation loop

### 3a. What runtime gets

Stage 2 emits two scalar fields over position:

```
dist_to_shore(p)   — km to the nearest non-ocean thing (continent, island, lake edge)
access_width(p)    — narrowest bottleneck on the widest corridor from p to open ocean
```

Legality is the pair of comparisons derived in 2d, evaluated per query. Nothing about
`K` or `W` is baked into the artifact.

### 3b. Legality at the resolution the situation demands

`legal(p)` is a property of two continuous fields, so it can be answered at whatever
resolution is sufficient: coarse in open water, fine near a coast. Store a pyramid
over H3 resolutions holding, per cell, the min and max of each field over that cell's
footprint:

```
cell -> (min_d, max_d, min_a, max_a)
```

A query descends from a coarse level:

```
if min_d ≥ K and min_a ≥ W + 2K:   LEGAL    — whole cell is clear, stop
if max_d < K  or  max_a < W + 2K:  ILLEGAL  — whole cell is excluded, stop
else:                              descend
```

Mid-Atlantic this settles at res 2–3 on the first comparison; approaching a coastline
it descends until the mixed cells resolve. Query cost scales with proximity to land.

Only mixed cells need children stored, so uniform cells are leaves and the pyramid is
a compressed encoding of the two fields rather than an index bolted on top of them.
That is what makes global coverage affordable: a fine global raster of `dist_to_shore`
at 500 m would be ~3×10⁹ samples.

Aggregate each level directly from the source, never by combining children. H3
hexagons do not nest — a res-`r+1` cell is not contained in its res-`r` parent, and
`cellToChildren` returns 7 cells whose union only approximates it — so taking a cell's
`min_d` as the min over its children is unsound near cell boundaries, which is exactly
where coastlines sit. Computing each level over its own footprint reduces the
resolution ladder to an indexing convenience with no correctness role. Descent
re-indexes at each level rather than following child links, for the same reason.

### 3c. Why H3

Near-equal-area cells everywhere, so a resolution level means the same thing at every
latitude and a distance measured on the grid is isotropic; global seamlessness,
including the antimeridian and both poles; and 64-bit integer ids that double as
spatial hash keys. Uniform neighbour spacing, which motivated the original choice, no
longer matters now that motion is continuous and headings are not quantized.

The domain is global, so the library is not optional. What needs a tested
implementation is exactly what global demands: tiling a sphere without seams —
icosahedron projection, cross-face transforms, the 12 pentagons, exact index encoding.
Those failures are silent, a precision slip near a face boundary mislabelling geometry
with nothing raised.

Pentagons will be hit here, unlike in most H3 applications. The icosahedron is
oriented so that all 12 fall in water, which spares land-based users and guarantees a
marine map meets every one. They have 5 neighbours and off-nominal area, so the flood
(2c), the fast marching (2b), and the cost-to-go (3d) each need the special case.

### 3d. Cost-to-go

A solver that looks only a few steps ahead will route into culs-de-sac and around the
wrong side of a peninsula. Precompute a terminal value function per goal: take the
legal cells at a working resolution, connect each to its `grid_disk(c, 1)` neighbours,
and run one Dijkstra from the goal, storing distance-to-goal per cell. It is a lookup
for "how much further from here", not a route, and it is the only remaining use of
cell adjacency.

Its resolution is bounded below by `W`, not free. `value_fn` is only a heuristic, so
the instinct is to compute it coarse: global res 5 is ~1.4×10⁶ ocean cells against res
7's ~7×10⁷. But res 5 centres sit ~15 km apart, so Gibraltar is not traversable on
that grid, and `value_fn` would report the Mediterranean reachable only around Africa
while `legal(p)` admits it. A heuristic that disagrees with legality about which
passages exist steers the solver away from the correct route rather than merely
slowing it down. So `value_fn` must resolve every passage the `(K, W)` in use admits,
which puts it at res 6 (~1×10⁷ ocean cells) at the coarsest.

### 3e. The seams

```
solve(state, env, value_fn)          -> request vector  (desired heading + speed, or mode)
advance(state, request, env)         -> achievable velocity through water, power draw
propagate(state, velocity, env, dt)  -> end lat/lon, battery, t + dt
legal(p) / value_fn(p)               -> map lookups
```

`solve` owns the decision and is deliberately unspecified here; see Stage 4.

`advance` owns the boat: what speed through water is achievable given battery and
mode, and what it costs in power. Future rate models (sea-state drag, hull fouling)
compose here as a pipeline of speed modifiers.

`propagate` owns the ocean. It samples the environment's vector fields and integrates
achievable velocity plus drift over `dt` to a new position, integrating battery across
the same interval (solar in via `Rad`, propulsion out). It is an integrator, not a
field — it consumes fields — and holding that distinction is what keeps the fields
pure, cacheable, and independently testable.

Because the request is a vector, set-and-drift falls out correctly. Holding a course
over ground against a current requires crabbing, and with speed through water `V` and
the current decomposed into along-course `c∥` and cross-course `c⊥`:

```
speed_over_ground = c∥ + sqrt(V² − c⊥²),   feasible only if V > |c⊥| and the result > 0
```

Below that threshold the requested course is unachievable — a weak boat cannot cross a
strong set. A scalar speed cannot express this; a vector can.

### 3f. Stepping, collisions, termination

`dt` is a chosen quantity, so four things a graph formulation would have given for
free have to be specified.

Swept-path legality: checking only the endpoint lets a step jump over a headland or an
island. The pyramid supplies a cheap sufficient condition — if `min_d` at the
terminating cell of the query exceeds the step distance, no collision is possible and
the check can be skipped. Sample along the segment only when that fails, which is
exactly when near land.

Endpoint policy: drift can push the boat into illegal water from a legal request. Pick
one contract and state it — reject the action, clamp to the legality boundary, or
terminate the rollout as a failure. Solvers behave very differently under each.

Arrival tolerance: continuous drift means never landing exactly on the goal, so
arrival is a radius and the solver needs it as a termination condition.

`dt` selection trades fidelity against rollout cost, and interacts with the radiation
frame spacing (Stage 1) and with step distance relative to clearance. A step long
enough to cross a channel is a step whose swept-path check cannot be skipped.

---

## Stage 4 — Handoff to the solver

The solver reaches the world only through the seams in 3e, and nothing it touches
knows about GRIB, coastlines, or H3:

- `solve` is called with the current state, the environment, and `value_fn`;
- `legal(p)` answers navigability in near-constant time, `K` and `W` already applied;
- `value_fn(p)` supplies distance-to-goal, so local decisions stay globally sane;
- `advance` and `propagate` together form a simulator: given a request vector they
  return the next state. Any solver that can call that in a loop can be swapped in
  without touching anything else.

Waiting needs no special machinery. A zero-speed request is an ordinary action:
`advance` reports station-keeping power draw, `propagate` integrates solar in and drift
over `dt`. Loitering to charge through the night is therefore expressible — but only
if station-keeping draw is non-zero, since if holding is free the solver will loiter
indefinitely to top up. That draw is what makes waiting a trade-off rather than a
dominant strategy.

The solver class is still open, and it changes what else is needed around it:

- MPC / receding horizon wants cheap rollouts over a short horizon and leans hard on
  `value_fn` as the terminal term.
- Sampling-based (RRT\*, MCTS) wants a steering function toward a sampled target and
  leans hard on the swept-path test; it needs the H3 id as a visited-state hash key,
  since continuous positions give no free revisit pruning.
- A learned policy (RL) wants `advance` and `propagate` fast enough to run millions of
  times, which puts the pyramid's early-out on the hot path.

All three want the same substrate, so Stages 1–3 can be built before the solver is
chosen; only the phasing past that point depends on it.

---

## Core data structures, at a glance

| Concern | Structure | Notes |
|---|---|---|
| Radiation / env input | Field sampler (`ScalarField` / `VectorField`) over the coarse grid | Trilinear; frames stamped at accumulation midpoints; backed by `FrameCache` |
| Ocean classification | Boolean ocean raster in lat/lon | Burned from the ocean polygon; no metric involved, so lat/lon is safe here |
| Keep-out field | `dist_to_shore` by multi-source fast marching on the H3 grid | Measured on equal-area cells so the metric is isotropic |
| Basin selection | `access_width` per cell, by max-bottleneck flood / union-find | Powers `W`; the same flood is the connectivity pass |
| Legality | H3 pyramid of per-cell min/max of both fields | Thresholded at the coarsest sufficient resolution; K and W applied at query; only mixed cells store children |
| Global guidance | Cost-to-go per cell, one Dijkstra from the goal | `value_fn(p)`; the only remaining use of cell adjacency; resolution bounded below by `W` |
| Simulation state | Continuous `(lat, lon, time, battery, …)` | No discrete node; H3 id serves as the spatial hash key for revisit pruning |
| Vehicle + future models | `advance`, plus a pipeline of speed modifiers | Boat is the first; sea-state and fouling compose without touching the solver |
| Ocean interaction | `propagate`, an integrator over vector fields | Drift and currents to an end lat/lon; crabbing makes some courses infeasible |

---

## Recommended phasing

1. Fields. Burn the ocean raster; fast-march `dist_to_shore` on the res-7 H3 grid;
   build the `Rad(t,p)` sampler with interpolation and `FrameCache`. Visualise the
   ocean mask and distance field as overlays. No map artifact yet.
2. The map. Flood for connectivity and `access_width`; build the min/max pyramid;
   implement `legal(p)` with query-time `(K, W)` and the cost-to-go Dijkstra. Three
   things to validate: `dist_to_shore` peaks near Point Nemo at ~2690 km; the K/W
   selection behaves (Gibraltar sealing the Med at `W = 30 km`); and legality is
   resolution-independent, meaning the same points classify identically whether the
   query terminates coarse or descends to a leaf. This is also the phase where the 12
   pentagons and the antimeridian either work or silently do not.
3. The simulator. Battery state, mode to power draw, drift integration, swept-path
   checking, endpoint policy. Drive it with a trivial hand-written policy (steer down
   `value_fn`) rather than a real solver. The low-power-to-slower feedback loop appears
   here, and this phase can be validated against known passages at fixed speed.
4. The solver. Whichever class is chosen; it consumes the substrate from phases 1–3
   unchanged.
5. Refinements. Currents and wind as vector fields; sea-state speed modifiers; a soft
   distance-to-shore preference on top of the hard `K`.

Each phase is independently useful, and each reuses the previous one's seams unchanged.

---

## Decisions so far

- Route representation: continuous position with a closed-loop simulator, not a graph
  search.
- Domain extent: global. This makes the H3 library a hard dependency, forces
  `dist_to_shore` onto the H3 grid rather than a raster, and guarantees the build meets
  all 12 pentagons (3c).
- Map indexing: an H3 pyramid, built at res 7, chosen for equal-area cells, global
  seamlessness, and integer ids — not for neighbour uniformity (3c).
- What counts as navigable: the ocean polygon, so inland water is treated as land; the
  ERA5 land–sea field is not used (2a).
- Keep-out and basin selection: `dist_to_shore ≥ K and access_width ≥ W + 2K`, which
  carries the coupled K→W semantics exactly and leaves both as query-time parameters
  over a single build (2d).
- Inland-water connectivity: closed, handled by the ocean polygon plus the
  `access_width` flood.
- Data regime: historical ERA5 reanalysis, so this backtests rather than routes
  operationally, and the simulation terminates when the data runs out (Stage 1).
- Radiation interpolation: trilinear, on frames stamped at accumulation midpoints. It
  does not conserve daily energy; that error is knowingly accepted, with two drop-in
  upgrades noted if it proves material (Stage 1).

## Still open (need intent, not code)

1. Solver class — MPC, sampling-based, or learned policy. Stage 4 lists what each
   additionally needs. Everything in Stages 1–3 is common substrate, so this can be
   deferred, but nothing past phase 3 can start without it. The largest open decision.
2. Objective — maximise energy collected, minimise travel time, minimise time subject
   to a battery floor, or a weighted blend. Since the solver is not Dijkstra this is
   unconstrained in form: it need not be additive or non-negative, and maximising
   objectives are fine. It still fixes what `solve` scores and what the vehicle model
   must expose.
3. Vehicle state model — what is in the boat's state beyond battery, and how `mode`
   maps to power draw and speed. Must include station-keeping draw (Stage 4).
4. Goal mobility — is the goal fixed per deployment or arbitrary per query? This
   decides whether `value_fn` bakes into the artifact or is solved per request, and at
   res 6–7 that is a 10⁷–10⁸-cell Dijkstra each time (3d).
5. Time horizon — over how long a route does radiation change meaningfully? This sets
   how much the time axis of `Rad` matters, and how much of the reanalysis a single
   simulation consumes before it terminates.
