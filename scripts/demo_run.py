"""demo_run.py -- a stand-in for the solver, so the rest of the pipeline has
something to carry.

This is NOT a solver and makes no claim to be one. It steers greedily at the
goal, sidesteps when the map says the next step is illegal, and integrates a toy
battery off a sine-wave sun. There is no objective function, no lookahead, and no
vehicle model. Stage 4 of docs/routing_proposal.md is where the real thing goes,
and it should replace this file wholesale.

What it is good for is exercising the map's runtime contract exactly as a real
solver will:

  * legality is a query against a baked (K, W), not a lookup in a list of cells;
  * most steps skip the map entirely, on the clearance budget;
  * arrival is a radius, because continuous motion never lands on a point;
  * a step that cannot be made legal ends the run rather than beaching the boat.

The sidestep is the crudest possible obstacle response -- fan out from the direct
bearing, take the first heading that is legal. It exists so the track stays in
water, not because it is a good idea.

Running it
----------

    uv run python map_gen.py -K 5 -W 10 --save med.npz
    uv run python demo_run.py med.npz -o track.npz
    uv run python vis_map.py med.npz --track track.npz

Two modes:

    demo_run.py med.npz                              # direct: head at the goal
    demo_run.py med.npz --mode coastal --hug 30      # stay within 30 km of shore

Options:

    -o NPZ           where to write the track (default track.npz)
    --mode {direct,coastal}
    --hug KM         coastal band width (default 40)
    --start LAT LON  departure point (mode-specific default)
    --goal LAT LON   destination (mode-specific default)
    --speed KMH      constant speed over ground (default 9)
    --dt MIN         simulation step in minutes (default 20)
    --arrive KM      arrival radius (default 25)
    --max-hours H    give up after this much simulated time (default 400)

Coastal mode needs the map to have been built with a `--floor-res` (map_gen's
default of 5 is enough). Without one, legality is decided at res 2 in open water
and the stored shore distance is a single number across cells hundreds of km
wide, so every candidate heading scores identically and there is nothing to steer
by.

The output .npz holds `lat`, `lng`, `time` (hours from departure) and `battery`
(0..1), all the same length, plus `shore_km` in coastal mode and the scalar
`speed_kmh`. Any 1-D array of the same length is treated as a plottable scalar by
vis_map.py, so adding a channel means adding a key.
"""

from __future__ import annotations

import argparse

import numpy as np

from map_utils import ClearanceBudget, NavMap, great_circle_km, initial_bearing, step_along

# headings tried, in order, when the direct bearing is blocked
SIDESTEP = [0, 20, -20, 40, -40, 65, -65, 90, -90, 115, -115]


def solar_fraction(hours: float, lng: float) -> float:
    """A sine-wave sun: 0 at night, 1 at local noon. Not an insolation model --
    Stage 1 of the proposal is where real GRIB radiation comes from."""
    local = (hours - lng / 15.0) / 24.0
    return float(max(0.0, np.sin(2 * np.pi * (local - 0.25))))


def run(
    nav: NavMap,
    start: tuple[float, float],
    goal: tuple[float, float],
    speed_kmh: float = 9.0,
    dt_min: float = 20.0,
    arrive_km: float = 25.0,
    max_hours: float = 400.0,
) -> dict[str, np.ndarray]:
    dt_h = dt_min / 60.0
    step_km = speed_kmh * dt_h

    lat, lng = start
    if not nav.legal(lat, lng):
        raise SystemExit(f"start {start} is not navigable under K={nav.K}, W={nav.W}")

    lats, lngs, times, batts = [lat], [lng], [0.0], [0.6]
    budget = ClearanceBudget(nav)
    battery, t = 0.6, 0.0
    blocked = 0

    while t < max_hours:
        if great_circle_km(lat, lng, *goal) <= arrive_km:
            break

        bearing = initial_bearing(lat, lng, *goal)
        for offset in SIDESTEP:
            cand = step_along(lat, lng, bearing + offset, step_km)
            # The budget is why this is not a lookup per step: mid-ocean one
            # query certifies hundreds of km and the rest cost two floats.
            probe = ClearanceBudget(nav)
            probe.budget = budget.budget
            if probe.step(*cand, step_km):
                lat, lng = cand
                budget.budget = probe.budget
                budget.steps += 1
                budget.lookups += probe.lookups
                break
        else:
            blocked += 1
            break

        t += dt_h
        # A clipped sine averages 1/pi, so 0.10 charge against 0.030 draw is very
        # slightly net positive over a day and swings ~0.4 within one. Tuned to
        # look like a duty cycle, not derived from anything.
        battery = float(np.clip(battery + (0.10 * solar_fraction(t, lng) - 0.030) * dt_h, 0, 1))
        lats.append(lat)
        lngs.append(lng)
        times.append(t)
        batts.append(battery)

    reached = great_circle_km(lat, lng, *goal) <= arrive_km
    print(f"  {len(times)} steps over {t:.1f} h, {'arrived' if reached else 'stopped short'}")
    if blocked:
        print("  ended blocked: no legal heading within +/-115 degrees of the goal")
    print(f"  {budget.lookups} map lookups for {budget.steps} steps "
          f"({100 * (1 - budget.lookups / max(budget.steps, 1)):.0f}% skipped)")
    print(f"  battery {min(batts):.2f}..{max(batts):.2f}, ended at {battery:.2f}")

    return {
        "lat": np.array(lats),
        "lng": np.array(lngs),
        "time": np.array(times),
        "battery": np.array(batts),
        "speed_kmh": np.float64(speed_kmh),
    }


# fan of headings tried each step, relative to the bearing at the goal
FAN = list(range(-150, 151, 15))


def run_coastal(
    nav: NavMap,
    start: tuple[float, float],
    goal: tuple[float, float],
    hug_km: float = 40.0,
    speed_kmh: float = 9.0,
    dt_min: float = 20.0,
    arrive_km: float = 25.0,
    max_hours: float = 400.0,
) -> dict[str, np.ndarray]:
    """Coast-hugging variant: make progress, but stay inside `hug_km` of shore.

    Each step scores a fan of headings and takes the best. The score trades
    distance still to run against how far outside the coastal band the candidate
    would put the boat:

        score = -distance_to_goal - HUG_WEIGHT * max(0, shore_distance - hug_km)

    Both terms are in km, so the weight is the only tuning. There is no gradient
    of shore distance stored anywhere -- the fan samples it, which is the cheapest
    thing that works and is why this costs a map query per candidate rather than
    riding the clearance budget.

    The inner edge of the corridor is not enforced here: `K` already does it, via
    legality. So the boat ends up between K and hug_km of the coast.

    Two guards stop it chattering, and both are needed. `shore_distance` is a step
    function -- it reports whichever cell decided the query, so it jumps at cell
    boundaries rather than varying smoothly. A big penalty term over a cliffed
    landscape will happily flip between two adjacent positions forever, which is
    exactly what the first version did: 3600 km sailed for 198 km of progress.

    * a turn cost, so holding a heading beats swapping between two that score
      almost the same;
    * a leash on how far the goal is allowed to recede from the closest approach
      so far, which permits a detour round a headland but not an endless one.
    """
    hug_weight = 3.0
    turn_weight = 0.35  # km of score per 90 degrees of course change
    leash_km = 4.0 * hug_km
    dt_h = dt_min / 60.0
    step_km = speed_kmh * dt_h
    probe_km = max(4.0 * step_km, 20.0)  # far enough to cross stored cells

    lat, lng = start
    if not nav.legal(lat, lng):
        raise SystemExit(f"start {start} is not navigable under K={nav.K}, W={nav.W}")

    lats, lngs, times, batts, shores = [lat], [lng], [0.0], [0.6], [nav.shore_distance(lat, lng)]
    battery, t, lookups, stuck = 0.6, 0.0, 0, False
    closest = great_circle_km(lat, lng, *goal)
    last_offset = 0.0

    while t < max_hours:
        if great_circle_km(lat, lng, *goal) <= arrive_km:
            break

        base = initial_bearing(lat, lng, *goal)
        best, best_score, best_d, best_offset = None, -np.inf, None, 0.0
        for offset in FAN:
            cand = step_along(lat, lng, base + offset, step_km)
            # Score the shore distance at a lookahead, not underfoot. One step is
            # a few km and the stored field is coarser than that, so probing at
            # the step distance puts the whole fan inside one cell and every
            # heading scores the same.
            probe = step_along(lat, lng, base + offset, probe_km)
            lookups += 1
            d = nav.shore_distance(*probe)
            if not np.isfinite(d) or not nav.legal(*cand):
                continue  # illegal water, or off the edge of the map
            to_goal = great_circle_km(*cand, *goal)
            if to_goal > closest + leash_km:
                continue
            score = (
                -to_goal
                - hug_weight * max(0.0, d - hug_km)
                - turn_weight * abs(offset - last_offset) / 90.0 * step_km
            )
            if score > best_score:
                best, best_score, best_d, best_offset = cand, score, d, float(offset)
        if best is None:
            stuck = True
            break
        closest = min(closest, great_circle_km(*best, *goal))
        last_offset = best_offset

        lat, lng = best
        t += dt_h
        battery = float(np.clip(battery + (0.10 * solar_fraction(t, lng) - 0.030) * dt_h, 0, 1))
        lats.append(lat)
        lngs.append(lng)
        times.append(t)
        batts.append(battery)
        shores.append(best_d)

    reached = great_circle_km(lat, lng, *goal) <= arrive_km
    sh = np.array(shores)
    inside = 100.0 * np.mean(sh <= hug_km)
    print(f"  {len(times)} steps over {t:.1f} h, {'arrived' if reached else 'stopped short'}")
    if stuck:
        print("  ended stuck: no legal heading in the fan")
    print(f"  shore distance {sh.min():.0f}..{sh.max():.0f} km, {inside:.0f}% inside {hug_km:g} km")
    print(f"  {lookups} map lookups ({len(FAN)} per step -- the fan, not the budget)")
    print(f"  battery {min(batts):.2f}..{max(batts):.2f}, ended at {battery:.2f}")

    return {
        "lat": np.array(lats),
        "lng": np.array(lngs),
        "time": np.array(times),
        "battery": np.array(batts),
        "shore_km": sh,
        "speed_kmh": np.float64(speed_kmh),
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("npz", help="map artifact from map_gen.py --save")
    p.add_argument("-o", "--out", default="track.npz", help="where to write the track")
    p.add_argument(
        "--mode",
        default="direct",
        choices=("direct", "coastal"),
        help="direct: head at the goal. coastal: stay within --hug km of shore.",
    )
    p.add_argument("--hug", type=float, default=40.0, metavar="KM", help="coastal band width")
    p.add_argument("--start", type=float, nargs=2, metavar=("LAT", "LON"))
    p.add_argument("--goal", type=float, nargs=2, metavar=("LAT", "LON"))
    p.add_argument("--speed", type=float, default=9.0, metavar="KMH")
    p.add_argument("--dt", type=float, default=20.0, metavar="MIN")
    p.add_argument("--arrive", type=float, default=25.0, metavar="KM")
    p.add_argument("--max-hours", type=float, default=400.0, metavar="H")
    args = p.parse_args()

    nav = NavMap.load(args.npz)
    print(f"map: K={nav.K} km, W={nav.W} km, res {nav.res_min}..{nav.res_base}")

    # Mode-specific defaults, because the two want different geometry. The direct
    # run heads for open water off Morocco; a coastal run given that goal can
    # never satisfy both objectives, since the goal sits ~200 km offshore, so it
    # burns the clock trading one against the other. Its default goal is on the
    # coast instead.
    start = tuple(args.start) if args.start else (
        (42.5, -9.6) if args.mode == "coastal" else (42.0, -10.5)
    )
    goal = tuple(args.goal) if args.goal else (
        (36.2, -6.4) if args.mode == "coastal" else (31.5, -11.5)
    )
    common = dict(
        speed_kmh=args.speed,
        dt_min=args.dt,
        arrive_km=args.arrive,
        max_hours=args.max_hours,
    )
    print(f"{start} -> {goal}")
    if args.mode == "coastal":
        print(f"mode: coastal, staying within {args.hug:g} km of shore")
        track = run_coastal(nav, start, goal, hug_km=args.hug, **common)
    else:
        print("mode: direct")
        track = run(nav, start, goal, **common)
    np.savez_compressed(args.out, **track)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
