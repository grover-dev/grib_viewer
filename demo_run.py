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

Options:

    -o NPZ           where to write the track (default track.npz)
    --start LAT LON  departure point (default 42.0 -10.5, off Galicia)
    --goal LAT LON   destination (default 31.5 -11.5, off Morocco)
    --speed KMH      constant speed over ground (default 9)
    --dt MIN         simulation step in minutes (default 20)
    --arrive KM      arrival radius (default 25)
    --max-hours H    give up after this much simulated time (default 400)

The output .npz holds `lat`, `lng`, `time` (hours from departure) and `battery`
(0..1), all the same length, plus the scalar `speed_kmh`. Any 1-D array of the
same length is treated as a plottable scalar by vis_map.py, so adding a channel
means adding a key.
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


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("npz", help="map artifact from map_gen.py --save")
    p.add_argument("-o", "--out", default="track.npz", help="where to write the track")
    p.add_argument("--start", type=float, nargs=2, default=(42.0, -10.5), metavar=("LAT", "LON"))
    p.add_argument("--goal", type=float, nargs=2, default=(31.5, -11.5), metavar=("LAT", "LON"))
    p.add_argument("--speed", type=float, default=9.0, metavar="KMH")
    p.add_argument("--dt", type=float, default=20.0, metavar="MIN")
    p.add_argument("--arrive", type=float, default=25.0, metavar="KM")
    p.add_argument("--max-hours", type=float, default=400.0, metavar="H")
    args = p.parse_args()

    nav = NavMap.load(args.npz)
    print(f"map: K={nav.K} km, W={nav.W} km, res {nav.res_min}..{nav.res_base}")
    track = run(
        nav,
        tuple(args.start),
        tuple(args.goal),
        speed_kmh=args.speed,
        dt_min=args.dt,
        arrive_km=args.arrive,
        max_hours=args.max_hours,
    )
    np.savez_compressed(args.out, **track)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
