"""Generate a synthetic ERA5-like GRIB file for testing the viewer.

Writes a regular lat/lon grid with an ERA5-ish mix of fields: 10 m wind
components, surface solar radiation downwards (accumulated J/m², as ERA5 stores
it), total cloud cover, and 2 m temperature.

Output lands in data/ (gitignored) as data/sample_<preset>.grib unless --out says
otherwise.

    uv run make_sample_grib.py --preset small     # -> data/sample_small.grib
    uv run make_sample_grib.py --preset medium
    uv run make_sample_grib.py --preset large
    uv run make_sample_grib.py --preset xlarge    # whole planet, ~870 MB
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import eccodes as ec
import numpy as np

DATA_DIR = Path(__file__).parent / "data"

# paramId -> synthetic field, given lon/lat grids (degrees), step index and
# absolute valid time. Fields are smooth in space and coherent in time so the
# animation looks like weather rather than noise.
PARAMS = {
    165: lambda lon, lat, t, when: (  # 10u [m/s] — travelling wave
        8 * np.sin(np.deg2rad(lon) * 3 + t / 8.0) * np.cos(np.deg2rad(lat) * 2)
    ),
    166: lambda lon, lat, t, when: (  # 10v [m/s]
        6 * np.cos(np.deg2rad(lat) * 3 - t / 11.0)
    ),
    169: lambda lon, lat, t, when: (  # ssrd [J/m² accumulated over 1 h]
        np.clip(np.sin((when.hour + lon / 15.0 - 6) / 12 * np.pi), 0, 1)
        * 3.6e6
        * np.clip(np.cos(np.deg2rad(lat)), 0.1, 1)
    ),
    164: lambda lon, lat, t, when: np.clip(  # tcc [0-1]
        0.5 + 0.45 * np.sin(np.deg2rad(lon) * 2 + np.deg2rad(lat) * 2 + t / 6.0), 0, 1
    ),
    167: lambda lon, lat, t, when: (  # 2t [K] — latitude gradient + diurnal cycle
        300 - 0.5 * (lat - 35) + 4 * np.sin((when.hour - 3) / 24 * 2 * np.pi)
    ),
}

# bbox is (lat_north, lat_south, lon_west, lon_east)
PRESETS = {
    # smoke test: English Channel, coarse, 4 steps
    "small": dict(bbox=(60.0, 50.0, -5.0, 10.0), res=0.5, steps=4, step_hours=6),
    # a week of 3-hourly over the UK + North Sea at 0.25° — everyday working size
    "medium": dict(bbox=(62.0, 48.0, -12.0, 10.0), res=0.25, steps=8 * 7, step_hours=3),
    # stress: a month hourly over Western Europe + N Atlantic at ERA5's native 0.25°
    "large": dict(bbox=(65.0, 35.0, -20.0, 20.0), res=0.25, steps=24 * 30, step_hours=1),
    # whole planet, 0.25°, 3-hourly for a week — the worst realistic case
    "xlarge": dict(bbox=(90.0, -90.0, -180.0, 179.75), res=0.25, steps=8 * 7, step_hours=3),
}


def write_grib(out, bbox, res, steps, step_hours, start: dt.datetime) -> None:
    lat0, lat1, lon0, lon1 = bbox
    nx = int(round((lon1 - lon0) / res)) + 1
    ny = int(round((lat0 - lat1) / res)) + 1

    lons = np.linspace(lon0, lon1, nx)
    lats = np.linspace(lat0, lat1, ny)  # north -> south, as ERA5 scans
    lon_g, lat_g = np.meshgrid(lons, lats)

    n_msg = steps * len(PARAMS)
    print(f"grid {ny}x{nx} ({nx * ny:,} points) x {steps} steps x {len(PARAMS)} fields "
          f"= {n_msg:,} messages, ~{n_msg * nx * ny * 4 / 1e9:.2f} GB raw")

    with open(out, "wb") as fh:
        for t in range(steps):
            when = start + dt.timedelta(hours=t * step_hours)
            for pid, gen in PARAMS.items():
                h = ec.codes_grib_new_from_samples("regular_ll_sfc_grib1")
                ec.codes_set(h, "paramId", pid)
                ec.codes_set(h, "Ni", nx)
                ec.codes_set(h, "Nj", ny)
                ec.codes_set(h, "latitudeOfFirstGridPointInDegrees", lat0)
                ec.codes_set(h, "latitudeOfLastGridPointInDegrees", lat1)
                ec.codes_set(h, "longitudeOfFirstGridPointInDegrees", lon0)
                ec.codes_set(h, "longitudeOfLastGridPointInDegrees", lon1)
                ec.codes_set(h, "iDirectionIncrementInDegrees", res)
                ec.codes_set(h, "jDirectionIncrementInDegrees", res)
                ec.codes_set(h, "jScansPositively", 0)
                ec.codes_set(h, "iScansNegatively", 0)
                ec.codes_set(h, "dataDate", int(when.strftime("%Y%m%d")))
                ec.codes_set(h, "hour", when.hour)
                ec.codes_set_values(h, gen(lon_g, lat_g, t, when).astype(float).ravel())
                ec.codes_write(h, fh)
                ec.codes_release(h)

            if steps > 24 and (t + 1) % 24 == 0:
                print(f"  {t + 1}/{steps} steps", end="\r", flush=True)

    print(f"\nwrote {out}: {Path(out).stat().st_size / 1e6:.1f} MB")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", help="output path (default: data/sample_<preset>.grib)")
    p.add_argument("--preset", choices=sorted(PRESETS), default="small")
    p.add_argument("--res", type=float, help="grid spacing in degrees (overrides preset)")
    p.add_argument("--steps", type=int, help="number of time steps (overrides preset)")
    p.add_argument("--step-hours", type=int, help="hours between steps (overrides preset)")
    p.add_argument("--bbox", type=float, nargs=4, metavar=("N", "S", "W", "E"),
                   help="domain bounds in degrees (overrides preset)")
    p.add_argument("--start", default="2024-07-01", help="first valid time, YYYY-MM-DD")
    args = p.parse_args()

    cfg = dict(PRESETS[args.preset])
    for key in ("res", "steps", "step_hours", "bbox"):
        if getattr(args, key) is not None:
            cfg[key] = tuple(args.bbox) if key == "bbox" else getattr(args, key)

    out = Path(args.out) if args.out else DATA_DIR / f"sample_{args.preset}.grib"
    out.parent.mkdir(parents=True, exist_ok=True)

    write_grib(str(out), start=dt.datetime.fromisoformat(args.start), **cfg)


if __name__ == "__main__":
    main()
