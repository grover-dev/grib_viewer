"""Regenerate the .npz fixtures the C++ tests read.

    uv run make_fixtures.py [path/to/data.grib]

They are real ERA5 ssrd, cut small enough to commit, because the point of the
tests is to validate this pipeline end to end -- a synthetic cube would only
check that the sampler agrees with itself.

The expected values in solar_test.cpp are derived from the fixtures at runtime
rather than hardcoded, so regenerating from a different source GRIB is fine as
long as each fixture keeps its *shape*: regional vs global, quantised vs float.
That is what the individual tests actually depend on, and it is asserted below.

Driven through solar_npz.py's command line rather than by importing it, so what
the fixtures exercise is exactly the interface the docs tell you to use.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
GENERATOR = REPO / "scripts" / "solar_npz.py"
DEFAULT_GRIB = REPO / "data" / "18fdfe51416fefbcd1cd10d5e52abfe5" / "data.grib"

# North Atlantic, 6 hours spanning local dawn -- so the window holds both night
# and daylight and the time interpolation has something to interpolate.
REGION = ["--frames", "6:12", "--bbox", "-30", "-10", "30", "50"]

FIXTURES = [
    # Regional and quantised. Non-wrapping, so it exercises out-of-range
    # rejection at all four edges.
    ("region_u16.npz", REGION),
    # The same window as float32, so the tests can prove the quantised path
    # agrees with the unquantised one instead of only being self-consistent.
    ("region_f32.npz", REGION + ["--dtype", "f32"]),
    # Global, thinned to 2 degrees: 180 columns x 2 deg == 360, so the longitude
    # axis wraps and the antimeridian seam becomes testable.
    ("global_u16.npz", ["--frames", "0:3", "--thin", "8"]),
    # A full 24 hours over one small patch of ocean at 40N 20W. Tiny in area but
    # complete in time, so a test can find where daylight peaks and check it
    # against local noon -- which is what proves the half-step centring.
    ("diurnal_u16.npz", ["--frames", "0:24", "--bbox", "-21", "-19", "39", "41"]),
]


def expectations(name: str, npz) -> list[tuple[str, bool]]:
    """What solar_test.cpp assumes about each fixture, checked here so a bad
    regeneration fails loudly rather than as a puzzling C++ assertion."""
    wraps = bool(npz["lon_wrap"])
    dtype = npz["data"].dtype
    nt, nlat, nlon = npz["data"].shape
    common = [
        ("at least 3 frames, to interpolate in time", nt >= 3),
        ("at least 2 points per spatial axis", nlat >= 2 and nlon >= 2),
        ("some daylight in the window", float(npz["data"].max()) > 0),
    ]
    if name.startswith("diurnal"):
        return common + [
            ("a full day of frames", nt >= 24),
            ("spans local noon at ~20W", True),
        ]
    if name.startswith("global"):
        return common + [
            ("longitude wraps", wraps),
            ("covers both poles", abs(float(npz["lat"][0])) == 90.0
                                  and abs(float(npz["lat"][-1])) == 90.0),
        ]
    return common + [
        ("longitude does not wrap", not wraps),
        ("quantised" if "u16" in name else "unquantised",
         dtype == np.uint16 if "u16" in name else dtype == np.float32),
    ]


def main() -> None:
    grib = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_GRIB
    if not grib.exists():
        raise SystemExit(f"no such GRIB: {grib}")

    for name, options in FIXTURES:
        out = HERE / name
        print(f"\n=== {name} " + "=" * (60 - len(name)))
        subprocess.run(
            [sys.executable, str(GENERATOR), str(grib), str(out), *options],
            cwd=GENERATOR.parent,  # solar_npz.py imports grib_utils as a sibling
            check=True,
        )

    print("\nverifying what the C++ tests rely on:")
    failed = False
    for name, _ in FIXTURES:
        path = HERE / name
        with np.load(path) as npz:
            checks = expectations(name, npz)
        print(f"  {name}  ({path.stat().st_size / 1024:.0f} KB)")
        for what, ok in checks:
            print(f"    {'ok  ' if ok else 'FAIL'}  {what}")
            failed |= not ok

    # The region pair must describe the same window, or the quantisation-error
    # test is comparing two different things and would pass for the wrong reason.
    with np.load(HERE / "region_u16.npz") as a, np.load(HERE / "region_f32.npz") as b:
        for axis in ("time", "lat", "lon"):
            same = np.array_equal(a[axis], b[axis])
            print(f"    {'ok  ' if same else 'FAIL'}  region pair shares its {axis} axis")
            failed |= not same

    if failed:
        raise SystemExit("\nfixtures do not satisfy what the tests assume")
    print("\nfixtures ready")


if __name__ == "__main__":
    main()
