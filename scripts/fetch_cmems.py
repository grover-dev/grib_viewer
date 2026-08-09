"""Download a CMEMS subset to a single NetCDF4 file.

    uv run fetch_cmems.py out/currents.nc
    uv run fetch_cmems.py out/med.nc --bbox -6 36 30 46 --start 2025-01-01 --end 2025-01-31
    uv run fetch_cmems.py out/waves.nc -d cmems_mod_glo_wav_my_0.2deg_PT3H-i -v VHM0 VMDR

Thin wrapper over `copernicusmarine.subset`, so the output is whatever the
service cuts server-side: one file, the requested variables only, on the
dataset's own grid. Feed it to `netcdf4_npz.py` to get the npz the C++ sampler
reads.

Defaults reproduce the global daily surface-current pull: `uo`/`vo` from
`cmems_mod_glo_phy_my_0.083deg_P1D-m` over January 2025 at the top depth level.

**Depth.** The default 0.494 m is the first level of the GLORYS z-grid, not a
round number the service snaps to -- asking for `0` and getting the same layer
is luck, not contract. `--depth Z` selects the single level nearest Z by asking
for the degenerate range [Z, Z]; `--depth-range A B` keeps every level in the
span, which makes the file 3-D and is what you want only if something downstream
actually reads depth.

**Credentials.** Run `copernicusmarine login` once (it writes
`~/.copernicusmarine/.copernicusmarine-credentials`), or export
`COPERNICUSMARINE_SERVICE_USERNAME` / `COPERNICUSMARINE_SERVICE_PASSWORD`.
Nothing here prompts, so a missing login fails the run rather than hanging it.

**Size.** The global default is ~170 MB per month of two surface variables and
scales with the box, the variable count and the number of frames; a multi-year
global pull is tens of gigabytes and the service will be the slow part. Cut the
box first, the time span second.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DATASET = "cmems_mod_glo_phy_my_0.083deg_P1D-m"
VARIABLES = ("uo", "vo")

# Global GLORYS extent. The east edge is one cell short of 180 because the grid
# is 4320 cells of 1/12 degree starting at -180 -- asking for 180 exactly just
# wraps onto the first column.
BBOX = (-180.0, -80.0, 179.9166717529297, 90.0)

SURFACE_DEPTH = 0.49402499198913574


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download a CMEMS subset to one NetCDF file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("output", type=Path, help="path of the .nc file to write")
    p.add_argument("-d", "--dataset-id", default=DATASET)
    p.add_argument("-v", "--variables", nargs="+", default=list(VARIABLES))
    p.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("W", "S", "E", "N"),
        default=list(BBOX),
        help="lon/lat box, degrees, -180..180",
    )
    p.add_argument("--start", default="2025-01-01", help="ISO date or datetime, inclusive")
    p.add_argument("--end", default="2025-01-31", help="ISO date or datetime, inclusive")

    depth = p.add_mutually_exclusive_group()
    depth.add_argument(
        "--depth",
        type=float,
        default=SURFACE_DEPTH,
        help="single level nearest this depth, metres",
    )
    depth.add_argument(
        "--depth-range",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        help="keep every level in this span instead (3-D output)",
    )
    depth.add_argument(
        "--all-depths", action="store_true", help="keep every level the dataset has"
    )

    p.add_argument(
        "--overwrite", action="store_true", help="replace the output file if it exists"
    )
    p.add_argument(
        "--dry-run", action="store_true", help="print the request and exit without downloading"
    )
    return p.parse_args(argv)


def build_request(args: argparse.Namespace) -> dict:
    """The kwargs for `copernicusmarine.subset`, minus output paths."""
    west, south, east, north = args.bbox
    if west > east:
        raise SystemExit(
            f"--bbox west ({west}) is east of east ({east}); the service cannot cut "
            "across the antimeridian, so pull the two sides separately"
        )
    if south > north:
        raise SystemExit(f"--bbox south ({south}) is north of north ({north})")

    req = {
        "dataset_id": args.dataset_id,
        "variables": list(args.variables),
        "minimum_longitude": west,
        "maximum_longitude": east,
        "minimum_latitude": south,
        "maximum_latitude": north,
        "start_datetime": normalise_time(args.start),
        "end_datetime": normalise_time(args.end),
    }

    if args.depth_range is not None:
        req["minimum_depth"], req["maximum_depth"] = sorted(args.depth_range)
    elif not args.all_depths:
        # A degenerate range: the service returns the one level nearest it.
        req["minimum_depth"] = req["maximum_depth"] = args.depth
    return req


def normalise_time(value: str) -> str:
    """Accept a bare date; the API wants a full timestamp."""
    return f"{value}T00:00:00" if len(value) == 10 else value


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    req = build_request(args)

    out = args.output
    if out.suffix != ".nc":
        out = out.with_suffix(".nc")
    if out.exists() and not args.overwrite:
        raise SystemExit(f"{out} exists; pass --overwrite to replace it")

    for key, value in req.items():
        print(f"  {key} = {value!r}", file=sys.stderr)
    print(f"  output = {out}", file=sys.stderr)
    if args.dry_run:
        return 0

    try:
        import copernicusmarine
    except ImportError:
        raise SystemExit(
            "copernicusmarine is not installed: pip install copernicusmarine "
            "(or uv run --with copernicusmarine fetch_cmems.py ...)"
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    copernicusmarine.subset(
        output_directory=str(out.parent),
        output_filename=out.name,
        file_format="netcdf",
        overwrite=True,  # the exists() check above is the real gate
        **req,
    )

    size = out.stat().st_size / 1e6 if out.exists() else 0.0
    print(f"wrote {out} ({size:.1f} MB)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
