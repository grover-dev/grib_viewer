"""Download a CMEMS subset as a series of NetCDF4 blocks.

    uv run fetch_cmems.py ../raw_data/currents.nc
    uv run fetch_cmems.py ../raw_data/med.nc --bbox -6 30 36 46 --start 2025-03-01 --end 2025-03-31
    uv run fetch_cmems.py ../raw_data/waves.nc -d cmems_mod_glo_wav_my_0.2deg_PT3H-i -v VHM0 VMDR

Thin wrapper over `copernicusmarine.subset`, so each file is whatever the
service cuts server-side: the requested variables only, on the dataset's own
grid. Feed them to `netcdf4_npz.py` to get the npz the C++ sampler reads.

Defaults reproduce the global daily surface-current pull: `uo`/`vo` from
`cmems_mod_glo_phy_my_0.083deg_P1D-m` over January 2025 at the top depth level.

**Blocks.** The request is split along time into `--chunk-days` spans and,
optionally, along longitude/latitude into `--tile` degree tiles; each block is
one `subset` call writing one file. This is about memory, not politeness: the
toolbox materialises a request before it writes it, so a global multi-month pull
is a single allocation of tens of gigabytes, whereas the same data in 5-day
blocks peaks at one block. Blocks are named after their own span

    currents_20250101-20250105.nc
    currents_20250101-20250105_W180S-080.nc      (with --tile)

so an interrupted run resumes: an existing block is skipped unless `--overwrite`
is given, and only the missing ones are fetched. A single block keeps the plain
name you asked for.

Chunk on time first. Time is the axis the service indexes cheaply and the one
that keeps each file a whole grid, which is what the npz converter wants;
`--tile` exists for when one *frame* is already too large and should stay unset
otherwise.

Blocks are contiguous, not merely adjacent: one ends at 23:59:59 of its last day
and the next starts at 00:00:00 of the following one, so a sub-daily product
keeps every frame and no timestamp appears in two files. `--start`/`--end` given
as bare dates are whole days for the same reason.

**Depth.** The default 0.494 m is the first level of the GLORYS z-grid, not a
round number the service snaps to -- asking for `0` and getting the same layer
is luck, not contract. `--depth Z` selects the single level nearest Z by asking
for the degenerate range [Z, Z]; `--depth-range A B` keeps every level in the
span, which makes the file 3-D and multiplies its size by the level count.

**Credentials.** Run `copernicusmarine login` once (it writes
`~/.copernicusmarine/.copernicusmarine-credentials`), or export
`COPERNICUSMARINE_SERVICE_USERNAME` / `COPERNICUSMARINE_SERVICE_PASSWORD`.
Nothing here prompts, so a missing login fails the run rather than hanging it.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

DATASET = "cmems_mod_glo_phy_my_0.083deg_P1D-m"
VARIABLES = ("uo", "vo")

# Global GLORYS extent. The east edge is one cell short of 180 because the grid
# is 4320 cells of 1/12 degree starting at -180 -- asking for 180 exactly just
# wraps onto the first column.
BBOX = (-180.0, -80.0, 179.9166717529297, 90.0)

SURFACE_DEPTH = 0.49402499198913574

CHUNK_DAYS = 5

# Offset from midnight to the last second of the same day.
LAST_INSTANT = timedelta(hours=23, minutes=59, seconds=59)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download a CMEMS subset as a series of NetCDF blocks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("output", type=Path, help="path of the .nc file; blocks get a span suffix")
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

    p.add_argument(
        "--chunk-days",
        type=int,
        default=CHUNK_DAYS,
        help="days of time per block; 0 puts the whole span in one block",
    )
    p.add_argument(
        "--tile",
        nargs=2,
        type=float,
        metavar=("DLON", "DLAT"),
        help="also split each block into tiles this many degrees across",
    )

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
        "--overwrite", action="store_true", help="refetch blocks that are already on disk"
    )
    p.add_argument(
        "--dry-run", action="store_true", help="list the blocks and exit without downloading"
    )
    return p.parse_args(argv)


def parse_time(value: str, *, end: bool = False) -> datetime:
    """Accept a bare date or a full ISO timestamp.

    A bare date as `--end` means the whole of that day, not its first instant:
    on a 3-hourly product the latter would silently drop 21 hours of frames.
    """
    if len(value) != 10:
        return datetime.fromisoformat(value)
    return datetime.fromisoformat(value) + (LAST_INSTANT if end else timedelta())


def time_blocks(start: datetime, end: datetime, chunk_days: int) -> list[tuple[datetime, datetime]]:
    """Split [start, end] into closed spans of at most `chunk_days`.

    The service reads start/end inclusively, so a block ends at the last instant
    of its final day and the next begins at midnight the following one. That
    covers every frame of a sub-daily product without handing any timestamp to
    two blocks at once -- ending a block on the next one's start would fetch the
    boundary frame twice, and ending it at that day's midnight would keep only
    the 00:00 frame and lose the rest of the day.
    """
    if chunk_days <= 0 or start >= end:
        return [(start, end)]

    step = timedelta(days=chunk_days)
    blocks = []
    lo = start
    while lo <= end:
        last_day = (lo + step - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        hi = min(last_day + LAST_INSTANT, end)
        blocks.append((lo, hi))
        lo = (hi + timedelta(seconds=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return blocks


def space_blocks(
    west: float, south: float, east: float, north: float, tile: tuple[float, float] | None
) -> list[tuple[float, float, float, float]]:
    """Split the box into tiles, or return it whole when --tile is unset.

    Tiles are half-open on the east/north side, again so a shared edge column
    lands in exactly one file.
    """
    if tile is None:
        return [(west, south, east, north)]

    dlon, dlat = tile
    if dlon <= 0 or dlat <= 0:
        raise SystemExit("--tile needs positive degree spans")

    eps = 1e-6
    out = []
    y = south
    while y < north:
        y_hi = min(y + dlat, north)
        x = west
        while x < east:
            x_hi = min(x + dlon, east)
            out.append((x, y, x_hi - eps if x_hi < east else x_hi, y_hi - eps if y_hi < north else y_hi))
            x = x_hi
        y = y_hi
    return out


def block_name(
    stem: Path,
    span: tuple[datetime, datetime],
    box: tuple[float, float, float, float],
    n_time: int,
    n_space: int,
) -> Path:
    """`currents.nc` plus whichever suffixes actually distinguish the block."""
    parts = [stem.stem]
    if n_time > 1:
        lo, hi = span
        parts.append(f"{lo:%Y%m%d}-{hi:%Y%m%d}")
    if n_space > 1:
        west, south, _, _ = box
        parts.append(f"{'W' if west < 0 else 'E'}{abs(west):06.1f}"
                     f"{'S' if south < 0 else 'N'}{abs(south):05.1f}")
    return stem.with_name("_".join(parts) + ".nc")


def build_request(
    args: argparse.Namespace,
    span: tuple[datetime, datetime],
    box: tuple[float, float, float, float],
) -> dict:
    """The kwargs for one `copernicusmarine.subset` call."""
    west, south, east, north = box
    lo, hi = span
    req = {
        "dataset_id": args.dataset_id,
        "variables": list(args.variables),
        "minimum_longitude": west,
        "maximum_longitude": east,
        "minimum_latitude": south,
        "maximum_latitude": north,
        "start_datetime": lo.isoformat(),
        "end_datetime": hi.isoformat(),
    }

    if args.depth_range is not None:
        req["minimum_depth"], req["maximum_depth"] = sorted(args.depth_range)
    elif not args.all_depths:
        # A degenerate range: the service returns the one level nearest it.
        req["minimum_depth"] = req["maximum_depth"] = args.depth
    return req


def plan(args: argparse.Namespace) -> list[tuple[Path, dict]]:
    west, south, east, north = args.bbox
    if west > east:
        raise SystemExit(
            f"--bbox west ({west}) is east of east ({east}); the service cannot cut "
            "across the antimeridian, so pull the two sides separately"
        )
    if south > north:
        raise SystemExit(f"--bbox south ({south}) is north of north ({north})")

    start, end = parse_time(args.start), parse_time(args.end, end=True)
    if start > end:
        raise SystemExit(f"--start {args.start} is after --end {args.end}")

    spans = time_blocks(start, end, args.chunk_days)
    boxes = space_blocks(west, south, east, north, args.tile)

    stem = args.output if args.output.suffix == ".nc" else args.output.with_suffix(".nc")
    return [
        (block_name(stem, span, box, len(spans), len(boxes)), build_request(args, span, box))
        for span in spans
        for box in boxes
    ]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    blocks = plan(args)

    todo = [(path, req) for path, req in blocks if args.overwrite or not path.exists()]
    skipped = len(blocks) - len(todo)

    print(f"{len(blocks)} block(s), {skipped} already on disk", file=sys.stderr)
    for path, req in todo:
        print(
            f"  {path.name}: {req['start_datetime']}..{req['end_datetime']} "
            f"lon {req['minimum_longitude']}..{req['maximum_longitude']} "
            f"lat {req['minimum_latitude']}..{req['maximum_latitude']}",
            file=sys.stderr,
        )
    if args.dry_run or not todo:
        return 0

    try:
        import copernicusmarine
    except ImportError:
        raise SystemExit(
            "copernicusmarine is not installed: pip install copernicusmarine "
            "(or uv run --with copernicusmarine fetch_cmems.py ...)"
        )

    total = 0.0
    for i, (path, req) in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {path.name}", file=sys.stderr)
        path.parent.mkdir(parents=True, exist_ok=True)
        copernicusmarine.subset(
            output_directory=str(path.parent),
            output_filename=path.name,
            file_format="netcdf",
            overwrite=True,  # the exists() check above is the real gate
            **req,
        )
        if path.exists():
            total += path.stat().st_size / 1e6

    print(f"wrote {len(todo)} block(s), {total:.1f} MB total", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
