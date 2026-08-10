"""Extract one NetCDF4 field into an .npz the C++ side can sample in O(1).

    uv run netcdf4_npz.py data/cmems.nc --list
    uv run netcdf4_npz.py data/cmems.nc uo current.npz --at depth=0
    uv run netcdf4_npz.py data/cmems.nc usi ice.npz --bbox -40 5 20 60
    uv run netcdf4_npz.py raw_data/ uo current.npz          # every .nc in there
    uv run netcdf4_npz.py raw_data/currents_2025*.nc vo v.npz

Several sources are joined along time, which is how a product that is downloaded
a month at a time -- `raw_data/currents_20250201-20250303.nc` and its siblings --
becomes one field. The files are ordered by their own first stamp rather than by
name or by the order given, timestamps present in more than one are kept once
(download windows usually share an endpoint), and the joined axis then has to be
uniform like any other: a gap between two files is a hard error naming them,
because `t0 + i*dt` cannot express a hole. Everything else -- the grid, the
dimension layout, the units -- must match across the files, and the packing
attributes need not: each file is unpacked with its own `scale_factor` /
`add_offset` / `_FillValue`, which for a product packed per delivery is the
difference between correct values and quietly wrong ones.

Same output as `grib_npz.py`, same reader on the other side -- see that file for
what the format promises and `npz_out.py` for the members that promise it. This
is only a different way in: HDF5 datasets instead of GRIB messages.

**Read through h5py, not netCDF4 or xarray.** NetCDF4 *is* HDF5 -- a variable is
a dataset, a dimension is an HDF5 dimension scale, and everything else is
attributes. Reading it directly means one dependency (h5py) instead of a C
netCDF library plus its bindings, and it means a batch of frames is one strided
`dset[a:b, ..., ys, xs]` that HDF5 satisfies from the file, with no lazy graph
to build and no scheduler to pay. The cost is that CF decoding is this script's
job rather than a library's, so it is done explicitly and in one place:

* **Packing.** `scale_factor` / `add_offset` are applied on read, and
  `_FillValue` / `missing_value` are matched against the *raw* values before
  unpacking, which is what CF specifies and the only point where the comparison
  is exact. `valid_min` / `valid_max` are deliberately ignored -- they are
  advisory, frequently stale, and masking on them silently deletes real data.

* **Time.** `units` is a CF "<unit> since <reference>" string, which is parsed
  here into epoch seconds. Only the real-world calendars are accepted; a
  `360_day` or `noleap` model calendar has no fixed-length step and cannot be
  addressed by `t0 + i*dt` at all, so it is refused rather than approximated.

* **Axes.** NetCDF usually stores lat/lon as float32, and a 1/12 degree step is
  not representable in it: on the CMEMS global grid consecutive differences
  range over 0.083328..0.083336 from rounding alone. That spread is an artefact,
  not a non-uniform grid, so the step is taken from the endpoints -- where the
  error is divided by n-1 -- and the axis is checked by fitting rather than by
  differencing. See `axis_step`.

Fields with dimensions beyond (time, lat, lon) -- depth, a member number -- are
reduced to one index with `--at DIM=INDEX`, defaulting to 0, because the npz
holds one cube and not a hypercube.

Longitude convention needs no handling here: the sampler resolves it modulo 360
(npz_field.cpp), so a 0..360 file and a -180..180 file address identically and
neither is rolled. `--bbox` is likewise given in -180..180 whatever the file
uses, and is refused only when the box would cross the file's own seam and make
the stored axis non-monotonic.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from npz_out import (
    Grid,
    add_output_args,
    add_selection_args,
    describe,
    fmt_time,
    parse_frames,
    plan_layout,
    uniform_step,
    write_field,
)

Y_NAMES = ("latitude", "lat", "y", "nav_lat")
X_NAMES = ("longitude", "lon", "x", "nav_lon")
T_NAMES = ("time", "valid_time", "t")

# Units that mean "accumulated over the step"; CF files usually also say so in
# cell_methods, which is checked first. See --accumulated for the fields that
# accumulate without advertising it either way.
ACCUMULATED_UNITS = ("j m**-2", "j/m2", "j m-2", "j m^-2")

# What an accumulated field becomes once divided by its step length.
RATE_UNITS = {"j m**-2": "W m**-2", "j/m2": "W m**-2",
              "j m-2": "W m**-2", "j m^-2": "W m**-2"}

# CF time units, in seconds. Months and years are not here on purpose: they are
# not fixed-length, so they cannot produce the constant dt the format needs.
TIME_UNITS = {
    "second": 1.0, "seconds": 1.0, "sec": 1.0, "secs": 1.0, "s": 1.0,
    "minute": 60.0, "minutes": 60.0, "min": 60.0, "mins": 60.0,
    "hour": 3600.0, "hours": 3600.0, "hr": 3600.0, "hrs": 3600.0, "h": 3600.0,
    "day": 86400.0, "days": 86400.0, "d": 86400.0,
}

# Calendars that agree with numpy's proleptic Gregorian for any date a forecast
# or reanalysis product carries. The model calendars (360_day, noleap, ...) do
# not, and are rejected where this is used.
REAL_CALENDARS = ("", "standard", "gregorian", "proleptic_gregorian")


def text(obj, name: str) -> str:
    """An HDF5 string attribute, whichever of the three ways it was stored."""
    v = obj.attrs.get(name)
    if v is None:
        return ""
    if isinstance(v, np.ndarray):
        v = v.flat[0] if v.size else ""
    if isinstance(v, (bytes, np.bytes_)):
        v = v.decode("utf-8", "replace")
    return str(v).strip()


def number(obj, name: str, dtype=None):
    """A numeric attribute as a 0-d array of `dtype` (its own, if not given).

    Kept as an array rather than a float because `_FillValue` is compared
    against the stored values exactly, and going through python float would
    round a float32 sentinel into something that matches nothing.
    """
    v = obj.attrs.get(name)
    if v is None:
        return None
    v = np.asarray(v).flat[0]
    return v if dtype is None else np.asarray(v, dtype=dtype)


def basename(path: str) -> str:
    return path.rsplit("/", 1)[-1]


def dim_names(f: h5py.File, dset: h5py.Dataset) -> list[str]:
    """The dimension name of each axis of `dset`.

    NetCDF4 records this as HDF5 dimension scales: `DIMENSION_LIST[i]` holds
    references to the scale datasets attached to axis i, and a scale's name is
    the dimension's name. Dereferencing them recovers the whole layout without a
    netCDF library. Files written by tools that skip the scales fall back to
    matching an axis's length against the 1-D variables, which is ambiguous only
    when two coordinates happen to be the same length -- and then the axis order
    of the file decides, as it would anyway.
    """
    names: list[str] = []
    refs = dset.attrs.get("DIMENSION_LIST")
    for axis in range(dset.ndim):
        name = ""
        if refs is not None and axis < len(refs) and len(refs[axis]):
            name = basename(f[refs[axis][0]].name)
        names.append(name)

    if all(names):
        return names

    # Fallback: the 1-D datasets are the candidate coordinates, matched by length.
    coords = {d.shape[0]: basename(d.name)
              for d in f.values() if isinstance(d, h5py.Dataset) and d.ndim == 1}
    return [n or coords.get(dset.shape[axis], f"dim{axis}")
            for axis, n in enumerate(names)]


def axis_of(names: list[str], candidates: tuple[str, ...]) -> int | None:
    """Which axis carries a dimension named like one of `candidates`."""
    lowered = [n.lower() for n in names]
    for want in candidates:
        if want in lowered:
            return lowered.index(want)
    return None


def parse_reference(when: str) -> np.datetime64:
    """The '<reference>' half of a CF time unit, as a UTC second.

    Written every way a writer can imagine -- '1970-01-01 00:00:00',
    '1950-01-01T00:00:00Z', '2000-1-1 0:0:0', some with an explicit offset --
    so it is taken apart by hand rather than handed to np.datetime64, which
    accepts only the strict ISO spelling.
    """
    s = re.sub(r"\s*(UTC|GMT)\s*$", "", when.strip(), flags=re.IGNORECASE).rstrip("Zz").strip()

    # An explicit offset is applied, not dropped: 'hours since 2020-01-01 00:00 +01:00'
    # starts an hour before the same stamp in UTC.
    offset_s = 0
    m = re.search(r"([+-])(\d{1,2}):?(\d{2})$", s)
    if m:
        sign = -1 if m.group(1) == "+" else 1
        offset_s = sign * (int(m.group(2)) * 3600 + int(m.group(3)) * 60)
        s = s[:m.start()].strip()

    date, _, clock = s.replace("T", " ").partition(" ")
    try:
        y, mo, d = (int(v) for v in date.split("-"))
        parts = [float(v) for v in clock.split(":")] if clock.strip() else []
    except ValueError:
        raise SystemExit(f"cannot read the reference time in {when!r}")
    parts += [0.0] * (3 - len(parts))

    day = np.datetime64(f"{y:04d}-{mo:02d}-{d:02d}", "s")
    seconds = int(round(parts[0] * 3600 + parts[1] * 60 + parts[2])) + offset_s
    return day + np.timedelta64(seconds, "s")


def decode_time(values: np.ndarray, units: str, calendar: str) -> np.ndarray:
    """A CF time coordinate as int64 epoch seconds."""
    m = re.match(r"\s*(\w+)\s+since\s+(.+)$", units, flags=re.IGNORECASE)
    if not m:
        raise SystemExit(
            f"time units {units!r} are not a CF '<unit> since <reference>' string; "
            "there is no way to place these values on a real time axis"
        )
    unit, when = m.group(1).lower(), m.group(2)
    if unit not in TIME_UNITS:
        raise SystemExit(
            f"time unit {unit!r} is not a fixed length, so the axis cannot have a "
            "constant step; the npz format addresses time as t0 + i*dt"
        )
    if calendar.lower() not in REAL_CALENDARS:
        raise SystemExit(
            f"calendar {calendar!r} is a model calendar, not a real one; its steps do "
            "not map onto wall-clock time and the npz format cannot represent it"
        )
    epoch = int(parse_reference(when).astype("int64"))
    return epoch + np.rint(values.astype("float64") * TIME_UNITS[unit]).astype("int64")


def axis_step(values: np.ndarray, what: str) -> tuple[np.ndarray, float]:
    """The step of a stored coordinate, plus the axis rebuilt exactly from it.

    Differencing the stored values is the wrong test when they are float32: a
    1/12 degree step is not representable, so neighbouring differences disagree
    in the last bits even though the grid is perfectly regular. The step is
    therefore taken from the endpoints, where that rounding is divided by n-1,
    and uniformity is checked by *fitting* -- every stored value must sit within
    a rounding of the axis the step generates. A genuinely irregular axis (a
    stretched vertical grid, two concatenated regions) fails that by orders of
    magnitude, so nothing is being waved through.

    The rebuilt axis is what gets written out. It is the one the sampler's
    `lat0 + j*dlat` actually walks, so storing it instead of the source values
    makes the `lat`/`lon` members agree with the arithmetic to the last bit.
    """
    if values.size < 2:
        raise SystemExit(f"{what} axis has {values.size} point(s); need at least 2")
    v = values.astype("float64")
    step = (v[-1] - v[0]) / (v.size - 1)
    if step == 0.0:
        raise SystemExit(f"{what} axis does not advance: every value is {v[0]:g}")
    rebuilt = v[0] + step * np.arange(v.size)

    eps = float(np.finfo(values.dtype).eps) if values.dtype.kind == "f" else 0.0
    tol = max(8.0 * eps * max(abs(v[0]), abs(v[-1])), 1e-9)
    err = float(np.abs(v - rebuilt).max())
    if err > tol:
        raise SystemExit(
            f"{what} axis is not uniform (worst point is {err:g} off a constant step "
            f"of {step:g}, tolerance {tol:g}); the npz format addresses it "
            "arithmetically and cannot represent that"
        )
    return rebuilt, step


def contiguous_run(keep: np.ndarray, what: str, detail: str) -> slice:
    """The kept indices as one slice, or a hard error if they are in two pieces."""
    idx = np.flatnonzero(keep)
    if idx.size == 0:
        raise SystemExit(f"no {what} points inside the requested range")
    if idx.size != idx[-1] - idx[0] + 1:
        raise SystemExit(f"the requested {what} range is split in this file: {detail}")
    return slice(int(idx[0]), int(idx[-1]) + 1)


def lon_window(lon: np.ndarray, west: float, east: float) -> slice:
    """Columns inside [west, east], whatever longitude convention the file uses.

    Both edges and every column are measured as degrees *east of west*, modulo
    360, which is the same trick the sampler uses: it makes a -180..180 box and
    a 0..360 file agree without touching either. A box that straddles the point
    where the file's own axis restarts selects two runs and cannot be stored as
    one uniform axis, so it is refused by `contiguous_run` rather than silently
    reordered.
    """
    width = (east - west) % 360.0 or 360.0
    return contiguous_run(
        ((lon.astype("float64") - west) % 360.0) <= width,
        "longitude",
        f"its axis runs {lon.min():g}..{lon.max():g} and the box crosses that seam. "
        "This tool does not cross the seam; slice_grib.py does."
    )


def find_fields(f: h5py.File) -> dict[str, tuple[h5py.Dataset, list[str]]]:
    """Every dataset that is a (time, ..., lat, lon) field, with its dim names."""
    found: dict[str, tuple[h5py.Dataset, list[str]]] = {}

    def visit(_name, obj):
        if not isinstance(obj, h5py.Dataset) or obj.ndim < 3:
            return
        if obj.attrs.get("CLASS", b"") == b"DIMENSION_SCALE":
            return
        names = dim_names(f, obj)
        if all(axis_of(names, c) is not None for c in (T_NAMES, Y_NAMES, X_NAMES)):
            found[basename(obj.name)] = (obj, names)

    f.visititems(visit)
    return found


def list_fields(path: Path, f: h5py.File, fields: dict) -> None:
    """Every field in the file, with what it would cost to extract."""
    print(f"\n{path}")
    title = text(f, "title")
    if title:
        print(f"  {title}")
    print(f"\n  {'name':<10} {'long name':<40} {'units':>10}  {'grid':>11} "
          f"{'frames':>7}  dims")
    for name, (dset, names) in sorted(fields.items()):
        t, y, x = (axis_of(names, c) for c in (T_NAMES, Y_NAMES, X_NAMES))
        units = (text(dset, "units") or "-")[:10]
        long_name = (text(dset, "long_name") or text(dset, "standard_name") or "?")[:40]
        mark = "*" if is_accumulated(dset, units) else " "
        print(f" {mark}{name:<10} {long_name:<40} {units:>10}  "
              f"{dset.shape[y]:>5}x{dset.shape[x]:<5} {dset.shape[t]:>7}  "
              f"({', '.join(names)})")
    print("\n  * accumulated over the step: converted to a rate and re-centred in time")
    print("  extra dims (depth, ...) are reduced with --at DIM=INDEX\n")


def is_accumulated(dset: h5py.Dataset, units: str) -> bool:
    """Whether the field is a sum over the step rather than an instant value.

    CF says so in cell_methods ('time: sum'), which is checked first because it
    is the statement of intent; the unit heuristic is the fallback for files
    that carry ERA5-style accumulations without a cell_methods entry.
    """
    methods = text(dset, "cell_methods").lower()
    if re.search(r"time\s*:\s*(sum|total)", methods):
        return True
    return units.strip().lower() in ACCUMULATED_UNITS


# What a directory argument picks up. NetCDF4 has no single blessed extension,
# and these are the ones products in the wild actually ship with.
NC_SUFFIXES = (".nc", ".nc4", ".netcdf", ".cdf")


def collect_paths(given: list[str]) -> list[Path]:
    """The files behind the arguments: a directory expands to the NetCDF in it.

    Names are only a sorting key here, and a weak one -- the real order comes
    from the time axes once the files are open. Sorting the expansion of a
    directory anyway keeps the "which file is which" reporting stable between
    runs, and a repeat (a shell glob overlapping an explicit name) is dropped
    rather than joined to itself.
    """
    found: list[Path] = []
    for name in given:
        path = Path(name)
        if path.is_dir():
            inside = sorted(c for c in path.iterdir()
                            if c.is_file() and c.suffix.lower() in NC_SUFFIXES)
            if not inside:
                raise SystemExit(
                    f"no NetCDF files in {path}/ (looking for "
                    f"{', '.join(NC_SUFFIXES)})"
                )
            found.extend(inside)
        elif path.exists():
            found.append(path)
        else:
            raise SystemExit(f"no such file or directory: {path}")

    seen, unique = set(), []
    for path in found:
        key = path.resolve()
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def open_hdf5(path: Path) -> h5py.File:
    try:
        return h5py.File(path, "r")
    except OSError as exc:
        raise SystemExit(
            f"{path} is not readable as HDF5 ({exc}). NetCDF *3* files are a different "
            "format entirely and this tool cannot open them; convert with "
            "`nccopy -k nc4 in.nc out.nc` first."
        )


@dataclass
class Source:
    """One open file's share of the field, decoded far enough to be joined.

    The packing attributes live here rather than at the top of `main` because
    they are per file: a product delivered in monthly parts is packed against
    each part's own range, so unpacking every file with the first one's
    `scale_factor` would be off by whatever the ranges differ by.
    """
    path: Path
    f: h5py.File
    dset: h5py.Dataset
    names: list[str]
    times: np.ndarray            # int64 epoch seconds, this file's own axis
    scale_factor: np.ndarray | None
    add_offset: np.ndarray | None
    fill_raw: np.ndarray | None
    missing_raw: np.ndarray | None

    def decode(self, raw: np.ndarray) -> np.ndarray:
        """Raw values as float32, sentinels turned into NaN, unpacked."""
        cube = raw.astype("float32")
        # Sentinels are matched against the raw values, before unpacking: that is
        # what CF specifies, and it is the only point where the comparison is
        # exact. A NaN sentinel needs no test -- it is already NaN in the cast.
        for sentinel in (self.fill_raw, self.missing_raw):
            if sentinel is not None and not (sentinel.dtype.kind == "f"
                                             and np.isnan(sentinel)):
                cube[raw == sentinel] = np.nan
        if self.scale_factor is not None:
            cube *= np.float32(self.scale_factor)
        if self.add_offset is not None:
            cube += np.float32(self.add_offset)
        return cube


def open_source(path: Path, var: str) -> Source:
    """One file, with `var` found in it and its time axis decoded."""
    f = open_hdf5(path)
    fields = find_fields(f)
    if not fields:
        raise SystemExit(f"no (time, lat, lon) field found in {path}")
    if var not in fields:
        raise SystemExit(
            f"{var!r} is not in {path}.\n"
            f"  available: {', '.join(sorted(fields))}"
        )
    dset, names = fields[var]

    t_axis = axis_of(names, T_NAMES)
    for dim in (names[t_axis], names[axis_of(names, Y_NAMES)],
                names[axis_of(names, X_NAMES)]):
        if dim not in f:
            raise SystemExit(
                f"dimension {dim!r} has no coordinate variable in {path}, so its "
                "axis values are unknown and the grid cannot be described"
            )

    tvar = f[names[t_axis]]
    return Source(
        path=path, f=f, dset=dset, names=names,
        times=decode_time(tvar[:], text(tvar, "units"), text(tvar, "calendar")),
        scale_factor=number(dset, "scale_factor"),
        add_offset=number(dset, "add_offset"),
        fill_raw=number(dset, "_FillValue", dtype=dset.dtype),
        missing_raw=number(dset, "missing_value", dtype=dset.dtype),
    )


def check_compatible(sources: list[Source], var: str) -> None:
    """That the files describe the same field on the same grid.

    Only the properties the joined cube actually depends on are compared. The
    packing is deliberately not among them -- see `Source.decode` -- and neither
    is anything cosmetic, because refusing a file over a reworded `long_name`
    would help nobody.
    """
    head = sources[0]
    t_axis = axis_of(head.names, T_NAMES)
    shape = tuple(n for a, n in enumerate(head.dset.shape) if a != t_axis)
    units = text(head.dset, "units")
    lat = head.f[head.names[axis_of(head.names, Y_NAMES)]][:]
    lon = head.f[head.names[axis_of(head.names, X_NAMES)]][:]

    for s in sources[1:]:
        where = f"{s.path} disagrees with {head.path}"
        if s.names != head.names:
            raise SystemExit(f"{where}: dims are ({', '.join(s.names)}), not "
                             f"({', '.join(head.names)})")
        other = tuple(n for a, n in enumerate(s.dset.shape) if a != t_axis)
        if other != shape:
            raise SystemExit(f"{where}: {var} is {other} outside time, not {shape}")
        if text(s.dset, "units") != units:
            raise SystemExit(f"{where}: units are {text(s.dset, 'units')!r}, "
                             f"not {units!r}; these are not the same quantity")
        for axis, mine, theirs in (("latitude", lat, s.f[s.names[axis_of(s.names, Y_NAMES)]][:]),
                                   ("longitude", lon, s.f[s.names[axis_of(s.names, X_NAMES)]][:])):
            if not np.array_equal(mine, theirs):
                raise SystemExit(
                    f"{where}: the {axis} axis is a different grid "
                    f"({theirs.min():g}..{theirs.max():g} against "
                    f"{mine.min():g}..{mine.max():g}); this tool joins along time "
                    "only and does not regrid"
                )


def order_sources(sources: list[Source]) -> list[Source]:
    """Time order, which is the only order the join is defined in.

    Not the order given and not the filenames: a name is a convention, and
    `currents_20250201-20250303.nc` sorts as intended only until someone drops a
    file named differently into the same directory.
    """
    return sorted(sources, key=lambda s: (int(s.times[0]), str(s.path)))


def join_time(sources: list[Source]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The joined time axis, and where each of its frames comes from.

    Returns `(times, src_of, row_of)`: frame i of the joined selection is row
    `row_of[i]` of `sources[src_of[i]]`. `sources` must already be in time order
    (`order_sources`); a stamp already covered by an earlier file is dropped --
    consecutive downloads normally share an endpoint, and keeping both would put
    a zero-length step in an axis that has to be uniform.
    """
    times, src_of, row_of = [], [], []
    for i, s in enumerate(sources):
        if uniform_step(s.times, f"time in {s.path}", tol=0.0) <= 0:
            raise SystemExit(f"the time axis in {s.path} does not increase")
        rows = np.flatnonzero(s.times > times[-1][-1]) if times \
            else np.arange(s.times.size)
        if rows.size == 0:
            print(f"note: {s.path.name} adds nothing, "
                  "every stamp in it is already covered by an earlier file")
            continue
        times.append(s.times[rows])
        src_of.append(np.full(rows.size, i, dtype="int64"))
        row_of.append(rows.astype("int64"))
    return (np.concatenate(times), np.concatenate(src_of), np.concatenate(row_of))


def check_joined(times: np.ndarray, src_of: np.ndarray, sources: list[Source]) -> float:
    """The step of the joined axis, or an error pointing at the file boundary.

    `uniform_step` would catch a gap on its own, but it can only say that the
    steps disagree. With several files the useful half of the answer is *which*
    ones, since the fix is usually a missing download rather than anything about
    the data.
    """
    if len(sources) == 1:
        return uniform_step(times, "time", tol=0.0)
    steps = np.diff(times)
    if steps.min() != steps.max():
        bad = int(np.argmax(np.abs(steps - np.median(steps))))
        left, right = sources[int(src_of[bad])], sources[int(src_of[bad + 1])]
        gap = f"{fmt_time(np.datetime64(int(times[bad]), 's'))} .. " \
              f"{fmt_time(np.datetime64(int(times[bad + 1]), 's'))}"
        raise SystemExit(
            f"the joined time axis is not uniform: steps span "
            f"{steps.min() / 3600:g}..{steps.max() / 3600:g} h, the worst at {gap} "
            f"({left.path.name} -> {right.path.name}). The npz format addresses time "
            "as t0 + i*dt, so a hole cannot be represented -- fetch the missing "
            "span, or extract each run separately with --start/--end"
        )
    return float(steps[0])


def same_source_runs(src: np.ndarray):
    """[a, b) ranges over which `src` does not change."""
    if src.size == 0:
        return
    edges = [0, *(np.flatnonzero(np.diff(src)) + 1).tolist(), src.size]
    for a, b in zip(edges, edges[1:]):
        yield a, b


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # One list, split by hand below. argparse cannot do "many sources then two
    # more things": a greedy nargs='+' followed by optional positionals simply
    # eats them, and making them required would break --list, which takes
    # sources alone.
    p.add_argument("rest", nargs="+", metavar="NC [NC ...] VAR NPZ",
                   help="source NetCDF4 files, or directories of them, then the "
                        "field to extract and the output .npz. Several sources "
                        "are joined along time, ordered by their own first stamp. "
                        "With --list, the sources alone")
    p.add_argument("--list", action="store_true",
                   help="show every field in the first file and exit")

    sel = add_selection_args(p)
    sel.add_argument("--at", action="append", default=[], metavar="DIM=INDEX",
                     help="index to take on a dimension that is not time/lat/lon, "
                          "such as --at depth=0 (the default for every such dim). "
                          "Repeat for several")
    add_output_args(
        p,
        accumulated_help="treat the field as accumulated over the step. 'auto' (default) "
                         "decides from cell_methods and the units; 'yes' for "
                         "accumulations that advertise it in neither, such as total "
                         "precipitation",
        batch_help="how much of the time axis to read at once (default 128). "
                   "This is the peak read buffer; HDF5 satisfies one strided "
                   "read per batch, so a bigger batch is fewer, longer seeks",
    )
    args = p.parse_args()

    if args.frames and (args.start or args.end):
        raise SystemExit("choose either --frames or --start/--end, not both")

    if args.list:
        args.var = args.npz = None
        paths = collect_paths(args.rest)
    elif len(args.rest) < 3:
        raise SystemExit(
            "give one or more sources, then a field and an output path:\n"
            "  netcdf4_npz.py raw_data/ uo current.npz --at depth=0\n"
            "or --list with the sources alone to see what is available"
        )
    else:
        *given, args.var, args.npz = args.rest
        paths = collect_paths(given)

    if args.list:
        # The fields of the first file. The rest are only checked against it once
        # a field has been named, and listing them all would repeat one table.
        head = open_hdf5(paths[0])
        fields = find_fields(head)
        if not fields:
            raise SystemExit(f"no (time, lat, lon) field found in {paths[0]}")
        list_fields(paths[0], head, fields)
        if len(paths) > 1:
            print(f"  {len(paths)} files given; the other {len(paths) - 1} are "
                  "checked against this one and joined along time\n")
        return

    sources = order_sources([open_source(path, args.var) for path in paths])
    check_compatible(sources, args.var)
    head = sources[0]
    f, dset, names = head.f, head.dset, head.names
    t_axis = axis_of(names, T_NAMES)
    y_axis = axis_of(names, Y_NAMES)
    x_axis = axis_of(names, X_NAMES)
    units = text(dset, "units")

    # Whether the two accumulation conversions apply. Deciding once, here, keeps
    # the unit change and the time shift from drifting apart -- they describe the
    # same property of the field and must agree.
    accumulated = is_accumulated(dset, units) if args.accumulated == "auto" \
        else args.accumulated == "yes"

    # --- the dims that are neither time nor grid --------------------------
    # The npz holds one cube, so anything else has to collapse to a single index
    # here rather than downstream.
    extra: dict[int, int] = {}
    wanted = {}
    for spec in args.at:
        dim, _, value = spec.partition("=")
        if not value.strip().lstrip("-").isdigit():
            raise SystemExit(f"--at wants DIM=INDEX, got {spec!r}")
        wanted[dim.strip().lower()] = int(value)
    known = {names[t_axis], names[y_axis], names[x_axis]}
    picked: list[str] = []
    for axis, dim in enumerate(names):
        if dim in known:
            continue
        index = wanted.pop(dim.lower(), 0)
        if not -dset.shape[axis] <= index < dset.shape[axis]:
            raise SystemExit(f"--at {dim}={index} is outside 0..{dset.shape[axis] - 1}")
        extra[axis] = index % dset.shape[axis]
        label = f"{dim}[{extra[axis]}]"
        if dim in f and f[dim].ndim == 1:
            label += f" = {float(f[dim][extra[axis]]):g} {text(f[dim], 'units')}".rstrip()
        picked.append(label)
    if wanted:
        raise SystemExit(
            f"--at names {', '.join(sorted(wanted))}, which {args.var} does not have; "
            f"its dims are ({', '.join(names)})"
        )

    # --- time selection ---------------------------------------------------
    # One axis across every file, with the frame -> (file, row) map that the read
    # below walks. Both are over the *joined* source, before any selection.
    all_times, src_of, row_of = join_time(sources)
    # Accumulation length comes from the *source* spacing, before any striding:
    # each frame still covers one model step no matter how many we keep.
    step_s = check_joined(all_times, src_of, sources)

    if args.frames:
        lo, hi = parse_frames(args.frames, all_times.size)
        how = f"frames {lo}:{hi}"
    else:
        start = (np.datetime64(args.start, "s").astype("int64") if args.start
                 else all_times[0])
        end = (np.datetime64(args.end, "s").astype("int64") if args.end
               else all_times[-1])
        if end < start:
            raise SystemExit(
                f"end {fmt_time(np.datetime64(int(end), 's'))} is before start "
                f"{fmt_time(np.datetime64(int(start), 's'))}"
            )
        keep = (all_times >= start) & (all_times <= end)
        span = contiguous_run(keep, "time", "the axis is not sorted")
        lo, hi = span.start, span.stop
        how = (f"{fmt_time(np.datetime64(int(start), 's'))} .. "
               f"{fmt_time(np.datetime64(int(end), 's'))}")
    if args.stride < 1:
        raise SystemExit(f"--stride must be >= 1, got {args.stride}")
    t_sel = slice(lo, hi, args.stride)
    times = all_times[t_sel]
    if times.size < 2:
        raise SystemExit(f"{times.size} time step(s) matched {how}; need at least 2")

    # --- area selection ---------------------------------------------------
    lat_all = f[names[y_axis]][:]
    lon_all = f[names[x_axis]][:]
    y_sel, x_sel = slice(None), slice(None)
    if args.bbox:
        west, east, south, north = args.bbox
        if not (-90 <= south < north <= 90):
            raise SystemExit(f"bad latitudes: need -90 <= S < N <= 90, got S={south} N={north}")
        if west >= east:
            raise SystemExit(
                f"bad longitudes: need W < E, got W={west} E={east}. "
                "This tool does not cross the dateline; slice_grib.py does."
            )
        y_sel = contiguous_run((lat_all >= south) & (lat_all <= north),
                               "latitude", "its axis is not monotonic")
        x_sel = lon_window(lon_all, west, east)

    if args.thin < 1:
        raise SystemExit(f"--thin must be >= 1, got {args.thin}")
    if args.thin > 1:
        # Decimation, not averaging: it keeps the axes exactly uniform, which the
        # format requires. The wrap flag below is computed from the result, so a
        # stride that does not divide a global axis evenly simply yields a
        # non-wrapping grid rather than a subtly wrong seam.
        y_sel = slice(y_sel.start, y_sel.stop, args.thin)
        x_sel = slice(x_sel.start, x_sel.stop, args.thin)

    lat = lat_all[y_sel]
    lon = lon_all[x_sel]
    if lat.size < 2 or lon.size < 2:
        raise SystemExit(f"the selected grid is {lat.size}x{lon.size}; need at least 2x2")

    # Latitude is stored ascending so both spatial axes have a positive step and
    # the C++ sampler needs one sign convention, not two. HDF5 has no negative
    # stride, so unlike the other selections this one happens after the read.
    flip_lat = lat[0] > lat[-1]
    if flip_lat:
        lat = lat[::-1]

    lat, dlat = axis_step(lat, "latitude")
    lon, dlon = axis_step(lon, "longitude")
    dt = uniform_step(times, "time", tol=0.0)

    convert = accumulated and not args.keep_units
    center = accumulated and not args.no_center
    if center:
        times = times - int(round(step_s / 2))
    out_units = RATE_UNITS.get(units.strip().lower(), units) if convert else units

    # A global axis wraps: the cell after the last one is the first one again, so
    # a query at 179.9 interpolates across the seam instead of falling off it.
    # The tolerance is a thousandth of a cell rather than a fixed 1e-6 degrees --
    # dlon comes from float32 endpoints, and on a 4320-column grid its rounding
    # error alone accumulates past 1e-6 over the full turn.
    lon_wrap = abs(lon.size * dlon - 360.0) < abs(dlon) * 1e-3

    grid = Grid(times=times, lat=lat, lon=lon,
                dt=dt, dlat=dlat, dlon=dlon, lon_wrap=lon_wrap)
    layout = plan_layout(grid, dtype=args.dtype,
                         max_mib=args.max_mib, batch_mib=args.batch_mib)
    level = 0 if args.no_compress else args.compress_level

    if len(sources) == 1:
        print(f"\n{head.path}")
    else:
        # In the order they are read, which is by time and not as given.
        print(f"\n{len(sources)} files, joined along time:")
        for s in sources:
            print(f"    {s.path}  "
                  f"{fmt_time(np.datetime64(int(s.times[0]), 's'))} .. "
                  f"{fmt_time(np.datetime64(int(s.times[-1]), 's'))}  "
                  f"({s.times.size} frames)")
    print(f"  field       {args.var}  "
          f"({text(dset, 'long_name') or text(dset, 'standard_name') or '?'}) "
          f"[{units or '?'}]" + (f" -> [{out_units}]" if convert else ""))
    print(f"  dims        ({', '.join(names)})  {dset.dtype}"
          + (f", taken at {', '.join(picked)}" if picked else ""))
    print(f"  treated as  {'accumulated over the step' if accumulated else 'instantaneous'}"
          + ("" if args.accumulated == "auto" else f"  (--accumulated {args.accumulated})"))
    print(f"  selected    {how}, stride {args.stride}"
          + (f", bbox {tuple(args.bbox)}" if args.bbox else "")
          + (f", thin {args.thin}" if args.thin > 1 else ""))
    describe(grid, layout, max_mib=args.max_mib, writers=args.writers, level=level,
             note=(f"(centred: stamps moved back {step_s / 2 / 60:g} min to the "
                   "middle of each accumulation window)") if center else "",
             grid_note="   (flipped to ascending)" if flip_lat else "")

    # --- read and unpack ---------------------------------------------------
    # A batch at a time, never the whole cube: a year of global hourly frames is
    # ~36 GB as float32, so materialising it just to rescale it is out of the
    # question.
    #
    # Which file each selected frame lives in, and which of its rows. A batch is
    # cut on those boundaries and no finer: a run inside one file is still the
    # single strided read it was before, so joining costs one extra read per
    # boundary crossed and nothing per frame.
    sel_src = src_of[t_sel]
    sel_row = row_of[t_sel]

    # Where (time, lat, lon) end up once the collapsed dims are gone, so a file
    # that stores its axes in another order is transposed back rather than read
    # wrong. Almost always already (0, 1, 2).
    surviving = [a for a in range(dset.ndim) if a not in extra]
    order = tuple(surviving.index(a) for a in (t_axis, y_axis, x_axis))

    def read(lo: int, hi: int) -> np.ndarray:
        """Frames [lo, hi) as a writable float32 cube, unpacked, fill -> NaN."""
        cube = np.empty((hi - lo, grid.nlat, grid.nlon), dtype="float32")
        for a, b in same_source_runs(sel_src[lo:hi]):
            s = sources[int(sel_src[lo + a])]
            rows = sel_row[lo + a:lo + b]
            source = [slice(None)] * s.dset.ndim
            source[y_axis], source[x_axis] = y_sel, x_sel
            # Rows within one file are the selection's stride apart, so the run
            # is a slice rather than a fancy index -- HDF5 satisfies it in one
            # pass, where a list of indices would be a gather.
            source[t_axis] = slice(int(rows[0]), int(rows[-1]) + 1, args.stride)
            for axis, index in extra.items():
                source[axis] = index
            raw = s.dset[tuple(source)]
            if order != (0, 1, 2):
                raw = np.transpose(raw, order)
            block = s.decode(raw)
            cube[a:b] = block[:, ::-1, :] if flip_lat else block
        return cube

    write_field(Path(args.npz), read, grid, layout,
                level=level, writers=args.writers,
                divide_by=step_s if convert else None, units=out_units)


if __name__ == "__main__":
    main()
