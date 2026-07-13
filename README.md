# boatforge — ERA5 GRIB viewer & slicer

Play ERA5 weather data back over time, switch between variables, zoom into a
region, and carve big downloads down to the bits you actually need.

Built for the shape of real ERA5 data: multi-gigabyte files, a 0–360 longitude
grid, wave fields on a coarser grid than the atmosphere, and accumulated fields
(precipitation, radiation) stored against a *reference* time plus a forecast step
rather than the hour they describe.

```bash
uv run grib_info.py  data/data.grib                            # what's in here?
uv run slice_grib.py data/data.grib data/jan5.grib --frames 96:120 --bbox -12 5 48 62
uv run era5_viewer.py data/jan5.grib --all --quiver            # play it
```

---

## Install

Everything runs through [uv](https://docs.astral.sh/uv/); no manual venv needed.

```bash
uv sync          # creates .venv and installs from uv.lock
uv run pytest    # 42 tests, ~11s
```

Cartopy downloads its coastline shapefiles from Natural Earth on first use, so
the very first render needs a network connection. They're cached afterwards.

---

## What GRIB actually is

Worth reading once, because nearly every awkward thing in this codebase is a
direct consequence of the format.

**A GRIB file is not a cube. It's a flat pile of independent records**, called
*messages*, concatenated back to back. There is no header, no index, no table of
contents — just message after message, each one self-describing.

**One message = one 2-D field at one instant.** It carries a small block of
metadata keys plus a packed grid of values:

```
┌─ message ────────────────────────────────────┐
│ paramId 165  ("10 metre U wind component")   │  what
│ validityDate 20250105, validityTime 0400     │  when
│ gridType regular_ll, Ni 1440, Nj 721         │  where
│ latitudeOfFirstGridPointInDegrees 90.0  ...  │
│ [ 1,038,240 packed values ]                  │  the data
└──────────────────────────────────────────────┘
```

So a month of hourly ERA5 with 14 variables isn't one array — it's
`744 × 14 ≈ 10,000 separate messages` in one file. The 3-D "cube" you think of
(`time × lat × lon`) doesn't exist on disk; **cfgrib reconstructs it** by reading
every message's headers and working out which ones stack together.

That single fact explains most of this project's design:

**Why there are two layers.** `grib_utils.py` has a *message layer* and an
*xarray layer*. Slicing works on raw messages — to cut a file down you just copy
the messages you want and skip the rest. No decoding, no cube, and every original
metadata key survives into the output. (Going the other way, through xarray, is
lossy: cfgrib's writer needs the original `GRIB_*` attrs intact, and unit
conversions destroy them.)

**Why scanning is cheap but loading is expensive.** Reading every message's
*headers* to list the time axis takes seconds even on 16 GB, because the packed
values are skipped. Reading the *values* means decompressing gigabytes.

**Why the time axis is a minefield.** Each message states the hour it's valid for
(`validityDate`/`validityTime`) — but *accumulated* fields (precipitation,
radiation) are stored against a **reference time plus a forecast step**:

```
tsr:  dataDate/dataTime  20250104 1800   +   endStep 10h
      validityDate/Time  20250105 0400        ← the hour it actually describes
```

(That's a real message out of `data.grib`, not a hypothetical.)

Read the reference time and your rainfall lands 10 hours away from your wind.
Worse, cfgrib presents these as a rectangular `(time × step)` grid, which implies
combinations the file never contained — so the cube grows phantom all-NaN frames
at the edges. This project reads `validityDate`/`validityTime` from the messages
and treats them as ground truth.

**Why longitudes are a minefield too.** Each message declares its own grid
corners. ERA5's global fields start at longitude 0 and run east to 359.75 — not
−180…180 — so a European bounding box is *already* a wrap-around in the source
data, spanning columns near 350 and columns near 5.

**Other things messages don't have to agree on.** Two messages in the same file
can be on completely different grids: in a real ERA5 download the wave fields
(`swh`, `mwp`, `mwd`) arrive at 0.5° while the atmospheric fields are at 0.25°.
Nothing forbids it, so the tooling has to cope.

**Names are not stable.** A message identifies its parameter by a numeric
`paramId` (165), and its `shortName` in the file is `10u` — but cfgrib hands it to
xarray as `u10`. So the same field has three names depending on which layer you're
standing in, and none of them is safe to key behaviour off. This is why the viewer
infers presentation from *units* and the long name rather than matching variable
names.

**GRIB1 vs GRIB2** are different encodings of the same idea; ecCodes reads both
and this project doesn't care which you have. (Your file is edition 1.)

Handy for poking at a file directly, if you have ecCodes' CLI installed:

```bash
grib_ls  data.grib          # one line per message: param, date, time, grid
grib_dump -O data.grib      # every key of every message (very verbose)
```

## The three tools

| tool | what it's for |
|---|---|
| `grib_info.py` | Inspect a file: fields, frames, grids, and what loading it would cost. |
| `slice_grib.py` | Cut a big GRIB down by time and/or area, writing a new GRIB. |
| `era5_viewer.py` | Animate the data, switch variables, zoom, export video or GRIB. |

They share `grib_utils.py`, which holds both the message-level GRIB handling and
the xarray loading layer.

### Start here: `grib_info.py`

Reads headers and coordinates only, so it's safe to point at a 16 GB file.

```bash
uv run grib_info.py data/data.grib
uv run grib_info.py data/data.grib --times     # also list every valid time
```

```
time: 744 frames
  first  [0]     2025-01-01T00:00Z
  last   [743]   2025-01-31T23:00Z
  spacing        1 h

fields: 14
  name       long name                              units       grid      frames  full load
  u10        10 metre U wind component            m s**-1   721x1440        744     2.9 GB
  tsr        Top net short-wave (solar) radiation  J m**-2   721x1440        744     2.9 GB
  swh        Significant height of combined wind…        m    361x720        744   737.7 MB
  ...

grids: 2
  361x720 @ 0.5°     lat -90.0..90.0   lon -180.0..179.5     mwd, mwp, swh
  721x1440 @ 0.25°   lat -90.0..90.0   lon -180.0..179.75    cdir, d2m, msl, sp, ...

loading everything costs roughly 33.8 GB of RAM
  -> slice it down first: uv run slice_grib.py ...
```

That last line is the one that matters. **If the estimate exceeds your RAM, slice
before you view.**

### Cutting it down: `slice_grib.py`

Copies raw GRIB messages, so nothing is decoded and all the original metadata
survives. The output is a normal GRIB any tool can read.

```bash
uv run slice_grib.py data/data.grib --list                  # just the time axis

# pick times by frame index (0-based, end-exclusive, like a python slice)
uv run slice_grib.py data/data.grib data/day1.grib  --frames 0:24
uv run slice_grib.py data/data.grib data/one.grib   --frames 100

# ...or by timestamp (inclusive)
uv run slice_grib.py data/data.grib data/week.grib  --start 2025-01-06 --end 2025-01-12

# thin a long run
uv run slice_grib.py data/data.grib data/6hr.grib   --stride 6

# crop an area: W E S N, degrees in -180..180. Composes with any time selection.
uv run slice_grib.py data/data.grib data/uk.grib    --frames 0:24 --bbox -12 5 48 62
```

A **frame** is a position in the file's sorted list of distinct valid times — the
same numbering the viewer's slider shows. `--frames` and `--start/--end` are
mutually exclusive; `--stride` composes with either.

**Crossing the dateline:** give `W > E`.

```bash
uv run slice_grib.py data/data.grib data/pac.grib --frames 0:24 --bbox 170 -170 -20 20
```

Typical savings: one hour of the global 16 GB file is ~23 MB; a UK box for that
hour is ~0.1 MB.

### Viewing: `era5_viewer.py`

```bash
uv run era5_viewer.py data/uk.grib                # prompts for how much to load
uv run era5_viewer.py data/uk.grib --all          # skip the prompt, load it all
uv run era5_viewer.py data/uk.grib --list         # show discovered layers, exit
```

With no time flags and a terminal attached, it shows the time axis and the RAM
cost, then asks: load all, load a range, or thin by every Nth step.

**In the window:**

- **Variable list** (left) — every field found in the file, plus derived wind
  speed (`ws10`) wherever a `u`/`v` pair exists.
- **Time slider** (bottom) — reads as a clock, showing the valid time and the
  run's span, not a frame counter.
- **▶ play** — animate. **⤢ reset zoom** — back to the full domain.
- **Drag a box on the map** to zoom. This re-crops the *data*, not just the axes,
  and re-fits the colour scale to what's visible.
- **global colors** checkbox — pin the colour limits to the full domain instead,
  so the scale stays comparable as you zoom around.
- **⬇ export selection to GRIB** — write the visible area and loaded time window
  out as a new GRIB, next to the source.

**Flags:**

| flag | effect |
|---|---|
| `--var NAME` | which layer to show first |
| `--quiver` | overlay wind vectors (needs a `u`/`v` pair) |
| `--start` / `--end` / `--stride` | load only part of the time axis (skips the prompt) |
| `--all` | load everything without prompting |
| `--bbox W E S N` | load only this area |
| `--no-map` | turn off coastlines/borders |
| `--save OUT.mp4` | render an animation instead of opening a window |
| `--fps` / `--dpi` | animation speed and resolution |
| `--export OUT.grib` | write the selection to a new GRIB and exit (never decodes) |

### Rendering video

```bash
uv run era5_viewer.py data/uk.grib --all --var ws10 --quiver --save data/wind.mp4 --fps 12
```

One frame per loaded time step, so the time flags control the video's length:

```bash
uv run era5_viewer.py data/data.grib \
  --start 2025-01-05 --end 2025-01-06 --stride 3 \
  --bbox -12 5 48 62 --var tsr --save data/solar.mp4
```

The exported figure is chrome-free — map, title and colorbar only, no widgets.
`.gif` works too; the extension picks the writer.

---

## How presentation is decided

There is no per-variable whitelist. Every field with a `(time, y, x)` shape
becomes a layer, and its colormap, units and scale come from its own GRIB/CF
metadata plus the data's own distribution:

- **Kelvin → °C**, **Pa → hPa**, and **J/m² → W/m²** (radiation is accumulated
  over the step, so it's divided by the step length).
- Colormap follows the quantity: irradiance is `inferno`, cloud/fraction
  `Blues_r`, wind speed `turbo`, signed components `RdBu_r`, temperature
  `coolwarm`.
- Colour limits are the 2nd/98th percentile, **sampled** (≤8 time slices, ≤40k
  points each) rather than read exhaustively — reading whole cubes just to pick
  limits is what made startup slow and memory-hungry.

An unrecognised field still renders, just with generic styling.

---

## Test data

`make_sample_grib.py` writes synthetic ERA5-like files, useful when you don't
want to touch a real download:

```bash
uv run make_sample_grib.py --preset small     # -> data/sample_small.grib  (39 KB)
uv run make_sample_grib.py --preset medium    # UK/North Sea, a week 3-hourly (3.8 MB)
uv run make_sample_grib.py --preset large     # W. Europe, a month hourly (186 MB)
uv run make_sample_grib.py --preset xlarge    # whole planet (832 MB)
```

`data/` is gitignored — GRIB files and rendered videos live there.

---

## Tests

```bash
uv run pytest          # 42 tests, ~11s
```

Each test in `tests/` pins a bug that actually shipped and had to be fixed on
real ERA5 data — the docstrings say what went wrong, so a regression gets an
explanation rather than just a red X. The fixtures deliberately reproduce ERA5's
awkward shapes (0–360 grids, step-ladder accumulations, single-frame files);
tidy fixtures would pass no matter what the code did.

---

## Notes and limits

- **Area subsetting requires a `regular_ll` grid** (ERA5's default lat/lon
  download). A Gaussian-grid file is rejected with a clear error rather than
  silently mangled.
- **Accumulations are folded onto valid time.** ERA5 stores `tp`/`tsr`/`cdir`
  against a reference time plus a step; the hour they describe is `time + step`.
  The loader folds that ladder onto a single time axis and vets it against the
  message headers, which are ground truth — cfgrib otherwise presents
  `(time × step)` as a rectangle and invents edge frames the file doesn't hold.
- **Opening is index-only, and the index is cached.** There is no cfgrib in the
  read path — a message index (what each message is, when it's valid, its byte
  offset) is built in one header pass and pickled to `.cache/`. A frame is then a
  seek plus one message decode. On the 16 GB file: **~16 s to index cold, ~1.6 s
  warm, ~50 ms per frame**, in ~250 MB of RAM. cfgrib's `open_datasets`
  reconstructs the whole cube geometry up front and took **~140 s every run** on
  the same file — its own index cache can't help, because it compares mtimes and
  a GRIB whose mtime is in the future (a zip extracted across a timezone, like
  this dataset) is always judged newer than its index. Our cache key is
  path + size + mtime, which doesn't care what the clock says.
- **Fields are dask-backed**, one task per frame, so nothing is read until a
  frame is actually drawn.
- **Mixed resolutions are fine.** Wave fields (0.5°) and atmospheric fields
  (0.25°) coexist in one file; switching between them works.
- **Longitudes:** a *global* 0–360 grid is rolled onto −180…180. A *regional*
  window is left in its own frame — a Pacific box runs 170…190, and the viewer
  re-centres the map on it rather than tearing it in half.
