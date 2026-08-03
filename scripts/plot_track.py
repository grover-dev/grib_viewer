"""plot_track.py -- plot the channels of a track against time.

A track is an .npz of parallel 1-D arrays: `lat`, `lng` and `time` (hours from
departure) are required, and every other array of the same length is a channel
-- solar power, stored energy, distance remaining, whatever the run recorded.

Stands alone, and is also what vis_map.py calls for its --track-plot. Nothing
here imports PyVista or VTK, so plotting a run costs numpy and matplotlib and
does not need the globe, a GPU, or a display when writing to a file.

    uv run python plot_track.py run.npz --list           # what is in there
    uv run python plot_track.py run.npz                  # every channel
    uv run python plot_track.py run.npz -c battery       # just one
    uv run python plot_track.py run.npz --save run.png   # off-screen to a file
    uv run python plot_track.py out/*.npz                # compare runs
    uv run python plot_track.py a.npz b.npz --labels jan feb -c power_stored_wh

Several channels become stacked panels sharing the time axis, and never two
scales on one pair of axes: watts and kilometres on a shared y is a comparison
the reader cannot make, while a shared x is the one that matters here. Several
tracks overlay instead -- one line per run in every panel -- which is the shape
for "same route, different departure day".
"""

from __future__ import annotations

import argparse
import multiprocessing
from pathlib import Path

import numpy as np

# The chart palette. These are the globe's own colours -- vis_map.py imports
# them from here rather than the other way round, so that this module keeps no
# dependency on it and a plot looks the same whichever end it was asked for.
SURFACE = "#0d0d0d"
COAST = "#c3c2b7"
GRATICULE = "#2c2c2a"
TRACK = "#eb6834"  # the one warm accent, reserved for the boat
INK = "#ffffff"

# Categorical slots for comparing runs, taken in this fixed order and never
# cycled: the order is the colourblind-safety mechanism, not decoration. These
# are the dark-surface steps of the eight-hue set, validated as a set for
# adjacent pairs -- which is the pairlist that applies to lines -- at worst CVD
# dE 8.4 and normal-vision dE 19.3 (OKLab x100). Identity never rests on colour
# alone: a legend is drawn whenever there is more than one run, and up to four
# runs are labelled on the lines as well.
SERIES = (
    "#3987e5",  # blue
    "#d95926",  # orange
    "#199e70",  # aqua
    "#c98500",  # yellow
    "#d55181",  # magenta
    "#008300",  # green
    "#9085e9",  # violet
    "#e66767",  # red
)
# Past the eight slots there is no ninth hue to reach for -- a generated one
# would not have been validated against the rest. Extra runs are dropped, and
# said so out loud rather than silently folded in.
MAX_RUNS = len(SERIES)

# The keys that say where and when rather than what, so they are axes and not
# channels to plot.
AXES = ("lat", "lng", "time")


def load_track(path: str) -> dict[str, np.ndarray]:
    """A track written by demo_run.py, or anything with the same keys.

    `lat`, `lng` and `time` are required. Every other 1-D array of the same
    length is offered as a plottable scalar, so a new channel is a new key.
    """
    z = np.load(path)
    track = {k: z[k] for k in z.files}
    missing = set(AXES) - set(track)
    if missing:
        raise SystemExit(f"{path}: track is missing {sorted(missing)}")
    return track


def channel_names(track: dict) -> list[str]:
    """Everything in `track` that is a per-sample channel: the same length as
    the course, and not one of the axes."""
    shape = track["lat"].shape
    return [
        k for k, v in track.items()
        if k not in AXES and getattr(v, "shape", None) == shape
    ]


def _spread(values, gap: float):
    """Nudge `values` apart so no two are closer than `gap`, order preserved.

    End-of-line labels for runs that finish at nearly the same value would
    otherwise print on top of each other, which is exactly the case when four
    runs of the same route are being compared.
    """
    order = np.argsort(values)
    out = np.array(values, dtype=float)
    for a, b in zip(order, order[1:]):
        if out[b] - out[a] < gap:
            out[b] = out[a] + gap
    return out


def _place_end_labels(ax, entries) -> None:
    """Lay the end labels out against the axis as it currently stands.

    Their y positions are data coordinates, so they cannot be computed once and
    left: hiding a run rescales the axis, and a label still sitting where a
    range 28 orders of magnitude wider put it is both meaningless and, with
    annotation_clip off, enough to overflow the text transform when drawn.
    Re-placing on every rescale keeps them beside their lines.
    """
    visible = [e for e in entries if e[0].get_visible()]
    if not visible:
        return
    lo, hi = ax.get_ylim()
    x0, x1 = ax.get_xlim()
    gap = (hi - lo) * 0.075
    for (ann, x, y_true), y in zip(visible, _spread([e[2] for e in visible], gap)):
        ann.xyann = (x + (x1 - x0) * 0.025, y)


def _attach_toggles(fig, legend, runs: dict, end_labels=None) -> None:
    """Click a legend entry, or a label at the end of a line, to hide that run.

    `runs` maps a label to every artist belonging to it -- its line in each
    panel, and the end labels with their leaders -- so one click takes the whole
    run out of all panels at once and another brings it back.

    The axes rescale to what is left visible, which is the point of hiding a
    run: one series ranging orders of magnitude wider than the rest flattens
    them all, and taking it out should let the others open up.

    Only does anything in a window. Saved figures keep every run visible.
    """
    owner = {}  # artist -> run label
    dimmable = {label: [] for label in runs}

    for handle, text in zip(legend.legend_handles, legend.get_texts()):
        label = text.get_text()
        if label not in runs:
            continue
        handle.set_picker(8)  # a few pixels of slack around a thin line
        text.set_picker(True)
        owner[handle] = owner[text] = label
        dimmable[label] += [handle, text]

    for label, artists in runs.items():
        for artist in artists:
            if artist.get_gid() == "endlabel":
                artist.set_picker(True)
                owner[artist] = label

    def on_pick(event):
        label = owner.get(event.artist)
        if label is None:
            return
        shown = not runs[label][0].get_visible()
        for artist in runs[label]:
            artist.set_visible(shown)
        for artist in dimmable[label]:
            artist.set_alpha(1.0 if shown else 0.25)
        for ax in fig.axes:
            ax.relim(visible_only=True)
            ax.autoscale_view()
            # After the rescale, not before: the labels are placed in data
            # coordinates against the limits they end up with.
            _place_end_labels(ax, (end_labels or {}).get(ax, ()))
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("pick_event", on_pick)


def _style_axis(ax, title: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=12, loc="left", pad=10)
    # Recessive frame: horizontal rules only, and no box around the data.
    ax.grid(True, axis="y", color=GRATICULE, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_color(GRATICULE)
    ax.tick_params(colors=COAST, labelsize=9)
    # Vertical breathing room so a label under a minimum sitting on the floor
    # does not land on the axis tick beneath it.
    ax.margins(x=0.01, y=0.16)


def build_figure(plt, runs):
    """The figure, given a live pyplot. Shared by the window and the file.

    `runs` is a list of (label, times, channels). One run per line, one channel
    per panel, all panels sharing the time axis -- never two scales on one pair
    of axes, since watts and kilometres on a shared y is a comparison the reader
    cannot make, while a shared x is the one that matters here.

    A lone run is drawn in the boat's own accent with its extremes labelled: a
    number on every point is unreadable at a hundred-odd samples, but the ends
    are worth reading. Several runs take categorical colours instead and the
    extremes give way to a legend -- with four runs in a panel, eight annotated
    points is noise, and which line is which is the question that matters.
    """
    if len(runs) > MAX_RUNS:
        dropped = [label for label, _, _ in runs[MAX_RUNS:]]
        print(f"  only {MAX_RUNS} runs fit the palette; dropped {dropped}")
        runs = runs[:MAX_RUNS]

    # Union in first-appearance order: a channel only one run recorded is still
    # worth a panel, it just has one line in it.
    names: list[str] = []
    for _, _, channels in runs:
        for name in channels:
            if name not in names:
                names.append(name)

    multi = len(runs) > 1
    fig, axes = plt.subplots(
        len(names), 1, sharex=True, squeeze=False,
        figsize=(10.0 if multi else 9.0, 1.0 + 2.4 * len(names)), facecolor=SURFACE,
    )

    handles = []
    # Every artist belonging to a run, so a click can take all of it out at once.
    owned: dict[str, list] = {label: [] for label, _, _ in runs}
    end_labels: dict = {}  # ax -> [(annotation, x, y_true)], re-placed on rescale
    for panel, (ax, name) in enumerate(zip(axes[:, 0], names)):
        _style_axis(ax, name)
        ends = []
        for slot, (label, times, channels) in enumerate(runs):
            values = channels.get(name)
            if values is None:
                continue
            times = np.asarray(times, dtype=float)
            values = np.asarray(values, dtype=float)
            color = SERIES[slot] if multi else TRACK
            line, = ax.plot(times, values, color=color, linewidth=2.0,
                            solid_capstyle="round", label=label)
            owned[label].append(line)
            if panel == 0:
                handles.append(line)

            if multi:
                ends.append((times[-1], float(values[-1]), label, color))
                continue

            span = float(times[-1] - times[0]) or 1.0
            for i, above in ((int(np.argmin(values)), False),
                             (int(np.argmax(values)), True)):
                # An extreme often falls on the first or last sample, where a
                # centred label overhangs the axis; anchor it inward instead.
                edge = (times[i] - times[0]) / span
                ha = "left" if edge < 0.05 else "right" if edge > 0.95 else "center"
                ax.plot(times[i], values[i], "o", color=color, markersize=6)
                ax.annotate(
                    f"{values[i]:.4g}", (times[i], values[i]),
                    textcoords="offset points", xytext=(0, 10 if above else -16),
                    ha=ha, color=INK, fontsize=9,
                )

        # Direct labels at the line ends, so identity does not rest on matching
        # a colour to a legend swatch. Only up to four -- past that the right
        # margin is a thicket and the legend has to carry it alone.
        if multi and len(runs) <= 4 and ends:
            entries = []
            for x, y_true, label, color in ends:
                # Anchored to where the line actually ends, with a leader to
                # wherever the label had to move: runs that finish together get
                # fanned out, and without the leader the label would appear to
                # claim a value its line never reached. The text position is
                # set by _place_end_labels, now and after every rescale.
                end = ax.annotate(
                    label, xy=(x, y_true), xytext=(x, y_true),
                    textcoords="data", ha="left", va="center",
                    color=color, fontsize=9, annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", color=color, linewidth=0.8,
                                    shrinkA=0, shrinkB=2),
                )
                end.set_gid("endlabel")  # picked up by _attach_toggles
                owned[label].append(end)
                entries.append((end, x, y_true))
            end_labels[ax] = entries
            _place_end_labels(ax, entries)

    axes[-1, 0].set_xlabel("hours since departure", color=COAST, fontsize=10)
    fig.tight_layout()
    if multi:
        # A legend is present for every multi-run figure, direct labels or not.
        legend = fig.legend(
            handles=handles, loc="lower center", ncol=min(len(handles), 4),
            frameon=False, labelcolor=INK, fontsize=10,
        )
        # Room for the legend below and the end labels to the right.
        fig.subplots_adjust(bottom=0.06 + 0.30 / len(names), right=0.86)
        _attach_toggles(fig, legend, owned, end_labels)
    return fig


def _plot_worker(runs) -> None:
    """Entry point for the plot subprocess: draw, then run matplotlib's loop."""
    import matplotlib.pyplot as plt

    build_figure(plt, runs)
    plt.show()


def write_figure(runs, path: str) -> None:
    """Render to a file, with no window and no display required."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = build_figure(plt, runs)
    fig.savefig(path, dpi=140, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {path}")


def spawn_plot(runs):
    """Open the plot in a window, in a process of its own. Returns that process.

    The separate process is not incidental. matplotlib's Qt backend and VTK's
    interactor are two GUI event loops and only one of them can own a thread, so
    when vis_map.py draws this in-process the window appears and then sits there
    dead -- the window manager eventually offers to kill it -- because VTK is
    running the loop and nothing is dispatching Qt's events. A child has a loop
    of its own, so both windows stay live. Called on its own from main() below
    the same machinery costs one extra process and behaves identically.
    """
    # spawn, not fork: a forked child would inherit the parent's VTK and Qt
    # state mid-flight, which is its own source of hangs. Not a daemon, so the
    # plot outlives the caller and the run ends when the window is closed.
    ctx = multiprocessing.get_context("spawn")
    proc = ctx.Process(
        target=_plot_worker,
        args=([(label,
                np.asarray(times, dtype=float),
                {k: np.asarray(v, dtype=float) for k, v in channels.items()})
               for label, times, channels in runs],),
    )
    proc.start()
    return proc


def plot_runs(runs, path: str | None = None):
    """Write the plot to `path`, or open it in a window if there is no path.

    Returns the child process for the window case, None for the file case.
    """
    if path is not None:
        write_figure(runs, path)
        return None
    return spawn_plot(runs)


def plot_channels(times, channels: dict, path: str | None = None, label: str = ""):
    """One run's channels -- the single-track case of plot_runs()."""
    return plot_runs([(label, times, channels)], path)


def select(track: dict, names: list[str] | None) -> dict[str, np.ndarray]:
    """The channels asked for, in the order asked for, or all of them.

    Names the file does not have are reported rather than skipped quietly: an
    empty plot and a misspelt channel look identical otherwise.
    """
    available = channel_names(track)
    if not names:
        return {k: track[k] for k in available}

    chosen, missing = {}, []
    for name in names:
        values = track.get(name)
        if values is None or getattr(values, "shape", None) != track["lat"].shape:
            missing.append(name)
        else:
            chosen[name] = values
    if missing:
        print(f"  no channel{'s' if len(missing) > 1 else ''} "
              f"{', '.join(repr(m) for m in missing)}; available: {available}")
    return chosen


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "npz", nargs="+",
        help="tracks: lat, lng, time and any channels. More than one overlays "
        "them, one line per run in each channel's panel.",
    )
    p.add_argument(
        "-c", "--channels", nargs="+", metavar="CHANNEL",
        help="channels to plot, in order (default: all of them)",
    )
    p.add_argument(
        "--labels", nargs="+", metavar="NAME",
        help="names for the runs in the legend (default: the file stems)",
    )
    p.add_argument("--list", action="store_true", help="list the channels and exit")
    p.add_argument("--save", metavar="PNG", help="write to a file instead of a window")
    args = p.parse_args()

    if args.labels and len(args.labels) != len(args.npz):
        p.error(f"{len(args.labels)} labels for {len(args.npz)} tracks")
    labels = args.labels or [Path(f).stem for f in args.npz]

    runs = []
    for path, label in zip(args.npz, labels):
        track = load_track(path)
        hours = track["time"]
        available = channel_names(track)
        print(f"{label}: {len(track['lat'])} points over {hours[-1]:.1f} h")

        if args.list:
            for name in available:
                values = np.asarray(track[name], dtype=float)
                print(f"  {name}  {values.min():.4g} .. {values.max():.4g}")
            continue

        chosen = select(track, args.channels)
        if not chosen:
            raise SystemExit(f"{path}: nothing to plot; channels are {available}")
        for name, values in chosen.items():
            values = np.asarray(values, dtype=float)
            print(f"  {name} ({values.min():.4g}..{values.max():.4g})")
        runs.append((label, hours, chosen))

    if args.list:
        return

    # A channel one run recorded and another did not still gets a panel; say so,
    # because a line missing from one panel otherwise looks like a plotting bug.
    everywhere = set.intersection(*(set(c) for _, _, c in runs))
    ragged = [n for _, _, c in runs for n in c if n not in everywhere]
    if ragged:
        print(f"  not in every run: {sorted(set(ragged))}")

    if args.save:
        write_figure(runs, args.save)
    else:
        # Drawn here rather than in a child, since nothing else owns the loop.
        import matplotlib.pyplot as plt

        build_figure(plt, runs)
        plt.show()


if __name__ == "__main__":
    main()
