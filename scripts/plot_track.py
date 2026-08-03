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

Several channels become stacked panels sharing the time axis, and never two
scales on one pair of axes: watts and kilometres on a shared y is a comparison
the reader cannot make, while a shared x is the one that matters here.
"""

from __future__ import annotations

import argparse
import multiprocessing

import numpy as np

# The chart palette. These are the globe's own colours -- vis_map.py imports
# them from here rather than the other way round, so that this module keeps no
# dependency on it and a plot looks the same whichever end it was asked for.
SURFACE = "#0d0d0d"
COAST = "#c3c2b7"
GRATICULE = "#2c2c2a"
TRACK = "#eb6834"  # the one warm accent, reserved for the boat
INK = "#ffffff"

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


def build_figure(plt, times, channels: dict):
    """The figure itself, given a live pyplot. Shared by the window and the file.

    Each panel is a single series, so its title names it and no legend is
    needed, and only the two extremes carry a value -- a number on every point
    is unreadable at a hundred-odd samples, and the ends are exactly what a
    colour ramp over the same channel would be stretched between.
    """
    times = np.asarray(times, dtype=float)
    span = float(times[-1] - times[0]) or 1.0

    fig, axes = plt.subplots(
        len(channels), 1, sharex=True, squeeze=False,
        figsize=(9.0, 1.0 + 2.4 * len(channels)), facecolor=SURFACE,
    )
    for ax, (label, values) in zip(axes[:, 0], channels.items()):
        values = np.asarray(values, dtype=float)
        ax.set_facecolor(SURFACE)
        ax.plot(times, values, color=TRACK, linewidth=2.0, solid_capstyle="round")

        for i, above in ((int(np.argmin(values)), False),
                         (int(np.argmax(values)), True)):
            # An extreme often falls on the first or last sample, where a
            # centred label overhangs the axis; anchor it inward there instead.
            edge = (times[i] - times[0]) / span
            ha = "left" if edge < 0.05 else "right" if edge > 0.95 else "center"
            ax.plot(times[i], values[i], "o", color=TRACK, markersize=6)
            ax.annotate(
                f"{values[i]:.4g}", (times[i], values[i]),
                textcoords="offset points", xytext=(0, 10 if above else -16),
                ha=ha, color=INK, fontsize=9,
            )

        ax.set_title(label, color=INK, fontsize=12, loc="left", pad=10)
        # Recessive frame: horizontal rules only, and no box around the data.
        ax.grid(True, axis="y", color=GRATICULE, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("bottom", "left"):
            ax.spines[side].set_color(GRATICULE)
        ax.tick_params(colors=COAST, labelsize=9)
        # Vertical breathing room so a label under a minimum sitting on the
        # floor does not land on the axis tick beneath it.
        ax.margins(x=0.01, y=0.16)

    axes[-1, 0].set_xlabel("hours since departure", color=COAST, fontsize=10)
    fig.tight_layout()
    return fig


def _plot_worker(times, channels: dict) -> None:
    """Entry point for the plot subprocess: draw, then run matplotlib's loop."""
    import matplotlib.pyplot as plt

    build_figure(plt, times, channels)
    plt.show()


def write_figure(times, channels: dict, path: str) -> None:
    """Render to a file, with no window and no display required."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = build_figure(plt, times, channels)
    fig.savefig(path, dpi=140, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {path}")


def spawn_plot(times, channels: dict):
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
        args=(np.asarray(times, dtype=float),
              {k: np.asarray(v, dtype=float) for k, v in channels.items()}),
    )
    proc.start()
    return proc


def plot_channels(times, channels: dict, path: str | None = None):
    """Write the plot to `path`, or open it in a window if there is no path.

    Returns the child process for the window case, None for the file case.
    """
    if path is not None:
        write_figure(times, channels, path)
        return None
    return spawn_plot(times, channels)


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
    p.add_argument("npz", help="a track: lat, lng, time and any channels")
    p.add_argument(
        "-c", "--channels", nargs="+", metavar="CHANNEL",
        help="channels to plot, in order (default: all of them)",
    )
    p.add_argument("--list", action="store_true", help="list the channels and exit")
    p.add_argument("--save", metavar="PNG", help="write to a file instead of a window")
    args = p.parse_args()

    track = load_track(args.npz)
    hours = track["time"]
    available = channel_names(track)
    print(f"track: {len(track['lat'])} points over {hours[-1]:.1f} h")

    if args.list:
        for name in available:
            values = np.asarray(track[name], dtype=float)
            print(f"  {name}  {values.min():.4g} .. {values.max():.4g}")
        return

    chosen = select(track, args.channels)
    if not chosen:
        raise SystemExit(f"{args.npz}: nothing to plot; channels are {available}")
    for name in chosen:
        values = np.asarray(chosen[name], dtype=float)
        print(f"  {name} ({values.min():.4g}..{values.max():.4g})")

    if args.save:
        write_figure(hours, chosen, args.save)
    else:
        # Drawn here rather than in a child, since nothing else owns the loop.
        import matplotlib.pyplot as plt

        build_figure(plt, hours, chosen)
        plt.show()


if __name__ == "__main__":
    main()
