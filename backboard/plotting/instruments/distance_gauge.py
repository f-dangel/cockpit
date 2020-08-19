"""Distance Gauge."""

from .utils_instruments import create_basic_plot


def distance_gauge(self, fig, gridspec, part="all"):
    """Distance gauge, showing the parameter L2-distance to the initialization
    versus iteration.

    Args:
        fig (matplotlib.figure): Figure of the Cockpit.
        gridspec (matplotlib.gridspec): GridSpec where the instrument should be
            placed
        part (str/int, optional): Defines which part of the network should be
            plotted. If a integer is given, it uses this part of the network.
            If "all" then it consideres the whole network. Defaults to "all".
    """
    # Plot Trace vs iteration
    title = "Distance"
    title += "" if isinstance(part, str) else " for Part " + str(part)
    plot_args = {
        "x": "iteration",
        "y": "d2init",
        "data": self.iter_tracking,
        "y_scale": "linear",
        "cmap": self.cmap,
        "EMA": "y",
        "EMA_alpha": self.EMA_alpha,
        "EMA_cmap": self.cmap2,
        "title": title,
        "xlim": "tight",
        "ylim": None,
        "fontweight": "normal" if type(part) is int else "bold",
        "facecolor": None if type(part) is int else "summary",
    }
    # part that should be plotted
    plot_args["y"] += "" if isinstance(part, str) else "_part_" + str(part)
    ax = fig.add_subplot(gridspec)
    create_basic_plot(**plot_args, ax=ax)
