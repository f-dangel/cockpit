"""Trace Gauge."""

from .utils_instruments import create_basic_plot


def trace_gauge(self, fig, gridspec, part="all"):
    """[summary]

    Args:
        fig ([type]): [description]
        gridspec ([type]): [description]
        part (str, optional): [description]. Defaults to "all".
    """
    # Plot Trace vs iteration
    plot_args = {
        "x": "iteration",
        "y": "trace",
        "data": self.iter_tracking,
        "y_scale": "log",
        "cmap": self.cmap,
        "EMA": "y",
        "EMA_alpha": self.EMA_alpha,
        "EMA_cmap": self.cmap2,
        "title": "Performance Plot",
    }
    # part that should be plotted
    plot_args["y"] += "" if isinstance(part, str) else "_part_" + str(part)
    ax = fig.add_subplot(gridspec)
    create_basic_plot(**plot_args, ax=ax)
