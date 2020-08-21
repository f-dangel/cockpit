"""(Average) Condition Number Gauge."""

from .utils_instruments import create_basic_plot


def cond_gauge(self, fig, gridspec):
    """Showing (average) condition number of the Hessian versus iteration.

    This (average) condition number is computed as the ratio of the largest
    eigenvalue and average eigenvalue.

    Args:
        self (cockpit.plotter): The cockpit plotter requesting this instrument.
        fig (matplotlib.figure): Figure of the Cockpit.
        gridspec (matplotlib.gridspec): GridSpec where the instrument should be
            placed.
    """
    # Plot Trace vs iteration
    title = "(Average) Conditon Number"
    plot_args = {
        "x": "iteration",
        "y": "avg_cond",
        "data": self.iter_tracking,
        "y_scale": "linear",
        "cmap": self.cmap,
        "EMA": "y",
        "EMA_alpha": self.EMA_alpha,
        "EMA_cmap": self.cmap2,
        "title": title,
        "xlim": "tight",
        "ylim": None,
        "fontweight": "bold",
        "facecolor": "summary",
    }
    # part that should be plotted
    ax = fig.add_subplot(gridspec)
    create_basic_plot(**plot_args, ax=ax)
