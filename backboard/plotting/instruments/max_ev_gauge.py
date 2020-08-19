"""Max EV Gauge."""

from .utils_instruments import create_basic_plot


def max_ev_gauge(self, fig, gridspec):
    """Maximal eigenvalue gauge, showing the largest eigenvalue of the Hessian
    versus iteration.

    Args:
        fig (matplotlib.figure): Figure of the Cockpit.
        gridspec (matplotlib.gridspec): GridSpec where the instrument should be
            placed.
    """
    # Plot Trace vs iteration
    title = "Max Eigenvalue"
    plot_args = {
        "x": "iteration",
        "y": "max_ev",
        "data": self.iter_tracking,
        "y_scale": "log",
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
