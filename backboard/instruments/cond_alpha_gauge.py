"""Condition number vs. Alpha Gauge."""

from backboard.instruments.utils_instruments import create_basic_plot


def cond_alpha_gauge(self, fig, gridspec):
    """(Average) condition number vs Alpha value.

    Args:
        self (cockpit.plotter): The cockpit plotter requesting this instrument.
        fig (matplotlib.figure): Figure of the Cockpit.
        gridspec (matplotlib.gridspec): GridSpec where the instrument should be
            placed.
    """
    # Condition number vs. alpha
    plot_args = {
        "x": "avg_cond",
        "y": "alpha",
        "data": self.iter_tracking,
        "EMA": "xy",
        "EMA_alpha": self.EMA_alpha,
        "EMA_cmap": self.cmap2,
        "x_scale": "log",
        "y_scale": "linear",
        "cmap": self.cmap,
        "title": "Condition Number vs. Alpha",
        "xlim": None,
        "ylim": None,
        "fontweight": "bold",
        "facecolor": "summary",
        "zero_lines": True,
        "center": [False, True],
    }
    ax = fig.add_subplot(gridspec)
    create_basic_plot(**plot_args, ax=ax)
