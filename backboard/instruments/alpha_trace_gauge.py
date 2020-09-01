"""Condition number vs. Alpha Gauge."""

from backboard.instruments.utils_instruments import create_basic_plot


def alpha_trace_gauge(self, fig, gridspec):
    """Alpha vs Trace.

    Args:
        self (cockpit.plotter): The cockpit plotter requesting this instrument.
        fig (matplotlib.figure): Figure of the Cockpit.
        gridspec (matplotlib.gridspec): GridSpec where the instrument should be
            placed.
    """
    # Condition number vs. alpha
    plot_args = {
        "x": "alpha",
        "y": "trace",
        "data": self.iter_tracking,
        "EMA": "xy",
        "EMA_alpha": self.EMA_alpha,
        "EMA_cmap": self.cmap2,
        "x_scale": "linear",
        "y_scale": "linear",
        "cmap": self.cmap,
        "title": "Alpha vs. Trace",
        "xlim": None,
        "ylim": None,
        "fontweight": "bold",
        "facecolor": "summary",
        "zero_lines": True,
        "center": [True, False],
    }
    ax = fig.add_subplot(gridspec)
    create_basic_plot(**plot_args, ax=ax)
