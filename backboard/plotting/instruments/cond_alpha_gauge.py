"""Cond Alpha Gauge."""

from .utils_instruments import create_basic_plot


def cond_alpha_gauge(self, fig, gridspec):
    """[summary]

    Args:
        fig ([type]): [description]
        gridspec ([type]): [description]
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
    }
    ax = fig.add_subplot(gridspec)
    create_basic_plot(**plot_args, ax=ax)
