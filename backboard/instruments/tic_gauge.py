"""TIC Gauge."""

from backboard.instruments.utils_instruments import create_basic_plot


def tic_gauge(self, fig, gridspec):
    """TIC gauge, showing the TIC versus iteration.

    Args:
        self (cockpit.plotter): The cockpit plotter requesting this instrument.
        fig (matplotlib.figure): Figure of the Cockpit.
        gridspec (matplotlib.gridspec): GridSpec where the instrument should be
            placed
    """
    # Plot Trace vs iteration
    title = "TIC"

    if "tic_diag" in self.tracking_data:
        plot_args = {
            "x": "iteration",
            "y": "tic_diag",
            "data": self.tracking_data,
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
        ax = fig.add_subplot(gridspec)
        create_basic_plot(**plot_args, ax=ax)

    if "tic_trace" in self.tracking_data:
        if "ax" in locals():
            ax2 = ax.twinx()
        else:
            ax2 = fig.add_subplot(gridspec)
        plot_args = {
            "x": "iteration",
            "y": "tic_trace",
            "data": self.tracking_data,
            "y_scale": "linear",
            "cmap": self.cmap_backup,
            "EMA": "y",
            "EMA_alpha": self.EMA_alpha,
            "EMA_cmap": self.cmap2_backup,
            "title": title,
            "xlim": "tight",
            "ylim": None,
            "fontweight": "bold",
            "facecolor": "summary",
        }
        create_basic_plot(**plot_args, ax=ax2)
