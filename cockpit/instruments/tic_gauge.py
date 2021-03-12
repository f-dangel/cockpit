"""TIC Gauge."""

from cockpit.instruments.utils_instruments import check_data, create_basic_plot


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

    if check_data(self.tracking_data, ["TICDiag"]):
        plot_args = {
            "x": "iteration",
            "y": "TICDiag",
            "data": self.tracking_data,
            "x_scale": "symlog" if self.show_log_iter else "linear",
            "y_scale": "linear",
            "cmap": self.cmap,
            "EMA": "y",
            "EMA_alpha": self.EMA_alpha,
            "EMA_cmap": self.cmap2,
            "title": title,
            "xlim": "tight",
            "ylim": None,
            "fontweight": "bold",
            "facecolor": self.bg_color_instruments,
        }
        ax = fig.add_subplot(gridspec)
        create_basic_plot(**plot_args, ax=ax)

    if check_data(self.tracking_data, ["TICTrace"]):
        if "ax" in locals():
            ax2 = ax.twinx()
        else:
            ax2 = fig.add_subplot(gridspec)
        plot_args = {
            "x": "iteration",
            "y": "TICTrace",
            "data": self.tracking_data,
            "x_scale": "symlog" if self.show_log_iter else "linear",
            "y_scale": "linear",
            "cmap": self.cmap_backup,
            "EMA": "y",
            "EMA_alpha": self.EMA_alpha,
            "EMA_cmap": self.cmap2_backup,
            "title": title,
            "xlim": "tight",
            "ylim": None,
            "fontweight": "bold",
            "facecolor": self.bg_color_instruments,
        }
        create_basic_plot(**plot_args, ax=ax2)
