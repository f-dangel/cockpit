"""TIC Gauge."""

from cockpit.instruments.utils_instruments import check_data, create_basic_plot


def tic_gauge(self, fig, gridspec):
    """TIC gauge, showing the TIC versus iteration.

    The TIC (either approximated via traces or using a diagonal approximation)
    describes the relation between the curvature and the gradient noise. `Recent
    work <https://arxiv.org/abs/1906.07774>`_ suggested that *at a local minimum*,
    this quantitiy can estimate the generalization gap. This instrument shows the
    TIC versus iteration, overlayed with an exponentially weighted average.

    **Preview**

    .. image:: ../../_static/instrument_previews/TIC.png
        :alt: Preview TIC Gauge

    **Requires**

    The trace instrument requires data from the :class:`~cockpit.quantities.TICDiag`
    or :class:`~cockpit.quantities.TICTrace` quantity class.

    Args:
        self (CockpitPlotter): The cockpit plotter requesting this instrument.
        fig (matplotlib.figure.Figure): Figure of the Cockpit.
        gridspec (matplotlib.gridspec.GridSpec): GridSpec where the instrument should be
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
