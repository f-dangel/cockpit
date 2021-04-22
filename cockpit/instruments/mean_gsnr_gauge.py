"""Mean GSNR gauge."""

import warnings

from cockpit.instruments.utils_instruments import check_data, create_basic_plot


def mean_gsnr_gauge(self, fig, gridspec):
    """Mean GSNR gauge, showing the mean GSNR versus iteration.

    The mean GSNR describes the average gradient signal-to-noise-ratio. `Recent
    work <https://arxiv.org/abs/2001.07384>`_ used this quantity to study the
    generalization performances of neural networks, noting "that larger GSNR during
    training process leads to better generalization performance. The instrument
    shows the mean GSNR versus iteration, overlayed with an exponentially weighted
    average.

    **Preview**

    .. image:: ../../_static/instrument_previews/MeanGSNR.png
        :alt: Preview MeanGSNR Gauge

    **Requires**

    This instrument requires data from the :class:`~cockpit.quantities.MeanGSNR`
    quantity class.

    Args:
        self (CockpitPlotter): The cockpit plotter requesting this instrument.
        fig (matplotlib.figure.Figure): Figure of the Cockpit.
        gridspec (matplotlib.gridspec.GridSpec): GridSpec where the instrument should be
            placed
    """
    # Plot Trace vs iteration
    title = "Mean GSNR"

    # Check if the required data is available, else skip this instrument
    requires = ["MeanGSNR"]
    plot_possible = check_data(self.tracking_data, requires)
    if not plot_possible:
        if self.debug:
            warnings.warn(
                "Couldn't get the required data for the " + title + " instrument",
                stacklevel=1,
            )
        return

    plot_args = {
        "x": "iteration",
        "y": "MeanGSNR",
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
