"""CABS Gauge."""

import warnings

from cockpit.instruments.utils_instruments import check_data, create_basic_plot


def cabs_gauge(self, fig, gridspec):
    """CABS gauge, showing the CABS rule versus iteration.

    The batch size trades-off more accurate gradient approximations with longer
    computation. The `CABS criterion <https://arxiv.org/abs/1612.05086>`_ describes
    the optimal batch size under certain assumptions.

    The instruments shows the suggested batch size (and an exponential weighted
    average) over the course of training, according to

    - `Balles, L., Romero, J., & Hennig, P.,
      Coupling adaptive batch sizes with learning rates (2017).
      <https://arxiv.org/abs/1612.05086>`_

    **Preview**

    .. image:: ../../_static/instrument_previews/CABS.png
        :alt: Preview CABS Gauge

    **Requires**

    This instrument requires data from the :class:`~cockpit.quantities.CABS`
    quantity class.

    Args:
        self (CockpitPlotter): The cockpit plotter requesting this instrument.
        fig (matplotlib.figure.Figure): Figure of the Cockpit.
        gridspec (matplotlib.gridspec.GridSpec): GridSpec where the instrument should be
            placed
    """
    # Plot Trace vs iteration
    title = "CABS"

    # Check if the required data is available, else skip this instrument
    requires = ["CABS"]
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
        "y": "CABS",
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
