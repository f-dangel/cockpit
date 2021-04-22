"""Early Stopping Gauge."""

import warnings

from cockpit.instruments.utils_instruments import check_data, create_basic_plot


def early_stopping_gauge(self, fig, gridspec):
    """Early Stopping gauge, showing the LHS of the stopping criterion versus iteration.

    Early stopping the training has been widely used to prevent poor generalization
    due to over-fitting. `Mahsereci et al. (2017) <https://arxiv.org/abs/1703.09580>`_
    proposed an evidence-based stopping criterion based on mini-batch statistics.
    This instruments visualizes this criterion versus iteration, overlayed
    with an exponentially weighted average. If the stopping criterion becomes
    positive, this suggests stopping the training according to

    - `Mahsereci, M., Balles, L., Lassner, C., & Hennig, P.,
      Early stopping without a validation set (2017).
      <https://arxiv.org/abs/1703.09580>`_

    **Preview**

    .. image:: ../../_static/instrument_previews/EarlyStopping.png
        :alt: Preview EarlyStopping Gauge

    **Requires**

    This instrument requires data from the :class:`~cockpit.quantities.EarlyStopping`
    quantity class.

    Args:
        self (CockpitPlotter): The cockpit plotter requesting this instrument.
        fig (matplotlib.figure.Figure): Figure of the Cockpit.
        gridspec (matplotlib.gridspec.GridSpec): GridSpec where the instrument should be
            placed
    """
    # Plot Trace vs iteration
    title = "Early stopping"

    # Check if the required data is available, else skip this instrument
    requires = ["EarlyStopping"]
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
        "y": "EarlyStopping",
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
