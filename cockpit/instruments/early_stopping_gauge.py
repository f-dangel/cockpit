"""Early Stopping Gauge."""

import warnings

from cockpit.instruments.utils_instruments import check_data, create_basic_plot


def early_stopping_gauge(self, fig, gridspec):
    """Early Stopping gauge, showing the LHS of the stopping criterion versus iteration.

    If the stopping criterion becomes positive, this suggests stopping the training
    according to

    - Mahsereci, M., Balles, L., Lassner, C., & Hennig, P.,
      Early stopping without a validation set (2017).

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
