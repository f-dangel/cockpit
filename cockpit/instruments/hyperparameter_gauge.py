"""Hyperparameter Gauge."""

import warnings

import seaborn as sns

from cockpit.instruments.utils_instruments import (
    _add_last_value_to_legend,
    _beautify_plot,
    check_data,
)


def hyperparameter_gauge(self, fig, gridspec):
    """Hyperparameter gauge, currently showing the learning rate over time.

    This instrument visualizes the hyperparameters values over the course of the
    training. Currently, it shows the learning rate, the most likely parameter to
    be adapted during training. The current learning rate is additionally shown
    in the figure's legend.

    **Preview**

    .. image:: ../../_static/instrument_previews/Hyperparameters.png
        :alt: Preview Hyperparameter Gauge

    **Requires**

    This instrument requires the learning rate data passed via the
    :func:`cockpit.Cockpit.log()` method.

    Args:
        self (CockpitPlotter): The cockpit plotter requesting this instrument.
        fig (matplotlib.figure.Figure): Figure of the Cockpit.
        gridspec (matplotlib.gridspec.GridSpec): GridSpec where the instrument should be
            placed
    """
    # Plot Trace vs iteration
    title = "Hyperparameters"

    # Check if the required data is available, else skip this instrument
    requires = ["iteration", "learning_rate"]
    plot_possible = check_data(self.tracking_data, requires)
    if not plot_possible:
        if self.debug:
            warnings.warn(
                "Couldn't get the required data for the " + title + " instrument",
                stacklevel=1,
            )
        return

    ax = fig.add_subplot(gridspec)

    clean_learning_rate = self.tracking_data[["iteration", "learning_rate"]].dropna()

    # Plot Settings
    plot_args = {
        "x": "iteration",
        "y": "learning_rate",
        "data": clean_learning_rate,
    }
    ylabel = plot_args["y"].replace("_", " ").title()
    sns.lineplot(
        **plot_args, ax=ax, label=ylabel, linewidth=2, color=self.secondary_color
    )

    _beautify_plot(
        ax=ax,
        xlabel=plot_args["x"],
        ylabel=ylabel,
        x_scale="symlog" if self.show_log_iter else "linear",
        title=title,
        xlim="tight",
        fontweight="bold",
        facecolor=self.bg_color_instruments2,
    )

    ax.legend()
    _add_last_value_to_legend(ax)
