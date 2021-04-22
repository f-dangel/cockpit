"""Performance Gauge."""

import warnings

import seaborn as sns

from cockpit.instruments.utils_instruments import (
    _add_last_value_to_legend,
    check_data,
    create_basic_plot,
)


def performance_gauge(self, fig, gridspec):
    """Plotting train/valid accuracy vs. epoch and mini-batch loss vs. iteration.

    This instruments visualizes the currently most popular diagnostic metrics. It
    shows the mini-batch loss in each iteration (overlayed with an exponentially
    weighted average) as well as accuracies for both the training as well as the
    validation set. The current accuracy numbers are also shown in the legend.

    **Preview**

    .. image:: ../../_static/instrument_previews/Performance.png
        :alt: Preview Performance Gauge

    **Requires**

    This instrument visualizes quantities passed via the
    :func:`cockpit.Cockpit.log()` method.

    Args:
        self (CockpitPlotter): The cockpit plotter requesting this instrument.
        fig (matplotlib.figure.Figure): Figure of the Cockpit.
        gridspec (matplotlib.gridspec.GridSpec): GridSpec where the instrument should be
            placed
    """
    # Plot Trace vs iteration
    title = "Performance Plot"

    # Check if the required data is available, else skip this instrument
    requires = ["iteration", "Loss"]
    plot_possible = check_data(self.tracking_data, requires)
    if not plot_possible:
        if self.debug:
            warnings.warn(
                "Couldn't get the loss data for the " + title + " instrument",
                stacklevel=1,
            )
        return

    # Mini-batch train loss
    plot_args = {
        "x": "iteration",
        "y": "Loss",
        "data": self.tracking_data,
        "EMA": "y",
        "EMA_alpha": self.EMA_alpha,
        "EMA_cmap": self.cmap2,
        "x_scale": "symlog" if self.show_log_iter else "linear",
        "y_scale": "linear",
        "cmap": self.cmap,
        "title": title,
        "xlim": "tight",
        "ylim": None,
        "fontweight": "bold",
        "facecolor": self.bg_color_instruments2,
    }
    ax = fig.add_subplot(gridspec)
    create_basic_plot(**plot_args, ax=ax)

    requires = ["iteration", "train_accuracy", "valid_accuracy"]
    plot_possible = check_data(self.tracking_data, requires)
    if not plot_possible:
        if self.debug:
            warnings.warn(
                "Couldn't get the accuracy data for the " + title + " instrument",
                stacklevel=1,
            )
        return
    else:
        clean_accuracies = self.tracking_data[
            ["iteration", "train_accuracy", "valid_accuracy"]
        ].dropna()

        # Train Accuracy
        plot_args = {
            "x": "iteration",
            "y": "train_accuracy",
            "data": clean_accuracies,
        }
        ax2 = ax.twinx()
        sns.lineplot(
            **plot_args,
            ax=ax2,
            label=plot_args["y"].title().replace("_", " "),
            linewidth=2,
            color=self.primary_color,
        )

        # Train Accuracy
        plot_args = {
            "x": "iteration",
            "y": "valid_accuracy",
            "data": clean_accuracies,
        }
        sns.lineplot(
            **plot_args,
            ax=ax2,
            label=plot_args["y"].title().replace("_", " "),
            linewidth=2,
            color=self.secondary_color,
        )

        # Customization
        ax2.set_ylim([0, 1])
        ax2.set_ylabel("Accuracy")
        _add_last_value_to_legend(ax2, percentage=True)
