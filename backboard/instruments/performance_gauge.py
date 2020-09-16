"""Performance Gauge."""

import warnings

import seaborn as sns

from backboard.instruments.utils_instruments import (
    _add_last_value_to_legend,
    check_data,
    create_basic_plot,
)


def performance_gauge(self, fig, gridspec):
    """Plotting train/valid accuracy vs. epoch and mini-batch loss vs. iteration.

    Args:
        self (cockpit.plotter): The cockpit plotter requesting this instrument.
        fig (matplotlib.figure): Figure of the Cockpit.
        gridspec (matplotlib.gridspec): GridSpec where the instrument should be
            placed
    """
    # Plot Trace vs iteration
    title = "Performance Plot"

    # Check if the required data is available, else skip this instrument
    requires = ["iteration", "train_accuracy", "valid_accuracy", "mini_batch_loss"]
    plot_possible = check_data(self.tracking_data, requires)
    if not plot_possible:
        warnings.warn(
            "Couldn't get the required data for the " + title + " instrument",
            stacklevel=1,
        )
        return

    # Mini-batch train loss
    plot_args = {
        "x": "iteration",
        "y": "mini_batch_loss",
        "data": self.tracking_data,
        "EMA": "y",
        "EMA_alpha": self.EMA_alpha,
        "EMA_cmap": self.cmap2,
        "x_scale": "linear",
        "y_scale": "linear",
        "cmap": self.cmap,
        "title": title,
        "xlim": "tight",
        "ylim": None,
        "fontweight": "bold",
        "facecolor": "summary",
    }
    ax = fig.add_subplot(gridspec)
    create_basic_plot(**plot_args, ax=ax)

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
