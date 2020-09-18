"""Alpha Gauge."""

import warnings

import numpy as np
import seaborn as sns
from scipy import stats

from backboard.instruments.utils_instruments import _beautify_plot, check_data


def alpha_gauge(self, fig, gridspec):
    """Showing a distribution of the alpha values since the last plot.

    Args:
        self (cockpit.plotter): The cockpit plotter requesting this instrument.
        fig (matplotlib.figure): Figure of the Cockpit.
        gridspec (matplotlib.gridspec): GridSpec where the instrument should be
            placed.
    """
    # Plot Alpha Distribution
    title = "Alpha Distribution"

    # Check if the required data is available, else skip this instrument
    requires = ["alpha"]
    plot_possible = check_data(self.tracking_data, requires)
    if not plot_possible:
        warnings.warn(
            "Couldn't get the required data for the " + title + " instrument",
            stacklevel=1,
        )
        return

    plot_args = {
        "xlabel": "Local Step Length",
        "ylabel": "Stand. Loss",
        "title": title,
        "xlim": [-1.5, 1.5],
        "ylim": [0, 1.75],
        "fontweight": "bold",
        "facecolor": self.bg_color_instruments,
        "zero_lines": True,
        "center": [True, False],
    }
    color_all = "gray"
    color_last = self.primary_color
    color_parabola = self.secondary_color

    ax = fig.add_subplot(gridspec)

    # Plot unit parabola
    x = np.linspace(plot_args["xlim"][0], plot_args["xlim"][1], 100)
    y = x ** 2
    ax.plot(x, y, linewidth=2, color=color_parabola)

    _beautify_plot(**plot_args, ax=ax)

    # Alpha Histogram
    ax2 = ax.twinx()
    # All alphas
    sns.distplot(
        self.tracking_data["alpha"].dropna(),
        ax=ax2,
        # norm_hist=True,
        fit=stats.norm,
        kde=False,
        color=color_all,
        fit_kws={"color": color_all},
        hist_kws={"linewidth": 0, "alpha": 0.5},
        label="all",
    )
    (mu_all, _) = stats.norm.fit(self.tracking_data["alpha"].dropna())
    # Last 10% alphas
    len_last_elements = int(len(self.tracking_data["alpha"]) / 10)
    try:
        sns.distplot(
            self.tracking_data["alpha"][-len_last_elements:].dropna(),
            ax=ax2,
            # norm_hist=True,
            fit=stats.norm,
            kde=False,
            color=color_last,
            fit_kws={"color": color_last},
            hist_kws={"linewidth": 0, "alpha": 0.85},
            label="last 10 %",
        )
        (mu_last, _) = stats.norm.fit(
            self.tracking_data["alpha"][-len_last_elements:].dropna()
        )
    except ValueError:
        mu_last = None

    # Manually beautify the plot:
    # Adding Zone Lines
    ax.axvline(0, ls="-", color="#ababba", linewidth=1.5, zorder=0)
    ax.axvline(-1, ls="-", color="#ababba", linewidth=1.5, zorder=0)
    ax.axvline(1, ls="-", color="#ababba", linewidth=1.5, zorder=0)
    ax.axhline(0, ls="-", color="#ababba", linewidth=1.5, zorder=0)
    ax.axhline(1, ls="-", color="#ababba", linewidth=0.5, zorder=0)
    # Labels
    ax.set_xlabel(r"Local step length $\alpha$")
    ax2.set_ylabel(r"$\alpha$ density")
    # Add indicator for outliers
    if max(self.tracking_data["alpha"][-len_last_elements:]) > plot_args["xlim"][1]:
        ax.annotate(
            "",
            xy=(1.8, 0.3),
            xytext=(1.7, 0.3),
            size=20,
            arrowprops=dict(color=color_last),
        )
    elif max(self.tracking_data["alpha"]) > plot_args["xlim"][1]:
        ax.annotate(
            "",
            xy=(1.8, 0.3),
            xytext=(1.7, 0.3),
            size=20,
            arrowprops=dict(color=color_all),
        )
    if min(self.tracking_data["alpha"][-len_last_elements:]) < plot_args["xlim"][0]:
        ax.annotate(
            "",
            xy=(-1.8, 0.3),
            xytext=(-1.7, 0.3),
            size=20,
            arrowprops=dict(color=color_last),
        )
    elif min(self.tracking_data["alpha"]) < plot_args["xlim"][0]:
        ax.annotate(
            "",
            xy=(-1.8, 0.3),
            xytext=(-1.7, 0.3),
            size=20,
            arrowprops=dict(color=color_all),
        )
    # Legend
    # Get the fitted parameters used by sns
    lines2, labels2 = ax2.get_legend_handles_labels()
    legend = []
    if mu_all is not None:
        legend.append("{0} ($\mu=${1:.2f})".format(labels2[0], mu_all))  # noqa: W605
    if mu_last is not None:
        legend.append("{0} ($\mu=${1:.2f})".format(labels2[1], mu_last))  # noqa: W605
    ax2.legend(legend)
