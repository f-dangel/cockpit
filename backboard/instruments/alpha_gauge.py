"""Alpha Gauge."""

import numpy as np
import seaborn as sns
from scipy import stats

from .utils_instruments import _beautify_plot


def alpha_gauge(self, fig, gridspec):
    """Showing a distribution of the alpha values since the last plot.

    Args:
        self (cockpit.plotter): The cockpit plotter requesting this instrument.
        fig (matplotlib.figure): Figure of the Cockpit.
        gridspec (matplotlib.gridspec): GridSpec where the instrument should be
            placed.
    """
    plot_args = {
        "xlabel": "Local Step Length",
        "ylabel": "Stand. Loss",
        "title": "Alpha Distribution",
        "xlim": [-1.5, 1.5],
        "ylim": [0, 1.75],
        "fontweight": "bold",
        "facecolor": "summary",
        "zero_lines": True,
        "center": [True, False],
    }
    color_all = "gray"
    color_last = sns.color_palette("muted")[1]
    color_parabola = sns.color_palette("muted")[0]

    ax = fig.add_subplot(gridspec)

    # Plot unit parabola
    x = np.linspace(plot_args["xlim"][0], plot_args["xlim"][1], 100)
    y = x ** 2
    ax.plot(x, y, linewidth=2, color=color_parabola)

    _beautify_plot(**plot_args, ax=ax)

    # Alpha Histogram
    ax2 = ax.twinx()
    # All alphas
    try:
        sns.distplot(
            self.iter_tracking["alpha"],
            ax=ax2,
            # norm_hist=True,
            fit=stats.norm,
            kde=False,
            color=color_all,
            fit_kws={"color": color_all},
            hist_kws={"linewidth": 0, "alpha": 0.25},
            label="all",
        )
        (mu_all, _) = stats.norm.fit(self.iter_tracking["alpha"])
    except ValueError:
        print("Alphas included NaN and could therefore not be plotted.")
        mu_all = None
    # Last 10% alphas
    try:
        len_last_elements = int(len(self.iter_tracking["alpha"]) / 10)
        sns.distplot(
            self.iter_tracking["alpha"][-len_last_elements:],
            ax=ax2,
            # norm_hist=True,
            fit=stats.norm,
            kde=False,
            color=color_last,
            fit_kws={"color": color_last},
            hist_kws={"linewidth": 0, "alpha": 0.65},
            label="last 10 %",
        )
        (mu_last, _) = stats.norm.fit(self.iter_tracking["alpha"][-len_last_elements:])
    except ValueError:
        print("Alphas included NaN and could therefore not be plotted.")
        mu_last = None

    # Manually beautify the plot:
    # Adding Zone Lines
    ax.axvline(0, ls="-", color="white", linewidth=1.5, zorder=0)
    ax.axvline(-1, ls="-", color="white", linewidth=1.5, zorder=0)
    ax.axvline(1, ls="-", color="white", linewidth=1.5, zorder=0)
    ax.axhline(0, ls="-", color="white", linewidth=1.5, zorder=0)
    ax.axhline(1, ls="-", color="gray", linewidth=0.5, zorder=0)
    # Labels
    ax.set_xlabel(r"Local step length $\alpha$")
    ax2.set_ylabel(r"$\alpha$ density")
    # Add indicator for outliers
    if max(self.iter_tracking["alpha"][-len_last_elements:]) > plot_args["xlim"][1]:
        ax.annotate(
            "",
            xy=(1.8, 0.3),
            xytext=(1.7, 0.3),
            size=20,
            arrowprops=dict(color=color_last),
        )
    elif max(self.iter_tracking["alpha"]) > plot_args["xlim"][1]:
        ax.annotate(
            "",
            xy=(1.8, 0.3),
            xytext=(1.7, 0.3),
            size=20,
            arrowprops=dict(color=color_all),
        )
    if min(self.iter_tracking["alpha"][-len_last_elements:]) < plot_args["xlim"][0]:
        ax.annotate(
            "",
            xy=(-1.8, 0.3),
            xytext=(-1.7, 0.3),
            size=20,
            arrowprops=dict(color=color_last),
        )
    elif min(self.iter_tracking["alpha"]) < plot_args["xlim"][0]:
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
    ax2.legend(
        [
            "{0} ($\mu=${1:.2f})".format(labels2[0], mu_all),  # noqa: W605
            "{0} ($\mu=${1:.2f})".format(labels2[1], mu_last),  # noqa: W605
        ]
    )
