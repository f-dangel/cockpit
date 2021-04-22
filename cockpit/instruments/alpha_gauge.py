"""Alpha Gauge."""

import math
import warnings

import numpy as np
import seaborn as sns
from scipy import stats

from cockpit.instruments.utils_instruments import _beautify_plot, check_data


def alpha_gauge(self, fig, gridspec):
    r"""Showing a distribution of the alpha values.

    This alpha instruments provides a summary of the alpha values of all tracked
    iterations (shown in gray) as well as the last 10% of them (shown in blue).

    The alpha value uses loss and gradient information to build a noise-informed
    univariate quadratic approximation of the loss function to assess to which
    point on this parabola the optimizer moves. The parabola (shown in orange) is
    always normalized such that the starting point of each iteration is at
    :math:`x=-1`. If the optimzer takes a single step to the local minimum (e.g.
    the valley floor) it would be indicated by an :math:`\alpha` value of :math:`0`.
    Analogously, taking a step to the exact opposite side of the valley, will be
    descrbied by :math:`\alpha=1`.

    The instruments shows a histogram and a distribution fit of all alpha values,
    as well as the last 10% of tracked iterations. The mean values of both distributions
    are also shown in the figure's legend.

    For a more detailed explanation of the alpha value and why - perhaps against
    intuition - values larger than zero might be desirable, have a look at the
    Cockpit paper:

    - `Schneider, F., Dangel, F., & Hennig, P.,
      Cockpit: A Practical Debugging Tool for Training Deep Neural Networks (2021).
      <https://arxiv.org/abs/2102.06604>`_

    .. image:: ../../_static/instrument_previews/Alpha.png
        :alt: Preview Alpha Gauge

    **Requires**

    This instrument requires data from the :class:`~cockpit.quantities.Alpha`
    quantity class.

    Args:
        self (CockpitPlotter): The cockpit plotter requesting this instrument.
        fig (matplotlib.figure.Figure): Figure of the Cockpit.
        gridspec (matplotlib.gridspec.GridSpec): GridSpec where the instrument
            should be placed.
    """
    # Plot Alpha Distribution
    title = "Alpha Distribution"

    # Check if the required data is available, else skip this instrument
    requires = ["Alpha"]
    plot_possible = check_data(self.tracking_data, requires, min_elements=2)
    if not plot_possible:
        if self.debug:
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
    sns.histplot(
        self.tracking_data["Alpha"].dropna(),
        ax=ax2,
        kde=True,
        color=color_all,
        kde_kws={"cut": 10},
        alpha=0.5,
        stat="probability",
        label="all",
    )
    (mu_all, _) = stats.norm.fit(self.tracking_data["Alpha"].dropna())
    # Last 10% alphas
    len_last_elements = int(len(self.tracking_data["Alpha"]) / 10)
    sns.histplot(
        self.tracking_data["Alpha"].dropna().tail(len_last_elements),
        ax=ax2,
        kde=True,
        color=color_last,
        kde_kws={"cut": 10},
        alpha=0.5,
        stat="probability",
        label="last 10 %",
    )
    if len_last_elements == 0:
        mu_last = math.nan
    else:
        (mu_last, _) = stats.norm.fit(
            self.tracking_data["Alpha"].dropna().tail(len_last_elements)
        )

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
    _add_indicators(
        self, ax, mu_last, plot_args, color_all, color_last, len_last_elements
    )

    # Legend
    # Get the fitted parameters used by sns
    lines2, labels2 = ax2.get_legend_handles_labels()
    for idx, lab in enumerate(labels2):
        if "all" in lab and not math.isnan(mu_all):
            labels2[idx] = lab + " ($\mu=${0:.2f})".format(mu_all)  # noqa: W605
        if "last 10 %" in lab and not math.isnan(mu_last):
            labels2[idx] = lab + " ($\mu=${0:.2f})".format(mu_last)  # noqa: W605
    ax2.legend(lines2, labels2)


def _add_indicators(
    self, ax, mu_last, plot_args, color_all, color_last, len_last_elements
):
    """Adds indicators that some alpha values were outside of ploting range."""
    # Add indicator for outliers
    if (
        not math.isnan(mu_last)
        and max(self.tracking_data["Alpha"].dropna().tail(len_last_elements))
        > plot_args["xlim"][1]
    ):
        ax.annotate(
            "",
            xy=(1.8, 0.3),
            xytext=(1.7, 0.3),
            size=20,
            arrowprops=dict(color=color_last),
        )
    elif max(self.tracking_data["Alpha"]) > plot_args["xlim"][1]:
        ax.annotate(
            "",
            xy=(1.8, 0.3),
            xytext=(1.7, 0.3),
            size=20,
            arrowprops=dict(color=color_all),
        )
    if (
        not math.isnan(mu_last)
        and min(self.tracking_data["Alpha"].dropna().tail(len_last_elements))
        < plot_args["xlim"][0]
    ):
        ax.annotate(
            "",
            xy=(-1.8, 0.3),
            xytext=(-1.7, 0.3),
            size=20,
            arrowprops=dict(color=color_last),
        )
    elif min(self.tracking_data["Alpha"]) < plot_args["xlim"][0]:
        ax.annotate(
            "",
            xy=(-1.8, 0.3),
            xytext=(-1.7, 0.3),
            size=20,
            arrowprops=dict(color=color_all),
        )
