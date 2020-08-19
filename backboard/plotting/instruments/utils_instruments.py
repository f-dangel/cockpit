"""Utility functions for the instruments."""

import warnings

import seaborn as sns


def create_basic_plot(
    x,
    y,
    data,
    ax,
    EMA="",
    EMA_alpha=0.2,
    x_scale="linear",
    y_scale="linear",
    cmap=None,
    EMA_cmap=None,
    xlabel=None,
    ylabel=None,
    title="",
    xlim=None,
    ylim=None,
    fontweight="normal",
    facecolor=None,
    zero_lines=False,
    center=False,
):
    sns.scatterplot(
        x=x, y=y, hue="iteration", palette=cmap, edgecolor=None, s=10, data=data, ax=ax,
    )

    # Save what is being ploted as labels, if not otherwise given
    xlabel = x if xlabel is None else xlabel
    ylabel = y if ylabel is None else ylabel

    if "y" in EMA:
        data["EMA_" + y] = data[y].ewm(alpha=EMA_alpha, adjust=False).mean()
        y = "EMA_" + y
    if "x" in EMA:
        data["EMA_" + x] = data[x].ewm(alpha=EMA_alpha, adjust=False).mean()
        x = "EMA_" + x
    if EMA != "":
        sns.scatterplot(
            x=x,
            y=y,
            hue="iteration",
            palette=EMA_cmap,
            edgecolor=None,
            marker=",",
            s=1,
            data=data,
            ax=ax,
        )

    _beautify_plot(
        ax=ax,
        x_scale=x_scale,
        y_scale=y_scale,
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        xlim=xlim,
        ylim=ylim,
        fontweight=fontweight,
        facecolor=facecolor,
        zero_lines=zero_lines,
        center=center,
    )


def _beautify_plot(
    ax,
    x_scale=None,
    y_scale=None,
    xlabel=None,
    ylabel=None,
    title="",
    xlim=None,
    ylim=None,
    fontweight="normal",
    facecolor=None,
    zero_lines=False,
    center=False,
):
    # Settings
    color_summary_plots = "#ababba"

    ax.set_title(title, fontweight=fontweight, fontsize="large")
    ax.get_legend().remove()

    if x_scale is not None:
        ax.set_xscale(x_scale)
    if y_scale is not None:
        ax.set_yscale(y_scale)

    if xlabel is not None:
        ax.set_xlabel(xlabel.title().replace("_", " "))
    if ylabel is not None:
        ax.set_ylabel(ylabel.title().replace("_", " "))

    xlim, ylim = _compute_plot_limits(ax, xlim, ylim, center)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    if facecolor == "summary":
        ax.set_facecolor(color_summary_plots)

    # Zero lines
    if zero_lines:
        ax.axvline(0, ls="-", color="white", linewidth=1.5, zorder=0)
        ax.axhline(0, ls="-", color="white", linewidth=1.5, zorder=0)


def _compute_plot_limits(ax, xlim, ylim, center=False):
    # extend shortend inputs
    if xlim is None:
        xlim = [None, None]
    elif xlim == "tight":
        xlim = ["tight", "tight"]
    if ylim is None:
        ylim = [None, None]
    elif ylim == "tight":
        ylim = ["tight", "tight"]
    if type(center) is list:
        pass
    elif center:
        center = [True, True]
    else:
        center = [False, False]

    lims = [xlim, ylim]
    auto_lims = [ax.get_xlim(), ax.get_ylim()]
    ax.autoscale(enable=True, tight=True)
    tight_limts = [ax.get_xlim(), ax.get_ylim()]

    # replace values according to inputs
    for lim in range(len(lims)):
        for direction in range(len(lims[lim])):
            if lims[lim][direction] is None:
                lims[lim][direction] = auto_lims[lim][direction]
            elif lims[lim][direction] == "tight":
                lims[lim][direction] = tight_limts[lim][direction]
            elif (
                type(lims[lim][direction]) == float or type(lims[lim][direction]) == int
            ):
                pass
            else:
                warnings.warn(
                    "Unknown input for limits, it is neither None, nor tight,"
                    "nor a float ",
                    stacklevel=1,
                )

    if center[0]:
        lims[0][1] = abs(max(lims[0], key=abs))
        lims[0][0] = -lims[0][1]
    if center[1]:
        lims[1][1] = abs(max(lims[1], key=abs))
        lims[1][0] = -lims[1][1]

    return lims[0], lims[1]


def _add_last_value_to_legend(ax, percentage=False):
    """Adds the last value of each line of a plot as a numeric value to its
    entry in the legend.

    Args:
        ax (matplotlib.axes): Axis of a matplotlib figure
        percentage (bool): Whether the value represents a percentage
    """
    # Formating
    if percentage:
        formating = "{0}: ({1:.2%})"
    else:
        formating = "{0}: ({1:.1E})"
    # Fix Legend
    lines, labels = ax.get_legend_handles_labels()
    plot_labels = []
    for line, label in zip(lines, labels):
        plot_labels.append(formating.format(label, line.get_ydata()[-1]))
    ax.get_legend().remove()
    ax.legend(lines, plot_labels)
