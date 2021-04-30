"""Utility functions for the instruments."""

import warnings

import matplotlib as mpl
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
    marker="o",
    EMA_marker=",",
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
    """Creates a basic plot of x vs. y values for the cockpit.

    Args:
        x (str): Name of the variable in data that should be plotted on the x-axis.
        y (str): Name of the variable in data that should be plotted on the y-axis.
        data (pandas.dataframe): Data Frame containing the plotting data.
        ax (matplotlib.axis): Axis where the plot should be created.
        EMA (str, optional): Signifies over which variables an exponentially
            moving average should be computed.E.g. "xy" would be an exponentially
            moving average over both variables. Defaults to "".
        EMA_alpha (float, optional): Decay parameter of the exponentially moving
            average. Defaults to 0.2.
        x_scale (str, optional): Whether to use a linear or log scale for the x-axis.
            Defaults to "linear".
        y_scale (str, optional): Whether to use a linear or log scale for the y-axis.
            Defaults to "linear".
        cmap (matplotlib.cmap, optional): A colormap for the individual data points.
            Defaults to None.
        EMA_cmap (matplotlib.cmap, optional): A colormap for the EMA.
            Defaults to None.
        marker (str, optional): Marker type to use in the plot. Defaults to "o".
        EMA_marker (str, optional): Marker for the EMA. Defaults to ",".
        xlabel (str, optional): Label for the x-axis. Defaults to None, meaning
            it uses `x`.
        ylabel (str, optional): Label for the y-axis. Defaults to None, meaning
            it uses `y`.
        title (str, optional): Title of this subfigure. Defaults to "".
        xlim (str, list, optional): Limits for the x-axis. Can be a (list of)
            strings, None or numbers. "tight" would shrink the x-limits to the
            data, None would use the default scaling, and float would use this
            limit. If it is given as a list, the first value is used as the lower
            bound and the second one as an upper bound. Defaults to None.
        ylim (str, list, optional): Limits for the y-axis. Can be a (list of)
            strings, None or numbers. "tight" would shrink the y-limits to the
            data, None would use the default scaling, and float would use this
            limit. If it is given as a list, the first value is used as the lower
            bound and the second one as an upper bound. Defaults to None.
        fontweight (str, optional): Fontweight of the title. Defaults to "normal".
        facecolor (tuple, optional): Facecolor of the plot. Defaults to None,
            which does not apply any color.
        zero_lines (bool, optional): Whether to highligh the x and y = 0.
            Defaults to False.
        center (bool, optional): Whether to center the limits of the plot.
            Can also be given as a list, where the first element is applied to
            the x-axis and the second to the y-axis. Defaults to False.
    """
    try:
        sns.scatterplot(
            x=x,
            y=y,
            hue="iteration",
            palette=cmap,
            edgecolor=None,
            marker=marker,
            s=10,
            data=data,
            ax=ax,
        )
    except TypeError:
        sns.scatterplot(
            x=x,
            y=y,
            palette=cmap,
            edgecolor=None,
            marker=marker,
            s=10,
            data=data,
            ax=ax,
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
        try:
            sns.scatterplot(
                x=x,
                y=y,
                hue="iteration",
                palette=EMA_cmap,
                edgecolor=None,
                marker=EMA_marker,
                s=1,
                data=data,
                ax=ax,
            )
        except TypeError:
            sns.scatterplot(
                x=x,
                y=y,
                palette=EMA_cmap,
                edgecolor=None,
                marker=EMA_marker,
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
    ax.set_title(title, fontweight=fontweight, fontsize="large")
    if ax.get_legend() is not None:
        ax.get_legend().remove()

    if x_scale is not None:
        ax.set_xscale(x_scale)
    if y_scale is not None:
        ax.set_yscale(y_scale)

    if xlabel is not None:
        if xlabel == "iteration":
            xlabel = "Iteration"
        ax.set_xlabel(xlabel.replace("_", " "))

    if ylabel is not None:
        ax.set_ylabel(ylabel.replace("_", " "))

    xlim, ylim = _compute_plot_limits(ax, xlim, ylim, center)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    if facecolor is not None:
        ax.set_facecolor(facecolor)

    # Zero lines
    if zero_lines:
        ax.axvline(0, ls="-", color="#ababba", linewidth=1.5, zorder=0)
        ax.axhline(0, ls="-", color="#ababba", linewidth=1.5, zorder=0)


def _compute_plot_limits(ax, xlim, ylim, center=False):
    xlim = _extend_input(xlim)
    ylim = _extend_input(ylim)
    center = _extend_input(center)

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


def _extend_input(shortend_input):
    # extend shortend inputs
    if type(shortend_input) is list:
        pass
    elif shortend_input is None:
        shortend_input = [None, None]
    elif shortend_input == "tight":
        shortend_input = ["tight", "tight"]
    elif shortend_input:
        shortend_input = [True, True]
    else:
        shortend_input = [False, False]

    return shortend_input


def _add_last_value_to_legend(ax, percentage=False):
    """Adds the last value of each line to the legend.

    This function takes every line in a plot, checks its last value and adds it
    in brackets to the corresponding label in the legend.

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


def check_data(data, requires, min_elements=1):
    """Checks if all elements of requires are available in data.

    Args:
        data (pandas.DataFrame): A dataframe holding the data.
        requires ([str]): A list of string that should be part of data.
        min_elements (int, optional): Minimal number of elements required for plotting.
            Defaults to 2. This is in general necessary, so that seaborn can apply
            its colormap.

    Returns:
        bool: Check whether all elements of requires exist in data
    """
    for r in requires:
        # Check fails if element does not exists in the data frame
        if r not in data.columns:
            return False
        # Or if it exists but has not enough elements
        else:
            if len(data[r].dropna()) < min_elements:
                return False

    return True


def _ticks_formatter(ticklabels, format_str="{:.2f}"):
    """Format the ticklabels.

    Args:
        ticklabels ([mpl.text.Text]): List of ticklabels.
        format_str (str, optional): Formatting string for the labels.
            Defaults to "{:.2f}".

    Returns:
        [mpl.text.Text]: Reformatted list of ticklabels.
    """
    new_ticks = []
    for tick in ticklabels:
        rounded_label = format_str.format(float(tick.get_text()))
        new_tick = mpl.text.Text(*tick.get_position(), rounded_label)
        new_ticks.append(new_tick)
    return new_ticks
