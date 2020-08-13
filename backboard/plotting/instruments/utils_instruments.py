"""Utility functions for the instruments."""

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
    title="",
):
    sns.scatterplot(
        x=x,
        y=y,
        hue="iteration",
        palette=cmap,
        edgecolor=None,
        s=10,
        data=data,
        ax=ax,
    )

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

    beautify_plot(ax=ax, title=title)


def beautify_plot(ax, title=""):
    ax.set_title(title)
    ax.get_legend().remove()
