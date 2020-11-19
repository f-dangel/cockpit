"""Utility functions for the paper experiments."""

import matplotlib.pyplot as plt
import seaborn as sns

import backobs
import deepobs


def register(cls, has_accuracy=True):
    """Register a new testproblem class in DeepOBS and BackOBS.

    It is assumed that the testproblem is supported by BackOBS.
    """
    dataset_name, net_name = cls.__name__.split("_")

    # hotfix
    dataset_name = dataset_name.replace("cifar10", "CIFAR-10")

    # DeepOBS
    setattr(deepobs.pytorch.testproblems, cls.__name__, cls)

    # for CockpitPlotter
    if dataset_name in deepobs.config.DATA_SET_NAMING.keys():
        if not deepobs.config.DATA_SET_NAMING[dataset_name] == dataset_name:
            raise ValueError(
                f"{deepobs.config.DATA_SET_NAMING[dataset_name]} != {dataset_name}"
            )
    else:
        deepobs.config.DATA_SET_NAMING[dataset_name] = dataset_name

    if net_name in deepobs.config.TP_NAMING.keys():
        assert deepobs.config.TP_NAMING[net_name] == net_name

    else:
        deepobs.config.TP_NAMING[net_name] = net_name

    # BackOBS
    backobs.utils.ALL += (cls,)
    backobs.utils.SUPPORTED += (cls,)
    backobs.integration.SUPPORTED += (cls,)
    if not has_accuracy:
        raise NotImplementedError()


def replace(module, trigger, make_new):
    """Check if layer should be replaced by calling trigger(m) → bool.

    If True, replace m with make_new(m)
    """

    def has_children(module):
        return bool(list(module.children()))

    for name, mod in module.named_children():
        if has_children(mod):
            replace(mod, trigger, make_new)
        else:
            if trigger(mod):
                new_mod = make_new(mod)
                setattr(module, name, new_mod)


def _set_plotting_params():
    """Set some consistent plotting settings and styles."""
    # Seaborn settings:
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.0)

    # Matplotlib settings (using tex font)
    tex_fonts = {
        # Use LaTeX to write all text
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": [
            "Times New Roman",
            "Times",
        ],  # Use Times New Roman, and Times as a back up
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 8,
        "font.size": 8,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        # Fix for missing \mathdefault command
        "text.latex.preamble": r"\newcommand{\Mathdefault}[1][]{}",
        # Less space between label and axis
        "xtick.major.pad": 0.0,
        "ytick.major.pad": 0.0,
        # More space between subplots
        "figure.subplot.hspace": 0.3,
        # Less space around the plot
        "savefig.pad_inches": 0.0,
        # dashed grid lines
        "grid.linestyle": "dashed",
        # width of grid lines
        "grid.linewidth": 0.4,
        # Show thin edge around each plot
        "axes.edgecolor": "black",
        "axes.linewidth": 0.4,
    }
    plt.rcParams.update(tex_fonts)


def _get_plot_size(
    textwidth="cvpr", fraction=1.0, height_ratio=(5 ** 0.5 - 1) / 2, subplots=(1, 1)
):
    r"""Returns the matplotlib plot size to fit with the LaTeX textwidth.

    Args:
        textwidth (float or string, optional): LaTeX textwidth in pt. Can be accessed
            directly in LaTeX via `\the\textwidth` and needs to be replaced
            accoring to the used template. Defaults to "cvpr", which automatically uses
            the size of the CVPR template.
        fraction (float, optional): Fraction of the textwidth the plot should occupy.
            Defaults to 1.0 meaning a full width figure.
        height_ratio (float, optional): Ratio of the height to width. A value of
            0.5 would result in a figure that is half as high as it is wide.
            Defaults to the golde ratio.
        subpllots (array-like, optional): The number of rows and columns of subplots.
            Defaults to (1,1)

    Returns:
        [float]: Desired height and width of the matplotlib figure.
    """
    if textwidth == "cvpr":
        width = 496.85625
    elif textwidth == "cvpr_col":
        width = 237.13594
    else:
        width = textwidth

    # Width of figure (in pts)
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Figure width in inches
    fig_width = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height = fig_width * height_ratio * (subplots[0] / subplots[1])

    return fig_width, fig_height
