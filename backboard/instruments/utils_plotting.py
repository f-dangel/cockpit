"""Utility functions for the CockpitPlotter."""

import warnings

import numpy as np
from matplotlib.colors import LinearSegmentedColormap


def _split_logpath(logpath):
    """Split the logpath to identify test problem, data set, etc.

    Args:
        logpath (str): Full logpath to the JSON file.

    Returns:
        [dict]: Dictioniary of logpath, testproblem, optimizer, etc.
    """
    try:
        dicty = {
            "logpath": logpath + ".json",
            "optimizer": logpath.split("/")[-3],
            "testproblem": logpath.split("/")[-4],
            "dataset": logpath.split("/")[-4].split("_", 1)[0],
            "model": logpath.split("/")[-4].split("_", 1)[1],
        }
    except Exception:
        warnings.warn(
            "Could not extract information about optimizer, dataset and model."
            + "Setting them to unknown."
        )
        dicty = {
            "logpath": logpath + ".json",
            "optimizer": "unknown_optimizer",
            "testproblem": "unknown_testproblem",
            "dataset": "unknown_dataset",
            "model": "unknown_model",
        }
    return dicty


def legend():
    """Creates the legend of the whole cockpit, combining the individual instruments."""
    pass


def _alpha_cmap(color, ncolors=256):
    """Create a Color map that goes from transparant to a given color.

    Args:
        color (tuple): A matplotlib-compatible color.
        ncolors (int, optional): Number of "steps" in the colormap.
            Defaults to 256.

    Returns:
        [matplotlib.cmap]: A matplotlib colormap
    """
    color_array = np.array(ncolors * [list(color)])

    # change alpha values
    color_array[:, -1] = np.linspace(0.0, 1.0, ncolors)
    # create a colormap object
    cmap = LinearSegmentedColormap.from_list(name="alpha_cmap", colors=color_array)

    return cmap
