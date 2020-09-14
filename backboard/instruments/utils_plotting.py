"""Utility functions for the CockpitPlotter."""


def _split_logpath(logpath):
    """Split the logpath to identify test problem, data set, etc.

    Args:
        logpath (str): Full logpath to the JSON file.

    Returns:
        [dict]: Dictioniary of logpath, testproblem, optimizer, etc.
    """
    dicty = {
        "logpath": logpath + ".json",
        "optimizer": logpath.split("/")[-3],
        "testproblem": logpath.split("/")[-4],
        "dataset": logpath.split("/")[-4].split("_")[0],
        "model": logpath.split("/")[-4].split("_")[1],
    }

    return dicty


def legend():
    """Creates the legend of the whole cockpit, combining the individual instruments."""
    pass
