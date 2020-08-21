"""Utility functions for the CockpitPlotter."""

import warnings


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


def _compute_layers_parts(self):
    """Computes the number of layers and 'parts' of the model.

    self here is an instance of a CockpitPlotter.

    Returns:
        [tuple]: Number of layers and parts
    """
    # hard-coded settings
    tp_model_parts = {"logreg": 2, "3c3d": 2, "2c2d": 4}

    # check number of layers using df0 variable
    layers = len(self.iter_tracking["df0"][0])

    # limit to four parts
    parts = min(layers, 4)

    # if given, overwrite with hard-coded settings
    if self.model in tp_model_parts:
        parts = tp_model_parts[self.model]
    return layers, parts


def _process_tracking_results(self):
    """Process the tracking results.

    self here is an instance of a CockpitPlotter.
    """
    # some util variables for the splitting/aggregation
    layers_per_part = self.layers // self.parts
    rest_ = self.layers % self.parts
    # splits = [layers_per_part + (1 if i < rest_ else 0)
    # for i in range(self.parts)]

    # split part-wise
    # Create new columns for each part
    for (columnName, columnData) in self.iter_tracking.items():
        # We only need to handle data that is non-scalar
        if isinstance(columnData[0], list):
            aggregate = _get_aggregate_function(self, columnName)
            # Create new parts
            for p in range(self.parts):
                start = p * layers_per_part + min(p, rest_)
                end = (p + 1) * layers_per_part + min(p + 1, rest_)
                self.iter_tracking[columnName + "_part_" + str(p)] = [
                    aggregate(x[start:end])
                    for x in self.iter_tracking[columnName].tolist()
                ]
            # Overall average
            self.iter_tracking[columnName] = [
                aggregate(x[:]) for x in self.iter_tracking[columnName].tolist()
            ]

    # Compute avg_ev & avg_cond
    # for that we need the number of parameters, which for now, we hardcode
    num_params = {
        "quadratic_deep": 100,
        "mnist_logreg": 7850,
        "cifar10_3c3d": 895210,
        "fmnist_2c2d": 3274634,
    }
    if self.testproblem in num_params:
        self.iter_tracking["avg_ev"] = (
            self.iter_tracking["trace"] / num_params[self.testproblem]
        )
        self.iter_tracking["avg_cond"] = (
            self.iter_tracking["max_ev"] / self.iter_tracking["avg_ev"]
        )
    else:
        warnings.warn(
            "Warning: Unknown testproblem "
            + self.testproblem
            + ", couldn't compute the average eigenvalue",
            stacklevel=1,
        )


def _get_aggregate_function(self, quantity):
    """Get the corresponding aggregation function for a given quantity.

    Args:
        quantity (str): Name of the quantity we want to aggregate

    Returns:
        [func]: The function that should be used to aggregate this quantity
    """
    if quantity in [
        "df0",
        "df1",
        "var_df0",
        "var_df1",
        "df_var0",
        "df_var1",
        "trace",
    ]:
        return sum
    elif quantity in ["grad_norms", "d2init", "dtravel"]:
        return _root_sum_of_squares
    elif quantity in [
        "norm_test_radius",
        "global_norm_test_radius",
        "inner_product_test_width",
        "global_inner_product_test_width",
        "acute_angle_test_sin",
        "global_acute_angle_test_sin",
        "mean_gsnr",
        "global_mean_gsnr",
    ]:
        warnings.warn(
            "Warning: Don't know how to aggregate "
            + quantity
            + ". Using `sum' for now, but this might be wrong.",
            stacklevel=2,
        )
        return sum
    else:
        warnings.warn("Warning: Don't know how to aggregate " + quantity, stacklevel=2)


def _root_sum_of_squares(list):
    """Returns the root of the sum of squares of a given list.

    Args:
        list (list): A list of floats

    Returns:
        [float]: Root sum of squares
    """
    return sum((el ** 2 for el in list)) ** (0.5)


def legend():
    """Creates the legend of the whole cockpit, combining the individual instruments."""
    pass
