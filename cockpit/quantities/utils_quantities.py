"""Utility Functions for the Quantities and the Tracking in General."""

import torch


def _layerwise_dot_product(x_s, y_s):
    """Computes the dot product of two parameter vectors layerwise.

    Args:
        x_s (list): First list of parameter vectors.
        y_s (list): Second list of parameter vectors.

    Returns:
        torch.Tensor: 1-D list of scalars. Each scalar is a dot product of one layer.
    """
    return [torch.sum(x * y).item() for x, y in zip(x_s, y_s)]


def _root_sum_of_squares(list):
    """Returns the root of the sum of squares of a given list.

    Args:
        list (list): A list of floats

    Returns:
        [float]: Root sum of squares
    """
    return sum((el ** 2 for el in list)) ** (0.5)


def abs_max(tensor):
    """Return maximum absolute entry in ``tensor``."""
    min_val, max_val = tensor.min(), tensor.max()
    return max(min_val.abs(), max_val.abs())
