"""Utility Functions for the Quantities and the Tracking in General."""

import torch


def _update_dicts(master_dict, update_dict):
    """Merge dicts of dicts by updating dict a with dict b.

    Args:
        master_dict (dict): [description]
        update_dict (dict): [description]
    """
    for key, value in update_dict.items():
        for subkey, subvalue in value.items():
            master_dict[key][subkey] = subvalue


def _unreduced_loss_hotfix(batch_loss):
    """A simple hotfix for the unreduced loss values.

    For the quadratic_deep problem of DeepOBS, the unreduced losses are a matrix
    and should be averaged over the second axis.

    Args:
        batch_loss (torch.Tensor): Mini-batch loss from current step, with the
            unreduced losses as an attribute.

    Returns:
        torch.Tensor: (Averaged) unreduced losses.
    """
    batch_losses = batch_loss._unreduced_loss
    if len(batch_losses.shape) == 2:
        batch_losses = batch_losses.mean(1)
    return batch_losses


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
