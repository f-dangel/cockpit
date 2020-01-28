"""Utility functions for the Cockpit. """

import numpy as np
import torch


def _layerwise_dot_product(x_s, y_s):
    """Computes the dot product of two parameter vectors layerwise.

    Args:
        x_s (list): First list of parameter vectors.
        y_s (list): Second list of parameter vectors.

    Returns:
        prod: 1-D list of scalars. Each scalar is a dot product of one layer.
    """
    return [torch.sum(x * y).item() for x, y in zip(x_s, y_s)]


def _fit_quadratic(t, fs, dfs, fs_var, dfs_var):
    # The first observation is always at 0, the second at t
    # Observation matrix (f(0), then f(t), then f'(0) and f'(t))
    # The matrix shows how we observe the parameters of the quadratic
    # i.e. for f(0) we see (0)**2 *w1 + (0)*w2 + w3
    # and for f'(t) we see 2*(t)*w1 + w2

    Phi = np.array([[1, 0, 0], [1, t, t ** 2], [0, 1, 0], [0, 1, 2 * t]]).T

    # Covariance matrix
    lambda_inv = np.linalg.inv(np.diag(fs_var + dfs_var))

    # Maximum Likelihood Estimation
    return np.linalg.inv(Phi @ lambda_inv @ Phi.T) @ Phi @ lambda_inv @ (fs + dfs)


def _get_alpha(mu, t):
    """Compute the local step size alpha.

    It will be expressed in terms of the local quadratic fit. A local step size
    of 1, is equal to `stepping on the other side of the quadratic`, while a 
    step size of 0 means `stepping to the minimum`.

    Args:
        mu (list): Parameters of the quadratic fit.
    """
    # get alpha_bar (the step size that is "the other side")
    alpha_bar = -mu[1] / mu[2]

    # scale everything on a standard quadratic (i.e. step of 0 = minimum)
    return (2 * t) / alpha_bar - 1
