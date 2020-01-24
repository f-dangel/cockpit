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


def _fit_quadratic(fs, dfs, fs_var, dfs_var):
    # We always assume that the observations will be at 0 and 1
    # Observation matrix (f(0), then f(1), then f'(0) and f'(1))
    # The matrix shows how we observe the parameters of the quadratic
    # i.e. for f(0) we see (0)**2 *w1 + (0)*w2 + w3
    # and for f'(1) we see 2*(1)*w1 + w2
    Phi = np.array([[1, 0, 0], [1, 1, 1], [0, 1, 0], [0, 1, 2]]).T

    # Covariance matrix
    lambda_inv = np.linalg.inv(np.diag(fs_var + dfs_var))

    # Maximum Likelihood Estimation
    return np.linalg.inv(Phi @ lambda_inv @ Phi.T) @ Phi @ lambda_inv @ (fs + dfs)
