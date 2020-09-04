"""Class for tracking the effective relative step size Alpha."""

import itertools
import warnings

import numpy as np

from backboard.quantities.quantity import Quantity
from backboard.quantities.utils_quantities import (
    _layerwise_dot_product,
    _root_sum_of_squares,
    _unreduced_loss_hotfix,
)
from backpack import extensions


class Alpha(Quantity):
    """Alpha Quantity Class."""

    def __init__(self, track_interval=1, verbose=False, check=False):
        super().__init__(track_interval=track_interval, verbose=verbose)

        self._check = check

        if self._check:
            raise NotImplementedError()

    def extensions(self, global_step):
        """Return list of BackPACK extensions required for the computation.

        Args:
            global_step (int): The current iteration number.

        Returns:
            list: (Potentially empty) list with required BackPACK quantities.
        """
        ext = []

        if self._is_start(global_step) or self._is_end(global_step):
            ext.append(extensions.BatchGrad())

        return ext

    def _is_start(self, global_step):
        """Return whether current iteration is the start of a quadratic fit."""
        return global_step % self._track_interval == 0

    def _is_end(self, global_step):
        """Return whether current iteration is the end of a quadratic fit."""
        first_step = global_step == 0

        return global_step % self._track_interval == 1 or first_step

    def compute(self, global_step, params, batch_loss):
        """Evaluate the current parameter distances.

        We track both the distance to the initialization, as well as the size of
        the last parameter update.

        Args:
            global_step (int): The current iteration number.
            params ([torch.Tensor]): List of torch.Tensors holding the network's
                parameters.
            batch_loss (torch.Tensor): Mini-batch loss from current step.
        """
        if self._is_end(global_step):
            self._store_values_at_end(global_step, params, batch_loss)
            self.output[global_step - 1]["alpha"] = self._compute_alpha()

        elif self._is_start(global_step):
            if self._is_end(global_step):
                # values were already computed and stored, they just need to be moved
                self._copy_values_end_to_start()
            else:
                self._store_values_at_start(global_step, params, batch_loss)

    def _store_values_at_start(self, global_step, params, batch_loss):
        """Update values at start point of the fit in ``self``."""
        # Store values and wait for next step #
        # (Variance of) loss
        self.f0 = batch_loss.item()
        self.var_f0 = _unreduced_loss_hotfix(batch_loss).var().item()

        # Store search direction
        # TODO this is currently only correctly implemented for SGD!
        # TODO raise NotImplementedError when optimizer is not SGD with 0 momentum
        self.search_dir = [-g for g in self._fetch_grad(params)]

        # Store current parameters (for update size)
        self.params0 = [p.data.clone().detach() for p in params]

        # (Variance of) projected gradient
        self.df0 = _projected_gradient(self._fetch_grad(params), self.search_dir)
        self.var_df0 = _exact_variance(self._fetch_batch_grad(params), self.search_dir)

    def _store_values_at_end(self, global_step, params, batch_loss):
        """Update values at end point of the fit in ``self``."""
        # Gather necessary values, compute alpha and store it #
        # (Variance of) loss
        self.f1 = batch_loss.item()
        self.var_f1 = _unreduced_loss_hotfix(batch_loss).var().item()

        # (Variance of) projected gradient
        print(global_step)
        self.df1 = _projected_gradient(self._fetch_grad(params), self.search_dir)
        self.var_df1 = _exact_variance(self._fetch_batch_grad(params), self.search_dir)

        self.params1 = self.params0 = [p.data.clone().detach() for p in params]

    def _copy_values_end_to_start(self):
        """Overwrite values at start point with values from end point."""
        self.f0 = self.f1
        self.df0 = self.df1
        self.var_df0 = self.var_df1
        self.params0 = self.params1

    def _compute_alpha(self):
        """Compute and return alpha, assuming all quantities have been stored."""
        # Update size
        t = _root_sum_of_squares(
            [(old_p - p).norm(2).item() for old_p, p in zip(self.params0, self.params1)]
        )

        # Combine
        fs = [self.f0, self.f1]
        dfs = [self.df0, self.df1]
        var_fs = [self.var_f0, self.var_f1]
        var_dfs = [1e-6, 1e-6]

        # Compute alpha
        mu = _fit_quadratic(t, fs, dfs, var_fs, var_dfs)
        alpha = _get_alpha(mu, t)

        if self._verbose:
            print(f"α: {alpha}")
            # TODO α sometimes seems to be None. Then this fails:
            # print(f"α: {alpha:.4f}")

        return alpha


def _projected_gradient(u, v):
    """Computes magnitude of a projection of one vector onto another.

    Args:
        u ([torch.Tensor]): First parameter vector, the vector to be projected.
        v ([torch.Tensor]): Second parameter vector, the direction.

    Returns:
        float: Scalar magnitude of the projection.
    """
    dot_product = sum(_layerwise_dot_product(u, v))
    norm = _root_sum_of_squares([v_el.norm(2).item() for v_el in v])
    return dot_product / norm


def _exact_variance(grads, search_dir):
    """Computes the exact variance of individual projected gradients.

    The gradients are projected onto the search direction before the variance
    is computed.

    Args:
        grads (list): List of individual gradients
        search_dir (list): List of search direction

    Returns:
        list: A list of the variances per layer.
    """
    # swap order of first two axis, so now it is:
    # grads[batch_size, layer, parameters]
    grads = [
        [i for i in element if i is not None]
        for element in list(itertools.zip_longest(*grads))
    ]
    proj_grad = []
    for grad in grads:
        proj_grad.append(_projected_gradient(grad, search_dir))
    return np.var(proj_grad, ddof=1)


def _fit_quadratic(t, fs, dfs, fs_var, dfs_var):
    """Fit a quadratic, given two (noisy) function & gradient observations.

    Args:
        t (float): Position of second observation.
        fs (float): Function values at 0 and t.
        dfs (float): Projected gradients at 0 and t.
        fs_var (list): Variances of function values at 0 and t.
        dfs_var (list): Variances of projected gradients at 0 and t.

    Returns:
        np.array: Parameters of the quadratic fit.
    """
    # The first observation is always at 0, the second at t
    # Observation matrix (f(0), then f(t), then f'(0) and f'(t))
    # The matrix shows how we observe the parameters of the quadratic
    # i.e. for f(0) we see (0)**2 *w1 + (0)*w2 + w3
    # and for f'(t) we see 2*(t)*w1 + w2

    Phi = np.array([[1, 0, 0], [1, t, t ** 2], [0, 1, 0], [0, 1, 2 * t]]).T

    # small value to use if one of the variances is 0.
    eps = 1e-10
    if 0 in fs_var:
        warnings.warn(
            "The variance of f is 0, using a small value instead.", stacklevel=2
        )
        fs_var = list(map(lambda i: eps if i == 0 else i, fs_var))
    if 0 in dfs_var:
        warnings.warn(
            "The variance of df is 0, using a small value instead.", stacklevel=2
        )
        dfs_var = list(map(lambda i: eps if i == 0 else i, dfs_var))

    try:
        # Covariance matrix
        lambda_inv = np.linalg.inv(np.diag(fs_var + dfs_var))
        # Maximum Likelihood Estimation
        mu = np.linalg.inv(Phi @ lambda_inv @ Phi.T) @ Phi @ lambda_inv @ (fs + dfs)
    except np.linalg.LinAlgError:
        warnings.warn(
            "Couldn't compute a quadratic fit due to Singular matrix, returning None",
            stacklevel=2,
        )
        mu = None

    return mu


def _get_alpha(mu, t):
    """Compute the local step size alpha.

    It will be expressed in terms of the local quadratic fit. A local step size
    of 1, is equal to `stepping on the other side of the quadratic`, while a
    step size of 0 means `stepping to the minimum`.

    If no quadratic approximation could be computed, return None.

    Args:
        mu (list): Parameters of the quadratic fit.
        t (float): Step size (taken).

    Returns:
        float: Local effective step size.
    """
    # If we couldn't compute a quadratic approx., return None.
    if mu is None:
        return None
    elif mu[2] < 0:
        # Concave setting: Since this means that it is still going downhill
        # where we stepped to, we will log it as -1.
        return -1
    else:
        # get alpha_bar (the step size that is "the other side")
        alpha_bar = -mu[1] / mu[2]

        # scale everything on a standard quadratic (i.e. step of 0 = minimum)
        return (2 * t) / alpha_bar - 1
