"""Class for tracking the effective relative step size Alpha."""

import itertools
import warnings

import numpy as np
import torch
from backpack import extensions

from cockpit.context import get_batch_size, get_individual_losses
from cockpit.quantities.quantity import Quantity, TwoStepQuantity
from cockpit.quantities.utils_quantities import (
    _layerwise_dot_product,
    _root_sum_of_squares,
)


class AlphaTwoStep(TwoStepQuantity):
    """Base class for α computation based on ``TwoStepQuantity``.

    Attributes:
        SAVE_SHIFT (int): Difference between iteration at which information is computed
            versus iteration under which it is stored. For instance, if set to ``1``,
            the information computed at iteration ``n + 1`` is saved under iteration
            ``n``. Default: ``1``.
    """

    SAVE_SHIFT = 1

    def extensions(self, global_step):
        """Return list of BackPACK extensions required for the computation.

        Args:
            global_step (int): The current iteration number.

        Returns:
            list: (Potentially empty) list with required BackPACK quantities.
        """
        ext = []

        if self.is_start(global_step) or self.is_end(global_step):
            ext.append(extensions.BatchGrad())

        return ext

    def is_start(self, global_step):
        """Return whether current iteration is start point.

        Args:
            global_step (int): The current iteration number.

        Returns:
            bool: Whether ``global_step`` is a start point.
        """
        return self._track_schedule(global_step)

    def is_end(self, global_step):
        """Return whether current iteration is end point.

        Args:
            global_step (int): The current iteration number.

        Returns:
            bool: Whether ``global_step`` is an end point.
        """
        return self._track_schedule(global_step - self.SAVE_SHIFT)

    def _compute_start(self, global_step, params, batch_loss):
        """Perform computations at start point (store info for α fit).

        Modifies ``self._cache``.

        Args:
            global_step (int): The current iteration number.
            params ([torch.Tensor]): List of torch.Tensors holding the network's
                parameters.
            batch_loss (torch.Tensor): Mini-batch loss from current step.
        """

        def block_fn(step):
            """Block deletion for current and next iteration.

            Args:
                step (int): Iteration number.

            Returns:
                bool: Whether deletion is blocked in the specified iteration
            """
            return 0 <= step - global_step <= self.SAVE_SHIFT

        # 0ᵗʰ order info
        f_start = batch_loss.item()
        self.save_to_cache(global_step, "f_start", f_start, block_fn)
        var_f_start = get_individual_losses(global_step).var().item()
        self.save_to_cache(global_step, "var_f_start", var_f_start, block_fn)

        # for projecting gradients along the step direction
        params_start = {id(p): p.data.clone().detach() for p in params}
        self.save_to_cache(global_step, "params_start", params_start, block_fn)
        grad_start = {id(p): p.grad.data.clone().detach() for p in params}
        self.save_to_cache(global_step, "grad_start", grad_start, block_fn)
        # L = ¹/ₙ ∑ᵢ ℓᵢ, BackPACK's BatchGrad computes ¹/ₙ ∇ℓᵢ, we have to rescale
        batch_size = get_batch_size(global_step)
        batch_grad_start = {
            id(p): batch_size * p.grad_batch.data.clone().detach() for p in params
        }
        self.save_to_cache(global_step, "batch_grad_start", batch_grad_start, block_fn)

    def _compute_end(self, global_step, params, batch_loss):
        """Compute and return α.

        Args:
            global_step (int): The current iteration number.
            params ([torch.Tensor]): List of torch.Tensors holding the network's
                parameters.
            batch_loss (torch.Tensor): Mini-batch loss from current step.

        Returns:
            float: Normalized step size α
        """

        def block_fn(step):
            """Block deletion for current step.

            Args:
                step (int): Iteration number.

            Returns:
                bool: Whether deletion is blocked in the specified iteration
            """
            return step == global_step

        # 0ᵗʰ order info
        f_end = batch_loss.item()
        self.save_to_cache(global_step, "f_end", f_end, block_fn)
        var_f_end = get_individual_losses(global_step).var().item()
        self.save_to_cache(global_step, "var_f_end", var_f_end, block_fn)

        params_end = {id(p): p.data.clone().detach() for p in params}
        self.save_to_cache(global_step, "params_end", params_end, block_fn)
        grad_end = {id(p): p.grad.data.clone().detach() for p in params}
        self.save_to_cache(global_step, "grad_end", grad_end, block_fn)
        # L = ¹/ₙ ∑ᵢ ℓᵢ, BackPACK's BatchGrad computes ¹/ₙ ∇ℓᵢ, we have to rescale
        batch_size = get_batch_size(global_step)
        batch_grad_end = {
            id(p): batch_size * p.grad_batch.data.clone().detach() for p in params
        }
        self.save_to_cache(global_step, "batch_grad_end", batch_grad_end, block_fn)

        self._cache_end(global_step)
        return self._compute_alpha(global_step)

    def _cache_end(self, end_step):
        """Cache information at end step."""

        start_step = end_step - self.SAVE_SHIFT

        start_params = self.load_from_cache(start_step, "params_start")
        end_params = self.load_from_cache(end_step, "params_end")

        search_dir = [
            end_params[key] - start_params[key] for key in start_params.keys()
        ]

        def block_fn(step):
            """Block deletion for current step.

            Args:
                step (int): Iteration number.

            Returns:
                bool: Whether deletion is blocked in the specified iteration
            """
            return step == end_step

        for step, step_name in zip([start_step, end_step], ["start", "end"]):
            grad = self.load_from_cache(step, f"grad_{step_name}")
            grad = [grad[key] for key in start_params.keys()]

            batch_grad = self.load_from_cache(step, f"batch_grad_{step_name}")
            batch_grad = [batch_grad[key] for key in start_params.keys()]

            # 1ˢᵗ order info
            self.save_to_cache(
                step, f"df_{step_name}", _projected_gradient(grad, search_dir), block_fn
            )
            self.save_to_cache(
                step,
                f"var_df_{step_name}",
                _exact_variance(batch_grad, search_dir),
                block_fn,
            )

    def _compute_alpha(self, end_step):
        """Compute and return alpha, assuming all information has been cached."""
        start_step = end_step - self.SAVE_SHIFT

        f_start = self.load_from_cache(start_step, "f_start")
        f_end = self.load_from_cache(end_step, "f_end")

        t = self._compute_step_length(start_step, end_step)

        df_start = self.load_from_cache(start_step, "df_start")
        df_end = self.load_from_cache(end_step, "df_end")

        var_f_start = self.load_from_cache(start_step, "var_f_start")
        var_f_end = self.load_from_cache(end_step, "var_f_end")

        var_df_start = self.load_from_cache(start_step, "var_df_start")
        var_df_end = self.load_from_cache(end_step, "var_df_end")

        # Compute alpha
        mu = _fit_quadratic(
            t,
            (f_start, f_end),
            (df_start, df_end),
            (var_f_start, var_f_end),
            (var_df_start, var_df_end),
        )
        alpha = _get_alpha(mu, t)

        return alpha

    def _compute_step_length(self, start_step, end_step):
        """Return distance between start and end point."""
        start_params = self.load_from_cache(start_step, "params_start")
        end_params = self.load_from_cache(end_step, "params_end")

        dists = [
            (end_params[key] - start_params[key]).norm(2).item()
            for key in start_params.keys()
        ]

        return _root_sum_of_squares(dists)


class _Alpha(Quantity):
    """Base class for α computation."""

    _positions = ["start", "end"]
    _start_end_difference = 1

    def __init__(self, track_schedule, verbose=False):
        super().__init__(track_schedule, verbose=verbose)
        self.clear_info()

    def clear_info(self):
        """Reset information of start and end points."""
        self._start_info = {}
        self._end_info = {}

    # TODO Rewrite to use parent class track method
    def track(self, global_step, params, batch_loss):
        """Evaluate the current parameter distances.

        We track both the distance to the initialization, as well as the size of
        the last parameter update.

        Args:
            global_step (int): The current iteration number.
            params ([torch.Tensor]): List of torch.Tensors holding the network's
                parameters.
            batch_loss (torch.Tensor): Mini-batch loss from current step.
        """
        if self._is_position(global_step, pos="end"):
            self._end_info = self._fetch_values(params, batch_loss, "end", global_step)

            alpha = self._compute_alpha()

            if self._verbose:
                print(f"[Step {global_step}] Alpha: {alpha:.4f}")

            self.output[global_step - self._start_end_difference] = alpha
            self.clear_info()

        if self._is_position(global_step, pos="start"):
            self._start_info = self._fetch_values(
                params, batch_loss, "start", global_step
            )

    def _fetch_values(self, params, batch_loss, pos, global_step):
        """Fetch values for quadratic fit. Return as dictionary.

        The entry "search_dir" is only initialized if ``pos`` is ``"start"``.

        Args:
            params ([torch.Tensor]): List of torch.Tensors holding the network's
                parameters.
            batch_loss (torch.Tensor): Mini-batch loss from current step.
            pos (str): Whether we are at the start or end of an iteration.
                One of ``start`` or ``end``.
            global_step (int): The current iteration number.

        Raises:
            NotImplementedError: If not defined. Should be implemented by subclass.
        """
        raise NotImplementedError

    def _compute_alpha(self):
        """Compute and return alpha, assuming all quantities have been stored."""
        t = self._compute_step_length()
        fs = self._get_info("f")
        dfs = self._get_info("df")
        var_fs = self._get_info("var_f")
        var_dfs = self._get_info("var_df")

        # Compute alpha
        mu = _fit_quadratic(t, fs, dfs, var_fs, var_dfs)
        alpha = _get_alpha(mu, t)

        return alpha

    def _compute_step_length(self):
        """Return distance between start and end point."""
        start_params, end_params = self._get_info("params")

        dists = [
            (end_params[key] - start_params[key]).norm(2).item()
            for key in start_params.keys()
        ]

        return _root_sum_of_squares(dists)

    def _get_info(self, key, start=True, end=True):
        """Return list with the requested information at start and/or end point.

        Args:
            key (str): Label of the information requested for start and end point
                of the quadratic fit.
            start (bool, optional): Whether to get info at start. Defaults to `True`.
            end (bool, optional): Whether to get info at end. Defaults to `True`.

        Returns:
            list: Requested information.
        """
        start_value, end_value = None, None

        if start:
            start_value = self._start_info[key]
        if end:
            end_value = self._end_info[key]

        return [start_value, end_value]

    def _is_position(self, global_step, pos):
        """Return whether current iteration is the start/end of a quadratic fit."""
        if pos == "start":
            step = global_step
        elif pos == "end":
            step = global_step - self._start_end_difference
        else:
            raise ValueError(f"Invalid position '{pos}'. Expect {self._positions}.")

        return self._track_schedule(step)


class AlphaGeneral(_Alpha):
    """Compute α but requires storing individual gradients."""

    def extensions(self, global_step):
        """Return list of BackPACK extensions required for the computation.

        Args:
            global_step (int): The current iteration number.

        Returns:
            list: (Potentially empty) list with required BackPACK quantities.
        """
        ext = []

        if self._is_position(global_step, "start") or self._is_position(
            global_step, "end"
        ):
            ext.append(extensions.BatchGrad())

        return ext

    def _fetch_values(self, params, batch_loss, pos, global_step):
        """Fetch values for quadratic fit. Return as dictionary.

        The entry "search_dir" is only initialized if ``pos`` is ``"start"``.

        Args:
            params ([torch.Tensor]): List of torch.Tensors holding the network's
                parameters.
            batch_loss (torch.Tensor): Mini-batch loss from current step.
            pos (str): Whether we are at the start or end of an iteration.
                One of ``start`` or ``end``.
            global_step (int): The current iteration number.

        Raises:
            ValueError: If position is not one of ``start`` or ``end``.

        Returns:
            dict: Holding the parameters, (variance of) loss and slope.
        """
        info = {}

        if pos in ["start", "end"]:
            # 0ᵗʰ order info
            info["f"] = batch_loss.item()
            info["var_f"] = get_individual_losses(global_step).var().item()

            # temporary information required to compute quantities used in fit
            info["params"] = {id(p): p.data.clone().detach() for p in params}
            info["grad"] = {id(p): p.grad.data.clone().detach() for p in params}
            # L = ¹/ₙ ∑ᵢ ℓᵢ, BackPACK's computes ¹/ₙ ∇ℓᵢ, we have to rescale
            batch_size = get_batch_size(global_step)
            info["batch_grad"] = {
                id(p): batch_size * p.grad_batch.data.clone().detach() for p in params
            }

        else:
            raise ValueError(f"Invalid position '{pos}'. Expect {self._positions}.")

        # compute all quantities used in fit
        # TODO Restructure base class and move to other function
        if pos == "end":
            start_params, _ = self._get_info("params", end=False)
            end_params = info["params"]

            search_dir = [
                end_params[key] - start_params[key] for key in start_params.keys()
            ]

            for info_dict in [self._start_info, info]:
                grad = [info_dict["grad"][key] for key in start_params.keys()]
                batch_grad = [
                    info_dict["batch_grad"][key] for key in start_params.keys()
                ]

                # 1ˢᵗ order info
                info_dict["df"] = _projected_gradient(grad, search_dir)
                info_dict["var_df"] = _exact_variance(batch_grad, search_dir)

        return info


class Alpha(_Alpha):
    """Optimized α Quantity Class. Does not require storing individual gradients."""

    def extensions(self, global_step):
        """Return list of BackPACK extensions required for the computation.

        Args:
            global_step (int): The current iteration number.

        Returns:
            list: (Potentially empty) list with required BackPACK quantities.
        """
        ext = []

        if self._is_position(global_step, "start"):
            ext.append(self._start_search_dir_projection_info())
        if self._is_position(global_step, "end"):
            ext.append(self._end_search_dir_projection_info())

        return ext

    def _fetch_values(self, params, batch_loss, pos, global_step):
        """Fetch values for quadratic fit. Return as dictionary.

        The entry "search_dir" is only initialized if ``pos`` is ``"start"``.

        Args:
            params ([torch.Tensor]): List of torch.Tensors holding the network's
                parameters.
            batch_loss (torch.Tensor): Mini-batch loss from current step.
            pos (str): Whether we are at the start or end of an iteration.
                One of ``start`` or ``end``.
            global_step (int): The current iteration number.

        Returns:
            dict: Holding the parameters, (variance of) loss and slope.
        """
        info = {}

        info["params"] = {id(p): p.data.clone().detach() for p in params}

        # 0ᵗʰ order info
        info["f"] = batch_loss.item()
        info["var_f"] = get_individual_losses(global_step).var().item()

        # 1ˢᵗ order info
        info["df"], info["var_df"] = self._fetch_df_and_var_df(params, pos)

        return info

    def _start_search_dir_projection_info(self):
        """Compute information for individual gradient projections onto search dir.

        The search direction at a start point depends on the optimizer.

        We want to compute dᵀgᵢ / ||d||₂² where d is the search direction and gᵢ are
        individual gradients. However, this fraction cannot be aggregated among
        parameters. We have to split the computation into components that can be
        aggregated, namely dᵀgᵢ and ||d||₂².

        Returns:
            BatchGradTransform: Transform to compute information to project
            individual gradients.
        """

        def compute_start_projection_info(batch_grad):
            """Compute information to project individual gradients onto the gradient."""
            batch_size = batch_grad.shape[0]

            # TODO Currently only correctly implemented for SGD! Make more general
            warnings.warn(
                "Alpha will only be correct if optimizer is SGD with momentum 0"
            )
            search_dir_flat = -1 * (batch_grad.data.sum(0).flatten())
            batch_grad_flat = batch_grad.data.flatten(start_dim=1)

            search_dir_l2_squared = (search_dir_flat ** 2).sum()
            dot_products = torch.einsum(
                "ni,i->n", batch_size * batch_grad_flat, search_dir_flat
            )

            return {
                "dot_products": dot_products,
                "search_dir_l2_squared": search_dir_l2_squared,
            }

        return extensions.BatchGradTransforms(
            {"start_projection_info": compute_start_projection_info}
        )

    def _end_search_dir_projection_info(self):
        """Compute information for individual gradient projections onto search dir.

        The search direction at an end point is inferred from the model parameters.

        We want to compute dᵀgᵢ / ||d||₂² where d is the search direction and gᵢ are
        individual gradients. However, this fraction cannot be aggregated among
        parameters. We have to split the computation into components that can be
        aggregated, namely dᵀgᵢ and ||d||₂².

        Returns:
            BatchGradTransform: Transform to compute information to project
            individual gradients.
        """

        def compute_end_projection_info(batch_grad):
            """Compute information to project individual gradients onto the gradient."""
            batch_size = batch_grad.shape[0]

            end_param = batch_grad._param_weakref()
            start_param = self._get_info("params", end=False)[0][id(end_param)]

            search_dir_flat = (end_param.data - start_param).flatten()
            batch_grad_flat = batch_grad.data.flatten(start_dim=1)

            search_dir_l2_squared = (search_dir_flat ** 2).sum()
            dot_products = torch.einsum(
                "ni,i->n", batch_size * batch_grad_flat, search_dir_flat
            )

            return {
                "dot_products": dot_products,
                "search_dir_l2_squared": search_dir_l2_squared,
            }

        return extensions.BatchGradTransforms(
            {"end_projection_info": compute_end_projection_info}
        )

    def _fetch_df_and_var_df(self, params, pos):
        """Compute projected gradient and variance from after-backward quantities."""
        if pos == "start":
            key = "start_projection_info"
        elif pos == "end":
            key = "end_projection_info"
        else:
            raise ValueError(f"Invalid position '{pos}'. Expect {self._positions}.")

        dot_products = sum(p.grad_batch_transforms[key]["dot_products"] for p in params)
        search_dir_l2_squared = sum(
            p.grad_batch_transforms[key]["search_dir_l2_squared"] for p in params
        )

        projections = dot_products / search_dir_l2_squared.sqrt()

        df = projections.mean().item()
        df_var = projections.var().item()

        return df, df_var


# TODO Move inside class
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


# TODO Move inside class
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


# TODO Move inside class
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

    # small value to use if one of the variances or t is 0.
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
    if t == 0.0:
        warnings.warn(
            "The two observations were (almost) at the same point.", stacklevel=2
        )

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


# TODO Move inside class
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
    # If we couldn't compute a quadratic approx., return None or -1 if it was
    # due to a very small step
    if mu is None:
        return -1 if t == 0.0 else None
    elif mu[2] < 0:
        # Concave setting: Since this means that it is still going downhill
        # where we stepped to, we will log it as -1.
        return -1
    else:
        # get alpha_bar (the step size that is "the other side")
        alpha_bar = -mu[1] / mu[2]

        # scale everything on a standard quadratic (i.e. step of 0 = minimum)
        return (2 * t) / alpha_bar - 1
