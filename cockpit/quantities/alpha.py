"""Class for tracking the effective relative step size Alpha."""

import itertools
import warnings

import numpy as np
import torch
from backpack import extensions

from cockpit.context import get_batch_size, get_individual_losses, get_optimizer
from cockpit.quantities.quantity import TwoStepQuantity
from cockpit.quantities.utils_quantities import (
    _layerwise_dot_product,
    _root_sum_of_squares,
)
from cockpit.quantities.utils_transforms import (
    BatchGradTransformsHook,
    get_first_n_alphabet,
)
from cockpit.utils.optim import ComputeStep


class Alpha(TwoStepQuantity):
    """Alpha Quantity class for the normalized step length α.

    The normalized step length α  uses a noise-aware quadratic loss landscape fit
    to estimate whether current local steps over- or undershoot the local minimum.

    The fit uses zero- and first-order information, including uncertainties, between
    two consecutive iterations which are referred to as ``'start'`` and ``'end'``
    point, respectively. This information needs to be projected onto the update
    step.

    Note:
        This quantity requires the optimizer be specified in the ``'optimizer'``
        ``info`` entry of a ``cockpit(...)`` context.

    Note:
        For SGD with default parameters the projections onto the search direction can
        be performed during a backward pass without storing large tensors between start
        and end point.
    """

    SAVE_SHIFT = 1
    """int: Difference between iteration at which information is computed
            versus iteration under which it is stored. For instance, if set to ``1``,
            the information computed at iteration ``n + 1`` is saved under iteration
            ``n``. Default: ``1``.
    """
    POINTS = ["start", "end"]
    """[str]:  Description of start and end point."""

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

    def extension_hooks(self, global_step):
        """Return list of BackPACK extension hooks required for the computation.

        Args:
            global_step (int): The current iteration number.

        Returns:
            [callable]: List of required BackPACK extension hooks for the current
                iteration.
        """
        hooks = []

        start = self.is_start(global_step)
        end = self.is_end(global_step)

        if (start or end) and self.__projection_with_backpack(global_step):
            if start:
                hooks.append(self._project_with_backpack_start(global_step))
            if end:
                hooks.append(self._project_with_backpack_end(global_step))

        return hooks

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

    def __projection_with_backpack(self, global_step):
        """Return whether the update step projection is computed through BackPACK.

        Currently, this can only be done for SGD with default parameters.

        Args:
            global_step (int): The current iteration number.

        Returns:
            bool: Truth value whether gradient projections and update length info
                are computed with BackPACK.
        """
        try:
            optimizer = get_optimizer(global_step)
            computed = ComputeStep().is_sgd_default_kwargs(optimizer)
        except KeyError:
            warnings.warn("No 'optimizer' handed in via info")
            computed = False
        except NotImplementedError:
            warnings.warn("No update step implemented for optimizer")
            computed = False

        return computed

    def _project_with_backpack_start(self, start_step):
        """Return hook that computes gradient info at start with known search direction.

        We want to compute ``dᵀgᵢ / ||d||₂²`` where ``d`` is the search direction and
        ``gᵢ`` are individual gradients. The information needs to be decomposed such
        that it can later be aggregated over parameters. Hence, the following
        operations are carried out for each parameter:

        - Compute and cache ``d`` under ``'update_start'``
        - Compute and cache ``dᵀgᵢ`` under ``'update_dot_grad_batch_start'``

        Remaining computations are finished up in ``_project_with_backpack_finalize``.

        Args:
            start_step (int): Iteration number of the start point.

        Returns:
            BatchGradTransform: Extension executed by BackPACK during a backward pass
                to perform the gradient projections.
        """
        block_fn = self._make_block_fn(start_step, start_step + self.SAVE_SHIFT)
        optimizer = get_optimizer(start_step)

        def hook(grad_batch):
            """Cache optimizer update and projected gradients.

            Modifies ``self._cache``, creating entries ``update_start`` and
            ``update_dot_grad_batch_start``.

            Args:
                grad_batch (torch.Tensor): Result of BackPACK's ``BatchGrad``
                    extension.

            Returns:
                dict: Empty dictionary.
            """
            param_id = id(grad_batch._param_weakref())

            search_dir = ComputeStep().compute_update_step(optimizer, [param_id])[
                param_id
            ]
            # L = ¹/ₙ ∑ᵢ ℓᵢ, BackPACK's BatchGrad computes ¹/ₙ ∇ℓᵢ, we have to rescale
            batch_size = get_batch_size(start_step)
            dot_products = batch_size * self.batched_dot_product(grad_batch, search_dir)

            # update or create cache for ``d``
            key = "update_start"
            try:
                update_dict = self.load_from_cache(start_step, key)
            except KeyError:
                update_dict = {}
            finally:
                update_dict[param_id] = search_dir
                self.save_to_cache(start_step, key, update_dict, block_fn)

            # update or create cache for ``dᵀgᵢ``
            key = "update_dot_grad_batch_start"
            try:
                update_dot_dict = self.load_from_cache(start_step, key)
            except KeyError:
                update_dot_dict = {}
            finally:
                update_dot_dict[param_id] = dot_products
                self.save_to_cache(start_step, key, update_dot_dict, block_fn)

            return {}

        return BatchGradTransformsHook({"_first_order_projections_start": hook})

    def _project_with_backpack_end(self, end_step):
        """Return hook that computes gradient info at end with known search direction.

        We want to compute ``dᵀgᵢ / ||d||₂²`` where ``d`` is the search direction at
        the start and ``gᵢ`` are individual gradients at the end. The information needs
        to be decomposed such that it can later be aggregated over parameters. Hence,
        the following operations are carried out for each parameter:

        - Compute and cache ``dᵀgᵢ`` under ``'update_dot_grad_batch_end'``

        Remaining computations are finished up in ``_project_with_backpack_finalize``.

        Args:
            end_step (int): Iteration number of the end point.

        Returns:
            BatchGradTransform: Extension executed by BackPACK during a backward pass
                to perform the gradient projections.
        """
        block_fn = self._make_block_fn(end_step, end_step)
        start_step = end_step - self.SAVE_SHIFT

        def hook(grad_batch):
            """Project the end point gradients onto the start point's update direction.

            Modifies ``self._cache``, creating entry ``update_dot_grad_batch_end``.

            Args:
                grad_batch (torch.Tensor): Result of BackPACK's ``BatchGrad``
                    extension.

            Returns:
                dict: Empty dictionary.
            """
            param_id = id(grad_batch._param_weakref())

            search_dir = self.load_from_cache(start_step, "update_start")[param_id]
            # L = ¹/ₙ ∑ᵢ ℓᵢ, BackPACK's BatchGrad computes ¹/ₙ ∇ℓᵢ, we have to rescale
            batch_size = get_batch_size(end_step)
            dot_products = batch_size * self.batched_dot_product(grad_batch, search_dir)

            # update or create cache for ``dᵀgᵢ``
            key = "update_dot_grad_batch_end"
            try:
                update_dot_dict = self.load_from_cache(end_step, key)
            except KeyError:
                update_dot_dict = {}
            finally:
                update_dot_dict[param_id] = dot_products
                self.save_to_cache(end_step, key, update_dot_dict, block_fn)

            return {}

        return BatchGradTransformsHook({"_first_order_projections_end": hook})

    def _compute_start(self, global_step, params, batch_loss):
        """Perform computations at start point (store info for α fit).

        Modifies ``self._cache``.

        Args:
            global_step (int): The current iteration number.
            params ([torch.Tensor]): List of torch.Tensors holding the network's
                parameters.
            batch_loss (torch.Tensor): Mini-batch loss from current step.
        """
        until = global_step + self.SAVE_SHIFT
        point = "start"

        self._save_0th_order_info(global_step, params, batch_loss, point, until)

        # fall back to storing parameters and individual gradients at start/end
        if not self.__projection_with_backpack(global_step):
            self._save_1st_order_info(global_step, params, batch_loss, point, until)

    def _save_0th_order_info(self, global_step, params, batch_loss, point, until):
        """Store 0ᵗʰ-order information about the objective in cache.

        Modifies ``self._cache``, creating entries ``f_*`` and ``var_f_*``.

        Args:
            global_step (int): The current iteration number.
            params ([torch.Tensor]): List of torch.Tensors holding the network's
                parameters.
            batch_loss (torch.Tensor): Mini-batch loss from current step.
            point (str): Description of point, ``'start'`` or ``'end'``.
            until (int): Iteration number until which deletion from cache is blocked.
        """
        block_fn = self._make_block_fn(global_step, until)

        f = batch_loss.item()
        self.save_to_cache(global_step, f"f_{point}", f, block_fn)

        var_f = get_individual_losses(global_step).var().item()
        self.save_to_cache(global_step, f"var_f_{point}", var_f, block_fn)

    def _save_1st_order_info(self, global_step, params, batch_loss, point, until):
        """Store information for projecting 1ˢᵗ-order info about the objective in cache.

        This is the go-to approach if the update step at the start point +projections
        cannot be computed automatically through BackPACK.

        Modifies ``self._cache``, creating the fields ``params_*``, ``grad_*``, and
        ``grad_batch_*``. Parameters are required to compute the update step, the
        gradients are required to project them onto the step at a later stage.

        Args:
            global_step (int): The current iteration number.
            params ([torch.Tensor]): List of torch.Tensors holding the network's
                parameters.
            batch_loss (torch.Tensor): Mini-batch loss from current step.
            point (str): Description of point, ``'start'`` or ``'end'``.
            until (int): Iteration number until which deletion from cache is blocked.
        """
        block_fn = self._make_block_fn(global_step, until)

        params_dict = {id(p): p.data.clone().detach() for p in params}
        self.save_to_cache(global_step, f"params_{point}", params_dict, block_fn)

        grad_dict = {id(p): p.grad.data.clone().detach() for p in params}
        self.save_to_cache(global_step, f"grad_{point}", grad_dict, block_fn)

        # L = ¹/ₙ ∑ᵢ ℓᵢ, BackPACK's BatchGrad computes ¹/ₙ ∇ℓᵢ, we have to rescale
        batch_size = get_batch_size(global_step)
        grad_batch_dict = {
            id(p): batch_size * p.grad_batch.data.clone().detach() for p in params
        }
        self.save_to_cache(
            global_step, f"grad_batch_{point}", grad_batch_dict, block_fn
        )

    @staticmethod
    def _make_block_fn(start_step, end_step):
        """Create a function that blocks cache deletion in interval ``[start; end]``.

        Args:
            start_step (int): Left boundary of blocked interval
            end_step (int): Right boundary of blocked interval

        Returns:
            callable: Block function that can be used as ``block_fn`` argument in
                ``save_to_cache``.
        """

        def block_fn(step):
            """Block deletion in ``[start; end]``.

            Args:
                step (int): Iteration number.

            Returns:
                bool: Whether deletion is blocked in the specified iteration
            """
            return step in range(start_step, end_step + 1)

        return block_fn

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
        point = "end"
        until = global_step

        self._save_0th_order_info(global_step, params, batch_loss, point, until)

        if self.__projection_with_backpack(global_step):
            self._project_with_backpack_finalize(global_step)
        # fall back to storing parameters and individual gradients at start/end
        else:
            self._save_1st_order_info(global_step, params, batch_loss, point, until)
            self._project_end(global_step)

        return self._compute_alpha(global_step)

    def _project_with_backpack_finalize(self, end_step):
        """Finish up the computations for ``df`` and ``var_df`` at start and end point.

        We need ``dᵀgᵢ / ||d||₂²`` (at start and end) and ``||d||₂²`` (at start) from
        the parameter-wise quantities ``dᵀgᵢ`` (at start and end) and ``d`` (at start).
        Thus, the following operations are performed:

        - Compute ``||d||₂²`` and cache under ``update_size_start``
        - Compute ``dᵀgᵢ / ||d||₂²`` with individual gradients at the start point, cache
          mean in ``df_start`` and variance in ``var_df_start``.
        - Compute ``dᵀgᵢ / ||d||₂²`` with individual gradients at the end point, cache
          mean in ``df_end`` and variance in ``var_df_end``.

        Args:
            end_step (int): Iteration number of end step.
        """
        block_fn = self._make_block_fn(end_step, end_step)
        start_step = end_step - self.SAVE_SHIFT

        # compute the update step
        updates = self.load_from_cache(start_step, "update_start")
        update_size = sum((upd ** 2).sum() for upd in updates.values()).sqrt().item()
        self.save_to_cache(start_step, "update_size_start", update_size, block_fn)

        for step, p in [[start_step, "start"], [end_step, "end"]]:
            dot_produts = self.load_from_cache(step, f"update_dot_grad_batch_{p}")
            projections = sum(dp for dp in dot_produts.values()) / update_size

            df = projections.mean().item()
            self.save_to_cache(step, f"df_{p}", df, block_fn)

            df_var = projections.var().item()
            self.save_to_cache(step, f"var_df_{p}", df_var, block_fn)

    def _project_end(self, end_step):
        """Project first-order information at end step and associated start step.

        Assumes that parameters, gradients, and individual gradients are cached
        at the end step and its associated start point.

        Modifies ``self._cache``, creating the fields ``df_*`` and ``var_df``,
        as well as ``update_size_start``.

        Args:
            end_step (int): Iteration number of end step.
        """
        start_step = end_step - self.SAVE_SHIFT

        # update direction and its length
        start_params = self.load_from_cache(start_step, "params_start")
        end_params = self.load_from_cache(end_step, "params_end")

        search_dir = [
            end_params[key] - start_params[key] for key in start_params.keys()
        ]
        update_size = sum((s ** 2).sum() for s in search_dir).sqrt().item()

        block_fn = self._make_block_fn(end_step, end_step)
        self.save_to_cache(start_step, "update_size_start", update_size, block_fn)

        # projections
        for step, p in [[start_step, "start"], [end_step, "end"]]:
            # projected gradient
            grad = self.load_from_cache(step, f"grad_{p}")
            grad = [grad[key] for key in start_params.keys()]
            self.save_to_cache(
                step, f"df_{p}", _projected_gradient(grad, search_dir), block_fn
            )

            # projected gradient variance
            grad_batch = self.load_from_cache(step, f"grad_batch_{p}")
            grad_batch = [grad_batch[key] for key in start_params.keys()]
            self.save_to_cache(
                step,
                f"var_df_{p}",
                _exact_variance(grad_batch, search_dir),
                block_fn,
            )

    @staticmethod
    def batched_dot_product(batched_tensor, tensor):
        """Compute scalar product between a batched and an unbatched tensor.

        Args:
            batched_tensor (torch.Tensor): Batched tensor along first axis.
            tensor (torch.Tensor): Unbatched tensor. All axes are feature dimensions.

        Returns:
            torch.Tensor: Tensor of shape ``[N]`` where ``N`` is the batch dimension.
                Contains scalar products for each sample along the batch axis.

        Raises:
            ValueError: If the batched tensor's trailing dimensions don't match the
                unbatched tensor's shape.
        """
        if tensor.shape != batched_tensor.shape[1:]:
            raise ValueError(
                "Tensors don't share same feature dimensions."
                + f" Got {tensor.shape} and f{batched_tensor.shape}"
            )

        letters = get_first_n_alphabet(batched_tensor.dim())
        equation = f"{letters},{letters[1:]}->{letters[0]}"

        return torch.einsum(equation, batched_tensor, tensor)

    def _compute_alpha(self, end_step):
        """Compute and return alpha, assuming all information has been cached.

        Requires the following cached entries:
        - ``update_size_start``
        - ``f_start``, ``f_end``, ``var_f_start``, ``var_f_end``
        - ``df_start``, ``df_end``, ``var_df_start``, ``var_df_end``

        Args:
            end_step (int): Iteration number of the end point.

        Returns:
            float: Local effective step size α.
        """
        start_step = end_step - self.SAVE_SHIFT

        t = self.load_from_cache(start_step, "update_size_start")

        points = [[start_step, "start"], [end_step, "end"]]

        fs = [self.load_from_cache(step, f"f_{p}") for step, p in points]
        dfs = [self.load_from_cache(step, f"df_{p}") for step, p in points]
        var_fs = [self.load_from_cache(step, f"var_f_{p}") for step, p in points]
        var_dfs = [self.load_from_cache(step, f"var_df_{p}") for step, p in points]

        # Compute alpha
        mu = _fit_quadratic(t, fs, dfs, var_fs, var_dfs)
        alpha = _get_alpha(mu, t)

        return alpha


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
