"""Base class for a tracked quantity."""

from collections import defaultdict

import numpy
import torch


class Quantity:
    """Base class for a tracked quantity with the Cockpit.

    Quantities can modify the backward pass:

    1. They can ask that the forward pass computation graph be restored.
       This may be useful if their computation requires differentiating through
       the mini-batch loss.
    2. They can ask for certain BackPACK extensions being computed.

    New `Quantity` objects can be defined by inheriting from this class and
    implementing the `extensions` and `track` methods.

    Instead of writing your own `track` function, this base class already decouples
    storing and computing results using the `should_compute` and `compute` methods.
    You may therefore implement the latter two instead of overwriting `track`.
    """

    def __init__(self, track_schedule, verbose=False):
        """Initialization sets the tracking schedule & creates the output dict.

        Args:
            track_schedule (callable): Function that maps the ``global_step``
                to a boolean, which determines if the quantity should be computed.
            verbose (bool, optional): Turns on verbose mode. Defaults to ``False``.
        """
        self._track_schedule = track_schedule
        self._verbose = verbose

        self.output = defaultdict(dict)

    def create_graph(self, global_step):
        """Return whether access to the forward pass computation graph is needed.

        Args:
            global_step (int): The current iteration number.

        Returns:
            bool: ``True`` if the computation graph shall not be deleted,
            else ``False``.
        """
        return False

    def extensions(self, global_step):
        """Return list of BackPACK extensions required for the computation.

        Args:
            global_step (int): The current iteration number.

        Raises:
            NotImplementedError: If not implemented. Should be defined by subclass.
        """
        raise NotImplementedError

    def extension_hooks(self, global_step):
        """Return list of BackPACK extension hooks required for the computation.

        This method may be overwritten by descendants.

        Args:
            global_step (int): The current iteration number.

        Returns:
            [callable]: List of required BackPACK extension hooks for the current
                iteration.
        """
        return []

    def track(self, global_step, params, batch_loss):
        """Perform scheduled computations and store result.

        Args:
            global_step (int): The current iteration number.
            params ([torch.Tensor]): List of torch.Tensors holding the network's
                parameters.
            batch_loss (torch.Tensor): Mini-batch loss from current step.
        """
        if self.should_compute(global_step):
            iteration, result = self.compute(global_step, params, batch_loss)
            if result is not None:
                if self._verbose:
                    print(
                        f"{self._verbose_prefix(global_step)}:"
                        + f" Store iteration {iteration}, value {result}"
                    )

                self._save(iteration, result)
        else:
            if self._verbose:
                print(f"{self._verbose_prefix(global_step)}: No computation scheduled")

    def _verbose_prefix(self, global_step):
        return f"[Step {global_step} | {self.__class__.__name__}]"

    def should_compute(self, global_step):
        """Return if computations need to be performed at a specific iteration.

        Args:
            global_step (int): The current iteration number.

        Raises:
            NotImplementedError: If not implemented. Should be defined by subclass.
        """
        raise NotImplementedError

    def _save(self, global_step, result):
        """Store computation result.

        Args:
            global_step (int): The current iteration number.
            result (arbitrary): The result to be stored.
        """
        self.output[global_step] = self._apply_save_format(result)

    def _apply_save_format(self, value):
        """Apply formatting rules for saved data.

        ``torch.Tensor``s are detached, loaded to CPU and converted to ``numpy`` arrays.
        Items of ``dict``, ``list``, and ``tuple`` are converted recursively.
        ``float``, ``int``, and ``numpy.ndarray`` values are unaffected.

        Args:
            value (Any): Value to be saved.

        Returns:
            Any: Converted value.

        Raises:
            NotImplementedError: If there is no formatting rule for the data type.
        """
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()

        elif isinstance(value, dict):
            for key, val in value.items():
                value[key] = self._apply_save_format(val)

        elif isinstance(value, list):
            for idx, val in enumerate(value):
                value[idx] = self._apply_save_format(val)

        elif isinstance(value, tuple):
            value = tuple(self._apply_save_format(val) for val in value)

        elif isinstance(value, (float, int, numpy.ndarray)):
            pass

        else:
            raise NotImplementedError(f"No formatting rule for type {type(value)}")

        return value

    def compute(self, global_step, params, batch_loss):
        """Evaluate quantity at a step in training.

        Args:
            global_step (int): The current iteration number.
            params ([torch.Tensor]): List of torch.Tensors holding the network's
                parameters.
            batch_loss (torch.Tensor): Mini-batch loss from current step.

        Raises:
            NotImplementedError: If not implemented. Should be defined by subclass.
        """
        raise NotImplementedError

    def get_output(self):
        """Return a dictionary that stores the results.

        Keys correspond to the iteration and values represent the computational result.
        Values are kept as ``numpy.ndarray`` on CPU rather than ``torch.Tensor`` to
        avoid unnecessary data traffic during plotting.

        Example:
            >>> quantity = quantities.MaxEV()
            >>> # information tracked at iteration 3
            >>> global_step = 3
            >>> global_step_output = quantity.get_output()[global_step]

        Returns:
            dict: Dictionary with the stored results.
        """
        return self.output

    @staticmethod
    def _fetch_grad(params, aggregate=False):
        """Return parameter gradients.

        Gradients can be aggregated among parameters (vectorized). In this case,
        the return value is a vector of shape ``[D]`` where ``D`` is the total
        number of parameters.

        Args:
            params ([torch.Tensor]): List of parameters whose gradient will be fetched.
            aggregate (bool): Flatten and concatenate all gradients.

        Returns:
            [torch.Tensor] if ``aggregate`` is False: List of same length as ``params``
            with items of same shape containing the gradients.
            torch.Tensor if ``aggregate`` is True: Gradient of vectorized parameters.
        """
        grads = [p.grad for p in params]

        if aggregate:
            grads = torch.cat([g.flatten() for g in grads])

        return grads

    @staticmethod
    def _fetch_grad_l2_squared(params, aggregate=False):
        """Return gradient squared ℓ₂ norm, || ∇L ||₂².

        Norms can be aggregated among parameters (summed). In this case, the return
        value is a scalar. Without aggregation, the parameter-wise squared gradient
        ℓ₂ norm is returned

        Args:
            params ([torch.Tensor]): List of parameters whose gradient squared ℓ₂ norm
                will be fetched.
            aggregate (bool): Sum squared norms over parameter blocks.

        Returns:
            [torch.Tensor] if ``aggregate`` is False: List of same length as ``params``
            with scalar items containing block-wise gradient squared norms.
            torch.Tensor if ``aggregate`` is True: A scalar, the squared gradient norm.
        """
        grad_l2_squared = [p.grad.pow(2).sum() for p in params]

        if aggregate:
            grad_l2_squared = sum(grad_l2_squared)

        return grad_l2_squared

    @staticmethod
    def _fetch_batch_grad(params, aggregate=False):
        """Return individual gradients from ``backpack.extensions.BatchGrad``.

        Individual gradients can be aggregated among parameters. With a batch size
        ``N``, the aggregated individual gradients of a ``D``-dimensional model have
        shape ``[N, D]``.

        Args:
            params ([torch.Tensor]): List of parameters whose gradient squared ℓ₂ norm
                will be fetched.
            aggregate (bool): Flatten and concatenate individual gradients over the
                parameter dimension.

        Returns:
            [torch.Tensor] if ``aggregate`` is False: List of same length as ``params``
            with items of shape ``[N, *]`` containing block-wise individual
            gradients. Here, ``*`` denotes the shape of the associated parameter
            block.
            torch.Tensor if ``aggregate`` is True: A vector of shape ``[N, D]`` with
            individual gradients for the vectorized model parameters.
        """
        grad_batch = [p.grad_batch for p in params]

        if aggregate:
            grad_batch = torch.cat([g.flatten(start_dim=1) for g in grad_batch], dim=1)

        return grad_batch

    @staticmethod
    def _fetch_batch_l2_squared(params, aggregate=False):
        """Return individual gradient squared ℓ₂ norms, || ∇Lᵢ ||₂².

        Norms can be aggregated among parameters (summed). In this case, the return
        value has shape ``[N]`` for a mini-batch size of ``N``. Without aggregation,
        the parameter-wise individual squared gradient ℓ₂ norm is returned.

        Args:
            params ([torch.Tensor]): List of parameters whose individual gradient
                squared ℓ₂ norm will be fetched.
            aggregate (bool): Sum norms over parameter blocks.

        Returns:
            [torch.Tensor] if ``aggregate`` is False: List of same length as ``params``
            with tensors of shape ``[N]`` each.
            torch.Tensor if ``aggregate`` is True: A tensor of shape ``[N]``, containing
            the individual gradient squared norm.
        """
        batch_l2 = [p.batch_l2 for p in params]

        if aggregate:
            batch_l2 = sum(batch_l2)

        return batch_l2

    @staticmethod
    def _fetch_batch_l2_squared_via_batch_grad_transforms(params, aggregate=False):
        """Same as _fetch_batch_l2_squared, assumes BatchGradTransforms computation."""
        batch_l2 = [p.grad_batch_transforms["batch_l2"] for p in params]

        if aggregate:
            batch_l2 = sum(batch_l2)

        return batch_l2

    @staticmethod
    def _fetch_batch_dot(params, aggregate=False):
        """Return individual gradient pairwise dot products.

        Pairwise dot products can be aggregated (summed) among layers. In this case,
        the return value has shape ``[N, N]`` for a mini-batch size of ``N``. Without
        aggregation, the block-wise individual gradient pairwise dot products are
        returned.

        Args:
            params ([torch.Tensor]): List of parameters whose individual gradient
                pairwise dot products will be fetched.
            aggregate (bool): Sum dot products over parameter blocks.

        Returns:
            [torch.Tensor] if ``aggregate`` is False: List of same length as ``params``
            with tensors of shape ``[N, N]`` each.
            torch.Tensor if ``aggregate`` is True: A tensor of shape ``[N, N]``.
            containing pairwise dot products of aggregated individual gradients.
        """
        batch_dot_grad = [p.batch_dot for p in params]

        if aggregate:
            batch_dot_grad = sum(batch_dot_grad)

        return batch_dot_grad

    @staticmethod
    def _fetch_batch_dot_via_batch_grad_transforms(params, aggregate=False):
        """Same as _fetch_batch_dot, assumes BatchGradTransforms computation."""
        batch_dot_grad = [p.grad_batch_transforms["batch_dot"] for p in params]

        if aggregate:
            batch_dot_grad = sum(batch_dot_grad)

        return batch_dot_grad

    @staticmethod
    def _fetch_sum_grad_squared(params, aggregate=False):
        """Return sum of squared individual gradients.

        The sum of squared gradients can be aggregated (vectorized) among parameters.
        For a model with D parameters, the aggregated sum of squared gradients has
        shape ``[D]``. Without aggregation, the block-wise sum of squared individual
        gradients are returned.

        Args:
            params ([torch.Tensor]): List of parameters whose sum of squared individual
                gradients will be fetched.
            aggregate (bool): Concatenate (vectorize) results over parameter blocks.

        Returns:
            [torch.Tensor] if ``aggregate`` is False: List of same length as ``params``
            with items of same shape containing the gradient's second moment.
            torch.Tensor if ``aggregate`` is True: Gradient second moment of vectorized
            parameters.
        """
        sum_grad_squared = [p.sum_grad_squared for p in params]

        if aggregate:
            sum_grad_squared = torch.cat([sgs.flatten() for sgs in sum_grad_squared])

        return sum_grad_squared

    @staticmethod
    def _fetch_sum_grad_squared_via_batch_grad_transforms(params, aggregate=False):
        """Same as _fetch_sum_grad_squared, assumes BatchGradTransforms computation."""
        sum_grad_squared = [p.grad_batch_transforms["sum_grad_squared"] for p in params]

        if aggregate:
            sum_grad_squared = torch.cat([sgs.flatten() for sgs in sum_grad_squared])

        return sum_grad_squared

    @staticmethod
    def _fetch_diag_curvature(params, savefield, aggregate=True):
        """Return diagonal curvature approximation.

        Diagonal curvature can be aggregated (vectorized) among parameters.
        For a model with D parameters, the aggregated diagonal curvature has
        shape ``[D]``. Without aggregation, the block-wise diagonal curvature
        is returned.

        Args:
            params ([torch.Tensor]): List of parameters whose diagonal curvature
                will be fetched.
            savefield (str): Field name where BackPACK stores the curvature
            aggregate (bool): Concatenate (vectorize) results over parameter blocks.

        Returns:
            [torch.Tensor] if ``aggregate`` is False: List of same length as ``params``
            with items of same shape containing the diagonal curvatures.
            torch.Tensor if ``aggregate`` is True: Diagonal curvature of vectorized
            parameters.
        """
        diag_curvature = [getattr(p, savefield) for p in params]

        if aggregate:
            diag_curvature = torch.cat([c.flatten() for c in diag_curvature])

        return diag_curvature


class SingleStepQuantity(Quantity):
    """Quantity that only accessed information at one point in time.

    Child classes must implement the following methods:

    - ``_compute``

    """

    def should_compute(self, global_step):
        """Return if computations need to be performed at a specific iteration.

        Args:
            global_step (int): The current iteration number.

        Returns:
            bool: Truth value whether computations need to be performed.
        """
        return self._track_schedule(global_step)

    def compute(self, global_step, params, batch_loss):
        """Evaluate quantity at a step in training.

        Args:
            global_step (int): The current iteration number.
            params ([torch.Tensor]): List of torch.Tensors holding the network's
                parameters.
            batch_loss (torch.Tensor): Mini-batch loss from current step.

        Returns:
            (int, arbitrary): The second value is the result that will be stored at
            the iteration indicated by the first entry (important for multi-step
            quantities whose values are computed in later iterations).
        """
        return (global_step, self._compute(global_step, params, batch_loss))

    def _compute(self, global_step, params, batch_loss):
        raise NotImplementedError


class ByproductQuantity(SingleStepQuantity):
    """Quantity that is just tracked and does not require additional computation."""

    def extensions(self, global_step):
        """Return list of BackPACK extensions required for the computation.

        Args:
            global_step (int): The current iteration number.

        Returns:
            list: (Potentially empty) list with required BackPACK quantities.
        """
        return []


class TwoStepQuantity(Quantity):
    """Quantity that accesses information from two points in time.

    The earlier point is referred to as 'start', while the latter is referred to
    as 'end'.

    Child classes must implement the following:

    - ``is_start``, ``is_end``
    - ``_compute_start``, ``_compute_end``
    - (maybe) adapt ``SAVE_SHIFT``
    """

    SAVE_SHIFT = 0
    """int: Difference between iteration at which information is computed versus 
       iteration under which it is stored. For instance, if set to ``1``, the 
       information computed at iteration ``n + 1`` is saved under iteration ``n``. 
       Defaults to ``0``.
    """

    def __init__(self, track_schedule, verbose=False):
        """Initialization sets the tracking schedule & creates the output dict.

        Args:
            track_schedule (callable): Function that maps the ``global_step``
                to a boolean, which determines if the quantity should be computed.
            verbose (bool, optional): Turns on verbose mode. Defaults to ``False``.
        """
        super().__init__(track_schedule, verbose=verbose)

        # Cache dictionary to save information between start and end point
        self._cache = defaultdict(dict)

    def compute(self, global_step, params, batch_loss):
        """Evaluate quantity at a step in training.

        After the computation, temporarily cached info is deleted from the internal
        cache.

        Args:
            global_step (int): The current iteration number.
            params ([torch.Tensor]): List of torch.Tensors holding the network's
                parameters.
            batch_loss (torch.Tensor): Mini-batch loss from current step.

        Returns:
            (int, arbitrary): The second value is the result that will be stored at
            the iteration indicated by the first entry (important for multi-step
            quantities whose values are computed in later iterations).
        """
        save_iter = global_step - self.SAVE_SHIFT
        save_result = self._compute(global_step, params, batch_loss)

        # assume next iteration already started to discard irrelevant information
        virtual_step = global_step + 1
        self.free_cache(global_step, virtual_step)

        return save_iter, save_result

    def _compute(self, global_step, params, batch_loss):
        """Perform start and end point computation if necessary.

        Args:
            global_step (int): The current iteration number.
            params ([torch.Tensor]): List of torch.Tensors holding the network's
                parameters.
            batch_loss (torch.Tensor): Mini-batch loss from current step.

        Returns:
            Any: Result of the computation. If ``None``, the step served to compute
            intermediate information rather than computing the quantity's value.
        """
        if self.is_start(global_step):
            self._compute_start(global_step, params, batch_loss)

        result = None
        if self.is_end(global_step):
            result = self._compute_end(global_step, params, batch_loss)

        return result

    def _compute_start(self, global_step, params, batch_loss):
        """Compute and cache information at a start point.

        Use ``self.save_to_cache`` to store information temporarily.

        Args:
            global_step (int): The current iteration number.
            params ([torch.Tensor]): List of torch.Tensors holding the network's
                parameters.
            batch_loss (torch.Tensor): Mini-batch loss from current step.

        Raises:
            NotImplementedError: Must be implemented by child classes.
        """
        raise NotImplementedError

    def _compute_end(self, global_step, params, batch_loss):
        """Compute info at end, combine with start info and return quantity's value.

        Use ``self.load_from_cache`` to retrieve cached info from the start point.

        Args:
            global_step (int): The current iteration number.
            params ([torch.Tensor]): List of torch.Tensors holding the network's
                parameters.
            batch_loss (torch.Tensor): Mini-batch loss from current step.

        Returns: # noqa: DAR202
            Any: The quantity's value.

        Raises:
            NotImplementedError: Must be implemented by child classes.
        """
        raise NotImplementedError

    def should_compute(self, global_step):
        """Return if computations need to be performed at a specific iteration.

        Args:
            global_step (int): The current iteration number.

        Returns:
            bool: Truth value whether computations need to be performed.
        """
        return self.is_start(global_step) or self.is_end(global_step)

    def is_start(self, global_step):
        """Return whether current iteration is start point.

        Args:
            global_step (int): The current iteration number.

        Returns: # noqa: DAR202
            bool: Whether ``global_step`` is a start point.

        Raises:
            NotImplementedError: Must be implemented by child classes.
        """
        raise NotImplementedError

    def is_end(self, global_step):
        """Return whether current iteration is end point.

        Args:
            global_step (int): The current iteration number.

        Returns: # noqa: DAR202
            bool: Whether ``global_step`` is an end point.

        Raises:
            NotImplementedError: Must be implemented by child classes.
        """
        raise NotImplementedError

    def save_to_cache(self, global_step, key, value, block_fn):
        """Save information to cache. Overwrite existing entry.

        Modifies the class-internal cache in ``self._cache``, a nested dictionary
        with iterations as outer, and custom keys as inner, nesting.

        Args:
            global_step (int): The iteration number associated with the cache event.
            key (str): Custom key under which ``value`` is cached.
            value (any): Object to be cached.
            block_fn (callable): Function mapping integers (iteration count) to a
                boolean. This value indicates whether deletion is blocked for the
                cached ``(key, value)`` pair at a certain iteration.
        """
        if self._verbose:
            message = f"Create cache entry ('{global_step}', '{key}')"
            print(f"{self._verbose_prefix(global_step)}: {message}")

        self._cache[global_step][key] = (value, block_fn)

    def free_cache(self, global_step, virtual_step):
        """Delete information from cache that is not required anymore.

        Removes empty iterations and unblocked entries from ``self._cache``.

        Args:
            global_step (int): Current iteration number. Used for verbose output.
            virtual_step (int): Virtual future iteration number that is used to
                determine which information can be freed.
        """
        self._free_unblocked(global_step, virtual_step)
        self._free_empty(global_step)

    def _free_unblocked(self, global_step, virtual_step):
        """Remove unblocked entries from class-internal cache.

        Modifies ``self._cache``.

        Args:
            global_step (int): Current iteration number. Used for verbose output.
            virtual_step (int): Virtual future iteration number that is used to
                determine which information can be freed.
        """
        unblocked = []

        for step, info in self._cache.items():
            for key, (_, block_fn) in info.items():
                blocked = block_fn(virtual_step)

                if not blocked:
                    unblocked.append((step, key))

        for (step, key) in unblocked:
            self._cache[step].pop(key)

            if self._verbose:
                message = f"Delete cache entry ('{step}', '{key}')"
                print(f"{self._verbose_prefix(global_step)}: {message}")

    def _free_empty(self, global_step):
        """Remove empty iteration entries from class-internal cache.

        Modifies ``self._cache``.

        Args:
            global_step (int): Current iteration number. Used for verbose output.
        """
        empty = [key for key, value in self._cache.items() if value == {}]

        for key in empty:
            self._cache.pop(key)

            if self._verbose:
                message = f"Delete empty cache ('{key}')"
                print(f"{self._verbose_prefix(global_step)}: {message}")

    def load_from_cache(self, global_step, key):
        """Load cached information.

        Args:
            global_step (int): The iteration number associated with the cache event.
            key (str): Custom key under which the information is cached.

        Returns:
            Any: Cached item.
        """
        value, _ = self._cache[global_step][key]

        return value
