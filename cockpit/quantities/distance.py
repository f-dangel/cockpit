"""Class for tracking distance to initialization."""

from cockpit.quantities.quantity import TwoStepQuantity


class Distance(TwoStepQuantity):
    """Distance from initialization Quantity Class."""

    def __init__(self, track_schedule, verbose=False):
        """Initialize the Quantity by storing the track interval.

        Crucially, it creates the output dictionary, that is meant to store all
        values that should be stored.

        Args:
            track_schedule (callable): Function that maps the ``global_step``
                to a boolean, which determines if the quantity should be computed.
            verbose (bool, optional): Turns on verbose mode. Defaults to ``False``.
        """
        save_shift = 0
        super().__init__(save_shift, track_schedule, verbose=verbose)

        self._cache_key = "params"
        self._init_global_step = 0

    def extensions(self, global_step):
        """Return list of BackPACK extensions required for the computation.

        Args:
            global_step (int): The current iteration number.

        Returns:
            list: (Potentially empty) list with required BackPACK quantities.
        """
        return []

    def is_start(self, global_step):
        """Return whether current iteration is start point.

        Only the initializtion (first iteration) is a start point.

        Args:
            global_step (int): The current iteration number.

        Returns:
            bool: Whether ``global_step`` is a start point.
        """
        return global_step == self._init_global_step

    def is_end(self, global_step):
        """Return whether current iteration is end point.

        Args:
            global_step (int): The current iteration number.

        Returns:
            bool: Whether ``global_step`` is an end point.
        """
        return self._track_schedule(global_step)

    def _compute_start(self, global_step, params, batch_loss):
        """Perform computations at start point (store initial parameter values).

        Modifies ``self._cache``.

        Args:
            global_step (int): The current iteration number.
            params ([torch.Tensor]): List of torch.Tensors holding the network's
                parameters.
            batch_loss (torch.Tensor): Mini-batch loss from current step.
        """
        params_copy = [p.data.clone().detach() for p in params]

        def block_fn(step):
            """Block deletion of parameters for all non-negative iterations.

            Args:
                step (int): Iteration number.

            Returns:
                bool: Whether deletion is blocked in the specified iteration
            """
            return step >= self._init_global_step

        self.save_to_cache(global_step, self._cache_key, params_copy, block_fn)

    def _compute_end(self, global_step, params, batch_loss):
        """Compute and return the current distance to initialization.

        Args:
            global_step (int): The current iteration number.
            params ([torch.Tensor]): List of torch.Tensors holding the network's
                parameters.
            batch_loss (torch.Tensor): Mini-batch loss from current step.

        Returns:
            [float]: Layer-wise L2-distances to initialization.
        """
        params_init = self.load_from_cache(self._init_global_step, self._cache_key)

        distance = [
            (p.data - p_init).norm(2).item() for p, p_init in zip(params, params_init)
        ]

        return distance
