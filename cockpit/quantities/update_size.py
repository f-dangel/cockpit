"""Class for tracking the update size."""

from cockpit.quantities.quantity import TwoStepQuantity


class UpdateSize(TwoStepQuantity):
    """Quantity for tracking parameter update sizes."""

    def __init__(self, track_schedule, verbose=False):
        """Initialize the Quantity by storing the track interval.

        Crucially, it creates the output dictionary, that is meant to store all
        values that should be stored.

        Args:
            track_schedule (callable): Function that maps the ``global_step``
                to a boolean, which determines if the quantity should be computed.
            verbose (bool, optional): Turns on verbose mode. Defaults to ``False``.
        """
        save_shift = 1
        super().__init__(save_shift, track_schedule, verbose=verbose)

        self._cache_key = "params"

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
        return self._track_schedule(global_step - self._save_shift)

    def _compute_start(self, global_step, params, batch_loss):
        """Perform computations at start point (store current parameter values).

        Modifies ``self._cache``.

        Args:
            global_step (int): The current iteration number.
            params ([torch.Tensor]): List of torch.Tensors holding the network's
                parameters.
            batch_loss (torch.Tensor): Mini-batch loss from current step.
        """
        params_copy = [p.data.clone().detach() for p in params]

        def block_fn(step):
            """Block deletion of parameters for current and next iteration.

            Args:
                step (int): Iteration number.

            Returns:
                bool: Whether deletion is blocked in the specified iteration
            """
            return 0 <= step - global_step <= self._save_shift

        self.save_to_cache(global_step, self._cache_key, params_copy, block_fn)

    def _compute_end(self, global_step, params, batch_loss):
        """Compute and return update size.

        Args:
            global_step (int): The current iteration number.
            params ([torch.Tensor]): List of torch.Tensors holding the network's
                parameters.
            batch_loss (torch.Tensor): Mini-batch loss from current step.

        Returns:
            [float]: Layer-wise L2-norms of parameter updates.
        """
        params_start = self.load_from_cache(
            global_step - self._save_shift, self._cache_key
        )

        update_size = [
            (p.data - p_start).norm(2).item()
            for p, p_start in zip(params, params_start)
        ]

        return update_size
