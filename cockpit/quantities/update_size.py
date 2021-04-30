"""Class for tracking the update size."""

from cockpit.quantities.quantity import TwoStepQuantity


class UpdateSize(TwoStepQuantity):
    """Quantity class for tracking parameter update sizes."""

    CACHE_KEY = "params"
    """str: String under which the parameters are cached for computation.
       Default: ``'params'``.
    """
    SAVE_SHIFT = 1
    """int: Difference between iteration at which information is computed versus
       iteration under which it is stored. For instance, if set to ``1``, the
       information computed at iteration ``n + 1`` is saved under iteration ``n``.
       Defaults to ``1``.
    """

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
        return self._track_schedule(global_step - self.SAVE_SHIFT)

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
            return 0 <= step - global_step <= self.SAVE_SHIFT

        self.save_to_cache(global_step, self.CACHE_KEY, params_copy, block_fn)

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
            global_step - self.SAVE_SHIFT, self.CACHE_KEY
        )

        update_size = [
            (p.data - p_start).norm(2).item()
            for p, p_start in zip(params, params_start)
        ]

        return update_size
