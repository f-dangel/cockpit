"""Classes for tracking the distance to initialization."""

from cockpit.quantities.quantity import SingleStepQuantity


class Distance(SingleStepQuantity):
    """Distance from initialization Quantity Class."""

    def extensions(self, global_step):
        """Return list of BackPACK extensions required for the computation.

        Args:
            global_step (int): The current iteration number.

        Returns:
            list: (Potentially empty) list with required BackPACK quantities.
        """
        return []

    def should_compute(self, global_step):
        """Return if computations need to be performed at a specific iteration.

        Args:
            global_step (int): The current iteration number.

        Returns:
            bool: Truth value whether computations need to be performed.
        """
        return global_step == 0 or self._track_schedule(global_step)

    def _compute(self, global_step, params, batch_loss):
        """Evaluate the update size."""
        distance = None

        # Store initial parameters
        if global_step == 0:
            self.params_init = [p.data.clone().detach() for p in params]

        if self._track_schedule(global_step):
            distance = [
                (init - p).norm(2).item()
                for init, p in zip(self.params_init, params)
                if p.requires_grad
            ]

        return distance
