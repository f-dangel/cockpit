"""Classes for tracking the Parameter Distances."""

from cockpit.quantities.quantity import Quantity, SingleStepQuantity


class UpdateSize(Quantity):
    """Update size Quantity Class."""

    _start_end_difference = 1

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
        return self._is_start(global_step) or self._is_end(global_step)

    def _is_start(self, global_step):
        """Return whether current iteration is start of update size computation."""
        return self._track_schedule(global_step)

    def _is_end(self, global_step):
        """Return whether current iteration is end of update size computation."""
        return self._track_schedule(global_step - self._start_end_difference)

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
        update_size = None

        if self._is_end(global_step):
            update_size = [
                (old_p - p).norm(2).item() for old_p, p in zip(self.old_params, params)
            ]
            del self.old_params

        if self._is_start(global_step):
            self.old_params = [p.data.clone().detach() for p in params]

        return global_step - self._start_end_difference, update_size


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
