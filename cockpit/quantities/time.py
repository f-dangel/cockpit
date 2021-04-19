"""Class for tracking the time."""

import time

from cockpit.quantities.quantity import ByproductQuantity


class Time(ByproductQuantity):
    """Time Quantity Class tracking the time during training."""

    def _compute(self, global_step, params, batch_loss):
        """Return the time at the current point.

        Args:
            global_step (int): The current iteration number.
            params ([torch.Tensor]): List of torch.Tensors holding the network's
                parameters.
            batch_loss (torch.Tensor): Mini-batch loss from current step.

        Returns:
            float: Current time as given by ``time.time()``.
        """
        return time.time()
