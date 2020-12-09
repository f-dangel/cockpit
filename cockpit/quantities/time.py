"""Class for tracking the time."""

import time

from cockpit.quantities.quantity import ByproductQuantity


class Time(ByproductQuantity):
    """Time Quantity Class."""

    def __init__(self, track_schedule, verbose=False):
        super().__init__(track_schedule, verbose=verbose)

        self._last = None

    # TODO Rewrite to use parent class track method
    def track(self, global_step, params, batch_loss):
        """Track the time at the current point.

        Args:
            global_step (int): The current iteration number.
            params ([torch.Tensor]): List of torch.Tensors holding the network's
                parameters.
            batch_loss (torch.Tensor): Mini-batch loss from current step.
        """
        if self.is_active(global_step):
            current = time.time()
            self.output[global_step]["time"] = current

            if self._verbose and self._last is not None:
                elapsed = current - self._last
                print(
                    f"[Step {global_step}] Time: {current:.3f}s, Last: {elapsed:.3f}s"
                )

            self._last = current
