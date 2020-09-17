"""Class for tracking the time."""

import time

from backboard.quantities.quantity import ByproductQuantity


class Time(ByproductQuantity):
    """Time Quantity Class."""

    def __init__(self, track_interval=1, track_offset=0, verbose=False):
        super().__init__(
            track_interval=track_interval, track_offset=track_offset, verbose=verbose
        )
        self._last = None

    def compute(self, global_step, params, batch_loss):
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
                print(f"Time: {current:.3f}s, elapsed since last: {elapsed:.3f}s")

            self._last = current
