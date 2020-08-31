"""Class for tracking the loss."""

from .quantity import Quantity


class Loss(Quantity):
    """Loss Quantitiy Class."""

    def extensions(self, global_step):
        """Return list of BackPACK extensions required for the computation.

        Args:
            global_step (int): The current iteration number.

        Returns:
            list: (Potentially empty) list with required BackPACK quantities.
        """
        return []

    def compute(self, global_step, params, batch_loss):
        """Track the loss at the current point.

        Args:
            global_step (int): The current iteration number.
            params ([torch.Tensor]): List of parameters considered in the computation.
            batch_loss (torch.Tensor): Mini-batch loss from current step.

        Returns:
            torch.Tensor: The quantity's value.
        """
        if global_step % self._track_interval == 0:
            self.output[global_step]["mini_batch_loss"] = [batch_loss.item()]
        else:
            pass
