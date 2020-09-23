"""Class for tracking the Individual Parameters."""

from backboard.quantities.quantity import Quantity


class Parameters(Quantity):
    """Parameter Quantitiy Class."""

    def extensions(self, global_step):
        """Return list of BackPACK extensions required for the computation.

        Args:
            global_step (int): The current iteration number.

        Returns:
            list: (Potentially empty) list with required BackPACK quantities.
        """
        return []

    def compute(self, global_step, params, batch_loss):
        """Store the current parameter.

        Args:
            global_step (int): The current iteration number.
            params ([torch.Tensor]): List of torch.Tensors holding the network's
                parameters.
            batch_loss (torch.Tensor): Mini-batch loss from current step.
        """
        if global_step % self._track_interval == 0:
            self.output[global_step]["params"] = [p.data.tolist() for p in params]
