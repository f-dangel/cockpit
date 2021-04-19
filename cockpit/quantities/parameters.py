"""Class for tracking the Individual Parameters."""

from cockpit.quantities.quantity import ByproductQuantity


class Parameters(ByproductQuantity):
    """Parameter Quantitiy class tracking the current parameters in each iteration."""

    def _compute(self, global_step, params, batch_loss):
        """Store the current parameter.

        Args:
            global_step (int): The current iteration number.
            params ([torch.Tensor]): List of torch.Tensors holding the network's
                parameters.
            batch_loss (torch.Tensor): Mini-batch loss from current step.

        Returns:
            list: Current model parameters.
        """
        return [p.data.tolist() for p in params]
