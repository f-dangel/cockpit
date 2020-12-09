"""Class for tracking the Individual Parameters."""

from cockpit.quantities.quantity import ByproductQuantity


class Parameters(ByproductQuantity):
    """Parameter Quantitiy Class."""

    # TODO Rewrite to use parent class track method
    def track(self, global_step, params, batch_loss):
        """Store the current parameter.

        Args:
            global_step (int): The current iteration number.
            params ([torch.Tensor]): List of torch.Tensors holding the network's
                parameters.
            batch_loss (torch.Tensor): Mini-batch loss from current step.
        """
        if self.is_active(global_step):
            parameters = [p.data.tolist() for p in params]
            self.output[global_step]["params"] = parameters

            if self._verbose:
                print(f"[Step {global_step}] Parameters: {parameters}")
