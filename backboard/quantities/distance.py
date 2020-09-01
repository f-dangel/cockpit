"""Class for tracking the Paramter Distances."""

from backboard.quantities.quantity import Quantity


class Distance(Quantity):
    """Parameter Distance Quantitiy Class."""

    def extensions(self, global_step):
        """Return list of BackPACK extensions required for the computation.

        Args:
            global_step (int): The current iteration number.

        Returns:
            list: (Potentially empty) list with required BackPACK quantities.
        """
        return []

    def compute(self, global_step, params, batch_loss):
        """Evaluate the current parameter distances.

        We track both the distance to the initialization, as well as the size of
        the last parameter update.

        Args:
            global_step (int): The current iteration number.
            params ([torch.Tensor]): List of parameters considered in the computation.
            batch_loss (torch.Tensor): Mini-batch loss from current step.
        """
        self._compute_d2init(global_step, params, batch_loss)
        self._compute_update_size(global_step, params, batch_loss)

    def _compute_d2init(self, global_step, params, batch_loss):
        """Evaluate the parameter distances to its initialization.

        Args:
            global_step (int): The current iteration number.
            params ([torch.Tensor]): List of parameters considered in the computation.
            batch_loss (torch.Tensor): Mini-batch loss from current step.
        """
        # Store initial parameters
        if global_step == 0:
            self.parameter_init = [p.data.clone().detach() for p in params()]

        if global_step % self._track_interval == 0:
            self.output[global_step]["d2init"] = [
                (init - p).norm(2).item()
                for init, p in zip(self.parameter_init, params())
                if p.requires_grad
            ]
        else:
            pass

    def _compute_update_size(self, global_step, params, batch_loss):
        """Evaluate the size of the last parameter update.

        Args:
            global_step (int): The current iteration number.
            params ([torch.Tensor]): List of parameters considered in the computation.
            batch_loss (torch.Tensor): Mini-batch loss from current step.
        """
        pass
        # Store initial parameters
        # if global_step == 0:
        #     self.parameter_init = [p.data.clone().detach() for p in params()]

        # if global_step % self._track_interval == 0:
        #     self.output[global_step]["d2init"] = [
        #         (init - p).norm(2).item()
        #         for init, p in zip(self.parameter_init, params())
        #         if p.requires_grad
        #     ]
        # else:
        #     pass
