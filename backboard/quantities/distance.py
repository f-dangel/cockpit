"""Class for tracking the Paramter Distances."""

from copy import deepcopy

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
            params ([torch.Tensor]): List of torch.Tensors holding the network's
                parameters.
            batch_loss (torch.Tensor): Mini-batch loss from current step.
        """
        self._compute_d2init(global_step, params, batch_loss)
        self._compute_update_size(global_step, params, batch_loss)

    def _compute_d2init(self, global_step, params, batch_loss):
        """Evaluate the parameter distances to its initialization.

        Args:
            global_step (int): The current iteration number.
            params ([torch.Tensor]): List of torch.Tensors holding the network's
                parameters.
            batch_loss (torch.Tensor): Mini-batch loss from current step.
        """
        # Store initial parameters
        if global_step == 0:
            self.parameter_init = deepcopy(params)

        if global_step % self._track_interval == 0:
            d2init = [
                (init - p).norm(2).item()
                for init, p in zip(self.parameter_init, params)
                if p.requires_grad
            ]
            self.output[global_step]["d2init"] = d2init

            if self._verbose:
                print(f"Distance to initialization: {sum(d2init):.4f}")
        else:
            pass

    def _compute_update_size(self, global_step, params, batch_loss):
        """Evaluate the size of the last parameter update.

        Args:
            global_step (int): The current iteration number.
            params ([torch.Tensor]): List of torch.Tensors holding the network's
                parameters.
            batch_loss (torch.Tensor): Mini-batch loss from current step.
        """
        if self._track_interval == 1:
            # Special case if we want to track every iteration, since then the two
            # computation steps of the quantity overlap.
            if hasattr(self, "old_params"):
                # compute update size of last (!) step
                update_size = [
                    (old_p - p).norm(2).item()
                    for old_p, p in zip(self.old_params, params)
                ]
                self.output[global_step - 1]["update_size"] = update_size

                if self._verbose:
                    print(f"Update size: {sum(update_size):.4f}")
            # store current parameters
            self.old_params = deepcopy(params)
        else:
            if global_step % self._track_interval == 0:
                # store current parameters
                self.old_params = deepcopy(params)
            elif global_step % self._track_interval == 1:
                # Compute update size
                update_size = [
                    (old_p - p).norm(2).item()
                    for old_p, p in zip(self.old_params, params)
                ]
                self.output[global_step - 1]["update_size"] = update_size

                if self._verbose:
                    print(f"Update size: {sum(update_size):.4f}")
