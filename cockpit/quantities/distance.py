"""Class for tracking the Paramter Distances."""

from cockpit.quantities.quantity import Quantity
from cockpit.quantities.utils_quantities import _root_sum_of_squares


class Distance(Quantity):
    """Parameter Distance Quantitiy Class."""

    _positions = ["start", "end"]
    _start_end_difference = 1

    def extensions(self, global_step):
        """Return list of BackPACK extensions required for the computation.

        Args:
            global_step (int): The current iteration number.

        Returns:
            list: (Potentially empty) list with required BackPACK quantities.
        """
        return []

    # TODO Rewrite to use parent class track method
    def track(self, global_step, params, batch_loss):
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
            self.parameter_init = [p.data.clone().detach() for p in params]

        if self._track_schedule(global_step):
            d2init = [
                (init - p).norm(2).item()
                for init, p in zip(self.parameter_init, params)
                if p.requires_grad
            ]
            self.output[global_step]["d2init"] = d2init

            if self._verbose:
                print(
                    f"[Step {global_step}] D2Init: {_root_sum_of_squares(d2init):.4f}"
                )

    def _compute_update_size(self, global_step, params, batch_loss):
        """Evaluate the size of the last parameter update.

        Args:
            global_step (int): The current iteration number.
            params ([torch.Tensor]): List of torch.Tensors holding the network's
                parameters.
            batch_loss (torch.Tensor): Mini-batch loss from current step.
        """
        if self._is_position(global_step, "end"):
            update_size = [
                (old_p - p).norm(2).item() for old_p, p in zip(self.old_params, params)
            ]
            self.output[global_step - self._start_end_difference][
                "update_size"
            ] = update_size

            if self._verbose:
                print(
                    f"[Step {global_step}] Update size:"
                    + f" {_root_sum_of_squares(update_size):.4f}"
                )
            del self.old_params

        if self._is_position(global_step, "start"):
            self.old_params = [p.data.clone().detach() for p in params]

    def _is_position(self, global_step, pos):
        """Return whether current iteration is start/end of update size computation."""
        if pos == "start":
            step = global_step
        elif pos == "end":
            step = global_step - self._start_end_difference
        else:
            raise ValueError(f"Invalid position '{pos}'. Expect {self._positions}.")

        return self._track_schedule(step)
