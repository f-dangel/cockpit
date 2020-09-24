"""Class for tracking the Trace of the Hessian."""

from backboard.quantities.quantity import SingleStepQuantity
from backpack import extensions


class Trace(SingleStepQuantity):
    """Trace Quantitiy Class."""

    def extensions(self, global_step):
        """Return list of BackPACK extensions required for the computation.

        Args:
            global_step (int): The current iteration number.

        Returns:
            list: (Potentially empty) list with required BackPACK quantities.
        """
        if self.is_active(global_step):
            ext = [extensions.DiagHessian()]
        else:
            ext = []

        return ext

    def compute(self, global_step, params, batch_loss):
        """Evaluate the trace of the Hessian at the current point.

        Args:
            global_step (int): The current iteration number.
            params ([torch.Tensor]): List of torch.Tensors holding the network's
                parameters.
            batch_loss (torch.Tensor): Mini-batch loss from current step.
        """
        if self.is_active(global_step):
            trace = [p.diag_h.sum().item() for p in params]

            if self._verbose:
                print(f"[Step {global_step}] Trace: {sum(trace):.4f}")

            self.output[global_step]["trace"] = trace
