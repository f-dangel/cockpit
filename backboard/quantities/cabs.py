"""Class for tracking the CABS criterion for adaptive batch size."""

from backboard.context import get_batch_size
from backboard.quantities.quantity import SingleStepQuantity
from backpack import extensions


class CABS(SingleStepQuantity):
    """CABS criterion, coupling adaptive batch sizes and learning rates.

    Only applies to SGD without momentum.

    Note: Proposed in

        - Balles, L., Romero, J., & Hennig, P.,
          Coupling adaptive batch sizes with learning rates (2017).
    """

    def __init__(
        self,
        track_interval=1,
        track_offset=0,
        lr=1.0,
        verbose=False,
        track_schedule=None,
    ):
        """Initialize.

        Args:
            track_interval (int): Tracking rate.
            lr (float): Learning rate. Defaults to 1.0.
            verbose (bool): Turns on verbose mode. Defaults to ``False``.
        """
        super().__init__(
            track_interval=track_interval,
            track_offset=track_offset,
            verbose=verbose,
            track_schedule=track_schedule,
        )
        self.set_lr(lr)

    def set_lr(self, lr):
        """Set value for current learning rate."""
        self._lr = lr

    def extensions(self, global_step):
        """Return list of BackPACK extensions required for the computation.

        Args:
            global_step (int): The current iteration number.

        Returns:
            list: (Potentially empty) list with required BackPACK quantities.
        """
        ext = []

        if self.is_active(global_step):
            ext.append(extensions.SumGradSquared())

        return ext

    def compute(self, global_step, params, batch_loss):
        """Evaluate CABS rule.

        Args:
            global_step (int): The current iteration number.
            params ([torch.Tensor]): List of torch.Tensors holding the network's
                parameters.
            batch_loss (torch.Tensor): Mini-batch loss from current step.
        """
        if self.is_active(global_step):
            cabs = self._compute(global_step, params, batch_loss).item()

            if self._verbose:
                print(f"[Step {global_step}] CABS(lr={self._lr}): {cabs:.4f}")

            self.output[global_step]["cabs"] = cabs

    def _compute(self, global_step, params, batch_loss):
        """Compute the CABS rule. Return suggested batch size.

        Evaluates Equ. 22 of

        - Balles, L., Romero, J., & Hennig, P.,
          Coupling adaptive batch sizes with learning rates (2017).
        """
        B = get_batch_size(global_step)
        lr = self._lr

        grad_squared = self._fetch_grad(params, aggregate=True) ** 2
        # # compensate BackPACK's 1/B scaling
        sgs_compensated = B ** 2 * self._fetch_sum_grad_squared(params, aggregate=True)

        return lr * (sgs_compensated - B * grad_squared).sum() / (B * batch_loss.item())
