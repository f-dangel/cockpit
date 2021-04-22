"""Class for tracking the CABS criterion for adaptive batch size."""

from backpack.extensions import BatchGrad

from cockpit.context import get_batch_size, get_optimizer
from cockpit.quantities.quantity import SingleStepQuantity
from cockpit.quantities.utils_transforms import BatchGradTransformsHook_SumGradSquared
from cockpit.utils.optim import ComputeStep


class CABS(SingleStepQuantity):
    """CABS Quantity class for the suggested batch size using the CABS criterion.

    CABS uses the current learning rate and variance of the stochastic gradients
    to suggest an optimal batch size.

    Only applies to SGD without momentum.

    Note: Proposed in

        - Balles, L., Romero, J., & Hennig, P.,
          Coupling adaptive batch sizes with learning rates (2017).
    """

    def get_lr(self, optimizer):
        """Extract the learning rate.

        Args:
            optimizer (torch.optim.Optimizer): A PyTorch optimizer.

        Returns:
            float: Learning rate

        Raises:
            ValueError: If the learning rate varies over parameter groups.
        """
        lrs = {group["lr"] for group in optimizer.param_groups}

        if len(lrs) != 1:
            raise ValueError(f"Found non-unique learning rates {lrs}")

        return lrs.pop()

    def extensions(self, global_step):
        """Return list of BackPACK extensions required for the computation.

        Args:
            global_step (int): The current iteration number.

        Returns:
            list: (Potentially empty) list with required BackPACK quantities.
        """
        ext = []

        if self.should_compute(global_step):
            ext.append(BatchGrad())

        return ext

    def extension_hooks(self, global_step):
        """Return list of BackPACK extension hooks required for the computation.

        Args:
            global_step (int): The current iteration number.

        Returns:
            [callable]: List of required BackPACK extension hooks for the current
                iteration.
        """
        hooks = []

        if self.should_compute(global_step):
            hooks.append(BatchGradTransformsHook_SumGradSquared())

        return hooks

    def _compute(self, global_step, params, batch_loss):
        """Compute the CABS rule. Return suggested batch size.

        Evaluates Equ. 22 of

        - Balles, L., Romero, J., & Hennig, P.,
          Coupling adaptive batch sizes with learning rates (2017).

        Args:
            global_step (int): The current iteration number.
            params ([torch.Tensor]): List of torch.Tensors holding the network's
                parameters.
            batch_loss (torch.Tensor): Mini-batch loss from current step.

        Returns:
            float: Batch size suggested by CABS.

        Raises:
            ValueError: If the optimizer differs from SGD with default arguments.
        """
        optimizer = get_optimizer(global_step)
        if not ComputeStep.is_sgd_default_kwargs(optimizer):
            raise ValueError("This criterion only supports zero-momentum SGD.")

        B = get_batch_size(global_step)
        lr = self.get_lr(optimizer)

        grad_squared = self._fetch_grad(params, aggregate=True) ** 2
        # # compensate BackPACK's 1/B scaling
        sgs_compensated = (
            B ** 2
            * self._fetch_sum_grad_squared_via_batch_grad_transforms(
                params, aggregate=True
            )
        )

        return (
            lr * (sgs_compensated - B * grad_squared).sum() / (B * batch_loss)
        ).item()
