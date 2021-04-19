"""Class for tracking the EB criterion for early stopping."""

from backpack.extensions import BatchGrad

from cockpit.context import get_batch_size, get_optimizer
from cockpit.quantities.quantity import SingleStepQuantity
from cockpit.quantities.utils_transforms import BatchGradTransformsHook_SumGradSquared
from cockpit.utils.optim import ComputeStep


class EarlyStopping(SingleStepQuantity):
    """Quantity class for the evidence-based early-stopping criterion.

    This criterion uses local statistics of the gradients to indicate when training
    should be stopped. If the criterion exceeds zero, training should be stopped.

    Note: Proposed in

        - Mahsereci, M., Balles, L., Lassner, C., & Hennig, P.,
          Early stopping without a validation set (2017).
    """

    def __init__(self, track_schedule, verbose=False, epsilon=1e-5):
        """Initialization sets the tracking schedule & creates the output dict.

        Args:
            track_schedule (callable): Function that maps the ``global_step``
                to a boolean, which determines if the quantity should be computed.
            verbose (bool, optional): Turns on verbose mode. Defaults to ``False``.
            epsilon (float): Stabilization constant. Defaults to 0.0.
        """
        super().__init__(track_schedule, verbose=verbose)

        self._epsilon = epsilon

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
        """Compute the EB early stopping criterion.

        Evaluates the left hand side of Equ. 7 in

        - Mahsereci, M., Balles, L., Lassner, C., & Hennig, P.,
          Early stopping without a validation set (2017).

        If this value exceeds 0, training should be stopped.

        Args:
            global_step (int): The current iteration number.
            params ([torch.Tensor]): List of torch.Tensors holding the network's
                parameters.
            batch_loss (torch.Tensor): Mini-batch loss from current step.

        Returns:
            float: Result of the Early stopping criterion. Training should stop
                if it is larger than 0.

        Raises:
            ValueError: If the used optimizer differs from SGD with default parameters.
        """
        if not ComputeStep.is_sgd_default_kwargs(get_optimizer(global_step)):
            raise ValueError("This criterion only supports zero-momentum SGD.")

        B = get_batch_size(global_step)

        grad_squared = self._fetch_grad(params, aggregate=True) ** 2

        # compensate BackPACK's 1/B scaling
        sgs_compensated = (
            B ** 2
            * self._fetch_sum_grad_squared_via_batch_grad_transforms(
                params, aggregate=True
            )
        )

        diag_variance = (sgs_compensated - B * grad_squared) / (B - 1)

        snr = grad_squared / (diag_variance + self._epsilon)

        return 1 - B * snr.mean().item()
