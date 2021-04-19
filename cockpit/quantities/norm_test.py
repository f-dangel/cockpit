"""Class for tracking the Norm Test."""

from backpack.extensions import BatchGrad

from cockpit.quantities.quantity import SingleStepQuantity
from cockpit.quantities.utils_transforms import BatchGradTransformsHook_BatchL2Grad


class NormTest(SingleStepQuantity):
    """Quantitiy Class for the norm test.

    Note: Norm test as proposed in

        - Byrd, R., Chin, G., Nocedal, J., & Wu, Y.,
          Sample size selection in optimization methods for machine learning (2012).
          https://link.springer.com/article/10.1007%2Fs10107-012-0572-5
    """

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
            hooks.append(BatchGradTransformsHook_BatchL2Grad())

        return hooks

    def _compute(self, global_step, params, batch_loss):
        """Track the practical version of the norm test.

        Return maximum θ for which the norm test would pass.

        The norm test is defined by Equation (3.9) in byrd2012sample.

        Args:
            global_step (int): The current iteration number.
            params ([torch.Tensor]): List of torch.Tensors holding the network's
                parameters.
            batch_loss (torch.Tensor): Mini-batch loss from current step.

        Returns:
            float: Maximum θ for which the norm test would pass.
        """
        batch_l2_squared = self._fetch_batch_l2_squared_via_batch_grad_transforms(
            params, aggregate=True
        )
        grad_l2_squared = self._fetch_grad_l2_squared(params, aggregate=True)
        batch_size = batch_l2_squared.size(0)

        var_l1 = self._compute_variance_l1(
            batch_size, batch_l2_squared, grad_l2_squared
        )

        return self._compute_theta_max(batch_size, var_l1, grad_l2_squared).item()

    def _compute_theta_max(self, batch_size, var_l1, grad_l2_squared):
        """Return maximum θ for which the norm test would pass.

        Args:
            batch_size (int): Mini-batch size.
            var_l1 (torch.Tensor): [description]
            grad_l2_squared (torch.Tensor): Squared ℓ₂ norm of mini-batch gradient.

        Returns:
            [type]: [description]
        """
        return (var_l1 / batch_size / grad_l2_squared).sqrt()

    def _compute_variance_l1(self, batch_size, batch_l2_squared, grad_l2_squared):
        """Compute the sample variance ℓ₁ norm.

        It shows up in Equations (3.9) and (3.11) in byrd2012sample and relies
        on the sample variance (Equation 3.6). The ℓ₁ norm can be computed using
        individual gradient squared ℓ₂ norms and the mini-batch gradient squared
        ℓ₂ norm.

        Args:
            batch_size (int): Mini-batch size.
            batch_l2_squared (torch.Tensor): [description]
            grad_l2_squared (torch.Tensor): Squared ℓ₂ norm of mini-batch gradient.

        Returns:
            torch.Tensor: The sample variance ℓ₁ norm.
        """
        return (1 / (batch_size - 1)) * (
            batch_size ** 2 * batch_l2_squared.sum() - batch_size * grad_l2_squared
        )
