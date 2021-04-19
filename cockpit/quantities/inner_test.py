"""Class for tracking the Inner Product Test."""


from backpack.extensions import BatchGrad

from cockpit.quantities.quantity import SingleStepQuantity
from cockpit.quantities.utils_transforms import BatchGradTransformsHook_BatchDotGrad


class InnerTest(SingleStepQuantity):
    """Quantitiy Class for tracking the result of the inner product test.

    Note: Inner Product test as proposed in

        - Bollapragada, R., Byrd, R., &  Nocedal, J.,
          Adaptive Sampling Strategies for Stochastic Optimization (2017).
          https://arxiv.org/abs/1710.11258
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
            hooks.append(BatchGradTransformsHook_BatchDotGrad())

        return hooks

    def _compute(self, global_step, params, batch_loss):
        """Track the practical version of the inner product test.

        Return maximum θ for which the inner product test would pass.

        The inner product test is defined by Equation (2.6) in bollapragada2017adaptive.

        Args:
            global_step (int): The current iteration number.
            params ([torch.Tensor]): List of torch.Tensors holding the network's
                parameters.
            batch_loss (torch.Tensor): Mini-batch loss from current step.

        Returns:
            float: Maximum θ for which the inner product test would pass.
        """
        batch_dot = self._fetch_batch_dot_via_batch_grad_transforms(
            params, aggregate=True
        )
        grad_l2_squared = self._fetch_grad_l2_squared(params, aggregate=True)
        batch_size = batch_dot.size(0)

        var_projection = self._compute_projection_variance(
            batch_size, batch_dot, grad_l2_squared
        )

        return self._compute_theta_max(
            batch_size, var_projection, grad_l2_squared
        ).item()

    def _compute_theta_max(self, batch_size, var_projection, grad_l2_squared):
        """Return maximum θ for which the inner product test would pass.

        Args:
            batch_size (int): Mini-batch size.
            var_projection (torch.Tensor): The sample variance of individual
                gradient projections on the mini-batch gradient.
            grad_l2_squared (torch.Tensor): Squared ℓ₂ norm of mini-batch gradient.

        Returns:
            [type]: [description]
        """
        return (var_projection / batch_size / grad_l2_squared ** 2).sqrt()

    def _compute_projection_variance(self, batch_size, batch_dot, grad_l2_squared):
        """Compute sample variance of individual gradient projections onto the gradient.

        The sample variance of projections is given by Equation (line after 2.6) in
        bollapragada2017adaptive (https://arxiv.org/pdf/1710.11258.pdf)

        Args:
            batch_size (int): Mini-batch size.
            batch_dot (torch.Tensor): Individual gradient pairwise dot product.
            grad_l2_squared (torch.Tensor): Squared ℓ₂ norm of mini-batch gradient.

        Returns:
            torch.Tensor: The sample variance of individual gradient projections on the
                mini-batch gradient.
        """
        projections = batch_size * batch_dot.sum(1)

        return (1 / (batch_size - 1)) * (
            (projections ** 2).sum() - batch_size * grad_l2_squared ** 2
        )
