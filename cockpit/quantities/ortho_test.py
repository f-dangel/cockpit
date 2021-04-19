"""Class for tracking the Orthogonality Test."""

from backpack.extensions import BatchGrad

from cockpit.quantities.quantity import SingleStepQuantity
from cockpit.quantities.utils_transforms import BatchGradTransformsHook_BatchDotGrad


class OrthoTest(SingleStepQuantity):
    """Quantity Class for the orthogonality test.

    Note: Orthogonality test as proposed in

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
        """Track the practical version of the orthogonality test.

        Return maximum ν for which the orthogonality test would pass.

        The orthogonality test is defined by Equation (3.3) in bollapragada2017adaptive.

        Args:
            global_step (int): The current iteration number.
            params ([torch.Tensor]): List of torch.Tensors holding the network's
                parameters.
            batch_loss (torch.Tensor): Mini-batch loss from current step.

        Returns:
            float: Maximum ν for which the orthogonality test would pass.
        """
        batch_dot = self._fetch_batch_dot_via_batch_grad_transforms(
            params, aggregate=True
        )
        batch_size = batch_dot.size(0)
        grad_l2_squared = self._fetch_grad_l2_squared(params, aggregate=True)

        var_orthogonal_projection = self._compute_orthogonal_projection_variance(
            batch_size, batch_dot, grad_l2_squared
        )

        return self._compute_nu_max(
            batch_size, var_orthogonal_projection, grad_l2_squared
        ).item()

    def _compute_nu_max(self, batch_size, var_orthogonal_projection, grad_l2_squared):
        """Return maximum ν for which the orthogonality test would pass.

        The orthogonality test is defined by Equation (3.3) in
        bollapragada2017adaptive.

        Args:
            batch_size (int): Mini-batch size.
            var_orthogonal_projection (torch.Tensor): [description]
            grad_l2_squared (torch.Tensor): Squared ℓ₂ norm of mini-batch gradient.

        Returns:
            [type]: Maximum ν for which the orthogonality test would pass.
        """
        return (var_orthogonal_projection / batch_size / grad_l2_squared).sqrt()

    def _compute_orthogonal_projection_variance(
        self, batch_size, batch_dot, grad_l2_squared
    ):
        """Compute sample variance of individual gradient orthogonal projections.

        The sample variance of orthogonal projections shows up in Equation (3.3) in
        bollapragada2017adaptive (https://arxiv.org/pdf/1710.11258.pdf)

        Args:
            batch_size (int): Mini-batch size.
            batch_dot (torch.Tensor): Individual gradient pairwise dot product.
            grad_l2_squared (torch.Tensor): Squared ℓ₂ norm of mini-batch gradient.

        Returns:
            torch.Tensor: The sample variance of individual gradient orthogonal
                projections on the mini-batch gradient.
        """
        batch_l2_squared = batch_dot.diag()
        projections = batch_size * batch_dot.sum(1)

        return (1 / (batch_size - 1)) * (
            batch_size ** 2 * batch_l2_squared.sum()
            - (projections ** 2 / grad_l2_squared).sum()
        )
