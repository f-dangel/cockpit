"""Class for tracking the Inner Product Test."""

import torch

from cockpit.context import get_batch_size
from cockpit.quantities.quantity import SingleStepQuantity
from cockpit.quantities.utils_transforms import BatchGradTransforms_BatchDotGrad


class InnerTest(SingleStepQuantity):
    """Inner Product Quantitiy Class.

    Inner product test proposed in bollapragada2017adaptive.

    Link:
        - https://arxiv.org/pdf/1710.11258.pdf
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
            ext.append(BatchGradTransforms_BatchDotGrad())

        return ext

    def _compute(self, global_step, params, batch_loss):
        """Track the practical version of the inner product test.

        Return maximum θ for which the inner product test would pass.

        The inner product test is defined by Equation (2.6) in bollapragada2017adaptive.

        Args:
            global_step (int): The current iteration number.
            params ([torch.Tensor]): List of torch.Tensors holding the network's
                parameters.
            batch_loss (torch.Tensor): Mini-batch loss from current step.
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

    def _compute_adapted_batch_size(self, theta, var_projection, grad_l2_squared):
        """Compute the batch size suggested by the inner product test.

        The adaptation rule is defined by a modification of Equation (4.7)
        in bollapragada2017adaptive.

        Args:
            theta ([type]): [description]
            var_projection (torch.Tensor): The sample variance of individual
                gradient projections on the mini-batch gradient.
            grad_l2_squared (torch.Tensor): Squared ℓ₂ norm of mini-batch gradient.

        Returns:
            [int]: Suggested batch size.
        """
        batch_size_theta = var_projection / theta ** 2 / grad_l2_squared ** 2

        if self._verbose:
            print(f"Inner product test θ={theta:.4f} proposes B={batch_size_theta:.4f}")

        return batch_size_theta

    # TODO Move to tests
    def __run_check(self, global_step, params, batch_loss):
        """Run sanity checks to verify math rearrangements."""

        def _compute_projection_variance_from_batch_grad(params):
            """Compute variance of individual gradient projections on the gradient.

            The sample variance of projections is given by Equation (line after 2.6)
            in bollapragada2017adaptive (https://arxiv.org/pdf/1710.11258.pdf)
            """
            batch_grad = self._fetch_batch_grad(params, aggregate=True)
            batch_size = get_batch_size(global_step)
            grad = self._fetch_grad(params, aggregate=True)
            grad_l2_squared = self._fetch_grad_l2_squared(params, aggregate=True)

            projections = torch.einsum("ni,i->n", batch_size * batch_grad, grad)

            return (1 / (batch_size - 1)) * (
                (projections ** 2).sum() - batch_size * grad_l2_squared ** 2
            )

        # sanity check 1: Variances of projected individual gradients should match
        # result from computation with individual gradients.
        batch_dot = self._fetch_batch_dot_via_batch_grad_transforms(
            params, aggregate=True
        )
        grad_l2_squared = self._fetch_grad_l2_squared(params, aggregate=True)
        batch_size = batch_dot.size(0)

        var_from_batch_dot = self._compute_projection_variance(
            batch_size, batch_dot, grad_l2_squared
        )
        var_from_batch_grad = _compute_projection_variance_from_batch_grad(params)

        assert torch.allclose(var_from_batch_dot, var_from_batch_grad, rtol=5e-4), (
            "Variances of projected individual gradient norms from batch_grad"
            + " and batch_dot do not match:"
            + f" {var_from_batch_dot:.4f} vs. {var_from_batch_grad:.4f}"
        )

        # sanity check 2: Adaptive batch size implied by maximum value of θ
        # matches used batch size
        var_projection = var_from_batch_dot
        theta_max = self._compute_theta_max(batch_size, var_projection, grad_l2_squared)

        # sanity check: suggested batch size by θₘₐₓ is the used batch size
        batch_size_theta = self._compute_adapted_batch_size(
            theta_max, var_projection, grad_l2_squared
        )

        assert torch.allclose(
            torch.tensor(
                [batch_size],
                dtype=batch_size_theta.dtype,
                device=batch_size_theta.device,
            ),
            batch_size_theta,
        )

        # (not really a check) print suggested batch sizes from values of θ
        # bollapragada2017adaptive
        thetas = [0.9]
        for theta in thetas:
            batch_size_theta = self._compute_adapted_batch_size(
                theta, var_projection, grad_l2_squared
            )
