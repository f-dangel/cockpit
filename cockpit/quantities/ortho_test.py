"""Class for tracking the Orthogonality Test."""

import math

import torch
from cockpit.quantities.quantity import SingleStepQuantity
from cockpit.quantities.utils_transforms import BatchGradTransforms_BatchDotGrad


class OrthoTest(SingleStepQuantity):
    """Orthogonality Test Quantitiy Class.

    Orthogonality test proposed in bollapragada2017adaptive.

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
        """Track the practical version of the orthogonality test.

        Return maximum ν for which the orthogonality test would pass.

        The orthogonality test is defined by Equation (3.3) in bollapragada2017adaptive.

        Args:
            global_step (int): The current iteration number.
            params ([torch.Tensor]): List of torch.Tensors holding the network's
                parameters.
            batch_loss (torch.Tensor): Mini-batch loss from current step.
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

    def _compute_adapted_batch_size(
        self, nu, var_orthogonal_projection, grad_l2_squared
    ):
        """Compute the batch size suggested by the orthogonality test.

        The adaptation rule is defined by a modification of Equation (4.7)
        in bollapragada2017adaptive.

        Args:
            nu ([type]): [description]
            var_orthogonal_projection (torch.Tensor): [description]
            grad_l2_squared (torch.Tensor): Squared ℓ₂ norm of mini-batch gradient.

        Returns:
            [int]: Suggested batch size.
        """
        batch_size_nu = var_orthogonal_projection / nu ** 2 / grad_l2_squared

        if self._verbose:
            print(f"Orthogonality test ν={nu:.4f} proposes B={batch_size_nu:.4f}")

        return batch_size_nu

    # TODO Move to tests
    def __run_check(self, params, batch_loss):
        """Run sanity checks to verify math rearrangements."""

        def _compute_orthogonal_projection_variance_from_batch_grad(params):
            """Compute variance of individual gradient orthogonal projections.

            The sample variance of orthogonal projections shows up in Equation (3.3)
            in bollapragada2017adaptive (https://arxiv.org/pdf/1710.11258.pdf)
            """
            batch_grad = self._fetch_batch_grad(params, aggregate=True)
            batch_size = batch_grad.size(0)
            grad = self._fetch_grad(params, aggregate=True)
            grad_l2_squared = self._fetch_grad_l2_squared(params, aggregate=True)

            batch_l2_squared = (batch_grad ** 2).sum(0)
            projections = torch.einsum("ni,i->n", batch_size * batch_grad, grad)

            return (1 / (batch_size - 1)) * (
                batch_size ** 2 * batch_l2_squared.sum()
                - (projections ** 2).sum() / grad_l2_squared
            )

        # sanity check 1: Variances of orthogonal projected individual gradients
        # should match result from computation with individual gradients.
        batch_dot = self._fetch_batch_dot_via_batch_grad_transforms(
            params, aggregate=True
        )
        grad_l2_squared = self._fetch_grad_l2_squared(params, aggregate=True)
        batch_size = batch_dot.size(0)

        var_from_batch_dot = self._compute_orthogonal_projection_variance(
            batch_size, batch_dot, grad_l2_squared
        )
        var_from_batch_grad = _compute_orthogonal_projection_variance_from_batch_grad(
            params
        )

        assert torch.allclose(var_from_batch_dot, var_from_batch_grad, rtol=5e-4), (
            "Variances of orthogonal projected individual gradient norms from"
            + " batch_grad and batch_dot do not match:"
            + f" {var_from_batch_dot:.4f} vs. {var_from_batch_grad:.4f}"
        )

        # sanity check 2: Adaptive batch size implied by maximum value of ν
        # matches used batch size
        var_orthogonal_projection = var_from_batch_dot
        nu_max = self._compute_nu_max(
            batch_size, var_orthogonal_projection, grad_l2_squared
        )

        batch_size_nu = self._compute_adapted_batch_size(
            nu_max, var_orthogonal_projection, grad_l2_squared
        )

        assert torch.allclose(
            torch.tensor(
                [batch_size],
                dtype=batch_size_nu.dtype,
                device=batch_size_nu.device,
            ),
            batch_size_nu,
        )

        # (not really a check) print suggested batch sizes from values of ν
        # bollapragada2017adaptive and bahamou2019dynamic
        nus = [5.84, math.sqrt(0.1 ** 3)]
        for nu in nus:
            batch_size_nu = self._compute_adapted_batch_size(
                nu, var_orthogonal_projection, grad_l2_squared
            )
