"""Class for tracking the Norm Test."""

import torch
from cockpit.quantities.quantity import SingleStepQuantity
from cockpit.quantities.utils_transforms import BatchGradTransforms_BatchL2Grad


class NormTest(SingleStepQuantity):
    """Norm Test Quantitiy Class.

    Norm test proposed in byrd2012adaptive.

    Link:
        - https://link.springer.com/content/pdf/10.1007/s10107-012-0572-5.pdf
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
            ext.append(BatchGradTransforms_BatchL2Grad())

        return ext

    def _compute(self, global_step, params, batch_loss):
        """Track the practical version of the norm test.

        Return maximum θ for which the norm test would pass.

        The norm test is defined by Equation (3.9) in byrd2012sample.

        Args:
            global_step (int): The current iteration number.
            params ([torch.Tensor]): List of torch.Tensors holding the network's
                parameters.
            batch_loss (torch.Tensor): Mini-batch loss from current step.
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

    def _compute_adapted_batch_size(self, theta, var_l1, grad_l2_squared):
        """Compute the batch size suggested by the norm test.

        The adaptation rule is defined by Equation (3.12) in byrd2012sample.

        Args:
            theta ([type]): [description]
            var_l1 ([type]): [description]
            grad_l2_squared (torch.Tensor): Squared ℓ₂ norm of mini-batch gradient.

        Returns:
            torch.Tensor: The sample variance ℓ₁ norm.
        """
        batch_size_theta = var_l1 / theta ** 2 / grad_l2_squared

        if self._verbose:
            print(f"Norm test θ={theta} proposes B={batch_size_theta:.4f}")

        return batch_size_theta

    # TODO Move to tests
    def __run_check(self, params, batch_loss):
        """Run sanity checks to verify math rearrangements."""

        def _compute_variance(params):
            """Compute the sample variance from individual gradients.

            The sample variance is given by Equation (3.6) in byrd2012sample
            (https://link.springer.com/content/pdf/10.1007/s10107-012-0572-5.pdf)
            """
            batch_grad = self._fetch_batch_grad(params, aggregate=True)
            grad = self._fetch_grad(params, aggregate=True)
            batch_size = batch_grad.size(0)

            return (1 / (batch_size - 1)) * ((batch_size * batch_grad - grad) ** 2).sum(
                0
            )

        # sanity check 1: ample variance ℓ₁  norm via individual gradients should match
        # result from computation with individual gradients ℓ₂ norms.
        batch_l2_squared = self._fetch_batch_l2_squared_via_batch_grad_transforms(
            params, aggregate=True
        )
        grad_l2_squared = self._fetch_grad_l2_squared(params, aggregate=True)
        batch_size = batch_l2_squared.size(0)

        var_l1_from_batch_l2 = self._compute_variance_l1(
            batch_size, batch_l2_squared, grad_l2_squared
        )
        var_l1_from_batch_grad = _compute_variance(params).norm(1)

        assert torch.allclose(
            var_l1_from_batch_l2, var_l1_from_batch_grad, rtol=5e-4
        ), (
            "Sample variance ℓ₁ norms from batch_grad and batch_l2 do not match:"
            + f" {var_l1_from_batch_grad:.4f} vs. {var_l1_from_batch_l2:.4f}"
        )

        # sanity check 2: Adaptive batch size implied by maximum value of θ
        # matches used batch size
        var_l1 = var_l1_from_batch_l2
        theta_max = self._compute_theta_max(batch_size, var_l1, grad_l2_squared)

        batch_size_theta = self._compute_adapted_batch_size(
            theta_max, var_l1, grad_l2_squared
        )
        assert torch.allclose(
            torch.tensor(
                [batch_size],
                dtype=batch_size_theta.dtype,
                device=batch_size_theta.device,
            ),
            batch_size_theta,
        ), "Batch size implied by θₘₐₓ does not match used batch size"

        # (not really a check) print suggested batch sizes from values of θ
        # byrd2012sample, bollapragada2017adaptive
        thetas = [0.5, 0.9]
        for theta in thetas:
            batch_size_theta = self._compute_adapted_batch_size(
                theta, var_l1, grad_l2_squared
            )
