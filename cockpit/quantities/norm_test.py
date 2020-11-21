"""Class for tracking the Norm Test."""

import torch

from backpack import extensions
from cockpit.quantities.quantity import SingleStepQuantity
from cockpit.quantities.utils_transforms import BatchGradTransforms_BatchL2Grad


class NormTest(SingleStepQuantity):
    """Norm Test Quantitiy Class.

    Norm test proposed in byrd2012adaptive.

    Link:
        - https://link.springer.com/content/pdf/10.1007/s10107-012-0572-5.pdf
    """

    def __init__(
        self,
        track_interval=1,
        track_offset=0,
        use_double=False,
        verbose=False,
        check=False,
        track_schedule=None,
    ):
        """Initialize.

        Args:
            track_interval (int): Tracking rate.
            use_double (bool): Whether to use doubles in computation. Defaults
                to ``False``.
            verbose (bool): Turns on verbose mode. Defaults to ``False``.
            check (bool): If True, this quantity will be computed via two different
                ways and compared. Defaults to ``False``.
        """
        super().__init__(
            track_interval=track_interval,
            track_offset=track_offset,
            verbose=verbose,
            track_schedule=track_schedule,
        )
        self._use_double = use_double
        self._check = check

    def extensions(self, global_step):
        """Return list of BackPACK extensions required for the computation.

        Args:
            global_step (int): The current iteration number.

        Returns:
            list: (Potentially empty) list with required BackPACK quantities.
        """
        if self.is_active(global_step):
            ext = [BatchGradTransforms_BatchL2Grad()]

            if self._check:
                ext.append(extensions.BatchGrad())

        else:
            ext = []

        return ext

    def compute(self, global_step, params, batch_loss):
        """Track the practical version of the norm test.

        Args:
            global_step (int): The current iteration number.
            params ([torch.Tensor]): List of torch.Tensors holding the network's
                parameters.
            batch_loss (torch.Tensor): Mini-batch loss from current step.
        """
        if self.is_active(global_step):
            norm_test = self._compute(params, batch_loss).item()

            if self._verbose:
                print(f"[Step {global_step}] NormTest: {norm_test:.4f}")

            self.output[global_step]["norm_test"] = norm_test

            if self._check:
                self.__run_check(params, batch_loss)
        else:
            pass

    def _compute(self, params, batch_loss):
        """Return maximum θ for which the norm test would pass.

        The norm test is defined by Equation (3.9) in byrd2012sample.

        Args:
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

        return self._compute_theta_max(batch_size, var_l1, grad_l2_squared)

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
        if self._use_double:
            batch_l2_squared = batch_l2_squared.double()
            grad_l2_squared = grad_l2_squared.double()

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

            if self._use_double:
                batch_grad = batch_grad.double()
                grad = grad.double()

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
