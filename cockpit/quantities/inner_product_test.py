"""Class for tracking the Inner Product Test."""

import torch

from backpack import extensions
from cockpit.context import get_batch_size
from cockpit.quantities.quantity import SingleStepQuantity
from cockpit.quantities.utils_transforms import BatchGradTransforms_BatchDotGrad


class InnerProductTest(SingleStepQuantity):
    """Inner Product Quantitiy Class.

    Inner product test proposed in bollapragada2017adaptive.

    Link:
        - https://arxiv.org/pdf/1710.11258.pdf
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

        Note:
            - The computation of this quantity suffers from precision errors.
              Enabling ``check`` will compute the quantity via two different
              BackPACK quantities and compare them. This is roughly 2x as expensive.
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
            ext = [BatchGradTransforms_BatchDotGrad()]

            if self._check:
                ext.append(extensions.BatchGrad())

        else:
            ext = []
        return ext

    def compute(self, global_step, params, batch_loss):
        """Track the practical version of the inner product test.

        Args:
            global_step (int): The current iteration number.
            params ([torch.Tensor]): List of torch.Tensors holding the network's
                parameters.
            batch_loss (torch.Tensor): Mini-batch loss from current step.
        """
        if self.is_active(global_step):
            inner_product_test = self._compute(params, batch_loss).item()

            if self._verbose:
                print(
                    f"[Step {global_step}] InnerProductTest: {inner_product_test:.4f}"
                )

            self.output[global_step]["inner_product_test"] = inner_product_test

            if self._check:
                self.__run_check(global_step, params, batch_loss)
        else:
            pass

    def _compute(self, params, batch_loss):
        """Return maximum θ for which the inner product test would pass.

        The inner product test is defined by Equation (2.6) in bollapragada2017adaptive.

        Args:
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

        return self._compute_theta_max(batch_size, var_projection, grad_l2_squared)

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
        if self._use_double:
            batch_dot = batch_dot.double()

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

            if self._use_double:
                batch_grad = batch_grad.double()
                grad = grad.double()
                grad_l2_squared = grad_l2_squared.double()

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
