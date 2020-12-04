"""Class for tracking the Maximum Hessian Eigenvalue."""

import warnings

import numpy as np
import torch
from scipy.sparse.linalg import LinearOperator, eigsh
from torch.nn.utils import parameters_to_vector

from backpack.hessianfree.hvp import hessian_vector_product
from backpack.utils.convert_parameters import vector_to_parameter_list
from cockpit.quantities.quantity import SingleStepQuantity


class MaxEV(SingleStepQuantity):
    """Maximum Hessian Eigenvalue Quantitiy Class."""

    def __init__(self, track_schedule, verbose=False, use_power=True):
        """Initialize maximum eigenvalue computation

        Args:
            use_power (bool): If ``True``, uses a power iteration that works on GPUs.
                If ``True``, use ``scipy.sparse.linalg.eigsh`` for eigenvalue computa-
                tion (requires transfers from GPU to CPU and ``torch`` to ``numpy``.)
        """
        super().__init__(track_schedule, verbose=verbose)

        self._use_power = use_power

    def create_graph(self, global_step):
        """Return whether access to the forward pass computation graph is needed.

        Args:
            global_step (int): The current iteration number.

        Returns:
            bool: ``True`` if the computation graph shall not be deleted,
                else ``False``.
        """
        return self.is_active(global_step)

    def extensions(self, global_step):
        """Return list of BackPACK extensions required for the computation.

        Args:
            global_step (int): The current iteration number.

        Returns:
            list: (Potentially empty) list with required BackPACK quantities.
        """
        return []

    def compute(self, global_step, params, batch_loss):
        """Compute the larges Hessian eigenvalue.

        Args:
            global_step (int): The current iteration number.
            params ([torch.Tensor]): List of torch.Tensors holding the network's
                parameters.
            batch_loss (torch.Tensor): Mini-batch loss from current step.
        """
        if self.is_active(global_step):
            max_ev = np.float64(self._compute_max_ev(global_step, params, batch_loss))
            self.output[global_step]["max_ev"] = max_ev

    def _compute_max_ev(self, global_step, params, batch_loss):
        """Helper Function to Compute the larges Hessian eigenvalue.

        Args:
            global_step (int): The current iteration number.
            params ([torch.Tensor]): List of torch.Tensors holding the network's
                parameters.
            batch_loss (torch.Tensor): Mini-batch loss from current step.
        """
        HVP = HVPLinearOperator(
            batch_loss, params, grad_params=self._fetch_grad(params)
        )

        if self._use_power:
            max_ev = HVP.power_iteration().item()
        else:
            max_ev = eigsh(HVP, k=1, which="LA", return_eigenvectors=False)[0]

        if self._verbose:
            print(f"[Step {global_step}] MaxEV: {max_ev:.4f}")

        return max_ev


# Utility Function and Classes #
class BaseLinearOperator(LinearOperator):
    """Base class for GGNVPs and HVPs."""

    def _preprocess(self, v_numpy):
        """Convert to `torch`, block into parameters."""
        v_torch = torch.from_numpy(v_numpy).to(self.device)
        return vector_to_parameter_list(v_torch, self.params)

    def _postprocess(self, v_torch):
        """Flatten and concatenate, then convert to `numpy` array."""
        v_torch_flat = [v.contiguous() for v in v_torch]
        return parameters_to_vector(v_torch_flat).cpu().numpy()


class HVPLinearOperator(BaseLinearOperator):
    """Scipy interface for multiplication with the Hessian in torch."""

    def __init__(self, loss, params, grad_params=None, dtype=np.float32):
        """Multiplication with the Hessian of loss w.r.t. params."""
        num_params = sum(p.numel() for p in params)
        shape = (num_params, num_params)
        super().__init__(dtype, shape)

        self.loss = loss
        self.params = params

        if grad_params is None:
            self.grad_params = torch.autograd.grad(
                loss, params, create_graph=True, retain_graph=True
            )
        else:
            self.grad_params = grad_params

        self.device = loss.device

    def _matvec(self, v_numpy):
        """Multiply with the Hessian."""
        v_torch = self._preprocess(v_numpy)
        Hv_torch = self.hessian_vector_product(v_torch)
        return self._postprocess(Hv_torch)

    def hessian_vector_product(self, v_torch):
        """Multiply by the Hessian using autodiff in torch."""
        return hessian_vector_product(
            self.loss, self.params, v_torch, grad_params=self.grad_params
        )

    def power_iteration(self, maxiter=100, rtol=1e-3, atol=1e-6):
        """Compute the largest eigenvalue by power iteration.

        Args:
            maxiter (int): Maximum number of iterations.
            rtol (float): Relative tolerance to determine convergence from
                consecutive eigenvalues.
            atol (float): Absolute tolerance to determine convergence from
                consecutive eigenvalues.

        Returns
            (torch.Tensor): Maximum eigenvalue. Warns if maximum number of iterations
                was reached and returns the potentially unconverged estimate.
        """

        def converged(old, new):
            return torch.allclose(old, new, rtol=rtol, atol=atol)

        def normalize(vecs):
            norm = sum((v ** 2).sum() for v in vecs).sqrt()
            for v in vecs:
                v /= norm

        def iteration(vecs):
            new_vecs = self.hessian_vector_product(vecs)
            new_eigval = sum((v * new_v).sum() for v, new_v in zip(vecs, new_vecs))

            normalize(new_vecs)

            return new_vecs, new_eigval

        vecs = [torch.rand_like(p) for p in self.params]
        normalize(vecs)
        eigval = torch.Tensor([float("inf")]).to(vecs[0].device)

        for _ in range(maxiter):
            vecs, new_eigval = iteration(vecs)

            if converged(eigval, new_eigval):
                return new_eigval

            eigval = new_eigval

        warnings.warn(f"Exceeded maximum number of {maxiter} iterations")
        return eigval
