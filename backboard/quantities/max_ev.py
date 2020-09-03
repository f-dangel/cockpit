"""Class for tracking the Maximum Hessian Eigenvalue."""

import numpy as np
import torch
from scipy.sparse.linalg import LinearOperator, eigsh
from torch.nn.utils import parameters_to_vector

from backboard.quantities.quantity import Quantity
from backpack.hessianfree.hvp import hessian_vector_product
from backpack.utils.convert_parameters import vector_to_parameter_list


class MaxEV(Quantity):
    """Maximum Hessian Eigenvalue Quantitiy Class."""

    def create_graph(self, global_step):
        """Return whether access to the forward pass computation graph is needed.

        Args:
            global_step (int): The current iteration number.

        Returns:
            bool: ``True`` if the computation graph shall not be deleted,
                else ``False``.
        """
        return True if global_step % self._track_interval == 0 else False

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
            params (method): Function to access the parameters.
            batch_loss (torch.Tensor): Mini-batch loss from current step.
        """
        if global_step % self._track_interval == 0:
            max_ev = np.float64(self._compute_max_ev(global_step, params, batch_loss))
            self.output[global_step]["max_ev"] = [max_ev]
        else:
            pass

    def _compute_max_ev(self, global_step, params, batch_loss):
        """Helper Function to Compute the larges Hessian eigenvalue.

        Args:
            global_step (int): The current iteration number.
            params (method): Function to access the parameters.
            batch_loss (torch.Tensor): Mini-batch loss from current step.
        """
        params = self._fetch_params(params)
        HVP = HVPLinearOperator(
            batch_loss, params, grad_params=self._fetch_grad(params)
        )

        max_ev = eigsh(HVP, k=1, which="LA", return_eigenvectors=False)[0]

        if self._verbose:
            print(f"Largest Hessian eigenvalue: λₘₐₓ={max_ev:.4f}")

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
