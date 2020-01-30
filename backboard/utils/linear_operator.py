"""
Hessian-vector products in torch with scipy.sparse.linalg.LinearOperator interface
"""

import numpy
import torch
from scipy.sparse.linalg import LinearOperator
from torch.nn.utils import parameters_to_vector

from backpack.hessianfree.ggnvp import ggn_vector_product_from_plist
from backpack.hessianfree.hvp import hessian_vector_product
from backpack.utils.convert_parameters import vector_to_parameter_list


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

    def __init__(self, loss, params, grad_params=None, dtype=numpy.float32):
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


class GGNVPLinearOperator(BaseLinearOperator):
    """Scipy interface for multiplication with the GGN in torch."""

    def __init__(self, loss, output, params, dtype=numpy.float32):
        """Multiplication with the Hessian of loss w.r.t. params."""
        num_params = sum(p.numel() for p in params)
        shape = (num_params, num_params)
        super().__init__(dtype, shape)

        self.loss = loss
        self.output = output
        self.params = params
        self.device = loss.device

    def _matvec(self, v_numpy):
        """Multiply with the Hessian."""
        v_torch = self._preprocess(v_numpy)
        GGNv_torch = self.ggn_vector_product(v_torch)
        return self._postprocess(GGNv_torch)

    def ggn_vector_product(self, v_torch):
        """Multiply by the GGN using autodiff in torch."""
        return ggn_vector_product_from_plist(
            self.loss, self.output, self.params, v_torch
        )
