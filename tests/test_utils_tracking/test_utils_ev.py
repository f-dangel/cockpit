"""Test Hessian-vector product interface via ``scipy``'s ``LinearOperator``.

We use ``scipy``'s wrapper of ARPACK to compute a subset of Hessian eigenvalues
via matrix-vector products.

.. note::

    Only the largest eigenvalue computation (spectral norm of the Hessian)
    is tested as it is our use case. It should work for other settings too.

"""

import numpy
import pytest
import torch
from scipy.sparse.linalg import eigsh

from backboard.tracking.utils_ev import HVPLinearOperator


def sym_mat(dim):
    """Create a random symmetric matrix of shape [dim, dim]."""
    mat = torch.rand(dim, dim)
    return 0.5 * (mat + mat.t())


def make_scipy_interface(hessian, linear_operator=False, use_grad=False):
    """Set up interface to ``scipy`` as ``numpy`` array or ``LinearOperator``.

    Returns:
       numpy.ndarray: Matrix representation if ``linear_operator`` is ``False``.
       scipy.sparse.linalg.LinearOperator: Linear operator interface to Hessian-
           vector products if ``linear_operator`` is ``True``.
    """
    if linear_operator:
        x = torch.rand(hessian.shape[1], requires_grad=True)
        batch_loss = 0.5 * torch.einsum("i,ij,j->", x, hessian, x)

        grad_params = None
        if use_grad:
            batch_loss.backward(create_graph=True)
            grad_params = [x.grad]

        return HVPLinearOperator(batch_loss, [x], grad_params=grad_params)

    else:
        return hessian.cpu().numpy()


def eigsh_mat_interface(hessian, k, which):
    """Compute eigenvalues with full matrix as interface."""
    A = make_scipy_interface(hessian)
    return eigsh(A, k=k, which=which, return_eigenvectors=False)


def eigsh_hvp_interface(hessian, k, which, use_grad):
    """Compute eigenvalues with linear operator as interface."""
    A = make_scipy_interface(hessian, linear_operator=True, use_grad=use_grad)
    return eigsh(A, k=k, which=which, return_eigenvectors=False)


def compare_eigsh_matrix_vs_linear_operator(k, which, use_grad, dim=42):
    """Compare eigsh result with Hessian matrix and Hessian-vector products."""
    hessian = sym_mat(dim)

    mat_eigvals = eigsh_mat_interface(hessian, k=k, which=which)
    hvp_eigvals = eigsh_hvp_interface(hessian, k=k, which=which, use_grad=use_grad)

    assert numpy.allclose(mat_eigvals, hvp_eigvals)


@pytest.mark.parametrize("use_grad", [False, True])
def test_largest_eigenvalue(use_grad, dim=42):
    """Test computation of λₘₐₓ."""
    torch.manual_seed(0)

    compare_eigsh_matrix_vs_linear_operator(k=1, which="SA", use_grad=use_grad)


@pytest.mark.parametrize("k,which", [[1, "SA"]])
def test_same_result_with_use_grad(k, which, dim=41):
    """Test that using gradients in HVP (faster) has same result."""
    torch.manual_seed(0)
    hessian = sym_mat(dim)

    eigvals_no_grad = eigsh_hvp_interface(hessian, k=k, which=which, use_grad=False)
    eigvals_with_grad = eigsh_hvp_interface(hessian, k=k, which=which, use_grad=True)

    assert numpy.allclose(eigvals_no_grad, eigvals_with_grad)
