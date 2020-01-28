import torch

from backboard.eigen import power_iteration
from backboard.utils import normalize


def test_power_iteration_example():
    """Verify largest eigenvalue and eigenvector from a test problem.

    Example from http://wwwf.imperial.ac.uk/metric/metric_public/matrices/
    eigenvalues_and_eigenvectors/eigenvalues2.html
    """
    mat = torch.Tensor([[-2.0, -4.0, 2.0], [-2.0, 1.0, 2.0], [4.0, 2.0, 5.0]])
    eigvals = torch.Tensor([3.0, -5.0, 6.0])
    # row-wise, not normalized
    eigvecs = torch.Tensor([[2.0, -3.0, -1.0], [2.0, -1.0, 1.0], [1.0, 6.0, 16.0]])

    dim = mat.shape[0]
    largest_eigval = eigvals[2]
    largest_eigvec = normalize(eigvecs[2])

    rtol, atol = 1e-6, 1e-6

    largest_eigval_power, largest_eigvec_power = power_iteration(
        mat.matmul, dim, rtol=rtol, atol=atol
    )
    eigval_close = torch.allclose(
        largest_eigval, largest_eigval_power, atol=atol, rtol=rtol
    )

    zero = torch.Tensor([0.0])
    eigvec_close = torch.allclose(
        (largest_eigvec - largest_eigvec_power).norm(),
        zero,
        atol=10 * atol,
        rtol=10 * rtol,
    )
    neg_eigvec_close = torch.allclose(
        (largest_eigvec + largest_eigvec_power).norm(),
        zero,
        atol=10 * atol,
        rtol=10 * rtol,
    )

    assert eigval_close, AssertionError("Eigenvalues not close")
    assert eigvec_close or neg_eigvec_close, AssertionError("Eigenvectors not close")
