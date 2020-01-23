import torch


def normalize(v):
    return v / v.norm()


def power_iteration(matmul, dim, v0=None, maxiter=100, rtol=1e-3, atol=1e-6):
    """Compute the largest eigenvalue from a matrix multiplication routine."""

    def converged(eigval_before, eigval_new):
        return torch.allclose(eigval_before, eigval_new, rtol=rtol, atol=atol)

    v = torch.randn(dim) if v0 is None else v0
    v = normalize(v)

    eigval = torch.Tensor([float("inf")])

    for _ in range(maxiter):
        new_v = matmul(v)
        new_eigval = new_v.norm() / v.norm()
        v = normalize(new_v)

        if converged(eigval, new_eigval):
            eigvec = v
            return eigval, eigvec

        eigval = new_eigval

    raise RuntimeError("Exceeded maximum number of iterations {}".format(maxiter))


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
        mat, dim, rtol=rtol, atol=atol
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


if __name__ == "__main__":
    test_power_iteration_example()
