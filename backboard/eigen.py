import torch

from backboard.utils import normalize


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
