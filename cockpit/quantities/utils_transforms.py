import string

from torch import einsum

from backpack import extensions


def BatchGradTransforms_BatchL2Grad():
    """Compute individual gradient ℓ₂ norms via individual gradients."""
    return extensions.BatchGradTransforms({"batch_l2": batch_l2_transform})


def batch_l2_transform(batch_grad):
    """Transform individual gradients into individual ℓ₂ norms."""
    sum_axes = list(range(batch_grad.dim()))[1:]
    return (batch_grad ** 2).sum(sum_axes)


def BatchGradTransforms_BatchDotGrad():
    """Compute pairwise individual gradient dot products via individual gradients."""
    return extensions.BatchGradTransforms({"batch_dot": batch_dot_transform})


def batch_dot_transform(batch_grad):
    """Transform individual gradients into pairwise dot products."""
    # make einsum string
    letters = get_first_n_alphabet(batch_grad.dim() + 1)
    n1, n2, sum_out = letters[0], letters[1], "".join(letters[2:])

    einsum_equation = f"{n1}{sum_out},{n2}{sum_out}->{n1}{n2}"

    return einsum(einsum_equation, batch_grad, batch_grad)


def get_first_n_alphabet(n):
    """Return the first n lowercase letters of the alphabet as a list."""
    return string.ascii_lowercase[:n]


def BatchGradTransforms_SumGradSquared():
    """Compute sum of squared individual gradients via individual gradients."""
    return extensions.BatchGradTransforms(
        {"sum_grad_squared": sum_grad_squared_transform}
    )


def sum_grad_squared_transform(batch_grad):
    """Transform individual gradients into second non-centered moment."""
    return (batch_grad ** 2).sum(0)
