from backpack import extensions


def BatchGradTransforms_BatchL2Grad():
    """Compute individual gradient ℓ₂ norms via individual gradients."""
    return extensions.BatchGradTransforms({"batch_l2": batch_l2_grad_transform})


def batch_l2_grad_transform(batch_grad):
    """Transform individual gradients into individual ℓ₂ norms."""
    sum_axes = list(range(batch_grad.dim()))[1:]
    return (batch_grad ** 2).sum(sum_axes)
