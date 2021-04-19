"""Utility functions for Transforms."""

import string
import weakref

from torch import einsum

from cockpit.quantities.hooks.base import ParameterExtensionHook


def BatchGradTransformsHook_BatchL2Grad():
    """Compute individual gradient ℓ₂ norms via individual gradients."""
    return BatchGradTransformsHook({"batch_l2": batch_l2_transform})


def batch_l2_transform(batch_grad):
    """Transform individual gradients into individual ℓ₂ norms."""
    sum_axes = list(range(batch_grad.dim()))[1:]
    return (batch_grad ** 2).sum(sum_axes)


def BatchGradTransformsHook_BatchDotGrad():
    """Compute pairwise individual gradient dot products via individual gradients."""
    return BatchGradTransformsHook({"batch_dot": batch_dot_transform})


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


def BatchGradTransformsHook_SumGradSquared():
    """Compute sum of squared individual gradients via individual gradients."""
    return BatchGradTransformsHook({"sum_grad_squared": sum_grad_squared_transform})


def sum_grad_squared_transform(batch_grad):
    """Transform individual gradients into second non-centered moment."""
    return (batch_grad ** 2).sum(0)


class BatchGradTransformsHook(ParameterExtensionHook):
    """Hook implementation of ``BatchGradTransforms``."""

    def __init__(self, transforms, savefield=None):
        """Store transformations and potential savefield.

        Args:
            transforms (dict): Values are functions that are evaluated on a parameter's
                ``grad_batch`` attribute. The result is stored in a dictionary stored
                under ``grad_batch_transforms``.
            savefield (str, optional): Attribute name under which the hook's result
                is saved in a parameter. If ``None``, it is assumed that the hook acts
                via side effects and no output needs to be stored.
        """
        super().__init__(savefield=savefield)
        self._transforms = transforms

    def param_hook(self, param):
        """Execute all transformations and store results as dictionary in the parameter.

        Delete individual gradients in the parameter.

        Args:
            param (torch.Tensor): Trainable parameter which hosts BackPACK quantities.
        """
        param.grad_batch._param_weakref = weakref.ref(param)
        # TODO Delete after backward pass with Cockpit
        param.grad_batch_transforms = {
            key: func(param.grad_batch) for key, func in self._transforms.items()
        }
        # TODO Delete with a separate hook that also knows which savefield should be
        # kept because it's protected by the user. See
        # https://github.com/f-dangel/cockpit-paper/issues/197
        del param.grad_batch
