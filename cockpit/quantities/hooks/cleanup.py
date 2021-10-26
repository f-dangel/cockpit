"""Contains hook that deletes BackPACK buffers during backpropagation."""

from typing import Set

from torch import Tensor

from cockpit.quantities.hooks.base import ParameterExtensionHook


class CleanupHook(ParameterExtensionHook):
    """Deletes specified BackPACK buffers during backpropagation."""

    def __init__(self, delete_savefields: Set[str]):
        """Store savefields to be deleted in the backward pass.

        Args:
            delete_savefields: Name of buffers to delete.
        """
        super().__init__()
        self._delete_savefields = delete_savefields

    def param_hook(self, param: Tensor):
        """Delete BackPACK buffers in parameter.

        Args:
            param: Trainable parameter which hosts BackPACK quantities.
        """
        for savefield in self._delete_savefields:
            if hasattr(param, savefield):
                delattr(param, savefield)
