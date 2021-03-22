"""Utility functions for random number generation."""

import torch


class restore_rng_state:
    """Restores PyTorch seed to the value of initialization.

    This has the effect that code inside this context does not influence the outer
    loop's random generator state.
    """

    def __init__(self):
        """Store the current PyTorch seed."""
        self._rng_state = torch.get_rng_state()

    def __enter__(self):
        """Do nothing."""
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        """Restore the random generator state at initialization."""
        torch.set_rng_state(self._rng_state)
