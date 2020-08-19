"""Test the normalize function."""

import random

import numpy as np
import pytest
import torch

from backboard.tracking.utils_tracking import _normalize

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


vecs = []
for _ in range(100):
    vecs.append(torch.randn(random.randint(1, 1000)))


@pytest.mark.parametrize("v", vecs)
def test_normalize(v):
    """Check that a bunch of random vectors have norm one.

    Args:
        v (torch.Tensor): A torch.Tensor vector
    """
    v_normalized = _normalize(v)

    assert np.isclose(v_normalized.norm(), 1)
