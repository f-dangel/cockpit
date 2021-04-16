"""Tests for using multiple batch grad transforms in Cockpit."""

import pytest

from cockpit.cockpit import Cockpit
from cockpit.quantities.utils_transforms import BatchGradTransformsHook


def test_merge_batch_grad_transforms():
    """Test merging of multiple ``BatchGradTransforms``."""
    bgt1 = BatchGradTransformsHook({"x": lambda t: t, "y": lambda t: t})
    bgt2 = BatchGradTransformsHook({"v": lambda t: t, "w": lambda t: t})

    merged_bgt = Cockpit._merge_batch_grad_transform_hooks([bgt1, bgt2])
    assert isinstance(merged_bgt, BatchGradTransformsHook)

    merged_keys = ["x", "y", "v", "w"]
    assert len(merged_bgt._transforms.keys()) == len(merged_keys)

    for key in merged_keys:
        assert key in merged_bgt._transforms.keys()

    assert id(bgt1._transforms["x"]) == id(merged_bgt._transforms["x"])
    assert id(bgt2._transforms["w"]) == id(merged_bgt._transforms["w"])


def test_merge_batch_grad_transforms_same_key_different_trafo():
    """Merging ``BatchGradTransforms`` with same key but different trafo should fail."""
    bgt1 = BatchGradTransformsHook({"x": lambda t: t, "y": lambda t: t})
    bgt2 = BatchGradTransformsHook({"x": lambda t: t, "w": lambda t: t})

    with pytest.raises(ValueError):
        _ = Cockpit._merge_batch_grad_transform_hooks([bgt1, bgt2])


def test_merge_batch_grad_transforms_same_key_same_trafo():
    """Test merging multiple ``BatchGradTransforms`` with same key and same trafo."""

    def func(t):
        return t

    bgt1 = BatchGradTransformsHook({"x": func})
    bgt2 = BatchGradTransformsHook({"x": func})

    merged = Cockpit._merge_batch_grad_transform_hooks([bgt1, bgt2])

    assert len(merged._transforms.keys()) == 1
    assert id(merged._transforms["x"]) == id(func)
