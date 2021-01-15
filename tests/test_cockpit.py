"""Tests for ``cockpit.cockpit``."""

import pytest
import torch

from backpack import extend
from backpack.extensions import (
    BatchGrad,
    BatchGradTransforms,
    DiagGGNExact,
    DiagHessian,
)
from cockpit import quantities
from cockpit.cockpit import Cockpit


def test_merge_batch_grad_transforms():
    """Test merging of multiple ``BatchGradTransforms``."""
    bgt1 = BatchGradTransforms({"x": lambda t: t, "y": lambda t: t})
    bgt2 = BatchGradTransforms({"v": lambda t: t, "w": lambda t: t})

    merged_bgt = Cockpit._merge_batch_grad_transforms([bgt1, bgt2])
    assert isinstance(merged_bgt, BatchGradTransforms)

    merged_keys = ["x", "y", "v", "w"]
    assert len(merged_bgt.get_transforms().keys()) == len(merged_keys)

    for key in merged_keys:
        assert key in merged_bgt.get_transforms().keys()

    assert id(bgt1.get_transforms()["x"]) == id(merged_bgt.get_transforms()["x"])
    assert id(bgt2.get_transforms()["w"]) == id(merged_bgt.get_transforms()["w"])


def test_merge_batch_grad_transforms_same_key_different_trafo():
    """
    Merging ``BatchGradTransforms`` with same key but different trafo should fail.
    """
    bgt1 = BatchGradTransforms({"x": lambda t: t, "y": lambda t: t})
    bgt2 = BatchGradTransforms({"x": lambda t: t, "w": lambda t: t})

    with pytest.raises(ValueError):
        _ = Cockpit._merge_batch_grad_transforms([bgt1, bgt2])


def test_merge_batch_grad_transforms_same_key_same_trafo():
    """Test merging multiple ``BatchGradTransforms`` with same key and same trafo."""

    def func(t):
        return t

    bgt1 = BatchGradTransforms({"x": func})
    bgt2 = BatchGradTransforms({"x": func})

    _ = Cockpit._merge_batch_grad_transforms([bgt1, bgt2])


def test_process_multiple_batch_grad_transforms_empty():
    """Test processing if no ``BatchGradTransforms`` is used."""
    ext1 = BatchGrad()
    ext2 = DiagGGNExact()

    extensions = [ext1, ext2]
    processed = Cockpit._process_multiple_batch_grad_transforms(extensions)

    assert processed == extensions


def test_automatic_call_track():
    """Make sure `track` is called automatically when a cockpit context is left."""
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(10, 2))
    loss_fn = torch.nn.MSELoss(reduction="mean")

    q_time = quantities.Time(track_interval=1)
    cp = Cockpit(model.parameters(), quantities=[q_time])

    global_step = 0

    batch_size = 3
    inputs = torch.rand(batch_size, 10)
    labels = torch.rand(batch_size, 2)

    loss = loss_fn(model(inputs), labels)

    with cp(global_step, info={"loss": loss}):
        loss.backward(create_graph=cp.create_graph(global_step))

        # cp.track should not have been called yet...
        # assert global_step not in q_time.output.keys()

    # ...but after the context is left
    assert global_step in q_time.output.keys()


def test_with_backpack_extensions():
    """Check if backpack quantities can be computed inside cockpit."""
    torch.manual_seed(0)
    model = extend(torch.nn.Sequential(torch.nn.Linear(10, 2)))
    loss_fn = extend(torch.nn.MSELoss(reduction="mean"))

    q_time = quantities.TICDiag(track_interval=1)
    cp = Cockpit(model.parameters(), quantities=[q_time])

    global_step = 0

    batch_size = 3
    inputs = torch.rand(batch_size, 10)
    labels = torch.rand(batch_size, 2)

    loss = loss_fn(model(inputs), labels)

    with cp(global_step, DiagHessian(), info={"loss": loss, "batch_size": batch_size}):
        loss.backward(create_graph=cp.create_graph(global_step))

        # BackPACK buffers exist...
        for param in model.parameters():
            # required by TICDiag and user
            assert hasattr(param, "diag_h")
            # required by TICDiag only
            assert hasattr(param, "grad_batch_transforms")
            assert "sum_grad_squared" in param.grad_batch_transforms

    # ... and are not deleted when specified by the user
    for param in model.parameters():
        assert hasattr(param, "diag_h")
        # not protected by user
        assert not hasattr(param, "grad_batch_transforms")
