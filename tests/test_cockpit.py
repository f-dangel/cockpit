"""Tests for ``cockpit.cockpit.py``."""

import pytest

from backpack.extensions import BatchGrad, BatchGradTransforms, DiagGGNExact
from cockpit import quantities
from cockpit.cockpit import Cockpit, configured_quantities
from deepobs.pytorch.testproblems import quadratic_deep
from tests.utils import set_up_problem


def set_up_cockpit_configuration(label):
    """Set up a dummy pre-configured cockpit."""
    tproblem = set_up_problem(quadratic_deep)
    logpath = "~/tmp/test_cockpit/quadratic_deep/SGD/hyperparams/log"
    quantities = configured_quantities(label)

    return Cockpit(tproblem, logpath, track_interval=1, quantities=quantities)


@pytest.mark.parametrize("label", ["full", "business", "economy"])
def test_cockpit_configuration(label):
    """Check cockpit quantities."""
    cockpit = set_up_cockpit_configuration(label)

    full_quantity_cls = configured_quantities("full")
    not_present = {
        "full": (),
        "business": (
            quantities.MaxEV,
            quantities.BatchGradHistogram2d,
        ),
        "economy": (
            quantities.MaxEV,
            quantities.TICDiag,
            quantities.TICTrace,
            quantities.Trace,
            quantities.BatchGradHistogram2d,
        ),
    }[label]

    for q_cls in full_quantity_cls:
        if q_cls in not_present:
            assert not any(isinstance(q, q_cls) for q in cockpit.quantities)
        else:
            assert len([q for q in cockpit.quantities if isinstance(q, q_cls)]) == 1


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
