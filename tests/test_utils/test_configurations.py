"""Tests for ``cockpit.utils.configuration``."""

import pytest

from cockpit import quantities
from cockpit.utils.configuration import quantities_cls_for_configuration


@pytest.mark.parametrize("label", ["full", "business", "economy"])
def test_quantities_cls_for_configuration(label):
    """Check cockpit configurations contain the correct quantities."""
    economy = [
        quantities.Alpha,
        quantities.GradNorm,
        quantities.UpdateSize,
        quantities.Distance,
        quantities.InnerTest,
        quantities.OrthoTest,
        quantities.NormTest,
        quantities.GradHist1d,
        quantities.Loss,
    ]
    business = economy + [quantities.TICDiag, quantities.HessTrace]
    full = business + [quantities.HessMaxEV, quantities.GradHist2d]

    configs = {
        "full": set(full),
        "business": set(business),
        "economy": set(economy),
    }

    quants = set(quantities_cls_for_configuration(label))

    assert quants == configs[label]
