"""Tests for ``cockpit.utils.configuration``."""

import pytest

from cockpit import quantities
from cockpit.utils.configuration import quantities_cls_for_configuration


@pytest.mark.parametrize("label", ["full", "business", "economy"])
def test_quantities_cls_for_configuration(label):
    """Check cockpit configurations contain the correct quantities."""
    economy = [
        quantities.AlphaOptimized,
        quantities.GradNorm,
        quantities.UpdateSize,
        quantities.Distance,
        quantities.InnerProductTest,
        quantities.OrthogonalityTest,
        quantities.NormTest,
        quantities.BatchGradHistogram1d,
    ]
    business = economy + [quantities.TICDiag, quantities.Trace]
    full = business + [quantities.MaxEV, quantities.BatchGradHistogram2d]

    configs = {
        "full": set(full),
        "business": set(business),
        "economy": set(economy),
    }

    quants = set(quantities_cls_for_configuration(label))

    assert quants == configs[label]
