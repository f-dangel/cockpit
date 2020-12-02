"""Tests for ``cockpit.quantities.quantity``."""

import pytest

from cockpit import quantities
from cockpit.quantities.distance import Distance
from cockpit.quantities.quantity import SingleStepQuantity
from cockpit.utils.schedules import linear
from tests.test_quantities.utils import train_small_mlp

PROBLEMS = [
    quantities.Time,
    quantities.Loss,
    quantities.CABS,
    quantities.MeanGSNR,
    quantities.EarlyStopping,
    quantities.Parameters,
    quantities.AlphaOptimized,
    quantities.GradNorm,
    quantities.Distance,
    quantities.InnerProductTest,
    quantities.OrthogonalityTest,
    quantities.NormTest,
    quantities.BatchGradHistogram1d,
    quantities.TICDiag,
    quantities.Trace,
    quantities.MaxEV,
    quantities.BatchGradHistogram2d,
]
IDS = [q_cls.__name__ for q_cls in PROBLEMS]


@pytest.mark.parametrize("quantity_cls", PROBLEMS, ids=IDS)
def test_correct_track_events_in_output(quantity_cls):
    """Check if a ``Cockpit`` with a single quanity writes to the output field."""
    interval, offset = 1, 2
    schedule = linear(interval, offset=offset)
    quantity = quantity_cls(track_schedule=schedule, verbose=True)

    iterations = 5
    train_small_mlp(iterations, [quantity], True)

    def is_track_event(iteration):
        # TODO Refactor ``quantities.Distance`` into two quantities:
        # 1) D2Init (SingleStepQuantity)
        # 2) Distance (MultiStepQuantity)
        # Currently, it can happen that ``d2init`` is already computed
        # ``update_size`` is only computed at the next iteration. This
        # leads to unexpected behavior in the output.
        if isinstance(quantity, (SingleStepQuantity, quantities.Distance)):
            return schedule(iteration)
        else:
            # multi-step quantity
            shift = quantity_cls._start_end_difference
            return schedule(iteration) and iteration + shift < iterations

    track_events = sorted([i for i in range(iterations) if is_track_event(i)])
    output_events = sorted(list(quantity.output.keys()))

    assert output_events == track_events
