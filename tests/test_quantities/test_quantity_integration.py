"""Intergation tests for ``cockpit.quantities``."""

import pytest
from cockpit import quantities
from cockpit.quantities import __all__
from cockpit.quantities.quantity import SingleStepQuantity
from cockpit.utils.schedules import linear
from tests.utils import SimpleTestHarness

QUANTITIES = [
    getattr(quantities, q) for q in __all__ if q != "Quantity" and q != "Parameters"
]
IDS = [q_cls.__name__ for q_cls in QUANTITIES]


@pytest.mark.parametrize("quantity_cls", QUANTITIES, ids=IDS)
def test_quantity_integration_and_track_events(quantity_cls):
    """Check if ``Cockpit`` with a single quantity works.

    Args:
        quantity_cls (Class): Quantity class that should be tested.
    """
    interval, offset = 1, 2
    schedule = linear(interval, offset=offset)
    quantity = quantity_cls(track_schedule=schedule, verbose=True)

    iterations = 5
    testing_harness = SimpleTestHarness("ToyData", iterations)
    cockpit_kwargs = {"quantities": [quantity]}
    testing_harness.test(cockpit_kwargs)

    def is_track_event(iteration):
        if isinstance(quantity, SingleStepQuantity):
            return schedule(iteration)
        else:
            shift = quantity_cls._start_end_difference
            return schedule(iteration) and iteration + shift < iterations

    track_events = sorted(i for i in range(iterations) if is_track_event(i))
    output_events = sorted(quantity.get_output().keys())

    assert output_events == track_events
