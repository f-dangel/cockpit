"""Intergation tests for ``cockpit.quantities``."""

import pytest

from cockpit import quantities
from cockpit.quantities import __all__
from cockpit.quantities.quantity import SingleStepQuantity, TwoStepQuantity
from cockpit.utils.schedules import linear
from tests.test_quantities.settings import PROBLEMS, PROBLEMS_IDS
from tests.utils.harness import SimpleTestHarness
from tests.utils.problem import instantiate

QUANTITIES = [
    getattr(quantities, q)
    for q in __all__
    if q != "Quantity"
    and q != "SingleStepQuantity"
    and q != "TwoStepQuantity"
    and q != "ByproductQuantity"
]
IDS = [q_cls.__name__ for q_cls in QUANTITIES]


@pytest.mark.parametrize("problem", PROBLEMS, ids=PROBLEMS_IDS)
@pytest.mark.parametrize("quantity_cls", QUANTITIES, ids=IDS)
def test_quantity_integration_and_track_events(problem, quantity_cls):
    """Check if ``Cockpit`` with a single quantity works.

    Args:
        problem (tests.utils.Problem): Settings for train loop.
        quantity_cls (Class): Quantity class that should be tested.
    """
    interval, offset = 1, 2
    schedule = linear(interval, offset=offset)
    quantity = quantity_cls(track_schedule=schedule, verbose=True)

    with instantiate(problem):
        iterations = problem.iterations
        testing_harness = SimpleTestHarness(problem)
        cockpit_kwargs = {"quantities": [quantity]}
        testing_harness.test(cockpit_kwargs)

    def is_track_event(iteration):
        if isinstance(quantity, SingleStepQuantity):
            return schedule(iteration)
        elif isinstance(quantity, TwoStepQuantity):
            end_iter = quantity.SAVE_SHIFT + iteration
            return quantity.is_end(end_iter) and end_iter < iterations
        else:
            raise ValueError(f"Unknown quantity: {quantity}")

    track_events = sorted(i for i in range(iterations) if is_track_event(i))
    output_events = sorted(quantity.get_output().keys())

    assert output_events == track_events
