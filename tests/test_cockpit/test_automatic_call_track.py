"""Intergation tests for ``cockpit.quantities``."""

from cockpit import quantities
from cockpit.utils.schedules import linear
from tests.utils import SimpleTestHarness


class CustomTestHarness(SimpleTestHarness):
    """Custom Test Harness checking that track gets called when leaving the context."""

    def check_in_context(self):
        """Check that track has not been called yet."""
        global_step = 0
        assert global_step not in self.cockpit.quantities[0].output.keys()

    def check_after_context(self):
        """Verify that track has been called after the context is left."""
        global_step = 0
        assert global_step in self.cockpit.quantities[0].output.keys()


def test_backpack_extensions():
    """Check if backpack quantities can be computed inside cockpit."""
    quantity = quantities.Time(track_schedule=linear(1))

    iterations = 1
    testing_harness = CustomTestHarness("ToyData", iterations)
    cockpit_kwargs = {"quantities": [quantity]}
    testing_harness.test(cockpit_kwargs)
