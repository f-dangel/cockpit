"""Check that ``track`` is called when leaving the ``cockpit`` context."""

import pytest

from cockpit import quantities
from cockpit.utils.schedules import linear
from tests.test_cockpit.settings import PROBLEMS, PROBLEMS_IDS
from tests.utils.harness import SimpleTestHarness
from tests.utils.problem import instantiate


# TODO Reconsider purpose of this test
class CustomTestHarness(SimpleTestHarness):
    """Custom Test Harness checking that track gets called when leaving the context."""

    def check_in_context(self):
        """Check that track has not been called yet."""
        assert self.problem.iterations == 1, "Test only checks the first step"
        global_step = 0
        assert global_step not in self.cockpit.quantities[0].output.keys()

    def check_after_context(self):
        """Verify that track has been called after the context is left."""
        assert self.problem.iterations == 1, "Test only checks the first step"
        global_step = 0
        assert global_step in self.cockpit.quantities[0].output.keys()


@pytest.mark.parametrize("problem", PROBLEMS, ids=PROBLEMS_IDS)
def test_track_writes_output(problem):
    """Test that a ``cockpit``'s ``track`` function writes to the output."""
    quantity = quantities.Time(track_schedule=linear(1))

    with instantiate(problem):
        testing_harness = CustomTestHarness(problem)
        cockpit_kwargs = {"quantities": [quantity]}
        testing_harness.test(cockpit_kwargs)
