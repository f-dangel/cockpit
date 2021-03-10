"""Intergation tests for ``cockpit.quantities``."""

from cockpit import quantities
from cockpit.utils.schedules import linear
from tests.utils import SimpleTestHarness


class CustomTestHarness(SimpleTestHarness):
    """Create a Custom Test Harness that checks whether the BackPACK buffers exist."""

    def check_in_context(self):
        """Check that the BackPACK buffers exists in the context."""
        for param in self.model.parameters():
            # required by TICDiag and user
            assert hasattr(param, "diag_h")
            # required by TICDiag only
            assert hasattr(param, "grad_batch_transforms")
            assert "sum_grad_squared" in param.grad_batch_transforms

    def check_after_context(self):
        """Check that the buffers are not deleted when specified by the user."""
        for param in self.model.parameters():
            # assert hasattr(param, "diag_h")
            # not protected by user
            assert not hasattr(param, "grad_batch_transforms")


def test_backpack_extensions():
    """Check if backpack quantities can be computed inside cockpit."""
    quantity = quantities.TICDiag(track_schedule=linear(1))

    iterations = 3
    testing_harness = CustomTestHarness("ToyData", iterations)
    cockpit_kwargs = {"quantities": [quantity]}
    testing_harness.test(cockpit_kwargs)
