"""Test for ``cockpit.utils.schedules``."""

import pytest

from cockpit.utils import schedules

MAX_STEP = 100


@pytest.mark.parametrize("interval", [1, 2, 3])
@pytest.mark.parametrize("offset", [0, 1, 2, -1])
def test_linear_schedule(interval, offset):
    """Check linear schedule.

    Args:
        interval (int): The regular tracking interval.
        offset (int, optional): Offset of tracking. Defaults to 0.
    """
    schedule = schedules.linear(interval, offset)
    tracking = []
    for i in range(MAX_STEP):
        tracking.append(schedule(i))

    # If offset is negativ, start from first true value
    if offset < 0:
        offset = interval + offset

    # check that all steps that should be tracked are true
    assert all(tracking[offset::interval])
    # Check that everything else is false
    assert sum(tracking[offset::interval]) == sum(tracking)
