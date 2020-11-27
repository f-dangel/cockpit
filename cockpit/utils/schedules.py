"""Convenient schedule functions."""

import torch


def linear(interval, offset=0):
    """Create schedule that tracks at iterations ``{offset + n interval | n >= 0}``."""

    docstring = "Track at iterations {" + f"{offset} + n {interval} " + "| n >= 0}."

    def schedule(global_step):
        if global_step < 0:
            return False
        else:
            return (global_step - offset) % interval == 0

    schedule.__doc__ = docstring

    return schedule


def logarithmic(start, end, steps=300, base=10, init=True):
    """Create schedule that tracks linearly in logspace."""

    # TODO Compute match and avoid array lookup
    scheduled_steps = torch.logspace(start, end, steps, base=base, dtype=int)

    if init:
        zero = torch.tensor([0], dtype=int)
        scheduled_steps = torch.cat((scheduled_steps, zero)).int()

    def schedule(global_step):
        return global_step in scheduled_steps

    return schedule
