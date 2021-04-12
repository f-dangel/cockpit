"""Convenient schedule functions."""

import torch


def linear(interval, offset=0):
    """Creates a linear schedule that tracks when ``{offset + n interval | n >= 0}``.

    Args:
        interval (int): The regular tracking interval.
        offset (int, optional): Offset of tracking. Defaults to 0.

    Returns:
        callable: Function that given the global_step returns whether it should track.
    """
    docstring = "Track at iterations {" + f"{offset} + n * {interval} " + "| n >= 0}."

    def schedule(global_step):
        shifted = global_step - offset
        if shifted < 0:
            return False
        else:
            return shifted % interval == 0

    schedule.__doc__ = docstring

    return schedule


def logarithmic(start, end, steps=300, base=10, init=True):
    """Creates a logarithmic tracking schedule.

    Args:
        start ([type]): The starting value.
        end ([type]): The end value.
        steps (int, optional): Number of log spaced points. Defaults to 300.
        base (int, optional): Logarithmic base. Defaults to 10.
        init (bool, optional): Whether 0 should be included. Defaults to True.

    Returns:
        callable: Function that given the global_step returns whether it should track.
    """
    # TODO Compute match and avoid array lookup
    scheduled_steps = torch.logspace(start, end, steps, base=base, dtype=int)

    if init:
        zero = torch.tensor([0], dtype=int)
        scheduled_steps = torch.cat((scheduled_steps, zero)).int()

    def schedule(global_step):
        return global_step in scheduled_steps

    return schedule
