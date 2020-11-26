"""Convenient schedule functions."""

import numpy


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


def logarithmic(start, stop, num=300, base=10, init=True):
    """Create schedule that tracks linearly in logspace."""

    # TODO Compute match and avoid array lookup
    scheduled_steps = numpy.logspace(start, stop, num=num, dtype=int)

    if init:
        zero = numpy.array([0], dtype=int)
        scheduled_steps = numpy.concatenate((scheduled_steps, zero), dtype=int)

    def schedule(global_step):
        return global_step in scheduled_steps

    return schedule
