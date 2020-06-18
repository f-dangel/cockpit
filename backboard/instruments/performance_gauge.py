"""Performance Gauge."""

from .instrument import Instrument


class PerformanceGauge(Instrument):
    def __init__(self):
        super().__init__()

    def other_stuff(self):
        print("doing something else")
