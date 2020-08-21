"""Airline Plotter."""


class AirlinePlotter:
    """AirlinePlotter Class."""

    def __init__(self):
        """Initialize the AirlinePlotter."""
        params = locals()
        del params["self"]
        self.__dict__ = params
