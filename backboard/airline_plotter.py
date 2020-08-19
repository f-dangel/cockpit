"""Airline Plotter"""


class AirlinePlotter:
    def __init__(self):
        params = locals()
        del params["self"]
        self.__dict__ = params
