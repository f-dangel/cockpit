"""Superclass for an instrument of the cockpit."""


class Instrument:
    def __init__(self):
        params = locals()
        del params["self"]
        self.__dict__ = params

    def plot(self):
        print("***Plotting...")
