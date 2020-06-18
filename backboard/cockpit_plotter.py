"""Plotting Part of the Cockpit"""


class CockpitPlotter:
    def __init__(self, logpath):
        params = locals()
        del params["self"]
        self.__dict__ = params

    def plot(self, show_plot=True, save_plot=False, savename_append=None):
        params = locals()
        del params["self"]
        print("Plotting with", params)
