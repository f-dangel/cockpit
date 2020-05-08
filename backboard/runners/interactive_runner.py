"""Interactive Runner to combine DeepOBS and Backboard."""

from backboard import CockpitTracker


class InteractiveCockpitRunner:
    def __init__(self):
        params = locals()
        del params["self"]
        self.__dict__ = params

        self.cockpit = CockpitTracker()
