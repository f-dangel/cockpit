from backpack import backpack, backpack_deactivate_io


class CockpitCTX:
    """Stores additional info handed in to a cockpit, that quantities may query."""

    INFO = {}

    @staticmethod
    def set(info, global_step):
        CockpitCTX.INFO[global_step] = info

    @staticmethod
    def get(name, global_step):
        try:
            return CockpitCTX.INFO[global_step][name]
        except KeyError as e:
            e_msg = getattr(e, "message", repr(e))
            raise KeyError(f"{e_msg}. Please hand in '{name}' via cockpit(info=...).")

    @staticmethod
    def erase():
        """Erase all stored information."""
        CockpitCTX.INFO = {}


def get_individual_losses(global_step):
    """Return the individual losses at the current iteration."""
    return CockpitCTX.get("individual_losses", global_step)


def get_batch_size(global_step):
    """Return the batch size at the current iteration."""
    return CockpitCTX.get("batch_size", global_step)


def get_loss(global_step):
    """Return the mini-batch loss at the current iteration."""
    return CockpitCTX.get("loss", global_step)


class BackwardCTX:
    """Context used by a ``Cockpit`` to handle computations in backward pass."""

    def __init__(self, cp, global_step, info, debug=False):
        CockpitCTX.set(info, global_step)
        self.cp = cp
        self.global_step = global_step

        # choose context
        ext = cp._get_extensions(global_step)
        if ext:
            self.ctx = backpack(*ext)
        else:
            self.ctx = backpack_deactivate_io()

        # update create graph
        self.cp.update_create_graph(global_step)

        if debug:
            print(f"[DEBUG, step {global_step}]")
            print(f" ↪Quantities  : {cp.quantities}")
            print(f" ↪Extensions  : {ext}")
            print(f" ↪Create graph: {cp.create_graph}")

    def __enter__(self):
        self.ctx.__enter__()

    def __exit__(self, type, value, traceback):
        self.ctx.__exit__(type, value, traceback)

        self.cp.track(self.global_step)

        CockpitCTX.erase()
