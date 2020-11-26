import warnings

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
    loss = CockpitCTX.get("loss", global_step)
    __warn_invalid_loss(loss, global_step)
    return loss


def __warn_invalid_loss(batch_loss, global_step):
    """Warn if the mini-batch loss has values that may break the computation."""
    if batch_loss.isnan().any():
        warnings.warn(
            f"[Step {global_step}] Mini-batch loss is {batch_loss}."
            + "This may break computation of quantities."
        )


class BackwardCTX:
    """Context used by a ``Cockpit`` to handle computations in backward pass."""

    def __init__(self, cp, global_step, custom_exts, info, debug=False):
        CockpitCTX.set(info, global_step)
        self.cp = cp
        self.global_step = global_step

        self.protected_savefields = [e.savefield for e in custom_exts]

        # choose context
        ext = cp._get_extensions(global_step, custom_exts=custom_exts)
        if ext:
            self.ctx = backpack(*ext, debug=debug)
        else:
            self.ctx = backpack_deactivate_io()

        if debug:
            print(f"[DEBUG, step {global_step}]")
            print(f" ↪Quantities  : {cp.quantities}")
            print(f" ↪Extensions  : {ext}")
            print(f" ↪Create graph: {cp.create_graph}")

    def __enter__(self):
        self.ctx.__enter__()

    def __exit__(self, type, value, traceback):
        self.ctx.__exit__(type, value, traceback)

        self.cp.track(self.global_step, protected_savefields=self.protected_savefields)

        CockpitCTX.erase()
