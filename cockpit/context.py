"""Cockpit Context."""

import warnings

from backpack import backpack, disable
from backpack.core.derivatives.convnd import weight_jac_t_save_memory


class CockpitCTX:
    """Stores additional info handed in to a cockpit, that quantities may query."""

    INFO = {}

    @staticmethod
    def set(info, global_step):
        """Store the given info for the global step.

        Args:
            info (dict): Dictionary that specifies additional information. Some
                quantities require additional information that is overly difficult
                to infer from a backward pass, like the individual losses.
            global_step (int): Current number of iteration/global step.
        """
        CockpitCTX.INFO[global_step] = info

    @staticmethod
    def get(name, global_step):
        """Get info from global step.

        Args:
            name (str): Name of the info that should be extracted.
            global_step (int): Current number of iteration/global step.

        Raises:
            KeyError: If no info is stored for `name`.

        Returns:
            [type]: Info stored under name for this step.
        """
        try:
            return CockpitCTX.INFO[global_step][name]
        except KeyError as e:
            raise KeyError(f"Please hand in '{name}' via cockpit(info=...).") from e

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


def get_optimizer(global_step):
    """Return the optimizer at the current iteration."""
    return CockpitCTX.get("optimizer", global_step)


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
        """Initialize context for the backward pass.

        Args:
            cp (Cockpit): ``Cockpit`` instance.
            global_step (int): Current number of iteration.
            custom_exts (list or tuple): Custom BackPACK extensions that will be
                computed on top.
            info (dict): Dictionary that specifies additional information. Some
                quantities require additional information that is overly difficult
                to infer from a backward pass, like the individual losses.
            debug (bool, optional): Switch on debug mode. Defaults to False.
        """
        CockpitCTX.set(info, global_step)
        self.cp = cp
        self.global_step = global_step

        self.protected_savefields = [e.savefield for e in custom_exts]

        # choose context
        ext = cp._get_extensions(global_step, custom_exts=custom_exts)
        ext_hook = cp._get_extension_hook(global_step)

        save_memory = cp.BACKPACK_CONV_SAVE_MEMORY

        if ext:
            self.contexts = (
                backpack(*ext, extension_hook=ext_hook, debug=debug),
                weight_jac_t_save_memory(save_memory=save_memory),
            )
        else:
            self.contexts = (disable(),)

        if debug:
            print(f"[DEBUG, step {global_step}]")
            print(f" ↪Quantities  : {cp.quantities}")
            print(f" ↪Extensions  : {ext}")
            print(f" ↪Hooks       : {ext_hook}")
            print(f" ↪Create graph: {cp.create_graph(global_step)}")
            print(f" ↪Save memory : {save_memory}")

    def __enter__(self):
        """Enter cockpit context(s)."""
        for ctx in self.contexts:
            ctx.__enter__()

    def __exit__(self, type, value, traceback):
        """Exist cockpit context(s) and call tracking function of cockpit."""
        for ctx in self.contexts:
            ctx.__exit__(type, value, traceback)

        self.cp.track(self.global_step, protected_savefields=self.protected_savefields)

        CockpitCTX.erase()
