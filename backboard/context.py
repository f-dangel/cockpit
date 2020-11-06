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
