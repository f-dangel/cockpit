"""Base class for a tracked quantity."""

from collections import defaultdict


class Quantity:
    """Base class for a tracked quantity with the Cockpit.

    Quantities can modify the backward pass:
    1. They can ask that the forward pass computation graph be restored.
       This may be useful if their computation requires differentiating through
       the mini-batch loss.
    2. They can ask for certain BackPACK extensions being computed.
    """

    def __init__(self, track_interval=1):
        """Initialize the Quantity by storing the track interval.

        Crucially, it creates the output dictionary, that is meant to store all
        values that should be stored.

        Args:
            track_interval (int, optional): Tracking rate. Defaults to 1.
        """
        self._track_interval = track_interval
        self.output = defaultdict(dict)

    def create_graph(self, global_step):
        """Return whether access to the forward pass computation graph is needed.

        Args:
            global_step (int): The current iteration number.

        Returns:
            bool: ``True`` if the computation graph shall not be deleted,
                else ``False``.
        """
        return False

    def extensions(self, global_step):
        """Return list of BackPACK extensions required for the computation.

        Args:
            global_step (int): The current iteration number.

        Returns:
            list: (Potentially empty) list with required BackPACK quantities.
        """
        raise NotImplementedError

    def compute(self, global_step, params, batch_loss):
        """Evaluate quantity at a step in training.

        Args:
            global_step (int): The current iteration number.
            params ([torch.Tensor]): List of parameters considered in the computation.
            batch_loss (torch.Tensor): Mini-batch loss from current step.

        Returns:
            torch.Tensor: The quantity's value.
        """
        raise NotImplementedError
