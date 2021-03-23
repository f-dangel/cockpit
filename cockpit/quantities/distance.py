"""Classes for tracking the Parameter Distances."""

from cockpit.quantities.quantity import Quantity, SingleStepQuantity, TwoStepQuantity


class UpdateSizeTwoStep(TwoStepQuantity):
    def __init__(self, track_schedule, verbose=False):
        """Initialize the Quantity by storing the track interval.

        Crucially, it creates the output dictionary, that is meant to store all
        values that should be stored.

        Args:
            track_schedule (callable): Function that maps the ``global_step``
                to a boolean, which determines if the quantity should be computed.
            verbose (bool, optional): Turns on verbose mode. Defaults to ``False``.
        """
        save_shift = 1
        super().__init__(save_shift, track_schedule, verbose=verbose)

        self._cache_key = "params"

    def extensions(self, global_step):
        """Return list of BackPACK extensions required for the computation.

        Args:
            global_step (int): The current iteration number.

        Returns:
            list: (Potentially empty) list with required BackPACK quantities.
        """
        return []

    def is_start(self, global_step):
        return self._track_schedule(global_step)

    def is_end(self, global_step):
        return self._track_schedule(global_step - self._save_shift)

    def _compute_start(self, global_step, params, batch_loss):
        params_copy = [p.data.clone().detach() for p in params]

        def block_fn(step):
            return 0 <= step - global_step <= self._save_shift

        self.save_to_cache(global_step, self._cache_key, params_copy, block_fn)

    def _compute_end(self, global_step, params, batch_loss):
        params_start = self.load_from_cache(
            global_step - self._save_shift, self._cache_key
        )

        update_size = [
            (p.data - p_start).norm(2).item()
            for p, p_start in zip(params, params_start)
        ]

        return update_size


class UpdateSize(Quantity):
    """Update size Quantity Class."""

    _start_end_difference = 1

    def extensions(self, global_step):
        """Return list of BackPACK extensions required for the computation.

        Args:
            global_step (int): The current iteration number.

        Returns:
            list: (Potentially empty) list with required BackPACK quantities.
        """
        return []

    def should_compute(self, global_step):
        """Return if computations need to be performed at a specific iteration.

        Args:
            global_step (int): The current iteration number.

        Returns:
            bool: Truth value whether computations need to be performed.
        """
        return self._is_start(global_step) or self._is_end(global_step)

    def _is_start(self, global_step):
        """Return whether current iteration is start of update size computation."""
        return self._track_schedule(global_step)

    def _is_end(self, global_step):
        """Return whether current iteration is end of update size computation."""
        return self._track_schedule(global_step - self._start_end_difference)

    def compute(self, global_step, params, batch_loss):
        """Evaluate quantity at a step in training.

        Args:
            global_step (int): The current iteration number.
            params ([torch.Tensor]): List of torch.Tensors holding the network's
                parameters.
            batch_loss (torch.Tensor): Mini-batch loss from current step.

        Returns:
            (int, arbitrary): The second value is the result that will be stored at
                the iteration indicated by the first entry (important for multi-step
                quantities whose values are computed in later iterations).
        """
        update_size = None

        if self._is_end(global_step):
            update_size = [
                (old_p - p).norm(2).item() for old_p, p in zip(self.old_params, params)
            ]
            del self.old_params

        if self._is_start(global_step):
            self.old_params = [p.data.clone().detach() for p in params]

        return global_step - self._start_end_difference, update_size


class Distance(SingleStepQuantity):
    """Distance from initialization Quantity Class."""

    def extensions(self, global_step):
        """Return list of BackPACK extensions required for the computation.

        Args:
            global_step (int): The current iteration number.

        Returns:
            list: (Potentially empty) list with required BackPACK quantities.
        """
        return []

    def should_compute(self, global_step):
        """Return if computations need to be performed at a specific iteration.

        Args:
            global_step (int): The current iteration number.

        Returns:
            bool: Truth value whether computations need to be performed.
        """
        return global_step == 0 or self._track_schedule(global_step)

    def _compute(self, global_step, params, batch_loss):
        """Evaluate the update size."""
        distance = None

        # Store initial parameters
        if global_step == 0:
            self.params_init = [p.data.clone().detach() for p in params]

        if self._track_schedule(global_step):
            distance = [
                (init - p).norm(2).item()
                for init, p in zip(self.params_init, params)
                if p.requires_grad
            ]

        return distance
