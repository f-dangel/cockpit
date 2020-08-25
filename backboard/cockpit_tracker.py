"""Cockpit Tracker."""

import contextlib
import json

import torch

from backpack import backpack, extensions

from .tracking import tracking, utils_tracking


class CockpitTracker:
    """Cockpit Tracker class."""

    def __init__(self, parameters, optimizer, logpath, track_interval=1):
        """Initialize the Cockpit tracking.

        Args:
            parameters (func): A function to access the parameters of the
                network
            optimizer (torch.optim): A PyTorch optimizer to track its
                internal hyperparameters (e.g. the learning rate)
            logpath (str): Path to the log file.
            track_interval (int, optional): Tracking rate.
                Defaults to 1 meaning every iteration is tracked.
        """
        # Store all parameters as attributes
        params = locals()
        del params["self"]
        self.__dict__ = params

        # Only true after we called both track_before and track_after
        # NOTE: The very last `track_before` is currently done without being
        # followed by a `track_after` operation. This is currently fixed in the
        # `write` operation, but there could be a better solution!
        self.iteration_complete = True

        # check whether we support this optimizer
        utils_tracking._check_optimizer(optimizer)

        # prepare logpath (create if necessary)
        utils_tracking._prepare_logpath(logpath)

        # Initialize tracked quantites
        self.iter_tracking, self.epoch_tracking = utils_tracking._init_tracking()

        # Additional quantites we need to keep track of but not log #
        # We need to store the current search direction
        self.search_dir = [torch.zeros_like(p) for p in self.parameters()]
        # We need to store the initial parameters
        self.p_init = [p.data.clone().detach() for p in self.parameters()]

    def __call__(self, global_step):
        """Returns the backpack extensions that should be used in this iteration.

        Args:
            global_step (int): Current number of iteration.

        Returns:
            backpack.backpack: BackPACK with the appropriate extensions, or the
                nullcontext
        """
        if (
            global_step % self.track_interval == 0
            or global_step % self.track_interval == 1
        ):
            ext = [
                extensions.SumGradSquared(),
                # TODO: Use SumGradSquared, Variance recomputes summed gradient
                extensions.Variance(),
                extensions.BatchGrad(),
                extensions.DiagHessian(),
                extensions.BatchL2Grad(),
            ]

            def context_manager():
                return backpack(*ext)

        else:
            context_manager = contextlib.nullcontext

        return context_manager()

    def track_epoch(
        self,
        epoch_count,
        global_step,
        train_losses,
        valid_losses,
        test_losses,
        train_accuracies,
        valid_accuracies,
        test_accuracies,
    ):
        """Tracking function for quantities computed at every epoch.

        Args:
            epoch_count (int): Number of epoch.
            global_step (int): Number of iteration/global step.
            train_losses (float): Loss on the train (eval) set.
            valid_losses (float): Loss on the validation set.
            test_losses (float): Loss on the test set.
            train_accuracies (float): Accuracy on the train (eval) set.
            valid_accuracies (float): Accuracy on the validation set.
            test_accuracies (float): Accuracy on the test set.
        """
        self.epoch_tracking["epoch"].append(epoch_count)
        self.epoch_tracking["iteration"].append(global_step)

        self.epoch_tracking["train_loss"].append(train_losses)
        self.epoch_tracking["valid_loss"].append(valid_losses)
        self.epoch_tracking["test_loss"].append(test_losses)

        self.epoch_tracking["train_accuracy"].append(train_accuracies)
        self.epoch_tracking["valid_accuracy"].append(valid_accuracies)
        self.epoch_tracking["test_accuracy"].append(test_accuracies)

        self.epoch_tracking["learning_rate"].append(
            self.optimizer.param_groups[0]["lr"]
        )

    def track_before(self, batch_losses, global_step):
        """Tracking function that is done before the forward/backward pass.

        This split is necessary since we want to store both projected gradient
        at the beginning of an iteration (computed here) as well as at the end
        of the iteration (computed in track_after()).

        Args:
            batch_losses (list): List of individual losses in a batch. Needed to
                compute the variance of the function value.
            global_step ([type]): Current number of iteration. Used for logging
                and to check whether logging is necessary, due to the
                `track_interval`.
        """
        if self._should_track(global_step) and self.iteration_complete:
            # Preparations
            batch_loss = batch_losses.mean()
            self.iter_tracking["iteration"].append(global_step)
            self._update_search_dir()

            # Tracking functions
            tracking.track_f(self, batch_loss, "0")
            tracking.track_var_f(self, batch_losses, "0")
            tracking.track_df(self, "0")
            tracking.track_var_df(self, "0")

            tracking.track_grad_norms(self)
            # dtravel re-uses the grad norms, so compute after
            tracking.track_dtravel(self, self.optimizer.param_groups[0]["lr"])
            tracking.track_trace(self)

            # Tracked the "before" part of the iteration, but not yet the "after"
            self.iteration_complete = False
        else:
            return

    def track_after(self, batch_losses, global_step):
        """Tracking function that is done after the forward/backward pass.

        This split is necessary since we want to store both projected gradient
        at the beginning of an iteration (computed in track_before()) as well as
        at the end of the iteration (computed in here).

        Args:
            batch_losses (list): List of individual losses in a batch. Needed to
                compute the variance of the function value.
            global_step (int): Current number of iteration. Used for logging
                and to check whether logging is necessary, due to the
                `track_interval`.
        """
        if self._should_track(global_step):
            # Verify that we already run track_before
            assert (
                self.iter_tracking["iteration"][-1] == global_step
            ), "Iterations before and after are inconsistent"

            # Preparations
            batch_loss = batch_losses.mean()

            tracking.track_f(self, batch_loss, "1")
            tracking.track_var_f(self, batch_losses, "1")
            tracking.track_df(self, "1")
            tracking.track_var_df(self, "1")

            tracking.track_d2init(self)
            tracking.track_alpha(self)

            tracking.track_ev(self, batch_loss)

            tracking.track_norm_test_radius(self)
            tracking.track_global_norm_test_radius(self)

            tracking.track_inner_product_test_width(self)
            tracking.track_global_inner_product_test_width(self)

            tracking.track_acute_angle_test_sin(self)
            tracking.track_global_acute_angle_test_sin(self)

            tracking.track_mean_gsnr(self)
            tracking.track_global_mean_gsnr(self)

            # Tracked a full iteration
            self.iteration_complete = True
        else:
            return

    def write(self):
        """Write all tracked data into a JSON file."""
        # trim the dictionary so all tracked quantites have the same number of
        # elements. This is a needed fix, since we very last `track_before` is
        # not followed by a `track_after`.
        trimmed_iter_tracking = utils_tracking._trim_dict(self.iter_tracking)

        tracking = {
            "iter_tracking": trimmed_iter_tracking,
            "epoch_tracking": self.epoch_tracking,
        }
        with open(self.logpath + ".json", "w") as json_file:
            json.dump(tracking, json_file, indent=4)
        print("Cockpit-Log written...")

    def _should_track(self, global_step):
        """Returns wether we want to track in the current step.

        Args:
            global_step ([type]): Current number of iteration, checked against
                the `track_interval`.

        Returns:
            bool: Whether we want to track in this iteration.
        """
        # Skip the first step, as nothing is computed yet
        if global_step == 0:
            return False
        # If the track_interval is 1, always track (except global_step=0)
        elif self.track_interval == 1:
            return True
        # Else, track every `track_interval`-th step.
        else:
            return global_step % self.track_interval == 1

    def _update_search_dir(self):
        """Updates the search direction to the negative current gradient.

        TODO this is currently only valid for SGD!
        """
        # update search direction of all trainable (!) parameters
        self.search_dir = [-p.grad.data for p in self.parameters() if p.requires_grad]
