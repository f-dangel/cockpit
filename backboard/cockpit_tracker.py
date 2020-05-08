"""Cockpit Tracker"""

import json

import torch

from backpack import extensions

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

        # check whether we support this optimizer
        utils_tracking._check_optimizer(optimizer)

        # prepare logpath (create if necessary)
        utils_tracking._prepare_logpath(logpath)

        # Initialize tracked quantites
        per_iter_quants = [
            "iteration",  # keep track which iteration we store
            "f0",  # function value at the beginning of a step
            "f1",  # function value at the end of a step
            "var_f0",  # variance of the function value
            "var_f1",  # -"-
            "df0",  # derivative, the projected gradient onto the search dir
            "df1",  # -"-
            "var_df0",  # variance of the derivative.
            "var_df1",  # -"-
            "grad_norms",  # gradient norm, is computed at theta_0 (position of f0)
            "dtravel",  # update step size, from theta_0 to theta_1
            "d2init",  # distance to parameter init, computed at theta_0
            "trace",  # hessian trace, computed at theta_0
            "alpha",  # local effective step size
            "max_ev",  # largest eigenvalue of hessian, computed at theta_0
        ]
        per_epoch_quants = [
            "iteration",  # keep track which iteration we store
            "epoch",  # keep track which epoch we store
            "train_loss",
            "valid_loss",
            "test_loss",
            "train_accuracy",
            "valid_accuracy",
            "test_accuracy",
            "learning_rate",
        ]
        self.iter_tracking, self.epoch_tracking = utils_tracking._init_tracking(
            per_iter_quants, per_epoch_quants
        )

        # Additional quantites we need to keep track of but not log #
        # We need to store the current search direction
        self.search_dir = [torch.zeros_like(p) for p in self.parameters()]
        # We need to store the initial parameters
        self.p_init = [p.data.clone().detach() for p in self.parameters()]

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
        """Tracking function for quantities computed at every epoch

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
        """Tracking function that is done before we get the new loss and
        gradients.

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
        if self._should_track(global_step):
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
            tracking.track_ev(self, batch_loss)
        else:
            return

    def track_after(self, batch_losses, global_step):
        """Tracking function that is done after we get the new loss and
        gradients.

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
        else:
            return

    def write(self):
        """Write all tracked data into a JSON file."""
        tracking = {
            "iter_tracking": self.iter_tracking,
            "epoch_tracking": self.epoch_tracking,
        }
        with open(self.logpath + ".json", "w") as json_file:
            json.dump(tracking, json_file, indent=4)
        print("Cockpit-Log written...")

    def extensions(self, global_step):
        """Returns the backpack extensions that should be used in this iteration

        Args:
            global_step (int): Current number of iteration.
        """
        # Use BackPACK either if we want to track this or the next iteration.
        # We need both because of the track_before, track_after split.
        if (
            global_step % self.track_interval == 0
            or global_step % self.track_interval == 1
        ):
            return (
                extensions.Variance(),
                extensions.BatchGrad(),
                extensions.DiagHessian(),
            )
        else:
            return []

    def _should_track(self, global_step):
        """Returns wether we want to track in the current step

        Args:
            global_step ([type]): Current number of iteration, checked against
                the `track_interval`.
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

        TODO this is currently only valid for SGD!"""
        # update search direction of all trainable (!) parameters
        self.search_dir = [-p.grad.data for p in self.parameters() if p.requires_grad]
