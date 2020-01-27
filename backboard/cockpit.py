"""Cockpit"""

import itertools
import json
import math
import os
import threading

import numpy as np
import torch

from backboard.cockpit_plotter import CockpitPlotter
from backboard.utils.cockpit_utils import (
    _fit_quadratic,
    _get_alpha,
    _layerwise_dot_product,
)


class Cockpit:
    """Cockpit class."""

    def __init__(self, get_parameters, opt, run_dir, file_name, plot_interval=1):
        """Initialize the Cockpit.

        Args:
            get_parameters ([func]): A function to access the parameters of the 
                network
            run_dir ([str]): Save directory of the run.
            file_name ([str]): File name of the run.
        """

        # Save inputs
        self.get_parameters = get_parameters
        self.plot_interval = plot_interval
        self.opt = opt

        # Make dir if necessary
        os.makedirs(run_dir, exist_ok=True)
        self.log_path = os.path.join(run_dir, file_name + "__log")

        # Create a CockpitPlotter class
        self.cockpit_plotter = CockpitPlotter(self.log_path)

        # Initialize tracking
        self.search_dir = []
        self.tracking, self.tracking_epoch = self.init_tracking()

        # Start listening for input
        self.keyinput = ""
        self.start_listening()

    def init_tracking(self):
        """Initialize the tracking by allocating the logs.

        Returns:
            [dict]: Dictionary to write the tracking quantities to.
        """

        self.p_init = [p.data.clone().detach() for p in self.get_parameters()]
        self.search_dir = [torch.zeros_like(p) for p in self.get_parameters()]

        tracking = dict()

        tracking_epoch = dict()

        # function value and its variance
        # implemented in a continuous list
        # will later be split into pairs of sequential values
        tracking["f_t"] = []
        tracking["var_f_t"] = []

        # derivative and its variance. This is the projected gradient
        # onto the search direction.
        tracking["df_0"] = []
        tracking["df_1"] = []
        # first project the gradients, then compute their variance.
        # Exact but slow way.
        tracking["var_df_0"] = []
        tracking["var_df_1"] = []
        # first compute the variance, then project it
        # Incorrect, but fast approximation.
        tracking["df_var_0"] = []
        tracking["df_var_1"] = []

        # Gradient Norms
        tracking["grad_norms"] = []

        # Distance traveled
        tracking["dtravel"] = []

        # Distance to Init
        tracking["d2init"] = []

        # Hessian Trace
        tracking["trace"] = []

        # Local Step Length
        tracking["alpha"] = []

        return tracking, tracking_epoch

    def start_listening(self):
        """Start a daeomen thread that waits for user input."""

        def check_input():
            self.keyinput = input()

        self.keyinput = ""
        self.keycheck = threading.Thread(target=check_input)
        self.keycheck.daemon = True
        self.keycheck.start()

    def check_listening(self):
        """Check for user input during last epoch and act accordingly.

        Returns:
            [Bool]: If training should be stopped.
        """
        if not self.keycheck.is_alive():
            if self.keyinput == "q":
                print("Quit after user intervention!")
                return True
            elif self.keyinput == "e":
                # If somebody pressed "e", now is the time to intervene.
                current_lr = self.opt.param_groups[0]["lr"]
                new_lr = input(
                    "Current rearning rate: "
                    + str(current_lr)
                    + "\nChange learning rate to: "
                )
                try:
                    new_lr = float(new_lr)
                    for param_group in self.opt.param_groups:
                        param_group["lr"] = new_lr
                except ValueError:
                    print("Invalid input. Leaving learning rate as it is")

            self.start_listening()

    def track(self, batch_losses):
        """Track all relevant quantities.

        Args:
            batch_losses ([array]): Array of individual losses in a batch.
        """

        # Compute the (mean) batch loss.
        batch_loss = torch.mean(batch_losses)

        # Track Statistics (before updating the search dir)
        self.track_d2init()
        self.track_grad_norm()
        self.track_dtravel()
        self.track_f(batch_loss)
        self.track_var_f(batch_losses)
        self.track_var_df_1()
        self.track_df_1()
        self.track_trace()
        self.track_alpha()

        # Update Search Direction
        self.update_search_dir()

        # Track Statistics (after updating the search dir)
        self.track_var_df_0()
        self.track_df_0()

    def epoch_track(self, train_accuracies, valid_accuracies):
        """[summary]

        Args:
            train_accuracies ([type]): [description]
            valid_accuracies ([type]): [description]
        """
        self.tracking_epoch["train_acc"] = train_accuracies
        self.tracking_epoch["valid_acc"] = valid_accuracies
        if "lr" in self.tracking_epoch:
            self.tracking_epoch["lr"].append(self.opt.param_groups[0]["lr"])
        else:
            self.tracking_epoch["lr"] = [self.opt.param_groups[0]["lr"]]

    def write(self):
        """ Write all tracking data into a JSON file."""
        tracking = {
            "iter_tracking": {
                "f0": self.tracking["f_t"][:-1],
                "f1": self.tracking["f_t"][1:],
                "var_f0": self.tracking["var_f_t"][:-1],
                "var_f1": self.tracking["var_f_t"][1:],
                "df0": self.tracking["df_0"][:-1],
                "df1": self.tracking["df_1"][1:],
                "var_df_0": self.tracking["var_df_0"][:-1],
                "var_df_1": self.tracking["var_df_1"][1:],
                "grad_norms": self.tracking["grad_norms"][:-1],
                "dtravel": self.tracking["dtravel"][:-1],
                "d2init": self.tracking["d2init"][:-1],
                "trace": self.tracking["trace"][:-1],
                "alpha": self.tracking["alpha"][:],
            },
            "epoch_tracking": self.tracking_epoch,
        }

        with open(self.log_path + ".json", "w") as json_file:
            json.dump(tracking, json_file, indent=4)
        print("Cockpit-Log written...")

    def update_search_dir(self):
        """Updates the search direction to the negative current gradient."""
        self.search_dir = [-p.grad.data for p in self.get_parameters()]

    def track_d2init(self):
        """Tracks the L2 distance of the current parameters to their init."""
        self.tracking["d2init"].append(
            [
                (init - p).norm(2).item()
                for init, p in zip(self.p_init, self.get_parameters())
            ]
        )

    def track_dtravel(self):
        """Tracks the distance traveled in each iteration"""
        self.tracking["dtravel"].append(
            [
                el * self.opt.param_groups[0]["lr"]
                for el in self.tracking["grad_norms"][-1]
            ]
        )

    def track_f(self, batch_loss):
        """Tracks the (mean) batch loss/function value of the 1-D search.

        Args:
            batch_loss ([float]): Mean loss value in the batch.
        """
        self.tracking["f_t"].append(batch_loss.item())

    def track_var_f(self, batch_losses):
        """Tracks the variance of the batch loss.

        Args:
            batch_losses ([array]): Array of individual losses in a batch.
        """
        self.tracking["var_f_t"].append(batch_losses.var().item())

    def track_grad_norm(self):
        """Tracks the gradient norm."""
        self.tracking["grad_norms"].append(
            [p.grad.data.norm(2).item() for p in self.get_parameters()]
        )

    def track_var_df_1(self):
        """Tracks the variance of the projected gradients at f(alpha).

        This method is exact (what we need), but slow."""
        grads = [p.grad_batch.data for p in self.get_parameters()]
        self.tracking["var_df_1"].append(self._exact_variance(grads))

    def track_var_df_0(self):
        """Tracks the variance of the projected gradients at f(0).

        This method is exact (what we need), but slow."""
        grads = [p.grad_batch.data for p in self.get_parameters()]
        self.tracking["var_df_0"].append(self._exact_variance(grads))

    def track_df_1(self):
        """Tracks the projected gradient at f(alpha) layerwise."""
        self.tracking["df_1"].append(
            _layerwise_dot_product(
                self.search_dir, [p.grad.data for p in self.get_parameters()]
            )
        )

    def track_df_0(self):
        """Tracks the projected gradient at f(0) layerwise."""
        self.tracking["df_0"].append(
            _layerwise_dot_product(
                self.search_dir, [p.grad.data for p in self.get_parameters()]
            )
        )

    def track_trace(self):
        """Tracks the trace of the Hessian."""
        self.tracking["trace"].append(
            [p.diag_h.sum().item() for p in self.get_parameters()]
        )

    def track_alpha(self):
        """Tracks the relative step size."""
        # TODO Use dtravel[-1] instead of f[1]!
        # Skip the first time this function is called
        # (since we don't have a full iteration yet.)
        if len(self.tracking["f_t"]) < 2:
            return

        # We need to find the size of the step taken,
        # since dtravel can be a list, we need to aggregate it
        if type(self.tracking["dtravel"][-1]) is list:
            t = math.sqrt(sum(t * t for t in self.tracking["dtravel"][-1]))
        else:
            t = self.tracking["dtravel"][-1]
        mu = _fit_quadratic(
            t,
            [self.tracking["f_t"][-2], self.tracking["f_t"][-1]],
            [sum(self.tracking["df_0"][-1]), sum(self.tracking["df_1"][-1])],
            [self.tracking["var_f_t"][-2], self.tracking["var_f_t"][-1]],
            [sum(self.tracking["var_df_0"][-1]), sum(self.tracking["var_df_1"][-1]),],
        )

        # Get the relative (or local) step size
        self.tracking["alpha"].append(_get_alpha(mu, t))

    def _exact_variance(self, grads):
        """Given a batch of individual gradients, it computes the exact variance
        of their projection onto the search direction.

        Args:
            grads ([list]): A batch of individual gradients.

        Returns:
            [list]: List of variances, split by layer.
        """
        # swap order of first two axis, so now it is:
        # grads[batch_size, layer, parameters]
        grads = [
            [i for i in element if i is not None]
            for element in list(itertools.zip_longest(*grads))
        ]
        proj_grad = []
        for grad in grads:
            proj_grad.append(_layerwise_dot_product(self.search_dir, grad))
        return np.var(np.array(proj_grad), axis=0, ddof=1).tolist()
