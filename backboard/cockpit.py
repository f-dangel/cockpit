"""Cockpit"""

import itertools
import json
import math
import os
import threading
import warnings

import numpy as np
import torch
from scipy.sparse.linalg import eigsh

from backboard.cockpit_plotter import CockpitPlotter
from backboard.utils.cockpit_utils import (_fit_quadratic, _get_alpha,
                                           _layerwise_dot_product)
from backboard.utils.linear_operator import HVPLinearOperator
from backpack import extensions


class Cockpit:
    """Cockpit class."""

    def __init__(
        self,
        get_parameters,
        opt,
        run_dir,
        file_name,
        track_interval=1,
        plot_interval=1,
    ):
        """Initialize the Cockpit.

        Args:
            get_parameters ([func]): A function to access the parameters of the 
                network
            run_dir ([str]): Save directory of the run.
            file_name ([str]): File name of the run.
        """

        # Save inputs
        self.get_parameters = get_parameters
        self.track_interval = track_interval
        self.plot_interval = plot_interval
        self.opt = opt

        self._check_optimizer()

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
        tracking["iteration"] = []

        # function value
        tracking["f0"] = []
        tracking["f1"] = []
        # and its variance
        tracking["var_f0"] = []
        tracking["var_f1"] = []

        # derivative and its variance. This is the projected gradient
        # onto the search direction.
        tracking["df0"] = []
        tracking["df1"] = []
        # first project the gradients, then compute their variance.
        # Exact but slow way.
        tracking["var_df0"] = []
        tracking["var_df1"] = []

        # Gradient tests
        tracking["norm_test_layers"] = []
        tracking["norm_test_network"] = []
        tracking["inner_product_test_layers"] = []
        tracking["inner_product_test_network"] = []
        tracking["orthogonality_test_layers"] = []
        tracking["orthogonality_test_network"] = []
        tracking["acute_angle_test_layers"] = []
        tracking["acute_angle_test_network"] = []

        # Gradient Norms
        # is alywas computed at theta_0 (the position of f0)
        tracking["grad_norms"] = []

        # Distance traveled
        # (aka the size of the step from theta_0 to theta_1)
        tracking["dtravel"] = []

        # Distance to Init
        # is alywas computed at theta_0 (the position of f0)
        tracking["d2init"] = []

        # Hessian Trace
        # is alywas computed at theta_0 (the position of f0)
        tracking["trace"] = []

        # Local Effective Step Length
        # (aka where do we end up on a quadratic fit with the step we took)
        tracking["alpha"] = []

        # Max Eigenvalue of the Hessian
        # is alywas computed at theta_0 (the position of f0)
        tracking["max_ev"] = []

        # Quantaties that are only tracked once per epoch
        tracking_epoch = dict()
        tracking_epoch["iteration"] = []
        tracking_epoch["epoch"] = []

        # Losses
        tracking_epoch["train_loss"] = []
        tracking_epoch["valid_loss"] = []
        tracking_epoch["test_loss"] = []
        # Accuracies
        tracking_epoch["train_accuracy"] = []
        tracking_epoch["valid_accuracy"] = []
        tracking_epoch["test_accuracy"] = []

        tracking_epoch["learning_rate"] = []

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

    def should_track(self, global_step):
        """Returns wether we want to track in the current step

        Args:
            global_step (int): Number of current iteration
        """
        if global_step == 0:
            return False
        elif self.track_interval == 1:
            return True
        else:
            return global_step % self.track_interval == 1

    def extensions(self, global_step):
        """Returns the backpack extensions that should be used in this iteration

        Args:
            global_step ([type]): [description]
        """
        if (
            global_step % self.track_interval == 0
            or global_step % self.track_interval == 1
        ):
            return (
                extensions.Variance(),
                extensions.BatchGrad(),
                extensions.DiagHessian(),
                extensions.BatchL2Grad(),
            )
        else:
            return []

    def track_before(self, batch_losses, global_step):
        """[summary]

        Args:
            batch_losses ([type]): [description]
        """
        batch_loss = torch.mean(batch_losses)

        self.tracking["iteration"].append(global_step)

        self.update_search_dir()

        self.track_f(batch_loss, "0")
        self.track_var_f(batch_losses, "0")
        self.track_df("0")
        self.track_var_df("0")

        self.track_grad_norms()
        self.track_dtravel()  # important to compute after grad_norms !
        self.track_trace()
        self.track_ev(batch_loss)

    def track_after(self, batch_losses, global_step):
        """[summary]

        Args:
            batch_losses ([type]): [description]
        """
        assert (
            self.tracking["iteration"][-1] == global_step
        ), "Iterations before and after are inconsistent"

        batch_loss = torch.mean(batch_losses)

        self.track_f(batch_loss, "1")
        self.track_var_f(batch_losses, "1")
        self.track_df("1")
        self.track_var_df("1")

        self.track_norm_test()
        self.track_inner_product_test()
        self.track_orthogonality_test()
        self.track_acute_angle_test()

        self.track_d2init()
        self.track_alpha()

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
        """Tracks quantities at the start of epoch number epoch_count.

        The epoch_count starts at 0. if a learning rate of [1, 2, 3] is tracked
        this means that a learning rate of 1 is used for epoch 0, a learning
        rate of 2 for epoch 1 and it was then switched to 3 for epoch 2 (but this
        epoch was never trained, since we only wanted to train for 2 epochs).

        Args:
            epoch_count ([type]): [description]
            global_step ([type]): [description]
            train_accuracies ([type]): [description]
            valid_accuracies ([type]): [description]

        Returns:
            [type]: [description]
        """
        self.tracking_epoch["epoch"].append(epoch_count)
        self.tracking_epoch["iteration"].append(global_step)

        self.tracking_epoch["train_loss"].append(train_losses)
        self.tracking_epoch["valid_loss"].append(valid_losses)
        self.tracking_epoch["test_loss"].append(test_losses)

        self.tracking_epoch["train_accuracy"].append(train_accuracies)
        self.tracking_epoch["valid_accuracy"].append(valid_accuracies)
        self.tracking_epoch["test_accuracy"].append(test_accuracies)

        self.tracking_epoch["learning_rate"].append(self.opt.param_groups[0]["lr"])

    # General tracking methods

    def update_search_dir(self):
        """Updates the search direction to the negative current gradient.

        TODO this is currently only valid for SGD!"""
        self.search_dir = [-p.grad.data for p in self.get_parameters()]

    def track_f(self, batch_loss, point):
        """[summary]

        Args:
            batch_losses ([type]): [description]
        """
        self.tracking["f" + point].append(batch_loss.item())

    def track_var_f(self, batch_losses, point):
        """[summary]

        Args:
            batch_losses ([type]): [description]
        """
        self.tracking["var_f" + point].append(batch_losses.var().item())

    def track_df(self, point):
        """[summary]"""
        self.tracking["df" + point].append(
            _layerwise_dot_product(
                self.search_dir, [p.grad.data for p in self.get_parameters()]
            )
        )

    def track_var_df(self, point):
        """Tracks the variance of the projected gradients at f(0).

        This method is exact (what we need), but slow."""
        self.tracking["var_df" + point].append(
            self._exact_variance([p.grad_batch.data for p in self.get_parameters()])
        )

    # Track_before methods

    def track_grad_norms(self):
        """Tracks the gradient norm."""
        self.tracking["grad_norms"].append(
            [p.grad.data.norm(2).item() for p in self.get_parameters()]
        )

    def track_trace(self):
        """Tracks the trace of the Hessian."""
        self.tracking["trace"].append(
            [p.diag_h.sum().item() for p in self.get_parameters()]
        )

    def track_ev(self, batch_loss):
        """Track the min and max eigenvalue of the Hessian.

        Args:
            batch_loss (torch): 
        """
        HVP = HVPLinearOperator(
            batch_loss,
            list(self.get_parameters()),
            grad_params=[p.grad for p in self.get_parameters()],
        )
        eigvals = eigsh(HVP, k=1, which="LA", return_eigenvectors=False)

        self.tracking["max_ev"].append(np.float64(eigvals))

    # Track_after methods

    def track_dtravel(self):
        """Tracks the distance traveled in each iteration.

        It is very important that this function is computed AFTER tracking
        grad_norms.
        """
        self.tracking["dtravel"].append(
            [
                el * self.opt.param_groups[0]["lr"]
                for el in self.tracking["grad_norms"][-1]
            ]
        )

    def track_d2init(self):
        """Tracks the L2 distance of the current parameters to their init."""
        self.tracking["d2init"].append(
            [
                (init - p).norm(2).item()
                for init, p in zip(self.p_init, self.get_parameters())
            ]
        )

    def track_alpha(self):
        """Tracks the relative step size."""
        # We need to find the size of the step taken,
        # since dtravel can be a list, we need to aggregate it
        if type(self.tracking["dtravel"][-1]) is list:
            t = math.sqrt(sum(t * t for t in self.tracking["dtravel"][-1]))
        else:
            t = self.tracking["dtravel"][-1]
        mu = _fit_quadratic(
            t,
            [self.tracking["f0"][-1], self.tracking["f1"][-1]],
            [sum(self.tracking["df0"][-1]), sum(self.tracking["df1"][-1])],
            [self.tracking["var_f0"][-1], self.tracking["var_f1"][-1]],
            [sum(self.tracking["var_df0"][-1]), sum(self.tracking["var_df1"][-1]),],
        )

        # Get the relative (or local) step size
        self.tracking["alpha"].append(_get_alpha(mu, t))

    def track_norm_test(self):
        """Track the norm test.

        The norm test estimates the variance of the residual length between
        the mini-batch and expected risk gradient. See Bollapragada (2017):
        'Adaptive Sampling Strategies for Stochastic Optimization'
        """

        def compute_norm_test(params):
            """
            Args:
               params (list(torch.Tensor)):
                    Parameters for computing the norm test
            """
            B = p.batch_l2.size(0)
            batch_l2_norm = sum(p.batch_l2.sum().item() for p in params)
            grad_l2_norm = sum(p.grad.data.norm(2).item() ** 2 for p in params)

            norm_test = 1.0 / (B - 1.0) * (B * batch_l2_norm / grad_l2_norm - 1)
            return norm_test

        # layer-wise
        self.tracking["norm_test_layers"].append(
            [compute_norm_test([p]) for p in self.get_parameters()]
        )

        # full network
        self.tracking["norm_test_layers_network"].append(
            compute_norm_test(list(self.get_parameters(0)))
        )

    def track_inner_product_test(self):
        """Track the inner product test.

        The inner product test estimates the variance of the mini-batch
        gradient projection on the expected risk gradient. See Bollapragada
        (2017): 'Adaptive Sampling Strategies for Stochastic Optimization'
        """
        def compute_inner_product_test(params):
            """
            Args:
               params (list(torch.Tensor)):
                    Parameters for computing the inner product test
            """
            inner_product_test = 0
            warnings.warn("Inner product test not implemented. Return dummy.")
            return inner_product_test

        # layer-wise
        self.tracking["inner_product_test_layers"].append(
            [compute_inner_product_test([p]) for p in self.get_parameters()]
        )

        # full network
        self.tracking["inner_product_test_layers_network"].append(
            compute_inner_product_test(list(self.get_parameters(0)))
        )


    def track_orthogonality_test(self):
        """Track the orthogonality test.

        The orthogonality test estimates the residual's variance between the
        mini-batch gradient projection on the expected risk gradient. See
        Bollapragada (2017): 'Adaptive Sampling Strategies for Stochastic
        Optimization'
        """
        def compute_orthogonality_test(params):
            """
            Args:
               params (list(torch.Tensor)):
                    Parameters for computing the orthogonality test
            """
            orthogonality_test = 0
            warnings.warn("Orthogonality test not implemented. Return dummy.")
            return orthogonality_test

        # layer-wise
        self.tracking["orthogonality_test_layers"].append(
            [compute_orthogonality_test([p]) for p in self.get_parameters()]
        )

        # full network
        self.tracking["orthogonality_test_layers_network"].append(
            compute_orthogonality_test(list(self.get_parameters(0)))
        )

    def track_acute_angle_test(self):
        """Track the acute angle test.

        The orthogonality test estimates the sin(angle)'s variance between the
        mini-batch gradient and the expected risk gradient. See Bahamou (2019):
        'A Dynamic Sampling Adaptive-SGD Method for Machine Learning'
        """
        def compute_acute_angle_test(params):
            """
            Args:
               params (list(torch.Tensor)):
                    Parameters for computing the acute angle test
            """
            acute_angle_test = 0
            warnings.warn("Acute angle test not implemented. Return dummy.")
            return acute_angle_test

        # layer-wise
        self.tracking["acute_angle_test_layers"].append(
            [compute_acute_angle_test([p]) for p in self.get_parameters()]
        )

        # full network
        self.tracking["acute_angle_test_layers_network"].append(
            compute_acute_angle_test(list(self.get_parameters(0)))
        )


    def write(self):
        """ Write all tracking data into a JSON file."""
        tracking = {
            "iter_tracking": self.tracking,
            "epoch_tracking": self.tracking_epoch,
        }

        with open(self.log_path + ".json", "w") as json_file:
            json.dump(tracking, json_file, indent=4)
        print("Cockpit-Log written...")

    def _check_optimizer(self):
        if self.opt.__class__.__name__ == "SGD":
            if self.opt.param_groups[0]["momentum"] != 0:
                warnings.warn(
                    "Warning: You are using SGD with momentum. Computation of "
                    "parameter update magnitude and search direction is "
                    "probably incorrect!",
                    stacklevel=2,
                )

        else:
            warnings.warn(
                "Warning: You are using an optimizer, with an unknown parameter "
                "update. Computation of parameter update magnitude and search "
                "direction is probably incorrect!",
                stacklevel=2,
            )

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
