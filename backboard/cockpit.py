"""Cockpit."""

import contextlib
import inspect
import json
import os
from collections import defaultdict

from backboard import quantities
from backboard.cockpit_plotter import CockpitPlotter
from backboard.quantities.utils_quantities import _update_dicts
from backobs import extend_with_access_unreduced_loss
from backpack import backpack
from backpack.extensions import BatchGradTransforms
from deepobs.pytorch.testproblems.testproblem import TestProblem


class Cockpit:
    """Cockpit class."""

    def __init__(self, tproblem, logpath, track_interval=1, quantities=None):
        """Initialize the Cockpit.

        Args:
            tproblem (deepobs.pytorch.testproblem): A DeepOBS testproblem.
                Alternatively, it ccould also be a general Pytorch Net.
            logpath (str): Path to the log file.
            track_interval (int, optional): Tracking rate.
                Defaults to 1 meaning every iteration is tracked.
            quantities (list, optional): List of quantities (classes or instances)
                that should be tracked. Defaults to None, which would use all
                implemented ones.
        """
        # Store all parameters as attributes
        params = locals()
        del params["self"]
        self.__dict__ = params
        self.create_graph = False
        self.output = defaultdict(dict)

        # Collect quantities
        self.quantities = self._collect_quantities(quantities, track_interval)

        # Extend testproblem
        if isinstance(tproblem, TestProblem):
            extend_with_access_unreduced_loss(tproblem)
        else:
            # TODO How do we handle general PyTorch nets?
            raise NotImplementedError

        # Prepare logpath
        self._prepare_logpath(logpath)

        # Create a Cockpit Plotter instance
        self.cockpit_plotter = CockpitPlotter(self.logpath)

    def __call__(self, global_step):
        """Returns the backpack extensions that should be used in this iteration.

        Args:
            global_step (int): Current number of iteration.

        Returns:
            backpack.backpack: BackPACK with the appropriate extensions, or the
                nullcontext
        """
        # Collect needed extensions
        ext = []
        for q in self.quantities:
            ext += q.extensions(global_step)

        ext = self._process_multiple_batch_grad_transforms(ext)

        ext = list(set(ext))

        ext = self._process_multiple_batch_grad_transforms(ext)
        ext = self._process_duplicate_extensions(ext)

        # Collect if create graph is needed and set switch
        self.create_graph = any(q.create_graph(global_step) for q in self.quantities)

        # return context_manager
        if ext:

            def context_manager():
                return backpack(*ext)

        else:
            context_manager = contextlib.nullcontext

        return context_manager()

    def track(self, global_step, batch_loss):
        """Tracking all quantities.

        Args:
            global_step (int): Current number of iteration.
            batch_loss (torch.Tensor): The batch loss of the current iteration.
        """
        params = [p for p in self.tproblem.net.parameters() if p.requires_grad]
        for q in self.quantities:
            q.compute(global_step, params, batch_loss)

    def log(
        self,
        global_step,
        epoch_count,
        train_loss,
        valid_loss,
        test_loss,
        train_accuracy,
        valid_accuracy,
        test_accuracy,
        learning_rate,
    ):
        """Tracking function for quantities computed at every epoch.

        Args:
            global_step (int): Current number of iteration/global step.
            epoch_count (int): Current number of epoch.
            train_loss (float): Loss on the train (eval) set.
            valid_loss (float): Loss on the validation set.
            test_loss (float): Loss on the test set.
            train_accuracy (float): Accuracy on the train (eval) set.
            valid_accuracy (float): Accuracy on the validation set.
            test_accuracy (float): Accuracy on the test set.
            learning_rate (float): Learning rate of the optimizer. We assume,
                that the optimizer uses a single global learning rate, which is
                used for all parameter groups.
        """
        # Loop over inputs, w/o self & global_step, adding them to the output dict
        params = locals()
        del params["self"]
        del params["global_step"]

        for k, v in params.items():
            self.output[global_step][k] = v

    def plot(self, *args, **kwargs):
        """Plot the Cockpit with the current state of the log file."""
        self.cockpit_plotter.plot(*args, **kwargs)

    def write(self):
        """Write the tracked Quantities of the Cockpit to file."""
        # Update the cockpit with the outputs from the individual quantities
        for q in self.quantities:
            _update_dicts(self.output, q.output)

        # Dump to file
        with open(self.logpath + ".json", "w") as json_file:
            json.dump(self.output, json_file, indent=4, sort_keys=True)
        print("Cockpit-Log written...")

    @staticmethod
    def _prepare_logpath(logpath):
        """Prepare the logpath by creating it if necessary.

        Args:
            logpath (str): The path where the logs should be stored
        """
        logdir, logfile = os.path.split(logpath)
        os.makedirs(logdir, exist_ok=True)

    @staticmethod
    def _collect_quantities(cockpit_quantities, track_interval):
        """Collect all quantities that should be used for tracking.

        Args:
            quantities (list, None) A list of quantities (classes or instances)
                that should be tracked. Can also be None, in this case all
                implemented quantities are being returned.
            track_interval (int, optional): Tracking rate.

        Returns:
            list: List of quantities (classes) that should be used for tracking.
        """
        if cockpit_quantities is None:
            cockpit_quantities = [
                # quantities.AlphaExpensive,
                quantities.AlphaOptimized,
                quantities.Distance,
                quantities.GradNorm,
                quantities.InnerProductTest,
                quantities.Loss,
                quantities.MaxEV,
                quantities.MeanGSNR,
                quantities.NormTest,
                quantities.OrthogonalityTest,
                quantities.TICDiag,
                # quantities.TICTrace,
                quantities.Trace,
            ]

        quants = []
        for q in cockpit_quantities:
            if inspect.isclass(q):
                quants.append(q(track_interval))
            else:
                quants.append(q)

        return quants

    def _process_duplicate_extensions(self, ext):
        """Remove duplicate BackPACK extensions.

        TODO Once we notice two instances of the same extensions, we just remove
        the latter one. This could be problematic if those extensions use different
        inputs (e.g. number of samples for MC extensions).

        Args:
            ext ([backpack.extensions]): A list of BackPACK extensions,
                potentially containing duplicates.

        Returns:
            [backpack.extensions]: A list of unique BackPACK extensions
        """
        ext_dict = dict()
        no_duplicate_ext = []
        for e in ext:
            if type(e) in ext_dict:
                pass
            else:
                no_duplicate_ext.append(e)
                ext_dict[type(e)] = True

        return no_duplicate_ext

    def _process_multiple_batch_grad_transforms(self, ext):
        """Handle multiple occurrences of ``BatchGradTransforms`` by combining them."""
        transforms = [e for e in ext if isinstance(e, BatchGradTransforms)]
        no_transforms = [e for e in ext if not isinstance(e, BatchGradTransforms)]

        batch_grad_transforms = self._merge_batch_grad_transforms(transforms)

        return no_transforms + [batch_grad_transforms]

    @staticmethod
    def _merge_batch_grad_transforms(batch_grad_transforms):
        """Merge multiple ``BatchGradTransform``s into a single one."""
        transforms = [t.get_transforms() for t in batch_grad_transforms]

        # Check for no duplicates. In principle this may be okay if same keys really
        # computed the same quantity. For now, simply avoid that.
        keys = []
        for t in transforms:
            keys += list(t.keys())
        if not len(keys) == len(set(keys)):
            raise ValueError(f"Found non-unique transforms: {keys}")

        combined_transforms = {}
        for t in transforms:
            combined_transforms.update(t)

        return BatchGradTransforms(combined_transforms)
