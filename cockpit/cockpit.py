"""Cockpit."""

import json
import os
from collections import defaultdict

from backpack import backpack_deactivate_io
from backpack.extensions import BatchGradTransforms
from backpack.extensions.backprop_extension import BackpropExtension
from cockpit import quantities
from cockpit.context import BackwardCTX, get_loss
from cockpit.quantities.utils_quantities import _update_dicts


class Cockpit:
    """Cockpit class."""

    def __init__(self, params, quantities=None):
        """Initialize a cockpit.

        Args:
            params (iterable): List or sequence of parameters on which the quantities
                will be evaluated. Every parameter must have ``require_grad = True``,
                otherwise the computation cannot be executed.
            quantities (list, optional): List of ``Quantity`` (instances) that will
                be tracked. Defaults to None, which will use no quantities.

        Returns:
            None
        """
        # check parameters
        self.params = list(params)
        for p in self.params:
            if not p.requires_grad:
                raise ValueError(f"Got parameter with requires_grad=False: {p}")

        # initialize output
        self.output = defaultdict(dict)

        # add quantities to cockpit
        if quantities is None:
            quantities = []

        self.quantities = []
        for q in quantities:
            self.add(q)

    def add(self, quantity):
        """Add quantity to tracked quantities.

        Args:
            quantity (Quantity): The quantity to be added.

        Returns:
            None
        """
        if not isinstance(quantity, quantities.Quantity):
            raise ValueError(
                f"Added quantities must be instances of Quantity. Got {quantity}"
            )
        else:
            self.quantities.append(quantity)

    def create_graph(self, global_step):
        """Return if computation graph should be kept for computing quantities.

        Args:
            global_step (int): Current number of iteration.

        Returns:
            bool: ``True`` if computation graph should be kept, else ``False``.
        """
        return any(q.create_graph(global_step) for q in self.quantities)

    def _get_extensions(self, global_step, custom_exts=()):
        """Collect BackPACK extensions required at current iteration.

        Args:
            custom_exts (list or tuple): Custom BackPACK extensions that will be
                computed on top.
        """
        ext = list(custom_exts)
        for q in self.quantities:
            ext += q.extensions(global_step)

        ext = self._process_multiple_batch_grad_transforms(ext)
        ext = self._process_duplicate_extensions(ext)

        return ext

    def __call__(self, global_step, *exts, info=None, debug=False):
        """Returns the backpack extensions that should be used in this iteration.

        Args:
            global_step (int): Current number of iteration.
            *exts: Custom BackPACK extensions that will be computed on top.
            info (dict): Dictionary that specifies additional information. Some
                quantities require additional information that is overly difficult
                to infer from a backward pass, like the individual losses.
            debug (bool): Enable debug mode.

        Returns:
            backpack.backpack: BackPACK with the appropriate extensions, or the
                backpack_disable_io context.
        """
        for e in exts:
            assert isinstance(
                e, BackpropExtension
            ), f"*exts must be tuple of backpack extensions. Got {e}"

        if info is None:
            info = {}

        return BackwardCTX(self, global_step, exts, info, debug=debug)

    def track(self, global_step, protected_savefields=()):
        """Tracking all quantities.

        Args:
            global_step (int): Current number of iteration.
            protected_savefields ([str]): List of strings containing attribute names
                of backpack extensions that will not be deleted after the backward pass
        """
        batch_loss = get_loss(global_step)

        before_cleanup = [
            q for q in self.quantities if not isinstance(q, quantities.MaxEV)
        ]

        for q in before_cleanup:
            q.track(global_step, self.params, batch_loss)

        self._free_backpack_buffers(global_step, protected_savefields)

        after_cleanup = [q for q in self.quantities if isinstance(q, quantities.MaxEV)]
        with backpack_deactivate_io():
            for q in after_cleanup:
                q.track(global_step, self.params, batch_loss)

    def _free_backpack_buffers(self, global_step, protected_savefields, verbose=False):
        """Manually free quantities computed by BackPACK to save memory.

        protected_savefields ([str]): List of strings containing attribute names
            of backpack extensions that will not be deleted after the backward pass
        """
        if verbose:
            print("Freeing BackPACK buffers")

        ext = self._get_extensions(global_step)

        for param in self.params:
            for e in ext:
                try:
                    field = e.savefield
                    if field not in protected_savefields:
                        delattr(param, field)

                        if verbose:
                            print(
                                f"Deleting '{field}' from param of shape {param.shape}"
                            )
                except AttributeError:
                    pass

    @staticmethod
    def _has_children(net):
        return len(list(net.children())) > 0

    @staticmethod
    def _remove_module_io(module):
        io_fields = ["input0", "output"]

        for field in io_fields:
            try:
                delattr(module, field)
            except AttributeError:
                pass

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
        # Store inputs
        self.output[global_step]["epoch_count"] = epoch_count

        self.output[global_step]["train_loss"] = train_loss
        self.output[global_step]["valid_loss"] = valid_loss
        self.output[global_step]["test_loss"] = test_loss

        self.output[global_step]["train_accuracy"] = train_accuracy
        self.output[global_step]["valid_accuracy"] = valid_accuracy
        self.output[global_step]["test_accuracy"] = test_accuracy

        self.output[global_step]["learning_rate"] = learning_rate

    def write(self, logpath):
        """Write tracked quantities to a json file.

        Args:
            logpath (str): Path to a log file without the ``.json`` suffix.

        Returns:
            None
        """
        self.update_output()

        # Dump to file
        logpath_with_suffix = logpath + ".json"
        print(f"[cockpit] writing output to {logpath_with_suffix}")

        os.makedirs(os.path.dirname(logpath_with_suffix), exist_ok=True)

        with open(logpath_with_suffix, "w") as json_file:
            json.dump(self.output, json_file, indent=4, sort_keys=True)

    def update_output(self):
        """Fetch outputs from tracked quantities into ``self.output``."""
        # Update the cockpit with the outputs from the individual quantities
        for q in self.quantities:
            _update_dicts(self.output, q.output)

    def get_output(self):
        self.update_output()
        return self.output

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

    @classmethod
    def _process_multiple_batch_grad_transforms(cls, ext):
        """Handle multiple occurrences of ``BatchGradTransforms`` by combining them."""
        transforms = [e for e in ext if isinstance(e, BatchGradTransforms)]
        no_transforms = [e for e in ext if not isinstance(e, BatchGradTransforms)]

        processed_transforms = no_transforms
        if transforms:
            processed_transforms.append(cls._merge_batch_grad_transforms(transforms))

        return processed_transforms

    @staticmethod
    def _merge_batch_grad_transforms(batch_grad_transforms):
        """Merge multiple ``BatchGradTransform``s into a single one.

        Non-uniqye transformations are identified via python's ``id`` function.
        """
        transforms = [t.get_transforms() for t in batch_grad_transforms]

        key_function_pairs = []
        for t in transforms:
            for key, value in t.items():
                key_function_pairs.append((key, value))

        unique_keys = {pair[0] for pair in key_function_pairs}
        combined_transforms = {}

        for key in unique_keys:
            functions = [pair[1] for pair in key_function_pairs if pair[0] == key]
            ids = [id(f) for f in functions]

            if len(set(ids)) != 1:
                raise ValueError(
                    f"Got non-unique transform functions with ids {ids} for key '{key}'"
                )
            else:
                combined_transforms[key] = functions[0]

        return BatchGradTransforms(combined_transforms)
