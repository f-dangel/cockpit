"""Cockpit."""

import os
from collections import defaultdict

import json_tricks
from backpack import disable
from backpack.extensions.backprop_extension import BackpropExtension

from cockpit import quantities
from cockpit.context import BackwardCTX, get_loss
from cockpit.quantities.quantity import Quantity
from cockpit.quantities.utils_transforms import BatchGradTransformsHook


class Cockpit:
    """Cockpit class."""

    BACKPACK_CONV_SAVE_MEMORY = True
    """bool: Tell BackPACK to use a more memory-efficient
            Jacobian-vector product algorithm for weights in convolution layers.
            Default: ``True``.
    """

    def __init__(self, params, quantities=None):
        """Initialize a cockpit.

        Args:
            params (iterable): List or sequence of parameters on which the quantities
                will be evaluated. Every parameter must have ``require_grad = True``,
                otherwise the computation cannot be executed.
            quantities (list, optional): List of ``Quantity`` (instances) that will
                be tracked. Defaults to None, which will use no quantities.

        Raises:
            ValueError: If not all passed parameters have ``required_grad=True``.
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
            quantity (cockpit.quantites.Quantity): The quantity to be added.

        Raises:
            ValueError: If passed quantity is not a ``cockpit.quantity``.
        """
        if not isinstance(quantity, Quantity):
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
            global_step (int): Current number of iteration.
            custom_exts (list or tuple): Custom BackPACK extensions that will be
                computed on top.

        Returns:
            list: List of required BackPACK extensions for the current iteration.
        """
        # TODO A user could introduce a bug here by running an extension which is
        # also required by one quantity, but uses different hyperparameters (for
        # instance the user picks ``DiagGGNMC(mc_samples=2)`` the Hessian trace
        # quantity uses ``DiagGGNMC(mc_samples=1)``.). Catching such a corner case
        # requires hyperparameter comparison of extensions. Considering the large
        # amount of required boilerplate, this is left for the future
        ext = list(custom_exts)
        for q in self.quantities:
            ext += q.extensions(global_step)

        ext = self._process_duplicate_extensions(ext)

        return ext

    def _get_extension_hook(self, global_step):
        """Build BackPACK extension hook for the current iteration.

        Args:
            global_step (int): Current number of iteration.

        Returns:
            callable or None: BackPACK extension hook for the current iteration.
                ``None`` indicates no hook.
        """
        hooks = []

        for q in self.quantities:
            hooks += q.extension_hooks(global_step)

        # Currently expects only ``BatchGradTransformsHook``s
        # This changes with https://github.com/f-dangel/cockpit-paper/issues/142
        assert all(isinstance(h, BatchGradTransformsHook) for h in hooks)

        if len(hooks) == 0:
            hook = None
        else:
            hook = self._merge_batch_grad_transform_hooks(hooks)

        return hook

    def __call__(self, global_step, *exts, info=None, debug=False):
        """Returns the backpack extensions that should be used in this iteration.

        Args:
            global_step (int): Current number of iteration.
            *exts: Custom BackPACK extensions that will be computed on top.
            info (dict): Dictionary that specifies additional information. Some
                quantities require additional information that is overly difficult
                to infer from a backward pass, like the individual losses.
            debug (bool, optional): Enable debug mode.. Defaults to False.

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

        if "optimizer" in info:
            self._optimizer_name = type(info["optimizer"]).__name__

        return BackwardCTX(self, global_step, exts, info, debug=debug)

    def track(self, global_step, protected_savefields=()):
        """Tracking all quantities.

        Args:
            global_step (int): Current number of iteration.
            protected_savefields ([str]): List of strings containing attribute names
                of backpack extensions that will not be deleted after the backward pass

        :meta private:
        """
        batch_loss = get_loss(global_step)

        before_cleanup = [
            q for q in self.quantities if not isinstance(q, quantities.HessMaxEV)
        ]

        for q in before_cleanup:
            q.track(global_step, self.params, batch_loss)

        self._free_backpack_buffers(global_step, protected_savefields)

        after_cleanup = [
            q for q in self.quantities if isinstance(q, quantities.HessMaxEV)
        ]
        with disable():
            for q in after_cleanup:
                q.track(global_step, self.params, batch_loss)

    def _free_backpack_buffers(self, global_step, protected_savefields, verbose=False):
        """Manually free quantities computed by BackPACK to save memory.

        Args:
            global_step (int): Current number of iteration.
            protected_savefields ([str]): List of strings containing attribute
                names of backpack extensions that will not be deleted after the
                backward pass
            verbose (bool, optional): Turns on verbose mode. Defaults to ``False``.
        """
        if verbose:
            print("Freeing BackPACK buffers")

        savefields = [ext.savefield for ext in self._get_extensions(global_step)]

        # TODO Determine hook savefields through ``ParameterExtensionHook`` and trigger
        # deletion. This can only happen after hooks have been introduced for all
        # quantities, see https://github.com/f-dangel/cockpit-paper/issues/142
        savefields.append("grad_batch_transforms")

        for param in self.params:
            for field in savefields:
                try:
                    if field not in protected_savefields:
                        delattr(param, field)

                        if verbose:
                            print(
                                f"Deleting '{field}' from param of shape {param.shape}"
                            )
                except AttributeError:
                    pass

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
        """
        logpath_with_suffix = logpath + ".json"
        print(f"[cockpit] writing output to {logpath_with_suffix}")

        os.makedirs(os.path.dirname(logpath_with_suffix), exist_ok=True)

        with open(logpath_with_suffix, "w") as json_file:
            json_tricks.dump(self.get_output(), json_file, indent=4, sort_keys=True)

    def _update_output(self):
        """Fetch outputs from tracked quantities into ``self.output``."""
        # Update the cockpit with the outputs from the individual quantities
        for q in self.quantities:
            key = q.__class__.__name__

            for iteration, value in q.get_output().items():
                self.output[iteration][key] = value

    def get_output(self):
        """Return a nested dictionary that stores the results of all tracked quantities.

        First key corresponds to the iteration, second key is the quantity class name,
        values represent the computational result of the quantity at that iteration.

        Example:
            >>> cockpit = Cockpit(...)
            >>> # information tracked at iteration 3
            >>> global_step = 3
            >>> global_step_output = cockpit.get_output()[global_step]
            >>> # information tracked at iteration 3 by Hessian max eigenvalue quantity
            >>> key = "HessMaxEV"
            >>> max_ev_global_step_output = cockpit.output[global_step][key]

        Returns:
            dict: Nested dictionary with the results of all tracked quantities.
        """
        self._update_output()
        return self.output

    def _process_duplicate_extensions(self, ext):
        """Remove duplicate BackPACK extensions.

        Note:
            Two extensions are considered equal if they are of the same class.

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

    @staticmethod
    def _merge_batch_grad_transform_hooks(batch_grad_transform_hooks):
        """Merge multiple ``BatchGradTransformHook``s, removing duplicate transforms.

        Note:
            Two transformations are identical if they have the same ``id``.

        Args:
            batch_grad_transform_hooks ([BatchGradTransformsHook]): List of
                ``BatchGradTransformHook``s.

        Raises:
            ValueError: If there is a non-unique transform.

        Returns:
            BatchGradTransformsHook: Single transform that includes all transforms.
        """
        transforms = [t._transforms for t in batch_grad_transform_hooks]

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

        return BatchGradTransformsHook(combined_transforms)
