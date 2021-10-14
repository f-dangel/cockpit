"""Base classes for BackPACK's extension hooks."""

from torch.nn import Module


class ModuleExtensionHook:
    """Extension hook class to act on a module right after BackPACK's extensions.

    Note:
        An instance of this class can only be used a single time during backpropagation
        and must be re-instantiated afterwards.

    Descendants must implement the following methods:

    - ``module_hook``: The action to perform.
    """

    def module_hook(self, param, module):
        """Perform an action right after BackPACK extensions were executed on a module.

        Args:
            param (torch.Tensor): Parameter of a neural net.
            module (torch.nn.Module): Layer that hosts `param`.

        Returns: # noqa: DAR202
            Any: Arbitrary output. If not ``None``, it will be stored under
                ``self.savefield`` in ``param``.

        Raises:
            NotImplementedError: Must be implemented by descendants.
        """
        raise NotImplementedError

    def __init__(self, savefield=None):
        """Store the potential savefield and create cache for visited parameters.

        Args:
            savefield (str, optional): Attribute name under which the hook's result
                is saved in a parameter. If ``None``, it is assumed that the hook acts
                via side effects and no output needs to be stored.
        """
        self.savefield = savefield
        self.processed = set()

    def __call__(self, module):
        """Execute hook on all module parameters. Skip already processed parameters.

        Args:
            module (torch.nn.Module): Module on which the extension hook is executed.
        """
        for param in module.parameters():
            if self.should_run_hook(param, module):
                self.run_hook(param, module)

    def should_run_hook(self, param, module):
        """Check if hooks should be executed on a parameter.

        Hooks are only executed once on every trainable parameter. Skip
        ``torch.nn.Sequential``s to prevent unexpected execution order of hooks.

        Args:
            param (torch.Tensor): Parameter of a neural net.
            module (torch.nn.Module): Layer that `param` is part of.

        Returns:
            bool: Whether the hook should be executed on the parameter.
        """
        if self._has_children(module):
            return False
        else:
            return id(param) not in self.processed and param.requires_grad

    def run_hook(self, param, module):
        """Execute the hook on parameter, add it to processed items and store result.

        Args:
            param (torch.Tensor): Parameter of a neural net.
            module (torch.nn.Module): Layer that `param` is part of.
        """
        value = self.module_hook(param, module)
        self._save(value, param)
        self.processed.add(id(param))

    @staticmethod
    def _has_children(net: Module) -> bool:
        """Check if module contains other modules.

        Args:
            net: A module.

        Returns:
            If the module contains other modules
        """
        return len(list(net.children())) > 0

    def _save(self, value, param):
        """Store value in parameter's ``savefield`` argument.

        Args:
            value (Any): Value to be saved. ``None`` means nothing will be saved.
            param (torch.Tensor): Parameter of a neural net.

        Raises:
            ValueError: If no savefield was specified byt the value is not ``None``.
        """
        if self.savefield is None:
            if value is not None:
                raise ValueError(f"Hook has no savefield but wants to save {value}")
        else:
            setattr(param, self.savefield, value)


class ParameterExtensionHook(ModuleExtensionHook):
    """Extension hook class to act on a parameter right after BackPACK's extensions.

    Note:
        An instance of this class can only be used a single time during backpropagation
        and must be re-instantiated afterwards.

    Descendants must implement the following methods:

    - ``param_hook``: The action to perform.
    """

    def param_hook(self, param):
        """Act on a parameter right after BackPACK's extensions.

        Args:
            param (torch.Tensor): Trainable parameter which hosts BackPACK quantities.

        Returns: # noqa: DAR202
            Any: Arbitrary output. If not ``None``, it will be stored under
                ``self.savefield`` in ``param``.

        Raises:
            NotImplementedError: Must be implemented by descendants.
        """
        raise NotImplementedError

    def module_hook(self, param, module):
        """Execute hook on the parameter.

        Args:
            param (torch.Tensor): Parameter of a neural net.
            module (torch.nn.Module): Layer that hosts `param`.

        Returns:
            Any: Arbitrary output. If not ``None``, it will be stored under
                ``self.savefield`` in ``param``.
        """
        return self.param_hook(param)
