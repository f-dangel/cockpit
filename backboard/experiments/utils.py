"""Utility functions for the paper experiments."""

import backobs
import deepobs


def register(cls, has_accuracy=True):
    """Register a new testproblem class in DeepOBS and BackOBS.

    It is assumed that the testproblem is supported by BackOBS.
    """
    dataset_name, net_name = cls.__name__.split("_")

    # hotfix
    dataset_name = dataset_name.replace("cifar10", "CIFAR-10")

    # DeepOBS
    setattr(deepobs.pytorch.testproblems, cls.__name__, cls)

    # for CockpitPlotter
    if dataset_name in deepobs.config.DATA_SET_NAMING.keys():
        if not deepobs.config.DATA_SET_NAMING[dataset_name] == dataset_name:
            raise ValueError(
                f"{deepobs.config.DATA_SET_NAMING[dataset_name]} != {dataset_name}"
            )
    else:
        deepobs.config.DATA_SET_NAMING[dataset_name] = dataset_name

    if net_name in deepobs.config.TP_NAMING.keys():
        assert deepobs.config.TP_NAMING[net_name] == net_name

    else:
        deepobs.config.TP_NAMING[net_name] = net_name

    # BackOBS
    backobs.utils.ALL += (cls,)
    backobs.utils.SUPPORTED += (cls,)
    backobs.integration.SUPPORTED += (cls,)
    if not has_accuracy:
        raise NotImplementedError()


def replace(module, trigger, make_new):
    """
    Check if layer should be replaced by calling trigger(m) → bool.
    If True, replace m with make_new(m)
    """

    def has_children(module):
        return bool(list(module.children()))

    for name, mod in module.named_children():
        if has_children(mod):
            replace(mod, trigger, make_new)
        else:
            if trigger(mod):
                new_mod = make_new(mod)
                setattr(module, name, new_mod)
