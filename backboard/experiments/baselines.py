"""Run DeepOBS baselines.

We have to re-run the baselines as some original testproblems use regularization.

The original baselines are hosted here:
    - https://github.com/fsschneider/DeepOBS_Baselines
"""

import numpy
from torch.optim import SGD, Adam

from deepobs.pytorch.runners import StandardRunner
from deepobs.tuner import GridSearch

# taken from: https://pytorch.org/docs/stable/optim.html#torch.optim.SGD
hyperparams_SGD = {
    "lr": {"type": float},
    "momentum": {"type": float, "default": 0.0},
    "dampening": {"type": float, "default": 0.0},
    "weight_decay": {"type": float, "default": 0.0},
    "nesterov": {"type": bool, "default": False},
}


def run_sgd_baseline(problem, output_dir):
    """Run DeepOBS grid search for SGD. Disable ℓ₂ regularization."""
    optimizer_class = SGD
    hyperparams = hyperparams_SGD
    num_lr = 36
    grid = {"lr": numpy.logspace(-5, 2, num_lr)}
    runner = StandardRunner

    tuner = GridSearch(
        optimizer_class, hyperparams, grid, runner=runner, ressources=num_lr
    )
    tuner.tune(
        problem,
        rerun_best_setting=True,
        skip_if_exists=True,
        output_dir=output_dir,
        l2_reg=0.0,
    )


# taken from: https://pytorch.org/docs/stable/optim.html#torch.optim.Adam
hyperparams_Adam = {
    "lr": {"type": float, "default": 0.001},
    "betas": {"type": float, "default": (0.9, 0.999)},
    "eps": {"type": float, "default": 1e-8},
    "weight_decay": {"type": float, "default": 0.0},
    "amsgrad": {"type": bool, "default": False},
}


def run_adam_baseline(problem, output_dir):
    """Run DeepOBS grid search for Adam. Disable ℓ₂ regularization."""
    optimizer_class = Adam
    hyperparams = hyperparams_Adam
    num_lr = 36
    grid = {"lr": numpy.logspace(-5, 2, num_lr)}
    runner = StandardRunner

    # grid search
    tuner = GridSearch(
        optimizer_class, hyperparams, grid, runner=runner, ressources=num_lr
    )
    tuner.tune(
        problem,
        rerun_best_setting=True,
        skip_if_exists=True,
        output_dir=output_dir,
        l2_reg=0.0,
    )
