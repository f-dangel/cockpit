"""Test for plotting bug (https://github.com/f-dangel/backboard/issues/121)."""

import pytest
from torch.optim import SGD

from backboard.cockpit import configured_quantities
from backboard.runners.scheduled_runner import ScheduleCockpitRunner
from tests.utils import hotfix_deepobs_argparse

optimizer_class = SGD
hyperparams = {
    "lr": {"type": float, "default": 0.01},
    "momentum": {"type": float, "default": 0.0},
    "nesterov": {"type": bool, "default": False},
}


def lr_schedule(num_epochs):
    return lambda epoch: 1.0


def track_schedule(global_step):
    steps = [0, 1]
    return global_step in steps


def plot_schedule(global_step):
    return track_schedule(global_step)


def test_bug_121():
    hotfix_deepobs_argparse()

    quants = [
        q(track_schedule=track_schedule, verbose=True)
        for q in configured_quantities("full")
    ]

    runner = ScheduleCockpitRunner(
        optimizer_class, hyperparams, quantities=quants, plot_schedule=plot_schedule
    )
    runner.run(
        testproblem="quadratic_deep",
        num_epochs=1,
        l2_reg=0.0,
        track_interval=1,
        show_plots=True,
        save_plots=True,
        plot_interval=1,
        save_final_plot=False,
        save_animation=False,
        lr_schedule=lr_schedule,
        plot_schedule=plot_schedule,
    )
