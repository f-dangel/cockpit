"""Example: Configure the cockpit with tracking and plotting schedules."""

from torch.optim import SGD

from cockpit.cockpit import configured_quantities
from cockpit.runners.scheduled_runner import ScheduleCockpitRunner
from cockpit.utils import fix_deepobs_data_dir

fix_deepobs_data_dir()

optimizer_class = SGD
hyperparams = {
    "lr": {"type": float, "default": 0.01},
    "momentum": {"type": float, "default": 0.0},
    "nesterov": {"type": bool, "default": False},
}


def lr_schedule(num_epochs):
    """Some Learning rate schedule.

    Example:
        >>> # Halving the learning rate every epoch:
        >>> lambda epoch: 0.5 ** epoch
        >>> # A less aggressive decay:
        >>> lambda epoch: 0.9 ** epoch
        >>> # Constant learning rate (using init lr):
        >>> lambda epoch: 1.0
    """
    return lambda epoch: 1.0


steps_per_epoch = 7


def track_schedule(global_step):
    steps = [0, 1, 4, 13, 30, 143]
    return global_step in steps


def plot_schedule(global_step):
    BUG = True
    if BUG:
        return track_schedule(global_step)
    else:
        return track_schedule(global_step) and global_step != 1


quants = [
    q(track_schedule=track_schedule, verbose=True)
    for q in configured_quantities("full")
]

runner = ScheduleCockpitRunner(
    optimizer_class, hyperparams, quantities=quants, plot_schedule=plot_schedule
)
runner.run(
    testproblem="quadratic_deep",
    l2_reg=0.0,  # necessary for backobs!
    track_interval=1,
    show_plots=True,
    save_plots=True,
    plot_interval=1,
    save_final_plot=True,
    save_animation=False,
    lr_schedule=lr_schedule,
    plot_schedule=plot_schedule,
)
