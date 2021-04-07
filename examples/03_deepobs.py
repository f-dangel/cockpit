"""An example of using Cockpit with DeepOBS."""

from torch.optim import SGD
from utils.deepobs_runner import DeepOBSRunner

from cockpit.utils import configuration, schedules

optimizer = SGD
hyperparams = {"lr": {"type": float, "default": 0.001}}

schedule = schedules.linear(20)
quantities = configuration.configuration("full", track_schedule=schedule)

runner = DeepOBSRunner(
    optimizer, hyperparams, quantities=quantities, plot_schedule=schedule
)


def const_schedule(num_epochs):
    """Constant learning rate schedule."""
    return lambda epoch: 1.0


runner.run(
    testproblem="quadratic_deep",
    l2_reg=0.0,  # necessary for backobs!
    num_epochs=15,
    show_plots=True,
    save_plots=False,
    save_final_plot=False,
    save_animation=False,
    lr_schedule=const_schedule,
)
