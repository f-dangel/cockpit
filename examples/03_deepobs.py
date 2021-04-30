"""An example of using Cockpit with DeepOBS."""

from _utils_deepobs import DeepOBSRunner
from _utils_examples import get_logpath
from torch.optim import SGD

from cockpit.utils import configuration, schedules

optimizer = SGD
hyperparams = {"lr": {"type": float, "default": 0.001}}

track_schedule = schedules.linear(10)
plot_schedule = schedules.linear(20)
quantities = configuration.configuration("full", track_schedule=track_schedule)

runner = DeepOBSRunner(optimizer, hyperparams, quantities, plot_schedule=plot_schedule)


def const_schedule(num_epochs):
    """Constant learning rate schedule."""
    return lambda epoch: 1.0


runner.run(
    testproblem="quadratic_deep",
    output_dir=get_logpath(),
    l2_reg=0.0,
    num_epochs=50,
    show_plots=True,
    save_plots=False,
    save_final_plot=True,
    save_animation=False,
    lr_schedule=const_schedule,
)
