"""Run SGD on the Quadratic Test Problem of DeepOBS."""

from torch.optim import SGD

from cockpit.runners.scheduled_runner import ScheduleCockpitRunner

# In this example, we use SGD. Replace this with your custom optimizer to tes
optimizer_class = SGD
# Define the hyperparameters of the optimizer and if applicable its default values.
hyperparams = {
    "lr": {"type": float, "default": 0.001},
    "momentum": {"type": float, "default": 0.0},
    "nesterov": {"type": bool, "default": False},
}


# You can define a learning rate schedule. Currently, this is set to a constant.
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


# This Runner is included in Cockpit and works like other DeepOBS runners, see
# https://deepobs.readthedocs.io/en/v1.2.0-beta0_a/api/pytorch/runner.html
runner = ScheduleCockpitRunner(optimizer_class, hyperparams)

# Start the training.
# We can fix several training parameters in this method, such as the testproblem
# Non-fixed parameters (in this case, for example, the batch size) can be passed
# via the command line.
# Some parameters (like the testproblem) are required (either by setting it here,
# or by passing it via the command line).
runner.run(
    testproblem="quadratic_deep",
    l2_reg=0.0,  # necessary for backobs!
    track_interval=1,
    plot_interval=10,
    show_plots=True,
    save_plots=False,
    save_final_plot=True,
    save_animation=False,
    lr_schedule=lr_schedule,
)
