"""Mockup implementations of runner and cockpit required for integration tests."""

import warnings

from torch.optim import SGD

from backboard.runners.scheduled_runner import _ScheduleCockpitRunner


class TestScheduleCockpitRunner(_ScheduleCockpitRunner):
    """Schedule Runner with a learning rate schedule used to run tests.

    Note:
        Computation of DeepOBS metrics is disabled and the runner performs
        only three steps per epoch

    """

    STOP_BATCH_COUNT_PER_EPOCH = 3

    def _maybe_stop_iteration(self, global_step, batch_count):
        """Stop after three steps of an epoch."""
        if batch_count == self.STOP_BATCH_COUNT_PER_EPOCH:
            warnings.warn(
                "The test runner performs only "
                + f"{self.STOP_BATCH_COUNT_PER_EPOCH} steps per epoch."
            )
            raise StopIteration

    def _should_eval(self):
        """Disable DeepOBS' evaluation of test/train/valid losses and accuracies."""
        return False


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


def run_sgd_test_runner(
    quantities,
    testproblem,
    num_epochs=1,
    batch_size=3,
    lr=0.01,
    momentum=0.0,
    l2_reg=0.0,
):
    """Perform short debug run (three steps per epoch) with SGD."""
    optimizer_class_sgd = SGD
    hyperparams_sgd = {
        "lr": {
            "type": float,
            "default": lr,
        },
        "momentum": {
            "type": float,
            "default": momentum,
        },
    }

    def plot_schedule(global_step):
        return False

    runner = TestScheduleCockpitRunner(
        optimizer_class_sgd,
        hyperparams_sgd,
        quantities=quantities,
        plot=False,
        plot_schedule=plot_schedule,
    )

    runner.run(
        testproblem=testproblem,
        num_epochs=num_epochs,
        batch_size=batch_size,
        l2_reg=l2_reg,
        track_interval=1,
        plot_interval=1,
        show_plots=False,
        save_plots=False,
        save_final_plot=False,
        save_animation=False,
        lr_schedule=lr_schedule,
    )
