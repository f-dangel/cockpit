"""Test the tracking part by running on the quadratic problem."""

import json
import os

import pytest
from torch.optim import SGD

from backboard.runners.schedule_runner import ScheduleCockpitRunner

optimizer_class = SGD
hyperparams = {"lr": {"type": float, "default": 1e-2}}


def lr_schedule_half(num_epochs):
    """Some Learning rate schedule."""
    return lambda epoch: 0.5 ** epoch


def lr_schedule_const(num_epochs):
    """Constant Learning rate schedule."""
    return lambda epoch: 1.0


@pytest.mark.parametrize(
    "lr,num_epochs,track_interval,lr_schedule",
    [(0.01, 5, 1, lr_schedule_half), (0.7, 3, 5, lr_schedule_const)],
)
def test_tracking(lr, num_epochs, track_interval, lr_schedule):
    """Test the tracking capabilities of the Cockpit.

    Args:
        lr (float): Learning rate of the optimizer.
        num_epochs (int): Number of epochs to train the model.
        track_interval (int): How often we want to track.
        lr_schedule (lambda): A lambda function defining a learning rate schedule.

    Raises:
        ValueError: [description]
    """
    runner = ScheduleCockpitRunner(optimizer_class, hyperparams)
    runner._run(
        testproblem="quadratic_deep",
        hyperparams={"lr": lr},
        batch_size=128,
        num_epochs=num_epochs,
        random_seed=42,
        data_dir=None,
        output_dir="./tests/results",
        l2_reg=None,
        no_logs=False,
        train_log_interval=10,
        print_train_iter=False,
        tb_log=False,
        tb_log_dir=None,
        track_interval=track_interval,
        plot_interval=10,
        show_plots=False,
        save_plots=False,
        save_final_plot=False,
        track_time=False,
        lr_schedule=lr_schedule,
    )

    with open(
        os.path.join(runner._run_directory, runner._file_name + "__log.json")
    ) as json_file:
        data = json.load(json_file)

        assert isinstance(data, dict)

        # Check if all lists have the same lengths
        for t in ["iter_tracking", "epoch_tracking"]:
            lengths = [len(v) for k, v in data[t].items()]
            result = all(elem == lengths[0] for elem in lengths)
            assert result

        # Check f0 and f1 are shifted versions, if track_interval is 1
        if track_interval == 1:
            assert all(
                data["iter_tracking"]["f0"][i + 1] == data["iter_tracking"]["f1"][i]
                for i in range(len(data["iter_tracking"]["f0"]) - 1)
            )

        # Check iteration starts with 1 and is spaced according to track_interval
        assert data["iter_tracking"]["iteration"][0] == 1
        diffs = [
            j - i
            for i, j in zip(
                data["iter_tracking"]["iteration"][:-1],
                data["iter_tracking"]["iteration"][1:],
            )
        ]
        assert all(elem == track_interval for elem in diffs)

        # Check epochs starts at 0 and has all values
        assert all(
            data["epoch_tracking"]["epoch"][i] == i for i in range(0, num_epochs + 1)
        )

        # Check if learning rate is initialized and decayed correctly
        assert data["epoch_tracking"]["learning_rate"][0] == lr

        decay = [
            j / i
            for i, j in zip(
                data["epoch_tracking"]["learning_rate"][:-1],
                data["epoch_tracking"]["learning_rate"][1:],
            )
        ]

        if lr_schedule.__name__ == "lr_schedule_half":
            decay_factor = 0.5
        elif lr_schedule.__name__ == "lr_schedule_const":
            decay_factor = 1.0
        else:
            raise ValueError("Unknown learning rate schedule")
        assert all(elem == decay_factor for elem in decay)
