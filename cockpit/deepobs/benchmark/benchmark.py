import os
import warnings
from collections import defaultdict

import pandas
from torch.optim import SGD

from cockpit.deepobs.benchmark.utils import get_train_size
from cockpit.deepobs.scheduled_runner import _ScheduleCockpitRunner
from cockpit.plotter import CockpitPlotter
from cockpit.quantities import Time
from deepobs.pytorch.config import set_default_device


def _check_timer(quantities, steps):
    """Run checks to make sure the benchmark has access to time information."""
    timers = [q for q in quantities if isinstance(q, Time)]
    num_timers = len(timers)

    if num_timers != 1:
        raise ValueError(f"Got {num_timers} Time quantities. Expect 1.")

    timer = timers[0]
    for step in [0, steps]:
        if not timer.should_compute(step):
            raise ValueError(f"Time quantity must track at step {step}")


def _check_quantities(quantities):
    """Run checks on quantities to make sure the benchmark runs."""
    if quantities is None:
        raise ValueError("Expect list of quantities but got None")

    for q in quantities:
        if not q.output == defaultdict(dict):
            raise ValueError(
                f"Quantity {q} has already been used or not been initialized"
            )


def _runner_stop_after(step=None):
    class BenchmarkRunner(_ScheduleCockpitRunner):
        """Runner with disabled computation DeepOBS' additional metrics."""

        STOP_AFTER = step

        def _maybe_stop_iteration(self, global_step, batch_count):
            """Don't interrupt."""
            if self.STOP_AFTER is None:
                return

            if getattr(self, "_already_stopped", False):
                raise RuntimeError("Runner must stop at last epoch.")

            if global_step > self.STOP_AFTER:
                warnings.warn(f"BenchmarkRunner stopping epoch at step {global_step}")
                self._already_stopped = True
                raise StopIteration

        def _should_eval(self):
            """Disable DeepOBS' evaluation of test/train/valid losses and accuracies."""
            return False

    return BenchmarkRunner


def _make_runner(quantities, steps):
    """Return a DeepOBS runner that tracks the specified quantities."""
    optimizer_class = SGD
    hyperparams = {
        "lr": {"type": float, "default": 0.001},
        "momentum": {"type": float, "default": 0.0},
        "nesterov": {"type": bool, "default": False},
    }

    def plot_schedule(global_step):
        """Never plot."""
        return False

    runner_cls = _runner_stop_after(step=steps)

    return runner_cls(
        optimizer_class,
        hyperparams,
        quantities=quantities,
        plot=False,
        plot_schedule=plot_schedule,
    )


def _get_num_epochs(runner, testproblem, steps):
    """Convert maximum number of steps into number of epochs."""
    batch_size = runner._use_default(testproblem, "batch_size")
    train_size = get_train_size(testproblem)
    steps_per_epoch, _ = divmod(train_size, batch_size)

    num_epochs, rest = divmod(steps, steps_per_epoch)
    if rest > 0:
        num_epochs += 1

    return num_epochs


def constant_lr_schedule(num_epochs):
    """Constant learning rate schedule."""
    return lambda epoch: 1.0


def _read_tracking_data(runner):
    """Return the tracked data from a completed run of a runner.

    Abuses the CockpitPlotter to read data
    """
    plotter = CockpitPlotter()
    plotter._read_tracking_results(runner._get_cockpit_logpath() + ".json")
    return plotter.tracking_data


def run_benchmark(testproblem, quantities, steps, random_seed):
    """Return average time per iteration.

    Args:
        testproblem (str): Label of a DeepOBS problem.
        quantities ([Quantity]): List of quantities used in the cockpit.
        steps (int): Maximum number of iterations used for average
            time estimation.
        random_seed (int): Random seed used at initialization.
    """
    _check_timer(quantities, steps)
    _check_quantities(quantities)

    runner = _make_runner(quantities, steps)
    num_epochs = _get_num_epochs(runner, testproblem, steps)

    runner.run(
        testproblem=testproblem,
        num_epochs=num_epochs,
        l2_reg=0.0,  # necessary for backobs!
        lr_schedule=constant_lr_schedule,
        random_seed=random_seed,
        track_interval=float("nan"),  # irrelevant
        # turn plotting off, everything below is irrelevant
        plot_interval=float("nan"),
        show_plots=False,
        save_plots=False,
        save_final_plot=False,
        save_animation=False,
    )

    data = _read_tracking_data(runner)

    return extract_average_time(data, steps)


def extract_average_time(data, steps):
    """Extract average run time per iteration from tracked data."""
    data = data[["iteration", "time"]].dropna()
    data = data.loc[data["iteration"].isin([0, steps])]

    iterations = data["iteration"].to_list()
    values = data["time"].to_list()

    assert iterations == [0, steps]
    assert len(values) == 2

    return (values[1] - values[0]) / (iterations[1] - iterations[0])


def _compute_steps(steps, track_events, track_interval):
    """
    Compute steps and check if large enough to allow at least ``track_events`` events.
    """
    if steps is None:
        return track_events * track_interval
    else:
        num_events = steps // track_interval
        if num_events < track_events:
            raise ValueError(
                f"steps is too small! Want {num_events}>={track_events} track events."
            )
        return steps


def benchmark(  # noqa: C901
    testproblems,
    configs,
    track_intervals,
    num_seeds,
    devices,
    steps=None,
    track_events=20,
    savefile=None,
    header=None,
):
    """Benchmark the cockpit.

    Maximum number of steps can be specified by ``steps``. If ``None``, it will
    be computed from from ``track_events`` and the current ``track_interval``.
    """
    columns = [
        "testproblem",
        "quantities",
        "track_interval",
        "steps",
        "random_seed",
        "device",
        "time_per_step",
    ]

    if savefile is not None and os.path.exists(savefile):
        # try loading
        data = pandas.read_csv(savefile, comment="#", index_col=[0])
        assert set(data.columns) == set(columns), (
            f"Loaded file {savefile} has inconsistent columns:"
            + f"\n\tFound:   {list(data.columns)}"
            + f"\n\tRequire: {columns}"
        )
    else:
        data = pandas.DataFrame(columns=columns)

    for device in devices:
        set_default_device(device)

        for testproblem in testproblems:
            for name, config in configs.items():
                for track_interval in track_intervals:
                    for random_seed in range(num_seeds):

                        this_steps = _compute_steps(steps, track_events, track_interval)

                        run_data = {
                            "testproblem": testproblem,
                            "quantities": name,
                            "track_interval": track_interval,
                            "steps": this_steps,
                            "random_seed": random_seed,
                            "device": device,
                        }

                        run_exists = (
                            (data["testproblem"] == testproblem)
                            & (data["quantities"] == name)
                            & (data["steps"] == this_steps)
                            & (data["track_interval"] == track_interval)
                            & (data["random_seed"] == random_seed)
                            & (data["device"] == device)
                        ).any()

                        if run_exists:
                            print(f"Setting already exists and is skipped: {run_data}")
                        else:
                            print(f"Running setting: {run_data}")

                            def track_schedule(global_step):
                                return (
                                    global_step >= 0
                                    and global_step % track_interval == 0
                                )

                            quantities = [
                                q(
                                    track_schedule=track_schedule,
                                    # verbose=True,
                                )
                                for q in config
                            ]
                            runtime = run_benchmark(
                                testproblem, quantities, this_steps, random_seed
                            )
                            run_data["time_per_step"] = runtime

                            data = data.append(run_data, ignore_index=True)

                            if savefile is not None:
                                with open(savefile, "w") as f:
                                    if header is not None:
                                        header_comment = "\n".join(
                                            "# " + line for line in header.splitlines()
                                        )
                                        f.write(header_comment + "\n")
                                    data.to_csv(f)

    return data
