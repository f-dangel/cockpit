# Project BackBoard

Idea: Use BackPACK to monitor quantities during training to tell the "pilot"
(person sitting in front of the computer performing neural network training)
what the "cockpit status of the airplane" is like.

## Table of Contents

- [Project BackBoard](#project-backboard)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
    - [Install Pre-Commit Hooks](#install-pre-commit-hooks)
  - [Usage](#usage)
    - [Runner](#runner)
    - [Cockpit Plotter](#cockpit-plotter)
    - [Airline Plotter](#airline-plotter)
  - [Customize Backboard](#customize-backboard)
    - [Track an Additional Quantity](#track-an-additional-quantity)
    - [Add an Instrument](#add-an-instrument)
    - [Customize the Cockpit (Change the Shown Instruments)](#customize-the-cockpit-change-the-shown-instruments)
  - [API Reference](#api-reference)
  - [License](#license)

## Installation

Set up a `conda` environment named `backboard`
  
```bash
conda env create -f .conda_env.yml
```

and activate it by running
  
```bash
conda activate backboard
```

### Install Pre-Commit Hooks

```bash
pre-commit install
```

## Usage

### Runner

The `runners` can be used to train on a DeepOBS test problem and simultaneously
track quantities to a log file. Depending on the `runner`, these log files can
be shown using the `CockpitPlotter` while training.

The output of the `runner` is among other things a `__log.json` log file that
can be read by the `CockpitPlotter` or `AirlinePlotter` to show the cockpit.

The `runners` in this repo can be used like the
[runner from DeepOBS](https://deepobs.readthedocs.io/en/v1.2.0-beta0_a/api/pytorch/runner.html).

For example to train any DeepOBS problem using SGD with CockpitTracking:

```python
from torch.optim import SGD
from backboard.runners.schedule_runner import ScheduleCockpitRunner

optimizer_class = SGD
hyperparams = {"lr": {"type": float}}

def lr_schedule():
    """Some lambda function that defines a lr_schedule"""
    pass

runner = ScheduleCockpitRunner(optimzier_class, hyperparams)

runner.run(
    # one can fix training parameters here, otherwise they have to be passed
    # via CLI
    lr_schedule=lr_schedule
)
```

### Cockpit Plotter

The cockpit plotter illustrates the cockpit from a given log (tracking) file.

```python
from backboard.cockpit_plotter import CockpitPlotter

log_file = "/log.json" # Path to the json log file

cockpit_plotter = CockpitPlotter(log_file)
cockpit_plotter.plot()
```

[TODO CHECK CODE]

This will result in a plot like this. The exact look of the cockpit
(e.g. which instruments are shown) is defined in the cockpit plotter and can
vary from version to version.

![Cockpit](docs/sample_cockpit.png)

### Airline Plotter

The airline plotter compares the cockpits of multiple runs with each other.
It takes a list of log files and creates an overview.

[TODO PREVIEW]

[TODO CODE FOR AIRLINE PLOTTER]

## Customize Backboard

[TODO]

### Track an Additional Quantity

In order to track an additional quantity, you need to tell the `CockpitTracker`
do track it. You can do so, by adding it to the list of `per_iter_quants`
(in the `__init__` method of the `CockpitTracker`).

Next, you have to decide whether you quantity should be tracked before or after
computing the forward-backward pass of the current iteration.
In general, it is advised to track it before, to stay consistent with other
quantites. However, for example to compute the local effective step size (`alpha`),
we need information from both before and after the iteration.

Either add your function to the `track_before` or `track_after` method of the
`CockpitTracker` class, e.g. by adding `tracking.track_my_quant(self)`.

The actual computation is done in the function that you define in
[`tracking.py`](backboard/tracking/tracking.py), where you would need to add
the definition of your tracking function. The result should not be returned but
appended to `self.iter_tracking[my_quant]`. Additional helper functions can be outsourced to [`utils_tracking.py`](backboard/tracking/utils_tracking.py) or separate utils files.

### Add an Instrument

[TODO]

### Customize the Cockpit (Change the Shown Instruments)

[TODO]

## API Reference

This is a rough sketch of what the API of the cockpit part looks like:
![API Sketch](docs/cockpit_api.png)

## License

[MIT](https://opensource.org/licenses/MIT)
