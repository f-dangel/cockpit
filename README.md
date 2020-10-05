# Project BackBoard - A Cockpit for Deep Learning

Idea: Use BackPACK to monitor quantities during training to tell the "pilot"
(person sitting in front of the computer performing neural network training)
what the "cockpit status of the airplane" is like.

---

## Table of Contents

- [Project BackBoard - A Cockpit for Deep Learning](#project-backboard---a-cockpit-for-deep-learning)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
    - [Install Pre-Commit Hooks](#install-pre-commit-hooks)
  - [Usage](#usage)
    - [Runner](#runner)
    - [Cockpit Plotter](#cockpit-plotter)
  - [Customizing the Cockpit](#customizing-the-cockpit)
    - [Removing Existing Quantities](#removing-existing-quantities)
    - [Changing Instruments](#changing-instruments)
    - [Adding a Novel Quantity](#adding-a-novel-quantity)
    - [Adding a Novel Instrument](#adding-a-novel-instrument)
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
track quantities to a log file.

The output of the `runner` is among other things a `__log.json` log file that
can be read by the `CockpitPlotter` to show the cockpit.

The `runners` in this repo can be used like the
[runners from DeepOBS](https://deepobs.readthedocs.io/en/v1.2.0-beta0_a/api/pytorch/runner.html).

For example to train any DeepOBS problem using SGD with CockpitTracking:

```python
"""Run SGD on the Quadratic with multiple LRs."""

from torch.optim import SGD
from backboard.runners.scheduled_runner import ScheduleCockpitRunner

optimizer_class = SGD
hyperparams = {"lr": {"type": float}}

def lr_schedule(num_epochs):
    """Some Learning rate schedule."""
    return lambda epoch: 0.95 ** epoch

runner = ScheduleCockpitRunner(optimizer_class, hyperparams)
runner.run(
    # one can fix training parameters here, otherwise they have to be passed via CLI
    testproblem="quadratic_deep",
    track_interval=1,
    plot_interval=10,
    show_plots=False,
    save_plots=False,
    save_final_plot=True,
    save_animation=False,
    lr_schedule=lr_schedule,
)
```

You can run `python exp/00_Quadratic_Example.py` for an example of this.

### Cockpit Plotter

The cockpit plotter illustrates the cockpit from a given log (tracking) file.

```python
from backboard import CockpitPlotter

log_file = "/log" # Path to the json log file (without the .json!)

cockpit_plotter = CockpitPlotter(log_file)
cockpit_plotter.plot()
```

This will result in a plot like this. The exact look of the cockpit
(e.g. which instruments are shown) is defined in the cockpit plotter and can
vary from version to version.

![Cockpit](docs/sample_cockpit.png)

And you can even create animated versions showing the Cockpit view over the course of training

![CockpitAnimation](docs/cockpit_animation.gif)

## Customizing the Cockpit

You can customize the Cockpit in various ways, e.g. by deciding which quantities should be tracked (and how often) or which instruments are shown. You could also add new quantities or instruments if you want.

### Removing Existing Quantities

When initializing the Cockpit, you can pass a list of quantities. These will decide which quantities will be tracked during training (and how often). By default, the Cockpit will use all available quantities (excluding some redundant ones such as the two `TIC` variants).

If you want to use a Cockpit with a custom list of quantities, just pass this to the Cockpit when initializing it. This can, for example, be helpful when debugging a single quantity or tracking multiple quantities is too expensive.

If you want to exclude a quantity by default, you can do so in the `_collect_quantities` method of the Cockpit.

### Changing Instruments

The CockpitPlotter uses a default collection and positioning of instruments. Currently, it is not possible to change this behaviour manually without changing the code.

The choice of instruments is defined in the `plot` method of the CockpitPlotter.

### Adding a Novel Quantity

In order to track an additional quantity there are three steps you need to take:

1. Create a subclass of [`Quantity`](backboard/quantities/quantity.py). This crucially includes the information how to [compute this quantity](https://github.com/f-dangel/backboard/blob/bc8be0592bfc17cf714af8d661d9105fd6c1242a/backboard/quantities/quantity.py#L55), which [`BackPACK` extensions are needed](https://github.com/f-dangel/backboard/blob/bc8be0592bfc17cf714af8d661d9105fd6c1242a/backboard/quantities/quantity.py#L44), and whether [access to the forward pass' computation graph is needed](https://github.com/f-dangel/backboard/blob/bc8be0592bfc17cf714af8d661d9105fd6c1242a/backboard/quantities/quantity.py#L32). Note that all these functions should be defined with respect to the `track_interval`, i.e. that the compute function only computes the quantity when we hit the tracking rate.
2. Add your class to the [`__init__.py` of the quantities](backboard/quantities/\_\_init\_\_.py).
3. Add it to the [*default* quantites](https://github.com/f-dangel/backboard/blob/bc8be0592bfc17cf714af8d661d9105fd6c1242a/backboard/cockpit.py#L195) tracked by the Cockpit.

### Adding a Novel Instrument

If you want to create a novel instrument (using tracked quantities), here are the steps you need to take:

1. Create the plotting function for this instrument in [instruments](backboard/instruments). The instrument will be plotted in a single `gridspec` element.
2. Add your class to the [`__init__.py` of the instruments](backboard/instruments/\_\_init\_\_.py).
3. Add it to the [instruments in the plot method of the CockpitPlotter](https://github.com/f-dangel/backboard/blob/bc8be0592bfc17cf714af8d661d9105fd6c1242a/backboard/cockpit_plotter.py#L31)

## API Reference

This is a rough sketch of what the API of the individual parts of the Cockpit look like and how they interact.
![API Sketch](docs/cockpit_package_structure.png)

## License

[MIT](https://opensource.org/licenses/MIT)
