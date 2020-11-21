<!-- PROJECT SHIELDS -->
[![Python 3.6+](https://img.shields.io/badge/python-3.6+-blue.svg?style=flat-square)](https://www.python.org/downloads/release/python-350/)
[![License: MIT](https://img.shields.io/github/license/fsschneider/deepobs?style=flat-square)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square)](https://github.com/psf/black)

<!-- PROJECT LOGO -->
<br />
<p align="center">
<a href="#"><img src="docs/assets/Logo.png" alt="Logo"/></a>


  <h3 align="center">A Practical Debugging Tool for Training Deep Neural Networks</h3>

  <p align="center">
    A better status screen for deep learning.
    <br />
    <a href="https://f-dangel.github.io/cockpit-paper/"><strong>Explore the docs »</strong></a>
    <br />
  </p>
</p>

<p align="center">
  <a href="#about-the-project">About The Project</a> •
  <a href="#getting-started">Getting Started</a> •
  <a href="#tutorials">Tutorials</a> •
  <a href="#documentation">Docs</a> •
  <a href="#license">License</a> •
  <a href="#citation">Citation</a>
</p>

---

<!-- ABOUT THE PROJECT -->
## About The Project

![CockpitAnimation](docs/assets/showcase.gif)

**Motivation:** Currently, training a deep neural network is often a pain! Succesfully training such a network usually requires either years of intuition or expensive parameter searches and lots of trial and error. Traditional debugger provide only limited help: They can help diagnose *syntactical errors* but they do not help with *training bugs* such as ill-chosen learning rates.

**Cockpit** is a visual and statistical debugger specifically designed for deep learning! With it, you can:

- **Track relevant diagnostics** that can tell you more about the state of the training process. The train/test loss might tell you *whether* your training is working or not, but not *why*. Statistical quantities, such as the *gradient norm*, *Hessian trace* and *histograms* over the network's gradient and parameters offer insight into the training process.
- **Visualize them in real-time** to get a *status screen* of your training. Cockpit compresses and visualizes the most important quantities into *instruments* providing more insight into the training.
- **Use these quantities** for novel and more sophisticated training algorithms or to build additional visualizations.

Cockpit uses [BackPACK](https://backpack.pt) in order to compute those quantities efficiently. In fact, the above animation shows training of the [All-CNN-C network](https://arxiv.org/abs/1412.6806) on [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html).


<!-- GETTING STARTED -->
## Getting Started

### Installation

To install **Cockpit** simply run

    pip install 'git+https://github.com/f-dangel/cockpit-paper.git'

### Customize & Contribute

If you plan to **customize Cockpit**, for example, by adding new quantities or your own visualizations, we suggest installing a local and modifiable version of Cockpit.

    pip install -e 'git+https://github.com/f-dangel/cockpit-paper.git'

If you plan to **contribute to Cockpit**, we suggest using our provided `conda` environment:

    conda env create -f .conda_env.yml
    conda activate cockpit

which will create a `conda` environment called `cockpit`.    
Clone this `repository`, install the package and all its (developer) requirements

    git clone https://github.com/f-dangel/cockpit-paper.git
    pip install -e .
    pip install -r requirements/requirements-dev.txt

<!-- TUTORIALS -->
## Tutorials

### Using the Cockpit with DeepOBS

It is very easy to use **Cockpit** together with [DeepOBS](https://deepobs.github.io/). DeepOBS is a benchmarking tool for optimization method and directly offers more than twenty test problems (i.e. data sets and deep networks) to train on.

If you want to use **Cockpit**, for example, to monitor your novel optimizer, you can simply use the runner provided with the Cockpit. The `ScheduleCockpitRunner` works analogously to other [DeepOBS Runners](https://deepobs.readthedocs.io/en/v1.2.0-beta0_a/api/pytorch/runner.html), with a minimal working example provided here:

```python
"""Run SGD on the Quadratic Problem of DeepOBS."""

from torch.optim import SGD
from backboard.runners.scheduled_runner import ScheduleCockpitRunner

# Replace with your optimizer, in this case we use SGD
optimizer_class = SGD
hyperparams = {"lr": {"type": float}}

def lr_schedule(num_epochs):
    """Some Learning rate schedule."""
    return lambda epoch: 0.95 ** epoch

runner = ScheduleCockpitRunner(optimizer_class, hyperparams)

# Fix training parameters, otherwise they can be passed via the command line
runner.run(
    testproblem="quadratic_deep",
    track_interval=1,
    plot_interval=10,
    lr_schedule=lr_schedule,
)
```

The output of this script is (among other files) a Cockpit log file ending in `__log.json` which holds all the tracke data. It can, for example, be read by the `CockpitPlotter` to visualize these quantities.

A more detailed example of using Cockpit and DeepOBS can be found in the [examples directory](examples/)

### Using the Cockpit for general Training Loops

### Using the Plotter

### Adding Quantities

### Adding Instruments

<!-- DOCUMENTATION -->
## Documentation

A more detailed documentation with the API can be found [here](https://f-dangel.github.io/cockpit-paper/)

<!-- LICENSE -->
## License

Distributed under the MIT License. See [`LICENSE`](LICENSE) for more information.

<!-- CITATION -->
## Citation
