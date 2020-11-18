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

<!-- TUTORIALS -->
## Tutorials

### Using the Cockpit

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
