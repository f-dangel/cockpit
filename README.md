<!-- PROJECT LOGO -->
<br />
<p align="center">
<a href="#"><img src="docs/rtd/assets/Logo.png" alt="Logo"/></a>
  <h3 align="center">A Practical Debugging Tool for Training Deep Neural Networks</h3>

  <p align="center">
    A better status screen for deep learning.
  </p>
</p>

<p align="center">
  <a href="#installation">Installation</a> •
  <a href="https://f-dangel.github.io/cockpit-paper/">Docs</a> •
  <a href="experiments/">Experiments</a> •
  <a href="#license">License</a> •
  <a href="#citation">Citation</a>
</p>

![CI](https://github.com/f-dangel/cockpit-paper/actions/workflows/CI.yml/badge.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/f-dangel/cockpit-paper/blob/master/LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![arXiv](https://img.shields.io/static/v1?logo=arxiv&logoColor=white&label=Preprint&message=2102.06604&color=B31B1B)](https://arxiv.org/abs/2102.06604)

---

**Cockpit is a visual and statistical debugger specifically desigend for deep learning.** Training a deep neural network is often a pain! Succesfully training such a network usually requires either years of intuition or expensive parameter searches involving lots of trial and error. Traditional debuggers provide only limited help: They can find *syntactical errors* but not *training bugs* such as ill-chosen learning rates. **Cockpit** offers a closer, more meaningful look into the training process with multiple well-chosen *instruments*.

---

![CockpitAnimation](docs/rtd/assets/showcase.gif)

<!-- Installation -->
## Installation

To install **Cockpit** simply run

```bash
git clone https://github.com/ahthie7u/cockpit.git
cd cockpit/
pip install -r requirements.txt
pip install .
```

<!-- Documentation -->
## Documentation

The [documentation](https://f-dangel.github.io/cockpit-paper/) provides a full tutorial on how to get started using **Cockpit** as well as a detailed documentation of its API.

<!-- Experiments -->
## Experiments

To showcase the capabilities of **Cockpit** we performed several experiments illustrating the usefulness of our debugging tool. The code for all experiments, as well as the generated data is presented in [experiments](experiments/). For a discussion of those experiments please refer to our [paper](https://arxiv.org/abs/2102.06604).

<!-- LICENSE -->
## License

Distributed under the MIT License. See [`LICENSE`](LICENSE) for more information.

<!-- Citation -->
## Citation

If you use **Cockpit**, please consider citing:

> [Frank Schneider, Felix Dangel, Philipp Hennig<br/>
> **Cockpit: A Practical Debugging Tool for Training Deep Neural Networks**<br/>
> *arXiv 2102.06604*](http://arxiv.org/abs/2102.06604)

```bibtex
@misc{schneider2021cockpit,
   title={{Cockpit: A Practical Debugging Tool for Training Deep Neural Networks}},
   author={Frank Schneider and Felix Dangel and Philipp Hennig},
   year={2021},
   eprint={2102.06604},
   archivePrefix={arXiv},
   primaryClass={cs.LG}
}
```
