=======
Cockpit
=======

|CI Status| |Lint Status| |Coverage| |License| |Code Style| |arXiv|

----

**Cockpit is a visual and statistical debugger specifically designed for deep
learning.** Training a deep neural network is often a pain! Successfully training
such a network usually requires either years of intuition or expensive parameter
searches involving lots of trial and error. Traditional debuggers provide only
limited help: They can find *syntactical errors* but not *training bugs* such as
ill-chosen learning rates. **Cockpit** offers a closer, more meaningful look
into the training process with multiple well-chosen *instruments*.

----

.. image:: _static/showcase.gif


To install **Cockpit** simply run

.. code:: bash

  pip install 'git+https://github.com/f-dangel/cockpit-paper.git@v1_release'


.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   examples/01_basic_fmnist
   examples/02_advanced_fmnist
   examples/03_deepobs
   introduction/good_to_know

.. toctree::
   :maxdepth: 1
   :caption: API Documentation

   api/cockpit
   api/plotter
   api/quantities
   api/instruments
   api/utils

.. toctree::
   :maxdepth: 1
   :caption: Other

   GitHub Repository <https://github.com/f-dangel/cockpit-paper>
   other/contributors
   other/license
   other/changelog


.. |CI Status| image:: https://github.com/f-dangel/cockpit-paper/actions/workflows/CI.yml/badge.svg
    :target: https://github.com/f-dangel/cockpit-paper/actions/workflows/CI.yml
    :alt: CI Status

.. |Lint Status| image:: https://github.com/f-dangel/cockpit-paper/actions/workflows/Lint.yml/badge.svg
    :target: https://github.com/f-dangel/cockpit-paper/actions/workflows/Lint.yml
    :alt: Lint Status

.. |Coverage| image:: https://coveralls.io/repos/github/f-dangel/cockpit-paper/badge.svg?branch=v1_release&t=piyZHm
    :target: https://coveralls.io/github/f-dangel/cockpit-paper?branch=v1_release
    :alt: CI Status

.. |License| image:: https://img.shields.io/badge/License-MIT-green.svg
    :target: https://github.com/f-dangel/cockpit-paper/blob/master/LICENSE
    :alt: License

.. |Code Style| image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black
    :alt: Code Style

.. |arXiv| image:: https://img.shields.io/static/v1?logo=arxiv&logoColor=white&label=Preprint&message=2102.06604&color=B31B1B
    :target: https://arxiv.org/abs/2102.06604
    :alt: arXiv
