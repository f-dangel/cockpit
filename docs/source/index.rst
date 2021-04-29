=======
Cockpit
=======

|CI Status| |Lint Status| |Doc Status| |Coverage| |License| |Code Style| |arXiv|

----

.. code:: bash

  pip install 'git+https://github.com/f-dangel/cockpit.git@development'

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

  pip install 'git+https://github.com/f-dangel/cockpit.git@development'


.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   introduction/quickstart
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

   GitHub Repository <https://github.com/f-dangel/cockpit>
   other/contributors
   other/license
   other/changelog


.. |CI Status| image:: https://github.com/f-dangel/cockpit/actions/workflows/CI.yml/badge.svg
    :target: https://github.com/f-dangel/cockpit/actions/workflows/CI.yml
    :alt: CI Status

.. |Lint Status| image:: https://github.com/f-dangel/cockpit/actions/workflows/Lint.yml/badge.svg
    :target: https://github.com/f-dangel/cockpit/actions/workflows/Lint.yml
    :alt: Lint Status

.. |Doc Status| image:: https://img.shields.io/readthedocs/cockpit/development.svg?logo=read%20the%20docs&logoColor=white&label=Doc
    :target: https://cockpit.readthedocs.io
    :alt: Doc Status

.. |Coverage| image:: https://coveralls.io/repos/github/f-dangel/cockpit/badge.svg?branch=development&t=piyZHm
    :target: https://coveralls.io/github/f-dangel/cockpit?branch=development
    :alt: CI Status

.. |License| image:: https://img.shields.io/badge/License-MIT-green.svg
    :target: https://github.com/f-dangel/cockpit/blob/master/LICENSE
    :alt: License

.. |Code Style| image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black
    :alt: Code Style

.. |arXiv| image:: https://img.shields.io/static/v1?logo=arxiv&logoColor=white&label=Preprint&message=2102.06604&color=B31B1B
    :target: https://arxiv.org/abs/2102.06604
    :alt: arXiv
