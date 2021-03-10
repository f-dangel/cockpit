=============
Basic Example
=============

In this snippet below you see an example of using **Cockpit** with a complete standard PyTorch training loop.
Lines that are highlighted in yellow include **Cockpit** specific code, but don't worry, most of these changes are simple plug-and-play solutions.

You can try out this basic example yourself by installing **Cockpit** and then running this :download:`example script <../../../examples/01_basic_fmnist.py>`  via

.. code:: bash

  python 01_basic_fmnist.py

In the following, we will break-down and explain each step of including **Cockpit** to the training loop.

.. literalinclude:: ../../../examples/01_basic_fmnist.py
   :language: python
   :emphasize-lines: 4-6,12-14,19-21,31, 33-42, 50
   :linenos:


Imports
=======

.. literalinclude:: ../../../examples/01_basic_fmnist.py
   :language: python
   :emphasize-lines: 4-6
   :linenos:
   :lines: 1-8

For this example weuse PyTorch and thus import ``torch``.
Additionally, we import ``BackPACK`` which will automatically installed when installing **Cockpit**.
We also import :func:`Cockpit<cockpit.Cockpit>` and :func:`CockpitPlotter<cockpit.CockpitPlotter>` to track and then visualize insightful quantities.

To simplify the code snippet, in line 8, we import from a utils file which will provide us with the Fashion-MNIST data.

Defining the Problem
====================

.. literalinclude:: ../../../examples/01_basic_fmnist.py
   :language: python
   :emphasize-lines: 3-5
   :linenos:
   :lines: 10-14
   :lineno-start: 10

Next, we build a simple classifier for our Fashion-MNIST data set.

The only change to a traditional training loop is that we need to `extend` both the model and the loss function using BackPACK.
This is as simple as wrapping the traditional model and loss function in the ``extend()`` function provided by BackPACK.
It lets BackPACK know that additional quantities (such as individual gradients) should be computed for these parameters.

For the :func:`Alpha<cockpit.quantities.AlphaOptimized>` quantity we require access to the individual loss values, which can be computed cheaply but is not usually part of a conventional training loop.
We can create this function analogously to the regular loss function just setting the ``reduction=None``.
There is no need to let BackPACK know about its existence, since these losses will not be differentiated.

Configuring the Cockpit
=======================

.. literalinclude:: ../../../examples/01_basic_fmnist.py
   :language: python
   :emphasize-lines: 2-3
   :linenos:
   :lines: 19-21
   :lineno-start: 19

Computation of quantities and storing of results are managed by the :func:`Cockpit<cockpit.Cockpit>` class. We have to pass the model parameters, and a list of quantities, which specify what should be tracked and when.

Cockpit offers configurations with different computational complexity: ``"economy"``, ``"business"``, and ``"full"``. We will use the provided utility function to configure the quantities.

Training Loop
=============

.. literalinclude:: ../../../examples/01_basic_fmnist.py
   :language: python
   :emphasize-lines: 9-16
   :linenos:
   :lines: 26-42
   :lineno-start: 26

Training itself is straightforward. At every iteration, we draw a mini-batch, compute the model predictions and losses, then perform a backward pass and update the parameters.

The main differences with **Cockpit** is that the backward call is surrounded by a ``with cockpit(...)`` context, that manages the extra computations during the backward pass. Additional information required by some quantities is passed through the ``info`` argument.

Plotting the Cockpit
====================

.. literalinclude:: ../../../examples/01_basic_fmnist.py
   :language: python
   :emphasize-lines: 1
   :linenos:
   :lines: 50
   :lineno-start: 50

At any point during the training, here we do it in every single iteration, the computed metrics can be visualized by calling the :func:`CockpitPlotter<cockpit.CockpitPlotter>`s plot functionality on the created :func:`Cockpit<cockpit.Cockpit>` instance.

