===============
DeepOBS Example
===============

**Cockpit** easily integrates with and can be used together with 
`DeepOBS <https://deepobs.github.io/>`_.
This will directly give you access to run on dozens of deep learning problems 
that you can explore with **Cockpit**.

.. Note::
   This example requires a **DeepOBS** installation. Currently, only the 1.2.0 beta
   version of it supports PyTorch. Installation instructions can be found
   `here <https://github.com/fsschneider/DeepOBS/tree/develop>`_.

In the following example, we use an :download:`example DeepOBS runner 
<../../../examples/utils/deepobs_runner.py>` that integrates the **Cockpit** 
with **DeepOBS**.

Assuming the structure from our `example files 
<https://github.com/f-dangel/cockpit/tree/development/examples>`_
from the repository, we just have to run

.. code:: bash

  python 03_deepobs.py

which exectues the following file:

.. literalinclude:: ../../../examples/03_deepobs.py
   :language: python
   :linenos:

Just like before, we can define a list of quantities (here we use the 
:mod:`~cockpit.utils.configuration` ``"full"``) that we this time pass to the
``DeepOBSRunner``. It will automatically pass it on to the :class:`~cockpit.Cockpit`.

With the arguments of the ``runner.run()`` function, we can define whether we want
the :class:`~cockpit.CockpitPlotter` plots to show and/or be stored.

The fifteen steps on the `deep quadratic 
<https://deepobs.readthedocs.io/en/v1.2.0-beta0_a/api/pytorch/testproblems/quadratic/quadratic_deep.html>`_ 
problem will result in a **Cockpit** plot similar to this:

.. image:: ../_static/03_deepobs.png
        :alt: Preview Cockpit DeepOBS Example
