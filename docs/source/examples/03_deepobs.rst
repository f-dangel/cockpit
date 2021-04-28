===============
DeepOBS Example
===============

**Cockpit** easily integrates with and can be used together with 
`DeepOBS <https://deepobs.github.io/>`_.
This will directly give you access to dozens of deep learning problems 
that you can explore with **Cockpit**.

.. Note::
   This example requires a `DeepOBS <https://github.com/fsschneider/DeepOBS/>`__ 
   and a `BackOBS <https://github.com/f-dangel/backobs>`_ installation. 
   You can install them by running

   .. code:: bash

      pip install 'git+https://github.com/fsschneider/DeepOBS.git@v1.2.0-beta0#egg=DeepOBS'

   and
   
   .. code:: bash

      pip install 'git://github.com/f-dangel/backobs.git@master#egg=backobs'

   Note, that currently, only the 1.2.0 beta version of DeepOBS supports PyTorch
   which will be installed by the above command.

In the following example, we will use an additional :download:`utility file 
<../../../examples/_utils_deepobs.py>` which automatically incorporates **Cockpit**
with the DeepOBS training loop.

Having the two `utility files from our repository 
<https://github.com/f-dangel/cockpit/tree/development/examples>`_ we can run

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
