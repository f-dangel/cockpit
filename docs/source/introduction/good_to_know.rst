============
Good to Know
============

We try to make Cockpit's usage as easy and convenient as possible. Still, there
are limitations. Here are some common pitfalls and recommendations.

BackPACK
########

Most of Cockpit's quantities use BackPACK_ as the back-end for efficient
computation. Please pay attention to the following points for smooth
integration:

- Don't forget to `extend the model and loss function
  <https://docs.backpack.pt/en/master/main-api.html#extending-the-model-and-loss-function>`_
  yourself [1]_ to activate BackPACK_.

- Verify that your model architecture is `supported by BackPACK
  <https://docs.backpack.pt/en/master/supported-layers.html>`_.

- Your loss function must use ``"mean"`` reduction, that is the loss is of the
  following structure

  .. math::

    \mathcal{L}(\mathbf{\theta}) = \frac{1}{N} \sum_{n=0}^{N}
    \ell(f(\mathbf{x}_n, \mathbf{\theta}), \mathbf{y}_n)\,.

  This avoids an ambiguous scale in individual gradients, which is documented in
  `BackPACK's individual gradient extension
  <https://docs.backpack.pt/en/master/extensions.html#backpack.extensions.BatchGrad>`_.
  Otherwise, Cockpit quantities will use incorrectly scaled individual gradients
  in their computation.

It's also a good idea to read through BackPACK's `Good to know
<https://docs.backpack.pt/en/master/good-to-know.html>`_ section.

Performance
###########

Slow run time and memory errors are annoying. Here are some tweaks to reduce run
time and memory consumption:

- Use schedules to reduce the tracking frequency. You can specify custom
  schedules to literally select any iteration to be tracked, or rely on
  pre-defined :mod:`~cockpit.utils.schedules`.

- Exclude :py:class:`GradHist2d <cockpit.quantities.GradHist2d>` from your quantities. The
  two-dimensional histogram implementation uses :py:func:`torch.scatter_add`,
  which can be slow on GPU due to atomic additions.

- Exclude :py:class:`HessMaxEV <cockpit.quantities.HessMaxEV>` from your quantities. It
  requires multiple Hessian-vector products, that are executed sequentially.
  Also, this requires the full computation be kept in memory.

- Spot :ref:`quantities <quantities>` whose constructor contains a ``curvature``
  argument. It defaults to the most accurate, but also most expensive type. You
  may want to sacrifice accuracy for memory and run time performance by
  selecting a cheaper option.


.. [1] Leaving this responsibility to users is a deliberate choice, as Cockpit
  does not always need the package. Specific configurations, that are very
  limited though, work without BackPACK_ as they rely only on functionality
  built into PyTorch_.

.. _BackPACK: https://backpack.pt/
.. _PyTorch: https://pytorch.org/
