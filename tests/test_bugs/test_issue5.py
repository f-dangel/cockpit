"""Reproduces the bug described in https://github.com/f-dangel/cockpit/issues/5."""

from backpack import extend
from pytest import raises
from torch import manual_seed, rand
from torch.nn import Flatten, Linear, MSELoss, Sequential
from torch.optim import Adam

from cockpit import Cockpit
from cockpit.quantities import Alpha, GradHist1d
from cockpit.utils.schedules import linear


def test_BatchGradTransformsHook_deletes_attribute_required_by_Alpha():
    """If the optimizer is not SGD, ``Alpha`` needs access to ``.grad_batch``.

    But if an extension that uses ``BatchGradTransformsHook`` is used at the same time,
    it will delete the ``grad_batch`` attribute during the backward pass. Consequently,
    ``Alpha`` cannot access the attribute anymore. This leads to the error.
    """
    manual_seed(0)

    N, D_in, D_out = 2, 3, 1
    model = extend(Sequential(Flatten(), Linear(D_in, D_out)))

    opt_not_sgd = Adam(model.parameters(), lr=1e-3)
    loss_fn = extend(MSELoss(reduction="mean"))
    individual_loss_fn = MSELoss(reduction="none")

    on_first = linear(1)
    alpha = Alpha(on_first)
    uses_BatchGradTransformsHook = GradHist1d(on_first)

    cockpit = Cockpit(
        model.parameters(), quantities=[alpha, uses_BatchGradTransformsHook]
    )

    global_step = 0
    inputs, labels = rand(N, D_in), rand(N, D_out)

    # forward pass
    outputs = model(inputs)
    loss = loss_fn(outputs, labels)
    losses = individual_loss_fn(outputs, labels)

    # backward pass
    with cockpit(
        global_step,
        info={
            "batch_size": N,
            "individual_losses": losses,
            "loss": loss,
            "optimizer": opt_not_sgd,
        },
    ):
        loss.backward(create_graph=cockpit.create_graph(global_step))
