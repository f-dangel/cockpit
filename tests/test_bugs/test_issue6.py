"""Reproduces the bug described in https://github.com/f-dangel/cockpit/issues/6."""

from backpack import extend
from torch import Tensor, manual_seed, rand
from torch.nn import Linear, Module, MSELoss, ReLU
from torch.optim import SGD

from cockpit import Cockpit
from cockpit.quantities import GradHist1d
from cockpit.utils.schedules import linear


def test_extension_hook_executes_on_custom_module():
    """Cockpit's extension hook is only skipped for known containers like Sequential.

    It will thus execute on custom containers and lead to crashes whenever a quantity
    that uses extension hooks is used.
    """
    manual_seed(0)
    N, D_in, D_out = 2, 3, 1

    # NOTE Inheriting from Sequential passes
    class CustomModule(Module):
        """Custom container that is not skipped by the extension hook."""

        def __init__(self):
            super().__init__()
            self.linear = Linear(D_in, D_out)
            self.relu = ReLU()

        def forward(self, x: Tensor) -> Tensor:
            return self.relu(self.linear(x))

    uses_extension_hook = GradHist1d(linear(interval=1))
    config = [uses_extension_hook]

    model = extend(CustomModule())
    cockpit = Cockpit(model.parameters(), quantities=config)

    opt = SGD(model.parameters(), lr=0.1)

    loss_fn = extend(MSELoss(reduction="mean"))
    individual_loss_fn = MSELoss(reduction="none")

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
            "optimizer": opt,
        },
    ):
        loss.backward(create_graph=cockpit.create_graph(global_step))
