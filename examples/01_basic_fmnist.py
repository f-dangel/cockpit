"""A basic example of using Cockpit with PyTorch for Fashion-MNIST."""

import torch
from _utils_examples import fmnist_data
from backpack import extend

from cockpit import Cockpit, CockpitPlotter
from cockpit.utils.configuration import configuration

# Build Fashion-MNIST classifier
fmnist_data = fmnist_data()
model = extend(torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(784, 10)))
loss_fn = extend(torch.nn.CrossEntropyLoss(reduction="mean"))
individual_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

# Create SGD Optimizer
opt = torch.optim.SGD(model.parameters(), lr=1e-2)

# Create Cockpit and a plotter
cockpit = Cockpit(model.parameters(), quantities=configuration("full"))
plotter = CockpitPlotter()

# Main training loop
max_steps, global_step = 5, 0
for inputs, labels in iter(fmnist_data):
    opt.zero_grad()

    # forward pass
    outputs = model(inputs)
    loss = loss_fn(outputs, labels)
    losses = individual_loss_fn(outputs, labels)

    # backward pass
    with cockpit(
        global_step,
        info={
            "batch_size": inputs.shape[0],
            "individual_losses": losses,
            "loss": loss,
            "optimizer": opt,
        },
    ):
        loss.backward(create_graph=cockpit.create_graph(global_step))

    # optimizer step
    opt.step()
    global_step += 1

    print(f"Step: {global_step:5d} | Loss: {loss.item():.4f}")

    plotter.plot(cockpit)

    if global_step >= max_steps:
        break

plotter.plot(cockpit, block=True)
