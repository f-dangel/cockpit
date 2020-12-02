"""Minimum working example: Training Loop using Cockpit."""

import torch
from examples_utils import MNISTData, create_logpath

from backpack import extend
from cockpit import Cockpit, CockpitPlotter
from cockpit.utils.configuration import configuration

train_loader, test_loader = MNISTData()
model = extend(torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(784, 10)))
lossfunc = extend(torch.nn.CrossEntropyLoss(reduction="mean"))
individual_lossfunc = torch.nn.CrossEntropyLoss(reduction="none")
opt = torch.optim.SGD(model.parameters(), lr=1e-2)

# COCKPIT #
cockpit = Cockpit(model.parameters(), quantities=configuration("full"))
plotter = CockpitPlotter()

max_iterations = 5
iteration = 0

for inputs, labels in iter(train_loader):
    # Zero Gradients
    opt.zero_grad()

    # Forward pass
    outputs = model(inputs)
    loss = lossfunc(outputs, labels)
    with torch.no_grad():
        individual_losses = individual_lossfunc(outputs, labels)

    # COCKPIT #
    with cockpit(
        iteration,
        info={
            "batch_size": inputs.shape[0],
            "individual_losses": individual_losses,
            "loss": loss,
        },
    ):
        # Backward pass
        loss.backward(create_graph=cockpit.create_graph(iteration))

    # Update step
    opt.step()
    iteration += 1

    plotter.plot(cockpit)

    if iteration >= max_iterations:
        break

cockpit.write(create_logpath())
