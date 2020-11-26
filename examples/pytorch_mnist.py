"""Example: Training Loop using Cockpit."""

import torch
from examples_utils import CNN, MNISTData, create_logpath, evaluate

from backpack import extend
from cockpit import Cockpit, CockpitPlotter
from cockpit.utils import schedules
from cockpit.utils.configuration import configuration

train_loader, test_loader = MNISTData()
model = extend(CNN())
lossfunc = extend(torch.nn.CrossEntropyLoss(reduction="mean"))
individual_lossfunc = torch.nn.CrossEntropyLoss(reduction="none")
opt = torch.optim.SGD(model.parameters(), lr=1e-2)

# COCKPIT #
cockpit = Cockpit(
    model.parameters(),
    quantities=configuration("full", track_schedule=schedules.linear(interval=30)),
)
plotter = CockpitPlotter()

num_epochs = 1
iteration = 0

for _ in range(num_epochs):
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

        if iteration % 30 == 0:
            print("** Iteration: ", iteration)
            cockpit.write(create_logpath())
            plotter.plot(create_logpath(suffix=".json"))

    evaluate(model, lossfunc, test_loader)
