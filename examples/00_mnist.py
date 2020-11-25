"""Example: Training Loop using Cockpit."""

import torch
from examples_utils import CNN, MNISTData, create_logpath, evaluate

from cockpit import Cockpit

train_loader, test_loader = MNISTData()
model = CNN()
lossfunc = torch.nn.CrossEntropyLoss(reduction="mean")
individual_lossfunc = torch.nn.CrossEntropyLoss(reduction="none")
opt = torch.optim.SGD(model.parameters(), lr=1e-2)

# COCKPIT #
cockpit = Cockpit([model, lossfunc], create_logpath(), track_interval=5)

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
            loss.backward(
                create_graph=cockpit.create_graph,
            )

        # Update step
        opt.step()
        iteration += 1

        if iteration % 10 == 0:
            print("** Iteration: ", iteration)
            cockpit.write()
            cockpit.plot()

    evaluate(model, lossfunc, test_loader)
