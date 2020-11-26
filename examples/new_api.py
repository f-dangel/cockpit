"""Example: Training Loop using Cockpit."""

import new_api_utils as utils
import torch

from backpack import extend, extensions
from cockpit import Cockpit, CockpitPlotter
from cockpit.utils.configuration import configuration

# load data
train_loader = utils.get_mnist_trainloader()

# create and extend model/loss function
model = extend(torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(784, 10)))
lossfunc = extend(torch.nn.CrossEntropyLoss(reduction="mean"))
individual_lossfunc = torch.nn.CrossEntropyLoss(reduction="none")

# create optimizer
opt = torch.optim.SGD(model.parameters(), lr=1e-2)

# cockpit configuration and plotter
cockpit = Cockpit(model.parameters(), quantities=configuration("full"))
plotter = CockpitPlotter()

# where to write results or save plots to
logpath = utils.create_logpath()

# training specification
num_epochs, global_step = 2, 0

for _ in range(num_epochs):
    # draw a batch
    for inputs, labels in iter(train_loader):
        opt.zero_grad()

        # forward pass
        outputs = model(inputs)
        loss = lossfunc(outputs, labels)
        losses = individual_lossfunc(outputs, labels).detach()

        # backward pass inside cockpit context that computes additional quantities
        with cockpit(
            global_step,
            extensions.DiagHessian(),  # you can compute/use other BackPACK quantities
            info={  # some quantities require additional information specified here
                "batch_size": inputs.shape[0],
                "individual_losses": losses,
                "loss": loss,
            },
        ):
            loss.backward(create_graph=cockpit.create_graph(global_step))

        # take step
        opt.step()
        global_step += 1

        print(f"Step: {global_step:5d} | Loss: {loss.item():.4f}")

        # Visualize computed quantities, also works with a logpath
        plotter.plot(cockpit)

    cockpit.write(logpath)
