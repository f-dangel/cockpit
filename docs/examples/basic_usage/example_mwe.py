"""Basic usage example
==============================

In this snippet you will see the minimum overhead required to make the full Cockpit work with PyTorch training loops.

We will focus on explaining what the additional steps are, and neglect details like further customization and technical aspects. For more details on the latter, check out the walk-through tutorial here. It will guide you through the steps to integrate Cockpit into an existing PyTorch train loop, and elaborate on technical aspects.

For demonstration, we will use the canonical 'logistic regression on MNIST,
trained with SGD' example.

"""

# %%
# Preliminaries
# -------------
# Let's get the imports out of our way and set training hyper-parameters

import os
import pprint

import torch

import backpack
from backpack.utils.examples import get_mnist_dataloder
from cockpit import Cockpit, CockpitPlotter
from cockpit.utils.configuration import configuration

# make deterministic
torch.manual_seed(0)

# training hyper-parameters
BATCH_SIZE = 128
LR = 0.1
MAX_ITER = 10
PLOT_ITERS = [MAX_ITER]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_accuracy(outputs, labels):
    """Helper function to print the accuracy"""
    predictions = outputs.argmax(dim=1, keepdim=True).view_as(labels)

    return predictions.eq(labels).float().mean().item()


# %%
# We are ready to set up mini-batch loading, define the model, and the loss function

train_loader = get_mnist_dataloder(batch_size=BATCH_SIZE)

model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(784, 10)).to(DEVICE)
lossfunc = torch.nn.CrossEntropyLoss(reduction="mean").to(DEVICE)

# %%
# To be able to compute all quantities included in the full Cockpit, we need
# to integrate BackPACK by :func:`extend <backpack.extend>`ing the model and
# the loss function

model = backpack.extend(model)
lossfunc = backpack.extend(lossfunc)

# %%
# Some Cockpit quantities require additional information that can be
# computed cheaply, but is usually not part of a conventional training loop.
# In particular, we need access to the individual losses, which can simply
# be computed from the model prediction by setting the loss function reduction
# to ``"none"``. No need to let BackPACK know about its existence, since these
# losses will not be differentiated.

individual_lossfunc = torch.nn.CrossEntropyLoss(reduction="none").to(DEVICE)

# %%
# Choose a configuration
# ----------------------
# Computation of quantities and storing of results are managed by the
# :func:`Cockpit<cockpit.Cockpit>` class. We have to pass the model parameters,
# and a list of quantities, which specify what should be tracked and when.
#
# Cockpit offers configurations with different computational complexity:
# ``"economy"``, ``"business"``, and ``"full"``. We will use the provided
# utility function to configure the quantities.

cockpit = Cockpit(model.parameters(), quantities=configuration("full"))
plotter = CockpitPlotter()

# %%
# Training loop
# -------------
# Training itself is straightforward. At every iteration, we draw a mini-batch,
# compute the model predictions and losses, then perform a backward pass and
# update the parameters.
#
# The main differences with Cockpit is that ``backward`` call is surrounded by
# a ``with cockpit(...)`` context, that manages the extra computations during
# the backward pass. Additional information required by some quantities is
# passed through the ``info`` argument.
#
# At any iteration, the computed metrics can be visualized by calling
# the :func:`CockpitPlotter<cockpit.CockpitPlotter>`'s ``plot`` functionality
# on the created :func:`Cockpit<cockpit.Cockpit>` instance.

opt = torch.optim.SGD(model.parameters(), lr=LR)

iteration = 0

for inputs, labels in iter(train_loader):
    opt.zero_grad()

    outputs = model(inputs)
    loss = lossfunc(outputs, labels)

    # compute individual losses
    with torch.no_grad():
        individual_losses = individual_lossfunc(outputs, labels)

    # only backward inside a call to cockpit
    with cockpit(
        iteration,
        info={
            "batch_size": inputs.shape[0],
            "individual_losses": individual_losses,
            "loss": loss,
        },
    ):
        loss.backward(create_graph=cockpit.create_graph(iteration))

    print(
        f"[{iteration:2d}/{MAX_ITER}]"
        + f" Loss: {loss.item():.4f},"
        + f" Acc: {get_accuracy(outputs, labels):.2f}"
    )

    if iteration in PLOT_ITERS:
        plotter.plot(cockpit)

    opt.step()
    iteration += 1

    if iteration >= MAX_ITER:
        break

# %%
# Computed metrics
# ----------------
# We can inspect the computed metrics and use them elsewhere.

output = cockpit.get_output()

# prettify
print(pprint.pformat(output))

# %%
# Writing results
# ---------------
# To save the computed metrics in a ``.json`` file, use ``cockpit``'s ``write`` method.

logfile = "./cockpit_log"
cockpit.write(logfile)
