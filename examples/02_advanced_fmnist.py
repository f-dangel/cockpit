"""A slightly advanced example of using Cockpit with PyTorch for Fashion-MNIST."""

import torch
from _utils_examples import cnn, fmnist_data, get_logpath
from backpack import extend, extensions

from cockpit import Cockpit, CockpitPlotter, quantities
from cockpit.utils import schedules

# Build Fashion-MNIST classifier
fmnist_data = fmnist_data()
model = extend(cnn())  # Use a basic convolutional network
loss_fn = extend(torch.nn.CrossEntropyLoss(reduction="mean"))
individual_loss_fn = extend(torch.nn.CrossEntropyLoss(reduction="none"))

# Create SGD Optimizer
opt = torch.optim.SGD(model.parameters(), lr=5e-1)

# Create Cockpit and a plotter
# Customize the tracked quantities and their tracking schedule
quantities = [
    quantities.GradNorm(schedules.linear(interval=1)),
    quantities.Distance(schedules.linear(interval=1)),
    quantities.UpdateSize(schedules.linear(interval=1)),
    quantities.HessMaxEV(schedules.linear(interval=3)),
    quantities.GradHist1d(schedules.linear(interval=10), bins=10),
]
cockpit = Cockpit(model.parameters(), quantities=quantities)
plotter = CockpitPlotter()

# Main training loop
max_steps, global_step = 50, 0
for inputs, labels in iter(fmnist_data):
    opt.zero_grad()

    # forward pass
    outputs = model(inputs)
    loss = loss_fn(outputs, labels)
    losses = individual_loss_fn(outputs, labels)

    # backward pass
    with cockpit(
        global_step,
        extensions.DiagHessian(),  # Other BackPACK quantities can be computed as well
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

    if global_step % 10 == 0:
        plotter.plot(
            cockpit,
            savedir=get_logpath(),
            show_plot=False,
            save_plot=True,
            savename_append=str(global_step),
        )

    if global_step >= max_steps:
        break

# Write Cockpit to json file.
cockpit.write(get_logpath())

# Plot results from file
plotter.plot(
    get_logpath(),
    savedir=get_logpath(),
    show_plot=False,
    save_plot=True,
    savename_append="_final",
)
