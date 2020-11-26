"""Example: Training Loop using Cockpit."""

import torch
import torchvision

from backpack import extend
from cockpit import Cockpit


def get_mnist_dataloder(batch_size=64, shuffle=True):
    """Returns a dataloader for MNIST"""
    mnist_dataset = torchvision.datasets.MNIST(
        root="./data",
        train=True,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,)),
            ]
        ),
        download=True,
    )

    return torch.utils.data.dataloader.DataLoader(
        mnist_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )


batch_size = 128
train_loader = get_mnist_dataloder(batch_size=batch_size)

model = extend(torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(784, 10)))
lossfunc = extend(torch.nn.CrossEntropyLoss(reduction="mean"))
individual_lossfunc = torch.nn.CrossEntropyLoss(reduction="none")

opt = torch.optim.SGD(model.parameters(), lr=1e-2)

# empty cockpit
# cockpit = Cockpit(model.parameters(), quantities=None)

num_epochs, global_step = 1, 0

for _ in range(num_epochs):
    for inputs, labels in iter(train_loader):
        opt.zero_grad()

        outputs = model(inputs)
        loss = lossfunc(outputs, labels)
        losses = individual_lossfunc(outputs, labels).detach()

        # with cockpit(
        #     global_step,
        #     info={
        #         "batch_size": inputs.shape[0],
        #         "individual_losses": losses,
        #         "loss": loss,
        #     },
        # ):
        loss.backward(
            # create_graph=cockpit.create_graph(global_step)
        )

        opt.step()
        global_step += 1

        print(f"Step: {global_step:5d} | Loss: {loss.item():.4f}")

    # cockpit.write(create_logpath())
    # cockpit.plot()
