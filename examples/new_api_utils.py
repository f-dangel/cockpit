import os

import torch
import torchvision


def get_mnist_trainloader(batch_size=64, shuffle=True):
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


def create_logpath(suffix=""):

    HERE = os.path.abspath(__file__)
    HEREDIR = os.path.dirname(HERE)
    SAVEDIR = os.path.join(HEREDIR, "new_api")
    os.makedirs(SAVEDIR, exist_ok=True)
    LOGPATH = os.path.join(SAVEDIR, f"cockpit_output{suffix}")

    return LOGPATH
