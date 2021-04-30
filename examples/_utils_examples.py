"""Utility functions for the basic PyTorch example."""

import os
import warnings

import torch
import torchvision

HERE = os.path.abspath(__file__)
HEREDIR = os.path.dirname(HERE)
EXAMPLESDIR = os.path.dirname(HEREDIR)

# Ignore the PyTorch warning that is irrelevant for us
warnings.filterwarnings("ignore", message="Using a non-full backward hook ")


def fmnist_data(batch_size=64, shuffle=True):
    """Returns a dataloader for Fashion-MNIST.

    Args:
        batch_size (int, optional): Batch size. Defaults to 64.
        shuffle (bool, optional): Whether the data should be shuffled. Defaults to True.

    Returns:
        torch.DataLoader: Dataloader for Fashion-MNIST data.
    """
    # Additionally set the random seed for reproducability
    torch.manual_seed(0)

    fmnist_dataset = torchvision.datasets.FashionMNIST(
        root=os.path.join(EXAMPLESDIR, "data"),
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
        fmnist_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )


def cnn():
    """Basic Conv-Net for (Fashion-)MNIST."""
    return torch.nn.Sequential(
        torch.nn.Conv2d(1, 11, 5),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2, 2),
        torch.nn.Conv2d(11, 16, 5),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2, 2),
        torch.nn.Flatten(),
        torch.nn.Linear(16 * 4 * 4, 80),
        torch.nn.ReLU(),
        torch.nn.Linear(80, 84),
        torch.nn.ReLU(),
        torch.nn.Linear(84, 10),
    )


def get_logpath(suffix=""):
    """Create a logpath and return it.

    Args:
        suffix (str, optional): suffix to add to the output. Defaults to "".

    Returns:
        str: Path to the logfile (output of Cockpit).
    """
    save_dir = os.path.join(EXAMPLESDIR, "logfiles")
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, f"cockpit_output{suffix}")
    return log_path
