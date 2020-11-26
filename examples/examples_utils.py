"""Utility Functions for the Examples."""

import os
from random import seed

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as dataloader
import torchvision
import torchvision.transforms as transforms

SEED = 42
seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def CNN():
    """Basic Conv-Net for MNIST."""
    return nn.Sequential(
        nn.Conv2d(1, 11, 5),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(11, 16, 5),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Flatten(),
        nn.Linear(16 * 4 * 4, 80),
        nn.ReLU(),
        nn.Linear(80, 84),
        nn.ReLU(),
        nn.Linear(84, 10),
    )


def MNISTData():
    """Load the MNIST Data into data loader."""
    # Create data path
    curr_path = os.path.dirname(os.path.abspath(__file__))
    res_path = os.path.join(curr_path, "data")
    os.makedirs(res_path, exist_ok=True)

    transform = transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    # Load data sets
    train_set = torchvision.datasets.MNIST(
        root=res_path, train=True, download=True, transform=transform
    )
    train_loader = dataloader.DataLoader(train_set, batch_size=256, shuffle=True)
    test_set = torchvision.datasets.MNIST(
        root=res_path, train=False, download=True, transform=transform
    )
    test_loader = dataloader.DataLoader(test_set, batch_size=256, shuffle=False)

    return train_loader, test_loader


def evaluate(model, lossfunc, test_loader):
    """Evaluate the model on a test set."""
    # model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += lossfunc(output, target).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )


def create_logpath(suffix=""):
    """Creates a logpath for the example."""
    # Create data path
    curr_path = os.path.dirname(os.path.abspath(__file__))
    res_path = os.path.join(
        curr_path, f"results/mnist_convnet/SGD/example_run/run{suffix}"
    )
    return res_path
