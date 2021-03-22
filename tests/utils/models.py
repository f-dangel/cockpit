"""Utility functions to create toy models for Cockpit's tests."""

import torch


def load_toy_model():
    """Build a tor model that can be used in conjunction with ``ToyData``."""
    return torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(25, 4),
        torch.nn.ReLU(),
        torch.nn.Linear(4, 4),
        torch.nn.ReLU(),
        torch.nn.Linear(4, 3),
    )
