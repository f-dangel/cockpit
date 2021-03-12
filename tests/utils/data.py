"""Utility functions to create toy input for Cockpit's tests."""

import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset


def load_toy_data(batch_size):
    """Build a ``DataLoader`` with specified batch size from the toy data."""
    return DataLoader(ToyData(), batch_size=batch_size)


class ToyData(Dataset):
    def __init__(self):
        super(ToyData, self).__init__()

    def __getitem__(self, index):
        item_input = torch.rand(1, 5, 5)
        item_label = torch.randint(size=(), low=0, high=3)
        return (item_input, item_label)

    def __len__(self):
        return 10000  # of how many examples(images?) you have
