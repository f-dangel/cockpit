"""Utility functions to create toy input for Cockpit's tests."""

import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset


def load_toy_data(batch_size):
    """Build a ``DataLoader`` with specified batch size from the toy data."""
    return DataLoader(ToyData(), batch_size=batch_size)


class ToyData(Dataset):
    """Toy data set used for testing. Consists of small random "images" and labels."""

    def __init__(self, center=0.1):
        """Init the toy data set.

        Args:
            center (float): Center around which the data is randomly distributed.
        """
        super(ToyData, self).__init__()
        self._center = center

    def __getitem__(self, index):
        """Return item with index `index` of data set.

        Args:
            index (int): Index of sample to access. Ignored for now.

        Returns:
            [tuple]: Tuple of (random) input and (random) label.
        """
        item_input = torch.rand(1, 5, 5) + self._center
        item_label = torch.randint(size=(), low=0, high=3)
        return (item_input, item_label)

    def __len__(self):
        """Length of dataset. Arbitrarily set to 10 000."""
        return 10000  # of how many examples(images?) you have
