"""Utility functions for Cockpit's tests."""

import torch
from backpack import extend
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset

from cockpit import Cockpit


class SimpleTestHarness:
    """Class for running a simple test loop with the Cockpit."""

    def __init__(
        self,
        problem,
        iterations,
        batch_size=5,
        opt_cls=torch.optim.SGD,
        opt_kwargs=None,
    ):
        """Init the Testing Harness.

        Args:
            problem (string): The problem to test on.
            iterations (int): Maximum number of iterations to run the train loop.
            batch_size (int, optional): Batch size. Defaults to 5.
            opt_cls (class, optional): Optimizer class. Defaults to torch.optim.SGD.
            opt_kwargs (dict, optional): Arguments of the optimizer. Defaults to None.

        Raises:
            NotImplementedError: If no `opt_kwargs` given for a non-SGD optimizer.
        """
        torch.manual_seed(0)

        self.iterations = iterations
        self.batch_size = batch_size
        self.opt_cls = opt_cls
        if opt_kwargs is None:
            if opt_cls == torch.optim.SGD:
                opt_kwargs = {"lr": 0.01}
            else:
                raise NotImplementedError
        self.opt_kwargs = opt_kwargs
        self.data, self.model = self.extract_problem(problem)

    def test(self, cockpit_kwargs, *backpack_exts):
        """Run the test loop.

        Args:
            cockpit_kwargs (dict): Arguments for the cockpit.
            *backpack_exts (list): List of user-defined BackPACK extensions.
        """
        # Extend
        model = extend(self.model)
        loss_fn = extend(torch.nn.CrossEntropyLoss(reduction="mean"))
        individual_loss_fn = extend(torch.nn.CrossEntropyLoss(reduction="none"))

        # Create Optimizer
        opt = self.opt_cls(model.parameters(), **self.opt_kwargs)

        # Initialize Cockpit
        self.cockpit = Cockpit(model.parameters(), **cockpit_kwargs)

        # print(cockpit_exts)

        # Main training loop
        global_step = 0
        for inputs, labels in iter(self.data):
            opt.zero_grad()

            # forward pass
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            losses = individual_loss_fn(outputs, labels).detach()

            # backward pass
            with self.cockpit(
                global_step,
                *backpack_exts,
                info={
                    "batch_size": inputs.shape[0],
                    "individual_losses": losses,
                    "loss": loss,
                },
            ):
                loss.backward(create_graph=self.cockpit.create_graph(global_step))
                self.check_in_context()

            self.check_after_context()

            # optimizer step
            opt.step()
            global_step += 1

            if global_step >= self.iterations:
                break

    def extract_problem(self, problem):
        """Return the problem (data, model, loss_fn) given its string.

        Args:
            problem (string): The problem to test on.

        Returns:
            [tuple]: Data, model (and potentially loss_fn) of this problem.
        """
        if problem == "ToyData":
            dataset = ToyData()
            data = DataLoader(dataset, batch_size=self.batch_size)
            model = torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(128, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, 10),
            )
            # loss_fn = torch.nn.CrossEntropyLoss
        return data, model  # , loss_fn

    def check_in_context(self):
        """Check that will be executed within the cockpit context."""
        pass

    def check_after_context(self):
        """Check that will be executed directly after the cockpit context."""
        pass


class ToyData(Dataset):
    """Toy data set used for testing. Consists of small random "images" and labels."""

    def __init__(self):
        """Init the toy data set."""
        super(ToyData, self).__init__()

    def __getitem__(self, index):
        """Return item with index `index` of data set.

        Args:
            index (int): Index of sample to access. Ignored for now.

        Returns:
            [tuple]: Tuple of (random) input and (random) label.
        """
        item_input = torch.rand(2, 8, 8)
        item_label = torch.randint(size=(), low=0, high=10)
        return (item_input, item_label)

    def __len__(self):
        """Length of dataset. Arbitrarily set to 10 000."""
        return 10000  # of how many examples(images?) you have
