"""Utility functions for Cockpit's tests."""

import torch
from backpack import extend
from cockpit import Cockpit
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset


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
        pass

    def check_after_context(self):
        pass


class ToyData(Dataset):
    def __init__(self):
        super(ToyData, self).__init__()

    def __getitem__(self, index):
        item_input = torch.rand(2, 8, 8)
        item_label = torch.randint(size=(), low=0, high=10)
        return (item_input, item_label)

    def __len__(self):
        return 10000  # of how many examples(images?) you have
