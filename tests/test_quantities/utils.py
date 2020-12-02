import torch

from backpack import extend
from cockpit import Cockpit


def make_small_mlp():
    """A simple MLP with ReLU activation functions for testing."""
    return torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(128, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 10),
    )


def classification_targets(N, num_classes):
    """Create random targets for classes 0, ..., `num_classes - 1`."""
    return torch.randint(size=(N,), low=0, high=num_classes)


def train_small_mlp(
    iterations,
    quantities,
    use_backpack,
    batch_size=5,
    opt_cls=torch.optim.SGD,
    opt_kwargs=None,
):
    """Train a small MLP with cockpit."""
    torch.manual_seed(0)

    if opt_kwargs is None:
        if opt_cls == torch.optim.SGD:
            opt_kwargs = {"lr": 0.01}
        else:
            raise NotImplementedError

    # model
    model = make_small_mlp()
    lossfunc = torch.nn.CrossEntropyLoss(reduction="mean")
    individual_lossfunc = torch.nn.CrossEntropyLoss(reduction="none")

    if use_backpack:
        model = extend(model)
        lossfunc = extend(lossfunc)

    # cockpit
    cockpit = Cockpit(model.parameters(), quantities=quantities)

    # optimizer
    opt = opt_cls(model.parameters(), **opt_kwargs)

    # data
    def next_batch():
        inputs = torch.rand(batch_size, 2, 8, 8)
        labels = classification_targets(batch_size, num_classes=10)

        return inputs, labels

    # train loop
    for iteration in range(iterations):
        inputs, labels = next_batch()

        opt.zero_grad()

        outputs = model(inputs)
        loss = lossfunc(outputs, labels)
        losses = individual_lossfunc(outputs, labels).detach()

        with cockpit(
            iteration,
            info={
                "batch_size": inputs.shape[0],
                "individual_losses": losses,
                "loss": loss,
            },
        ):
            loss.backward(create_graph=cockpit.create_graph(iteration))

        opt.step()

    return cockpit
