"""Class for problem definition used in the tests."""
import copy

import torch


class instantiate:
    """Instantiates the objects in a problem from their functional representation."""

    def __init__(self, problem):
        """Store problem configuration."""
        self.problem = problem

    def __enter__(self):
        """Set up the problem."""
        self.problem.set_up()

    def __exit__(self, type, value, traceback):
        """Tear down the problem."""
        self.problem.tear_down()


class Problem:
    """Class for encapsulating information that specifies a training loop."""

    def __init__(
        self,
        data_fn,
        model_fn,
        individual_loss_function_fn,
        loss_function_fn,
        optimizer_fn,
        iterations,
        device,
        seed,
        id_prefix,
    ):
        """Collection of information required to test extensions.

        Args:
            data_fn (callable): Function that creates a ``DataLoader`` instance.
            model_fn (callable): Function returning the network.
            individual_loss_function_fn (callable): Function returning the individual
                loss module.
            loss_function_fn (callable): Function returning the loss module.
            optimizer_fn (callable): Function returning the optimizer that operates on
                the parameters of ``model_fn()``.
            iterations (int): Number of optimization steps.
            device (torch.device): Device to run on.
            seed (int): Random seed to set before instantiating.
            id_prefix (str): Extra string added to test id.
        """
        self.data_fn = data_fn
        self.model_fn = model_fn
        self.individual_loss_function_fn = individual_loss_function_fn
        self.loss_function_fn = loss_function_fn
        self.optimizer_fn = optimizer_fn
        self.iterations = iterations
        self.device = device
        self.seed = seed
        self.id_prefix = id_prefix

    def set_up(self):
        """Instantiate the problem deterministically."""
        torch.manual_seed(self.seed)

        self.data = self.data_fn()

        self.model = self.model_fn().to(self.device)
        self.individual_loss_function = self.individual_loss_function_fn().to(
            self.device
        )
        self.loss_function = self.loss_function_fn().to(self.device)

        trainable_parameters = (p for p in self.model.parameters() if p.requires_grad)
        self.optimizer = self.optimizer_fn(trainable_parameters)

    def tear_down(self):
        """Delete all instances created in ``set_up``."""
        del self.data, self.model, self.loss_function, self.optimizer

    def make_id(self):
        """Return a human-readable id."""
        self.set_up()

        prefix = (self.id_prefix + "-") if self.id_prefix != "" else ""

        id_str = (
            (
                prefix
                + f"device={self.device}"
                + f"-data={self.data}"
                + f"-model={self.model}"
                + f"-individual-loss={self.individual_loss_function}"
                + f"-loss={self.loss_function}"
                + f"-optimizer={self.optimizer}"
            )
            .replace(" ", "")
            .replace("\n", "")
        )

        self.tear_down()

        return id_str


def make_problems_with_ids(settings):
    """Convert settings to problems and ids on all available devices.

    Add default for unspecified entries.

    Args:
        settings (list): List of settings to test.

    Returns:
        list: List of problems.
        str: Humanly readable ID of the problem.
    """
    problems = make_problems(settings)
    problems_ids = [p.make_id() for p in problems]

    return problems, problems_ids


def make_problems(settings):
    """Convert settings to problems on all available devices.

    Add default for unspecified entries.

    Args:
        settings (list): List of settings to test.

    Returns:
        list: List of problems.
    """
    problem_dicts = []

    for setting in settings:
        setting = add_missing_defaults(setting)
        devices = setting["device"]

        for dev in devices:
            problem = copy.deepcopy(setting)
            problem["device"] = dev
            problem_dicts.append(problem)

    return [Problem(**p) for p in problem_dicts]


def add_missing_defaults(setting):
    """Fill up missing values in specified setting with their default."""
    required = [
        "data_fn",
        "model_fn",
        "loss_function_fn",
        "individual_loss_function_fn",
    ]
    optional = {
        "id_prefix": "",
        "seed": 0,
        "device": get_available_devices(),
        "optimizer_fn": lambda parameters: torch.optim.SGD(parameters, lr=0.01),
        "iterations": 5,
    }

    for req in required:
        if req not in setting.keys():
            raise ValueError("Missing configuration entry for {}".format(req))

    for opt, default in optional.items():
        if opt not in setting.keys():
            setting[opt] = default

    for s in setting.keys():
        if s not in required and s not in optional.keys():
            raise ValueError("Unknown config: {}".format(s))

    return setting


def get_available_devices():
    """Return CPU and, if present, GPU device.

    Returns:
        [torch.device]: Available devices for `torch`.
    """
    devices = [torch.device("cpu")]

    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))

    return devices
