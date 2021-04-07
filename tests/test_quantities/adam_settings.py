"""Problem settings using Adam as optimizer.

Some quantities are only defined for zero-momentum SGD (``CABS``, ``EarlyStopping``),
or use a different computation strategy (``Alpha``). This behavior needs to be tested.
"""

import torch

from tests.utils.data import load_toy_data
from tests.utils.models import load_toy_model
from tests.utils.problem import make_problems_with_ids

ADAM_SETTINGS = [
    {
        "data_fn": lambda: load_toy_data(batch_size=4),
        "model_fn": load_toy_model,
        "individual_loss_function_fn": lambda: torch.nn.CrossEntropyLoss(
            reduction="none"
        ),
        "loss_function_fn": lambda: torch.nn.CrossEntropyLoss(reduction="mean"),
        "iterations": 5,
        "optimizer_fn": lambda parameters: torch.optim.Adam(parameters, lr=0.01),
    },
]

ADAM_PROBLEMS, ADAM_IDS = make_problems_with_ids(ADAM_SETTINGS)
