"""Settings used by the tests in this submodule."""

import torch

from tests.settings import SETTINGS as GLOBAL_SETTINGS
from tests.utils.data import load_toy_data
from tests.utils.models import load_toy_model
from tests.utils.problem import make_problems_with_ids

LOCAL_SETTINGS = [
    {
        "data_fn": lambda: load_toy_data(batch_size=5),
        "model_fn": load_toy_model,
        "individual_loss_function_fn": lambda: torch.nn.CrossEntropyLoss(
            reduction="none"
        ),
        "loss_function_fn": lambda: torch.nn.CrossEntropyLoss(reduction="mean"),
        "iterations": 1,
    },
]
SETTINGS = GLOBAL_SETTINGS + LOCAL_SETTINGS

PROBLEMS, PROBLEMS_IDS = make_problems_with_ids(SETTINGS)
