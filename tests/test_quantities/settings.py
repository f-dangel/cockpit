"""Settings used by the tests in this submodule."""

import torch

from cockpit.utils.schedules import linear, logarithmic
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
        "iterations": 5,
    },
]

SETTINGS = GLOBAL_SETTINGS + LOCAL_SETTINGS

PROBLEMS, PROBLEMS_IDS = make_problems_with_ids(SETTINGS)

INDEPENDENT_RUNS = [True, False]
INDEPENDENT_RUNS_IDS = [f"independent_runs={run}" for run in INDEPENDENT_RUNS]

CPU_PROBLEMS = []
CPU_PROBLEMS_ID = []
for problem, problem_id in zip(PROBLEMS, PROBLEMS_IDS):
    if "cpu" in str(problem.device):
        CPU_PROBLEMS.append(problem)
        CPU_PROBLEMS_ID.append(problem_id)

QUANTITY_KWARGS = [
    {
        "track_schedule": linear(interval=1, offset=2),  # [1, 3, 5, ...]
        "verbose": True,
    },
    {
        "track_schedule": logarithmic(
            start=0, end=1, steps=4, init=False
        ),  # [1, 2, 4, 10]
        "verbose": True,
    },
]
QUANTITY_KWARGS_IDS = [f"q_kwargs={q_kwargs}" for q_kwargs in QUANTITY_KWARGS]
