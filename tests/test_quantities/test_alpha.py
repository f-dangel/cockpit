"""Compare ``Alpha`` and ``AlphaGeneral`` quantities with ``torch.autograd``."""

import pytest

from cockpit.context import get_individual_losses
from cockpit.quantities import Alpha, AlphaGeneral, AlphaTwoStep
from cockpit.quantities.alpha import _exact_variance, _projected_gradient
from tests.test_quantities.settings import (
    INDEPENDENT_RUNS,
    INDEPENDENT_RUNS_IDS,
    PROBLEMS,
    PROBLEMS_IDS,
    QUANTITY_KWARGS,
    QUANTITY_KWARGS_IDS,
)
from tests.test_quantities.utils import autograd_individual_gradients, get_compare_fn


class AutogradAlphaGeneral(AlphaGeneral):
    """``torch.autograd`` implementation of ``AlphaGeneral``."""

    def extensions(self, global_step):
        """Return list of BackPACK extensions required for the computation.

        Args:
            global_step (int): The current iteration number.

        Returns:
            list: (Potentially empty) list with required BackPACK quantities.
        """
        return []

    def create_graph(self, global_step):
        """Return whether access to the forward pass computation graph is needed.

        Args:
            global_step (int): The current iteration number.

        Returns:
            bool: ``True`` if the computation graph shall not be deleted,
                else ``False``.
        """
        return any(self._is_position(global_step, pos) for pos in ["start", "end"])

    # TODO Use parent class track method after refactoring it
    def track(self, global_step, params, batch_loss):
        """Evaluate the current parameter distances.

        We track both the distance to the initialization, as well as the size of
        the last parameter update.

        Args:
            global_step (int): The current iteration number.
            params ([torch.Tensor]): List of torch.Tensors holding the network's
                parameters.
            batch_loss (torch.Tensor): Mini-batch loss from current step.
        """
        if self._is_position(global_step, pos="end"):
            self._end_info = self._fetch_values(params, batch_loss, "end", global_step)

            alpha = self._compute_alpha()

            if self._verbose:
                print(f"[Step {global_step}] Alpha: {alpha:.4f}")

            self.output[global_step - self._start_end_difference] = alpha
            self.clear_info()

        if self._is_position(global_step, pos="start"):
            self._start_info = self._fetch_values(
                params, batch_loss, "start", global_step
            )

    def _fetch_values(self, params, batch_loss, pos, global_step):
        """Fetch values for quadratic fit. Return as dictionary.

        The entry "search_dir" is only initialized if ``pos`` is ``"start"``.
        """
        info = {}

        if pos in ["start", "end"]:
            # 0ᵗʰ order info
            info["f"] = batch_loss.item()
            losses = get_individual_losses(global_step)
            info["var_f"] = losses.var().item()

            # temporary information required to compute quantities used in fit
            info["params"] = {id(p): p.data.clone().detach() for p in params}
            info["grad"] = {id(p): p.grad.data.clone().detach() for p in params}
            batch_grad = autograd_individual_gradients(losses, params)
            info["batch_grad"] = {
                id(p): bg.data.clone().detach() for p, bg in zip(params, batch_grad)
            }

        else:
            raise ValueError(f"Invalid position '{pos}'. Expect {self._positions}.")

        # compute all quantities used in fit
        # TODO Restructure base class and move to other function
        if pos == "end":
            start_params, _ = self._get_info("params", end=False)
            end_params = info["params"]

            search_dir = [
                end_params[key] - start_params[key] for key in start_params.keys()
            ]

            for info_dict in [self._start_info, info]:
                grad = [info_dict["grad"][key] for key in start_params.keys()]
                batch_grad = [
                    info_dict["batch_grad"][key] for key in start_params.keys()
                ]

                # 1ˢᵗ order info
                info_dict["df"] = _projected_gradient(grad, search_dir)
                info_dict["var_df"] = _exact_variance(batch_grad, search_dir)

        return info


@pytest.mark.parametrize("problem", PROBLEMS, ids=PROBLEMS_IDS)
@pytest.mark.parametrize("independent_runs", INDEPENDENT_RUNS, ids=INDEPENDENT_RUNS_IDS)
@pytest.mark.parametrize("q_kwargs", QUANTITY_KWARGS, ids=QUANTITY_KWARGS_IDS)
def test_alpha(problem, independent_runs, q_kwargs):
    """Compare BackPACK and ``torch.autograd`` implementation of Alpha.

    Args:
        problem (tests.utils.Problem): Settings for train loop.
        independent_runs (bool): Whether to use to separate runs to compute the
            output of every quantity.
        q_kwargs (dict): Keyword arguments handed over to both quantities.
    """
    compare_fn = get_compare_fn(independent_runs)
    compare_fn(problem, (Alpha, AutogradAlphaGeneral), q_kwargs)


@pytest.mark.parametrize("problem", PROBLEMS, ids=PROBLEMS_IDS)
@pytest.mark.parametrize("independent_runs", INDEPENDENT_RUNS, ids=INDEPENDENT_RUNS_IDS)
@pytest.mark.parametrize("q_kwargs", QUANTITY_KWARGS, ids=QUANTITY_KWARGS_IDS)
def test_alpha_general(problem, independent_runs, q_kwargs):
    """Compare BackPACK and ``torch.autograd`` implementation of AlphaGeneral.

    Args:
        problem (tests.utils.Problem): Settings for train loop.
        independent_runs (bool): Whether to use to separate runs to compute the
            output of every quantity.
        q_kwargs (dict): Keyword arguments handed over to both quantities.
    """
    compare_fn = get_compare_fn(independent_runs)
    compare_fn(problem, (AlphaGeneral, AutogradAlphaGeneral), q_kwargs)


@pytest.mark.parametrize("problem", PROBLEMS, ids=PROBLEMS_IDS)
@pytest.mark.parametrize("independent_runs", INDEPENDENT_RUNS, ids=INDEPENDENT_RUNS_IDS)
@pytest.mark.parametrize("q_kwargs", QUANTITY_KWARGS, ids=QUANTITY_KWARGS_IDS)
def test_alpha_two_step(problem, independent_runs, q_kwargs):
    """Compare BackPACK and ``torch.autograd`` implementation of AlphaTwoStep.

    Args:
        problem (tests.utils.Problem): Settings for train loop.
        independent_runs (bool): Whether to use to separate runs to compute the
            output of every quantity.
        q_kwargs (dict): Keyword arguments handed over to both quantities.
    """
    compare_fn = get_compare_fn(independent_runs)
    compare_fn(problem, (AlphaTwoStep, AutogradAlphaGeneral), q_kwargs)
