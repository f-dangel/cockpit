"""All Instruments for the Cockpit."""

from backboard.instruments.alpha_gauge import alpha_gauge
from backboard.instruments.distance_gauge import distance_gauge
from backboard.instruments.grad_norm_gauge import grad_norm_gauge
from backboard.instruments.gradient_tests_gauge import gradient_tests_gauge
from backboard.instruments.histogram_1d_gauge import histogram_1d_gauge
from backboard.instruments.histogram_2d_gauge import histogram_2d_gauge
from backboard.instruments.hyperparameter_gauge import hyperparameter_gauge
from backboard.instruments.max_ev_gauge import max_ev_gauge
from backboard.instruments.performance_gauge import performance_gauge
from backboard.instruments.tic_gauge import tic_gauge
from backboard.instruments.trace_gauge import trace_gauge

__all__ = [
    "alpha_gauge",
    "distance_gauge",
    "grad_norm_gauge",
    "histogram_1d_gauge",
    "histogram_2d_gauge",
    "gradient_tests_gauge",
    "hyperparameter_gauge",
    "max_ev_gauge",
    "performance_gauge",
    "tic_gauge",
    "trace_gauge",
]
