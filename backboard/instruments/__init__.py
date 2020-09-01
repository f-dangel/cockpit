"""All Instruments for the Cockpit."""

from backboard.instruments.alpha_gauge import alpha_gauge
from backboard.instruments.alpha_trace_gauge import alpha_trace_gauge
from backboard.instruments.cond_alpha_gauge import cond_alpha_gauge
from backboard.instruments.cond_gauge import cond_gauge
from backboard.instruments.distance_gauge import distance_gauge
from backboard.instruments.hyperparameter_gauge import hyperparameter_gauge
from backboard.instruments.max_ev_gauge import max_ev_gauge
from backboard.instruments.performance_gauge import performance_gauge
from backboard.instruments.trace_gauge import trace_gauge

__all__ = [
    "alpha_gauge",
    "alpha_trace_gauge",
    "cond_alpha_gauge",
    "cond_gauge",
    "distance_gauge",
    "hyperparameter_gauge",
    "max_ev_gauge",
    "performance_gauge",
    "trace_gauge",
]
