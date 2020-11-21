"""All Instruments for the Cockpit."""

from cockpit.instruments.alpha_gauge import alpha_gauge
from cockpit.instruments.cabs_gauge import cabs_gauge
from cockpit.instruments.distance_gauge import distance_gauge
from cockpit.instruments.early_stopping_gauge import early_stopping_gauge
from cockpit.instruments.grad_norm_gauge import grad_norm_gauge
from cockpit.instruments.gradient_tests_gauge import gradient_tests_gauge
from cockpit.instruments.histogram_1d_gauge import histogram_1d_gauge
from cockpit.instruments.histogram_2d_gauge import histogram_2d_gauge
from cockpit.instruments.hyperparameter_gauge import hyperparameter_gauge
from cockpit.instruments.max_ev_gauge import max_ev_gauge
from cockpit.instruments.mean_gsnr_gauge import mean_gsnr_gauge
from cockpit.instruments.performance_gauge import performance_gauge
from cockpit.instruments.tic_gauge import tic_gauge
from cockpit.instruments.trace_gauge import trace_gauge

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
    "mean_gsnr_gauge",
    "early_stopping_gauge",
    "cabs_gauge",
]
