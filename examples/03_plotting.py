"""Example: Only using Cockpit's plotting part, e.g. for a already existing log file."""

import os
import sys

from cockpit import CockpitPlotter

# Basedir, might need to be changed depending on where the results are stored
base = os.path.join(os.path.dirname(os.path.dirname(sys.path[0])), "results")

# File to be plotted, for example:
probpath = "cifar10_3c3d/SGD/"
settingpath = "num_epochs__100__batch_size__128__l2_reg__0.e+00__lr__1.e-02/"
runpath = "random_seed__42__2020-09-24-19-44-01__log"

filepath = os.path.join(base, probpath, settingpath, runpath)

cp = CockpitPlotter(filepath)
cp.plot(block=True, show_log_iter=True)
