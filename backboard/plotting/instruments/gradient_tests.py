"""MWE how plotting of gradient tests could look like.

TODO Share axes between lower left plots
TODO Rotate upper right plot clockwise by 90 degrees
TODO Align plots
TODO Integrate into plotter
"""

import random

import matplotlib.pyplot as plt
import numpy

# fake data
random.seed(0)

steps = 10
log = {}

for s in range(steps):
    step_log = {
        "norm_test_radius": 0.8 * random.random(),
        "inner_product_test_width": 0.8 * random.random(),
        "orthogonality_test_width": 0.8 * random.random(),
    }
    log[s] = step_log

# tiling
fig = plt.figure(constrained_layout=True)
gs = fig.add_gridspec(3, 3)

fig_all = fig.add_subplot(gs[1:, 1:])
fig_all.set_title("overview")

fig_norm = fig.add_subplot(gs[1, 0])
fig_norm.set_title("norm")

fig_inner = fig.add_subplot(gs[2, 0])
fig_inner.set_title("inner")

fig_ortho = fig.add_subplot(gs[0, 2])
fig_ortho.set_title("ortho")

# plot overview
fig_all.set_xscale("linear")
fig_all.set_yscale("linear")

x_min, x_max = -1, 1
fig_all.set_xlim([x_min, x_max])
y_min, y_max = 0, 2
fig_all.set_ylim([y_min, y_max])

fig_all.axhline(1, color="black")
fig_all.axvline(0, color="black")

norm_test_radii = [log[s]["norm_test_radius"] for s in range(steps)]
norm_test_circle = plt.Circle((0, 1), norm_test_radii[-1], color="blue", alpha=0.4)
fig_all.add_artist(norm_test_circle)
fig_all.plot(
    [x_min, x_min],
    [1, 1 + norm_test_radii[-1]],
    color="blue",
    linewidth=6,
)


inner_product_test_widths = [log[s]["inner_product_test_width"] for s in range(steps)]
fig_all.axhspan(
    1 - inner_product_test_widths[-1],
    1 + inner_product_test_widths[-1],
    color="red",
    alpha=0.4,
)
fig_all.plot(
    [x_min, x_min],
    [1 - inner_product_test_widths[-1], 1],
    color="red",
    linewidth=6,
)

orthogonality_test_widths = [log[s]["orthogonality_test_width"] for s in range(steps)]
fig_all.axvspan(
    -orthogonality_test_widths[-1],
    orthogonality_test_widths[-1],
    color="green",
    alpha=0.4,
)
fig_all.plot(
    [0, orthogonality_test_widths[-1]],
    [y_max, y_max],
    color="green",
    linewidth=6,
)


steps_array = numpy.arange(0, steps)

# plot norm
fig_norm.fill_between(steps_array, norm_test_radii, color="blue", alpha=0.4)
fig_norm.plot(steps_array, norm_test_radii, color="blue")

# plot inner
fig_inner.fill_between(steps_array, inner_product_test_widths, color="red", alpha=0.4)
fig_inner.plot(steps_array, inner_product_test_widths, color="red")

# plot ortho
fig_ortho.fill_between(steps_array, orthogonality_test_widths, color="green", alpha=0.4)
fig_ortho.plot(steps_array, orthogonality_test_widths, color="green")

plt.show()
