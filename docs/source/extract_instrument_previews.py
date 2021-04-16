"""Quick file to extract previews of instruments."""


import os

import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox

import cockpit

HERE_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = "_static"
FILE_NAME = "instrument_preview_run"
SECONDARY_FILE_NAME = FILE_NAME + "_secondary"
SAVE_PATH = "instrument_previews"

full_path = os.path.join(HERE_DIR, FILE_PATH, FILE_NAME)
cp = cockpit.CockpitPlotter(secondary_screen=False)
cp.plot(full_path, show_plot=False, show_log_iter=True, debug=True)

# Store preview images
preview_dict = {
    "Hyperparameters": [[3, 1], [11.2, 4.8]],
    "Performance": [[11.2, 1], [27.8, 4.8]],
    "GradientNorm": [[4, 5], [10.5, 7.75]],
    "Distances": [[4, 7.75], [10.6, 10.25]],
    "Alpha": [[4, 10.15], [10.7, 12.75]],
    "GradientTests": [[11.75, 10.15], [19, 12.75]],
    "HessMaxEV": [[20, 10.15], [26.5, 12.75]],
    "HessTrace": [[20, 7.75], [26.5, 10.25]],
    "TIC": [[20, 5.1], [26.5, 7.75]],
    "Hist1d": [[11.75, 7.75], [19, 10.25]],
    "Hist2d": [[11.75, 5.1], [18.5, 7.75]],
}

for instrument in preview_dict:
    plt.savefig(
        os.path.join(HERE_DIR, FILE_PATH, SAVE_PATH, instrument),
        bbox_inches=Bbox(preview_dict[instrument]),
    )

# plt.savefig(
#     os.path.join(HERE_DIR, FILE_PATH, SAVE_PATH, "cockpit.png"),
#     format="png",
#     bbox_inches=Bbox([[11.75, 5.1], [18.5, 7.75]]),
# )

# Secondary Screen
plt.close("all")
cp = cockpit.CockpitPlotter(secondary_screen=True)
secondary_full_path = os.path.join(HERE_DIR, FILE_PATH, SECONDARY_FILE_NAME)
cp.plot(secondary_full_path, show_plot=False, show_log_iter=False)

# Store preview images
preview_dict = {
    "MeanGSNR": [[3.8, 10.45], [11, 13.15]],
    "CABS": [[11.5, 10.45], [19, 13.15]],
    "EarlyStopping": [[19, 10.45], [26.5, 13.15]],
}

for instrument in preview_dict:
    plt.savefig(
        os.path.join(HERE_DIR, FILE_PATH, SAVE_PATH, instrument),
        bbox_inches=Bbox(preview_dict[instrument]),
    )
