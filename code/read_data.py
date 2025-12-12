import matplotlib
matplotlib.use("QtAgg")  # ensures external interactive window

import warnings
warnings.filterwarnings("ignore", message="QObject::connect")  # optional

import mne
mne.set_config('MNE_BROWSER_THEME', 'light', set_env=True)
raw = mne.io.read_raw_bdf(
    # "D:\EEG\data\sub-31\eeg\sub-31_task-RPS_eeg.bdf",
    "D:\EEG\data\sub-01\eeg\sub-01_task-RPS_eeg.bdf",
    preload=False
)
raw.plot(block=True)  # will use the Matplotlib-based viewer