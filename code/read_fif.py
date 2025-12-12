import mne
from pathlib import Path
import matplotlib

matplotlib.use("QtAgg")
# point to your derivatives directory
deriv_dir = Path("data/derivatives")  # adjust if different
fname = deriv_dir / "pair-01_player-1_task-RPS_eeg-epo.fif"

mne.set_config('MNE_BROWSER_THEME', 'light', set_env=True)
epochs = mne.read_epochs(fname, preload=True)
print(epochs)
# access data: epochs.get_data()  # shape (n_epochs, n_channels, n_times)
epochs.plot(block=True)  # interactive browser