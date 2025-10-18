import sys
import click
from pathlib import Path
from mne_bids import (BIDSPath,read_raw_bids)
from ccs_eeg_utils import read_annotations_core
from matplotlib import pyplot as plt
import numpy as np

# import ccs_eeg_utils.py
sys.path.insert(0, '.')

@click.command()
@click.option("--bids", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path), help="Path to bids/ dir.")
@click.option("--channel", required=False, type=int, default=10, help="EEG channel that should be analyzed.")
def main(bids, channel):
    subject_id = '030'
    
    # Path to data set and its format
    bids_path = BIDSPath(subject=subject_id, task="P3", session="P3", datatype='eeg', suffix='eeg', root=bids)

    # read the file
    raw = read_raw_bids(bids_path)
    print(raw.info)

    # fix the annotations readin
    read_annotations_core(bids_path, raw)

    # https://mne.tools/stable/generated/mne.Info.html -> sfreq is the sampling frequency
    print(f"Sampling frequency: {raw.info['sfreq']} Hz")

    # https://mne.tools/stable/generated/mne.io.Raw.html -> raw data is in SI units, so EEG uses volts
    data, times = raw[channel, :]
    data = data * 1e6
    print(f"min-y: {np.min(data):.2f} µV, max-y: {np.max(data):.2f} µV")

    # plot the 10th channel over the whole recorded time
    plt.plot(times, data[0])  # take the first row of `data`
    plt.xlabel("Time (s)")
    plt.ylabel("EEG (µV)")
    plt.title(f"Channel {channel} EEG")
    plt.show()

if __name__ == "__main__":
    main()