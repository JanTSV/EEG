import sys
import click
from pathlib import Path
import numpy as np

import mne
from mne_bids import (BIDSPath, read_raw_bids)
from ccs_eeg_utils import read_annotations_core
from matplotlib import pyplot as plt

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
    print(f"min-y: {np.min(raw[channel, :][0].T) * 1e6:.2f} µV, max-y: {np.max(raw[channel, :][0].T) * 1e6:.2f} µV")

    # plot the 10th channel over the whole recorded time
    plt.plot(raw[channel, :][0].T)  # take the first row of `data`
    #plt.plot(raw[10,:][0].T)
    plt.xlabel("Time (s)")
    plt.ylabel("EEG (V)")
    plt.title(f"Channel {channel} EEG")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    raw = raw.pick(["Cz"])
    print(raw.annotations)

    evts, evts_dict = mne.events_from_annotations(raw)
    print(evts)

    # get all keys which contain "stimulus"
    wanted_keys = [e for e in evts_dict.keys() if "stimulus" in e]
    # subset the large event-dictionairy
    evts_dict_stim = dict((k, evts_dict[k]) for k in wanted_keys if k in evts_dict)

    epochs = mne.Epochs(raw, evts, evts_dict_stim, tmin=-0.1, tmax=1)

    # extract  data
    data = epochs.get_data()
    times = epochs.times
    
    # plot epochs
    plt.title(f"Epochs")
    plt.xlabel("Time (s)")
    plt.ylabel("EEG (µV)")
    plt.grid(True)
    plt.tight_layout()
    for i, epoch in enumerate(data):
        data = epoch[0] * 1e6
        print(f"min-y: {np.min(data):.2f} µV, max-y: {np.max(data):.2f} µV")
        plt.plot(times, data, label=f"Epoch {i + 1}", alpha=0.5)
    plt.legend()
    plt.show()

    # but which epochs belong to targets and which to distractors?
    target = ["stimulus:{}{}".format(k,k) for k in [1,2,3,4,5]]
    distractor = ["stimulus:{}{}".format(k,j) for k in [1,2,3,4,5] for j in [1,2,3,4,5] if k!=j]
    
    target = epochs[target].average()
    distractor = epochs[distractor].average()

    mne.viz.plot_compare_evokeds(
        {"Target": target, "Distractor": distractor},
        picks="Cz",
        title="ERP: Target vs Distractor",
        show=True
    )

    # data is still in volts. print min and max again
    data = target.data * 1e6
    print(f"Target: min-y: {np.min(data):.2f} µV, max-y: {np.max(data):.2f} µV")
    data = distractor.data * 1e6
    print(f"Distractor: min-y: {np.min(data):.2f} µV, max-y: {np.max(data):.2f} µV")


if __name__ == "__main__":
    main()
