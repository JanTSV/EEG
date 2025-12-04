# ----- SKETCH -----
# - plot data to identify noisy channels
# - interpolate noisy channels
# - down-sample to 256 Hz (to make data easier to work with)
# - save
# - toolboxes used by authors: fieltrip

# ----- IMPORTS -----
import pandas as pd
import mne
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("QtAgg")
import warnings
warnings.filterwarnings("ignore", message="QObject::connect")


# ----- PARAMETERS -----
IDENTIFY_BAD_CHANNELS = False        # set to true if you want to identify bad channels (marked in participants.tsv)
INTERPOLATE_BAD_CHANNELS= False     # set to true if you want to fix bad channels
PLOT_DATA = False                    # set to false if you don't want to plot the data
DOWN_SAMPLE_TO_256HZ = True         # set to false if you don't want to down sample

NUM_TRIALS = 480                    # 480 games 
PAIR_IDS = [[1,9],[11,22],[25,34]]  # list of valid pairs (tuple of 2 indexes (1.: start index, 2.: end index))
FS = 2048                           # Biosemi sampling frequency

# ----- CODE -----
def pipeline():
    # Step 1: read demographics
    print("starting pipeline ...")
    demographics = read_table(
        path_to_file = "../data/participants.tsv",
        description = "demographics"
        )
    
    # step 2: plot data
    if PLOT_DATA:
        plot_data()
          
    # step 3: loop over pairs
    valid_pair_ids_list = extract_pair_ids(PAIR_IDS)
    
    for pair in valid_pair_ids_list:
        if pair != 31:
            continue
        print(f'loading pair {pair}')
        
        # step 3.1: get trigger times (for start of each trial)
        print("loading trigger times ...")
        events_filename = f"../data/sub-{pair:02d}/eeg/sub-{pair:02d}_task-RPS_events.tsv"
        events = read_table(
            path_to_file=events_filename,
            description="events"
        )
        onset_sample = events.onset_sample
        onset_sample = np.array(onset_sample, dtype=int)

        # step 3.2: specify epoch: -0.2 to 5 sec
        print("loading matrix ...")
        prestim = 0.2
        poststim = 5
        
        # step 3.3: trial matrix (1. value: beginn sample, 2. value: endsample, 3. value: position of t=0 )
        pre_samps = int(np.ceil(prestim * FS))
        post_samps = int(np.ceil(poststim * FS))
        n_trials = onset_sample.shape[0]
        trl = np.zeros((n_trials, 3), dtype=int)
        trl[:, 0] = onset_sample - pre_samps
        trl[:, 1] = onset_sample + post_samps
        trl[:, 2] = -pre_samps
        
        # step 3.4: read channels
        print("reading channels ...")
        raw_filename = f"../data/sub-{pair:02d}/eeg/sub-{pair:02d}_task-RPS_eeg.bdf"
        raw = mne.io.read_raw_bdf(raw_filename, preload=False)
        channel_names = np.array(raw.ch_names)
        
        # step 3.5: loop over player 1 and 2
        for player in [1,2]:
            print(f"  processing player {player} ...")
            
            if player == 1:
                mask = np.array(
                    [("2-A" in ch) or ("2-B" in ch) for ch in channel_names],
                    dtype=bool
                    )
            else:
                mask = np.array(
                    [("1-A" in ch) or ("1-B" in ch) for ch in channel_names],
                    dtype=bool
                    )
            orig_labels = channel_names[mask]         
            orig_labels_list = orig_labels.tolist()
            print(f"    found {len(orig_labels_list)} channels for player {player}")

            # step 3.5.1: select only channels of player
            raw_player = raw.copy().pick(orig_labels_list)
            
            # step 3.5.2: map to default name of biosemi
            biosemi_montage = mne.channels.make_standard_montage("biosemi64")
            std_ch_names = biosemi_montage.ch_names
            if len(raw_player.ch_names) < 64:
                raise RuntimeError(
                    f"    Player {player}: expected at least 64 channels, "
                    f"    got {len(raw_player.ch_names)}"
                )

            rename_map = {
                raw_player.ch_names[i]: std_ch_names[i]
                for i in range(64)
            }
            raw_player.rename_channels(rename_map)
            raw_player.set_montage(biosemi_montage, match_case=False)
            
            # step 3.5.3: restructure events 
            event_samples = onset_sample - 1  # shape: (n_trials,)
            mne_events = np.column_stack([
                event_samples,
                np.zeros_like(event_samples),
                np.ones_like(event_samples),
            ]).astype(int)
            
            # step 3.5.4: epoching
            epochs = mne.Epochs(
                raw_player,
                mne_events,
                event_id={"decision": 1},
                tmin=-prestim,
                tmax=poststim,
                baseline=None,   # attention: in matlab no baseline correction
                preload=True,
                detrend=None,
            )
            # print(
            #     f"    created epochs for player {player}: "
            #     f"    {len(epochs)} trials, {len(epochs.ch_names)} channels, "
            #     f"    {len(epochs.times)} time points"
            # )
            
            # step 3.5.5: bad channel identification
            if IDENTIFY_BAD_CHANNELS:
                print("    filtering for visual inspection ...")
                epochs_filt = epochs.copy().filter(l_freq=0.1, h_freq=100.0, picks="eeg")
                print("    opening plot window (mark bad channels manually)...")
                epochs_filt.plot(n_epochs=10, n_channels=32, scalings="auto")
            
            # step 3.5.6: 
            if INTERPOLATE_BAD_CHANNELS:
                sub_id = f"sub-{pair:02d}"
                chan_to_fix = demographics.loc[
                    demographics["participant_id"] == sub_id
                ]

                if player == 1:
                    bad_str = chan_to_fix.iloc[0, 6]  # column 7 (as in matlab)
                else:
                    bad_str = chan_to_fix.iloc[0, 11]  # column 12 (as in matlab)

                if isinstance(bad_str, str) and bad_str.strip():
                    bads = [ch.strip() for ch in bad_str.split(",")]
                    print(f"    interpolating bad channels: {bads}")
                    epochs.info["bads"] = bads
                    epochs = epochs.interpolate_bads(reset_bads=True, mode="accurate")
                else:
                    print("    no bad channels listed for this participant.")
                    
            # step 3.5.7: Downsampling to 256 HZ
            if DOWN_SAMPLE_TO_256HZ:
                print("    downsampling to 256 Hz ...")
                epochs = epochs.copy().resample(256.0)
            
            # step 3.5.8: save
            out_dir = Path("../data/derivatives")
            out_dir.mkdir(exist_ok=True, parents=True)
            out_fname = out_dir / f"pair-{pair:02d}_player-{player}_task-RPS_eeg-epo.fif"

            print(f"    saving to {out_fname}")
            epochs.save(out_fname, overwrite=True)

            # Cleanup
            del epochs, raw_player
            
        print("\n")

    print("done")
    
    
def extract_pair_ids(pair_ids: list):
    pair_ids_list = []
    for pair in pair_ids:
        pair_ids_list+= list(range(pair[0], pair[1]+1))
    
    return pair_ids_list
    

def read_table(
    path_to_file:str=None, 
    debug_print:bool=False,
    description:str=None
    ):
    if description:
        print(f"reading {description} ...")
    else:
        print("reading table ...")
    df = pd.read_csv(
        path_to_file,
        sep='\t'
    )
    if debug_print:
        print(df.head(10))
    return df

def plot_data():
    print("plotting data ...")
    mne.set_config('MNE_BROWSER_THEME', 'light', set_env=True)
    raw = mne.io.read_raw_bdf(
        # "D:\EEG\data\sub-31\eeg\sub-31_task-RPS_eeg.bdf",
        "/Users/yannikfruehwirth/Desktop/academia/semester4/EEG/EEG/data/sub-31/eeg/sub-31_task-RPS_eeg.bdf",
        preload=False
    )
    raw.plot(block=True)  # will use the Matplotlib-based viewer
    
    
    
if __name__ == "__main__":
    pipeline()
    
    
    

