"""
01_preprocessing.py
Replication of Moerel et al., 2025
Preprocessing: Load BIDS, Map Channels, Epoch on 'Decision', Interpolate, Rereference.
"""

import os
import yaml
import numpy as np
import pandas as pd
import mne
from pathlib import Path

# --- STANDARD BIOSEMI 64 to 10-20 MAPPING ---
BIOSEMI_TO_1020 = {
    # Bank A (1-32)
    'A1': 'Fp1', 'A2': 'AF7', 'A3': 'AF3', 'A4': 'F1', 'A5': 'F3', 'A6': 'F5', 'A7': 'F7', 'A8': 'FT7',
    'A9': 'FC5', 'A10': 'FC3', 'A11': 'FC1', 'A12': 'C1', 'A13': 'C3', 'A14': 'C5', 'A15': 'T7', 'A16': 'TP7',
    'A17': 'CP5', 'A18': 'CP3', 'A19': 'CP1', 'A20': 'P1', 'A21': 'P3', 'A22': 'P5', 'A23': 'P7', 'A24': 'P9',
    'A25': 'PO7', 'A26': 'PO3', 'A27': 'O1', 'A28': 'Iz', 'A29': 'Oz', 'A30': 'POz', 'A31': 'Pz', 'A32': 'CPz',
    
    # Bank B (1-32)
    'B1': 'Fp2', 'B2': 'AF8', 'B3': 'AF4', 'B4': 'F2', 'B5': 'F4', 'B6': 'F6', 'B7': 'F8', 'B8': 'FT8',
    'B9': 'FC6', 'B10': 'FC4', 'B11': 'FC2', 'B12': 'C2', 'B13': 'C4', 'B14': 'C6', 'B15': 'T8', 'B16': 'TP8',
    'B17': 'CP6', 'B18': 'CP4', 'B19': 'CP2', 'B20': 'P2', 'B21': 'P4', 'B22': 'P6', 'B23': 'P8', 'B24': 'P10',
    'B25': 'PO8', 'B26': 'PO4', 'B27': 'O2', 
    # Midline & Right Inion fillers for the end of Bank B
    'B28': 'I2',  'B29': 'FCz', 'B30': 'Cz',  'B31': 'AFz', 'B32': 'Fz'
}

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

CONFIG = load_config('code_new/config_preprocessing.yaml') 

def get_pairs_to_process():
    if CONFIG['subjects']['run_mode'] == 'single':
        return [CONFIG['subjects']['single_pair_id']]
    else:
        all_pairs = list(range(CONFIG['subjects']['pair_range_start'], 
                               CONFIG['subjects']['pair_range_end'] + 1))
        return [p for p in all_pairs if p not in CONFIG['subjects']['exclude_pairs']]

def get_bad_channels(pair_id, player_idx, participants_df):
    sub_str = f"sub-{pair_id:02d}"
    row = participants_df[participants_df['participant_id'] == sub_str]
    if row.empty: return []

    try:
        # Index 5 (Player 1) and 9 (Player 2) based on inspection
        cols = row.columns
        bad_col = cols[5] if player_idx == 0 else cols[9]
        bad_str = row[bad_col].values[0]
        
        if pd.isna(bad_str) or str(bad_str).lower() in ['n/a', 'nan', 'none']: return []
        return [ch.strip() for ch in str(bad_str).replace(' ', '').split(',')]
    except Exception as e:
        print(f"    Warning: Error parsing bad channels: {e}")
        return []

def get_events_from_tsv(events_path):
    """
    Reads the BIDS events.tsv.
    Since the file lists every trial, we take 'onset_sample' as the Decision start.
    """
    try:
        df = pd.read_csv(events_path, sep='\t')
        
        # Check if 'onset_sample' exists
        if 'onset_sample' not in df.columns:
            print("    Error: 'onset_sample' column missing in TSV.")
            return None, None

        # Every row is a 'Decision' event (Start of Trial)
        onsets = df['onset_sample'].values
        
        # Create MNE events array: [sample_index, 0, event_id]
        events = np.zeros((len(onsets), 3), dtype=int)
        events[:, 0] = onsets
        events[:, 2] = 1 # Event ID 1 = Decision Start
        
        return events, {'Decision': 1}
        
    except Exception as e:
        print(f"    Error reading events.tsv: {e}")
        return None, None

def run_pipeline():
    data_root = Path(CONFIG['paths']['data_root'])
    output_dir = Path(CONFIG['paths']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    participants_path = data_root / CONFIG['paths']['participants_file']
    participants_df = pd.read_csv(participants_path, sep='\t')
    
    pairs = get_pairs_to_process()
    
    for pair_id in pairs:
        print(f"\n{'='*40}\nProcessing Pair {pair_id}\n{'='*40}")
        
        sub_str = f"sub-{pair_id:02d}"
        eeg_path = data_root / sub_str / 'eeg' / f"{sub_str}_task-RPS_eeg.bdf"
        events_path = data_root / sub_str / 'eeg' / f"{sub_str}_task-RPS_events.tsv"
        
        if not eeg_path.exists():
            print(f"Skipping: File not found {eeg_path}")
            continue
            
        # 1. Load Raw (Stim channel 'Status' excluded from data)
        try:
            raw = mne.io.read_raw_bdf(eeg_path, preload=True, verbose='error', stim_channel='Status')
        except Exception as e:
            print(f"Failed to load BDF: {e}")
            continue

        # 2. Get Events
        events, event_id = get_events_from_tsv(events_path)
        if events is None: continue
        print(f"  -> Found {len(events)} trials.")
        
        # 3. Process Players
        for player_idx in [0, 1]: 
            player_num_output = player_idx + 1
            
            # SWAP Logic
            if CONFIG['processing']['swap_players']:
                target_prefix = '2-' if player_num_output == 1 else '1-'
            else:
                target_prefix = '1-' if player_num_output == 1 else '2-'
                
            print(f"  -> Extracting Player {player_num_output} (Source: {target_prefix}*)")
            
            # Pick & Rename
            player_chans = [ch for ch in raw.ch_names if ch.startswith(target_prefix) and ('-A' in ch or '-B' in ch)]
            raw_player = raw.copy().pick(player_chans)
            
            rename_prefix = {ch: ch.replace(target_prefix, '') for ch in raw_player.ch_names}
            raw_player.rename_channels(rename_prefix)
            
            safe_map = {k: v for k, v in BIOSEMI_TO_1020.items() if k in raw_player.ch_names}
            raw_player.rename_channels(safe_map)
            
            # Montage
            try:
                montage = mne.channels.make_standard_montage(CONFIG['processing']['montage_name'])
                raw_player.set_montage(montage, on_missing='ignore')
            except Exception: pass

            # Epoching
            tmin, tmax = CONFIG['processing']['epoch_tmin'], CONFIG['processing']['epoch_tmax']
            epochs = mne.Epochs(raw_player, events, event_id=event_id, tmin=tmin, tmax=tmax, 
                                baseline=(tmin, 0), preload=True, verbose=False)
            
            # Interpolation
            bads = get_bad_channels(pair_id, player_idx, participants_df)
            if bads:
                existing_bads = [b for b in bads if b in epochs.ch_names]
                if existing_bads:
                    epochs.info['bads'] = existing_bads
                    epochs.interpolate_bads(method=CONFIG['processing']['interpolation_method'], verbose=False)
                    print(f"    Interpolated: {existing_bads}")

            # Rereference (CAR)
            if CONFIG['processing']['apply_rereference']:
                print("    Applying Average Reference")
                epochs.set_eeg_reference('average', projection=False, verbose=False)

            # Resample & Save
            print(f"    Resampling to {CONFIG['processing']['target_sfreq']} Hz")
            epochs.resample(CONFIG['processing']['target_sfreq'])

            out_fname = output_dir / f"pair-{pair_id:02d}_player-{player_num_output}_task-RPS_epo.fif"
            epochs.save(out_fname, overwrite=True, verbose=False)
            print(f"    Saved: {out_fname.name}")

if __name__ == "__main__":
    run_pipeline()