"""
01_preprocess.py (FIXED)
------------------------
Step 1 of the Pipeline: Signal Processing & Cleaning.
Includes fixes for:
- Non-EEG channels (GSR, Temp, etc.) -> set to 'misc'
- Robust config loading
- Montage setting (best effort)
"""

import yaml
import numpy as np
import pandas as pd
import mne
from pathlib import Path

# Import our visualization module
import sys
sys.path.append('code_new') # Ensure we can import local modules
from visualize import plot_raw_quality, plot_epochs_quality

def load_config():
    possible_paths = [
        "code_new/config.yaml", 
    ]
    for p in possible_paths:
        if Path(p).exists():
            with open(p, 'r') as f: return yaml.safe_load(f)
    raise FileNotFoundError("Config file not found")

def get_events_from_tsv(tsv_path, config_col_name):
    """Reads BIDS events.tsv and extracts the onset samples."""
    try:
        df = pd.read_csv(tsv_path, sep='\t')
        if config_col_name not in df.columns:
            print(f"    [ERROR] Column '{config_col_name}' missing in events.tsv")
            return None, None
        
        onsets = df[config_col_name].values
        # MNE events: [sample, 0, id]
        events = np.zeros((len(onsets), 3), dtype=int)
        events[:, 0] = onsets
        events[:, 2] = 1 # ID 1 for 'Decision'
        return events, {'Decision': 1}
    except Exception as e:
        print(f"    [ERROR] Reading events.tsv: {e}")
        return None, None

def get_bads_from_tsv(participants_path, pair_id, player_idx):
    """Parses participants.tsv for bad channels."""
    try:
        df = pd.read_csv(participants_path, sep='\t')
        sub_id = f"sub-{pair_id:02d}"
        row = df[df['participant_id'] == sub_id]
        if row.empty: return []
        
        # Heuristic: Column 5 (P1) and 9 (P2) based on previous inspection
        col_idx = 5 if player_idx == 0 else 9
        bad_str = row.iloc[0, col_idx]
        
        if pd.isna(bad_str) or str(bad_str).lower() in ['n/a', 'nan', 'none']:
            return []
        return [ch.strip() for ch in str(bad_str).replace(' ', '').split(',')]
    except Exception as e:
        print(f"    [WARN] parsing bads: {e}")
        return []

def run_preprocessing():
    cfg = load_config()
    
    # Paths
    in_dir = Path(cfg['paths']['output_raw'])       # From Step 0 (FIXED PATH NAME)
    out_dir = Path(cfg['paths']['output_preprocess'])   # To Step 1
    out_dir.mkdir(parents=True, exist_ok=True)
    
    raw_data_root = Path(cfg['paths']['data_root']) # Needed for .tsv files
    
    # Subjects
    if cfg['subjects']['run_mode'] == 'single':
        pairs = [cfg['subjects']['single_pair_id']]
    else:
        all_p = range(cfg['subjects']['pair_range'][0], cfg['subjects']['pair_range'][1] + 1)
        pairs = [p for p in all_p if p not in cfg['subjects']['exclude_pairs']]

    print(f"--- STARTING STEP 1: PREPROCESSING for {len(pairs)} Pair(s) ---")

    for pair_id in pairs:
        print(f"\nProcessing Pair {pair_id}...")
        
        part_tsv = raw_data_root / "participants.tsv"
        
        for player_num in [1, 2]:
            fname_in = in_dir / f"pair-{pair_id:02d}_player-{player_num}_raw.fif"
            
            if not fname_in.exists():
                print(f"  [SKIP] Input not found: {fname_in}")
                continue
                
            print(f"  -> Player {player_num}")
            
            # 1. Load Data
            raw = mne.io.read_raw_fif(fname_in, preload=True, verbose='error')
            
            # --- FIX: HANDLE NON-EEG CHANNELS ---
            # Set GSR, Temp, etc. to 'misc' so they are excluded from CAR/Filtering
            non_eeg_channels = ['GSR1', 'GSR2', 'Erg1', 'Erg2', 'Resp', 'Plet', 'Temp']
            existing_chans = raw.ch_names
            ch_types = {}
            for ch in existing_chans:
                for bad in non_eeg_channels:
                    if bad in ch:
                        ch_types[ch] = 'misc'
            
            if ch_types:
                # print(f"    [INFO] Setting {len(ch_types)} channels to 'misc' (GSR, etc.)")
                try:
                    raw.set_channel_types(ch_types)
                except Exception as e:
                    print(f"    [WARN] Could not set channel types: {e}")

            # --- FIX: MONTAGE (Best Effort) ---
            try:
                # Try standard 10-20. If channels are named A1..A64, this might do nothing or warn.
                # But it's better than nothing.
                raw.set_montage('standard_1020', on_missing='ignore')
            except:
                pass 

            # 2. [VISUALIZATION] Raw State
            plot_raw_quality(raw, pair_id, player_num, step_name="00_raw_input")
            
            # 3. Load Events
            sub_str = f"sub-{pair_id:02d}"
            events_tsv = raw_data_root / sub_str / 'eeg' / f"{sub_str}_task-RPS_events.tsv"
            events, event_id = get_events_from_tsv(events_tsv, cfg['step1_preprocessing']['epoching']['event_name_in_tsv'])
            
            if events is None:
                print("    [SKIP] No events found.")
                continue

            # 4. Interpolation
            bads = get_bads_from_tsv(part_tsv, pair_id, player_num - 1)
            
            if bads:
                if raw.get_montage():
                    valid_chans = raw.get_montage().ch_names
                    missing = [ch for ch in raw.ch_names if ch not in valid_chans]
                    # Only drop if we are strictly using a montage, otherwise skip this check
                    pass 
                
                present_bads = [b for b in bads if b in raw.ch_names]
                if present_bads:
                    raw.info['bads'] = present_bads
                    print(f"    [INTERP] Interpolating: {present_bads}")
                    raw.interpolate_bads(method=cfg['step1_preprocessing']['interpolation']['method'], verbose='error')
            
            # 5. Rereference (CAR)
            if cfg['step1_preprocessing']['rereference']['apply']:
                print("    [REF] Applying Common Average Reference (CAR)")
                # 'misc' channels are automatically excluded here!
                raw.set_eeg_reference(cfg['step1_preprocessing']['rereference']['type'], projection=False, verbose='error')

            # 6. Filtering (Skipped per Config)
            if cfg['step1_preprocessing']['filter']['apply']:
                pass 

            # 7. Epoching
            tmin = cfg['step1_preprocessing']['epoching']['tmin']
            tmax = cfg['step1_preprocessing']['epoching']['tmax']
            baseline = tuple(cfg['step1_preprocessing']['epoching']['baseline'])
            
            epochs = mne.Epochs(raw, events, event_id=event_id, tmin=tmin, tmax=tmax, 
                                baseline=baseline, preload=True, verbose='error')
            
            # 8. Resample
            sfreq = cfg['step1_preprocessing']['resample']['sfreq']
            print(f"    [RESAMPLE] Downsampling to {sfreq} Hz")
            epochs.resample(sfreq)

            # 9. [VISUALIZATION] Final State
            plot_epochs_quality(epochs, pair_id, player_num, step_name="01_final_clean")

            # 10. Save
            fname_out = out_dir / f"pair-{pair_id:02d}_player-{player_num}_clean_epo.fif"
            epochs.save(fname_out, overwrite=True, verbose='error')
            print(f"    [SAVE] Done: {fname_out.name}")

if __name__ == "__main__":
    run_preprocessing()