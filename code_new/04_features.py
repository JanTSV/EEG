"""
04_features.py (SCIENTIFIC FIX: LOCAL BASELINE)
-----------------------------------------------
Implements 'Piecewise Baselining' to match MATLAB step2a_decoding.m.
Splits trial into Decision, Response, Feedback.
Re-baselines each part individually to remove drifts.
"""

import yaml
import numpy as np
import pandas as pd
import mne
from pathlib import Path

def load_config():
    with open("code_new/config.yaml", 'r') as f:
        return yaml.safe_load(f)

def get_labels(tsv_path, player_num, target_cfg):
    # (Label logic remains identical - copying for completeness)
    df = pd.read_csv(tsv_path, sep='\t')
    is_p1 = (player_num == 1)
    if target_cfg['source'] == 'self': col = 'player1_resp' if is_p1 else 'player2_resp'
    else: col = 'player2_resp' if is_p1 else 'player1_resp'
    if col not in df.columns: return None
    moves = pd.to_numeric(df[col], errors='coerce').values.astype(float)
    shift = target_cfg['shift']
    if shift > 0:
        moves = np.roll(moves, shift)
        moves[:shift] = np.nan
    return moves

def compute_piecewise_features(epochs, bin_size_sec, sfreq):
    """
    Replicates MATLAB logic:
    1. Cut Decision (0-2s), Baseline [-0.2, 0]
    2. Cut Response (2-4s), Baseline [1.8, 2.0] (Local!)
    3. Cut Feedback (4-5s), Baseline [3.8, 4.0] (Local!)
    4. Average into bins.
    5. Concatenate.
    """
    # Definitions from MATLAB code
    # Part A: Decision (0-2s). 
    #   MATLAB uses -0.2 to 2.0. Baseline -0.2 to 0.
    #   We want bins covering 0.0 to 2.0.
    # Part B: Response (2-4s).
    #   MATLAB uses 1.8 to 4.0. Baseline 1.8 to 2.0 (mapped to -0.2 to 0).
    #   We want bins covering 2.0 to 4.0.
    # Part C: Feedback (4-5s).
    #   MATLAB uses 3.8 to 5.0. Baseline 3.8 to 4.0.
    #   We want bins covering 4.0 to 5.0.
    
    phases = [
        {'name': 'A', 'tmin': -0.2, 'tmax': 2.0, 'base': (-0.2, 0.0), 'crop': (0.0, 2.0)},
        {'name': 'B', 'tmin': 1.8,  'tmax': 4.0, 'base': (1.8, 2.0),  'crop': (2.0, 4.0)},
        {'name': 'C', 'tmin': 3.8,  'tmax': 5.0, 'base': (3.8, 4.0),  'crop': (4.0, 5.0)}
    ]
    
    all_bins = []
    
    for p in phases:
        # 1. Crop larger window to include baseline
        # Note: We must work on a copy to avoid modifying original epochs in loop
        # But 'epochs' is big. Best to use crop on a fresh copy or handle data array.
        # MNE crop is destructive, so we use copy().
        
        # Optimization: We can't easily crop the same epochs object multiple times 
        # because crop() removes data. We need to reload or keep original.
        # Assuming 'epochs' passed here is the full -0.5 to 5.5s raw epoch.
        
        epo_phase = epochs.copy().crop(tmin=p['tmin'], tmax=p['tmax'], include_tmax=False)
        
        # 2. Apply Baseline (Local!)
        # For Part A, base is (-0.2, 0).
        # For Part B, base is (1.8, 2.0). 
        # epo_phase will handle this correctly using absolute times.
        epo_phase.apply_baseline(p['base'], verbose=False)
        
        # 3. Crop to Analysis Window (remove baseline part)
        epo_phase.crop(tmin=p['crop'][0], tmax=p['crop'][1], include_tmax=False)
        
        # 4. Binning
        data = epo_phase.get_data(copy=False) # (n_epochs, n_chans, n_times)
        n_epochs, n_chans, n_samples = data.shape
        
        samples_per_bin = int(bin_size_sec * sfreq)
        n_bins = n_samples // samples_per_bin
        
        # Truncate
        limit = n_bins * samples_per_bin
        data = data[:, :, :limit]
        
        # Reshape & Mean
        data_binned = data.reshape(n_epochs, n_chans, n_bins, samples_per_bin).mean(axis=3)
        all_bins.append(data_binned)
        
    # 5. Concatenate across time (axis 2)
    # A(8 bins) + B(8 bins) + C(4 bins) = 20 bins
    X_final = np.concatenate(all_bins, axis=2)
    
    return X_final

def run_feature_extraction():
    cfg = load_config()
    in_dir = Path(cfg['paths']['output_preprocess'])
    data_root = Path(cfg['paths']['data_root'])
    out_dir = Path("/Volumes/HardDiskYF/ACADEMIA/EEG/ds006761-download/derivatives_new/02_features")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    targets = cfg['decoding']['targets']
    
    # Select Pairs
    if cfg['subjects']['run_mode'] == 'single':
        pairs = [cfg['subjects']['single_pair_id']]
    else:
        all_p = range(cfg['subjects']['pair_range'][0], cfg['subjects']['pair_range'][1] + 1)
        pairs = [p for p in all_p if p not in cfg['subjects']['exclude_pairs']]

    print(f"--- EXTRACTING FEATURES (SCIENTIFIC: LOCAL BASELINE) ---")

    for pair_id in pairs:
        sub_str = f"sub-{pair_id:02d}"
        tsv_path = data_root / sub_str / 'eeg' / f"{sub_str}_task-RPS_events.tsv"
        
        for player_num in [1, 2]:
            fif_path = in_dir / f"pair-{pair_id:02d}_player-{player_num}_clean_epo.fif"
            if not fif_path.exists(): continue
            
            print(f"Pair {pair_id} Player {player_num}...")
            # Load Epochs
            # IMPORTANT: We need pre-baseline data (-0.2s) available.
            # 01_preprocess.py usually saves -0.2 to 5.0. Check if tmin is enough.
            epochs = mne.read_epochs(fif_path, preload=True, verbose='error')
            
            # --- CRITICAL STEP: Set Average Reference ---
            # Paper: "Re-referenced to the average reference"
            # We do it here to be safe, if not done in 01_preprocess
            epochs.set_eeg_reference('average', projection=False, verbose=False)
            
            # Check if we have enough pre-stim data for baseline
            if epochs.tmin > -0.2:
                print(f"  [WARN] Epochs start at {epochs.tmin}s. Need -0.2s for baseline!")
                # In this case, we proceed but baseline might fail or be suboptimal.
            
            # --- COMPUTE FEATURES WITH LOCAL BASELINE ---
            try:
                X = compute_piecewise_features(epochs, cfg['decoding']['time_bin_size'], epochs.info['sfreq'])
            except Exception as e:
                print(f"  [ERROR] Feature computation failed: {e}")
                continue
            
            # (Rest is identical: Loop targets, save)
            for tgt in targets:
                y_raw = get_labels(tsv_path, player_num, tgt)
                if y_raw is None: continue

                valid_mask = (~np.isnan(y_raw)) & (y_raw != 0)
                X_clean = X[valid_mask]
                y_clean = y_raw[valid_mask]
                
                prefix = f"pair-{pair_id:02d}_player-{player_num}_target-{tgt['name']}"
                np.save(out_dir / f"{prefix}_X.npy", X_clean)
                np.save(out_dir / f"{prefix}_y.npy", y_clean)
                print(f"  -> {tgt['name']}: {len(y_clean)} trials")

if __name__ == "__main__":
    run_feature_extraction()