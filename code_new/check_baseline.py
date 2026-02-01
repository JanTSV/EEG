"""
check_baseline_plot.py
----------------------
Diagnosis: Visualizes the effect of "Piecewise Baselining".
Compares "Global Baseline" (standard) vs. "Local Baseline" (Piecewise).
"""

import yaml
import numpy as np
import matplotlib.pyplot as plt
import mne
from pathlib import Path

# Style
plt.rcParams.update({'font.size': 12, 'figure.figsize': (12, 8)})

def load_config():
    with open("code_new/config.yaml", 'r') as f:
        return yaml.safe_load(f)

def apply_piecewise_baseline_no_binning(epochs):
    """
    Applies the piecewise baseline logic but returns the full time series
    (concatenated) instead of bins, so we can plot the ERP traces.
    """
    # Definitions (Same as 04_features.py)
    phases = [
        {'name': 'Decision', 'tmin': -0.2, 'tmax': 2.0, 'base': (-0.2, 0.0), 'crop': (0.0, 2.0)},
        {'name': 'Response', 'tmin': 1.8,  'tmax': 4.0, 'base': (1.8, 2.0),  'crop': (2.0, 4.0)},
        {'name': 'Feedback', 'tmin': 3.8,  'tmax': 5.0, 'base': (3.8, 4.0),  'crop': (4.0, 5.0)}
    ]
    
    data_segments = []
    time_segments = []
    current_time_offset = 0
    
    for p in phases:
        # 1. Crop & Copy
        epo = epochs.copy().crop(tmin=p['tmin'], tmax=p['tmax'], include_tmax=False)
        
        # 2. Apply Local Baseline
        epo.apply_baseline(p['base'], verbose=False)
        
        # 3. Crop to analysis window
        epo.crop(tmin=p['crop'][0], tmax=p['crop'][1], include_tmax=False)
        
        # 4. Get Data
        # Shape: (n_epochs, n_chans, n_times)
        d = epo.get_data(copy=False)
        t = epo.times
        
        data_segments.append(d)
        time_segments.append(t)
        
    # Concatenate along time axis (axis 2)
    full_data = np.concatenate(data_segments, axis=2)
    # Reconstruct time axis for plotting (0 to 5s continuous)
    # Note: There might be slight gaps or overlaps in 't', but for plotting 
    # we just want to see the concatenated result.
    n_samples = full_data.shape[2]
    full_times = np.linspace(0, 5.0, n_samples)
    
    return full_data, full_times

def run_baseline_check():
    cfg = load_config()
    in_dir = Path(cfg['paths']['output_preprocess'])
    out_dir = Path("figures/00_quality_checks")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Use Pair 1, Player 1 as example
    pair_id = 1
    player_num = 1
    
    fif_path = in_dir / f"pair-{pair_id:02d}_player-{player_num}_clean_epo.fif"
    if not fif_path.exists():
        print(f"File not found: {fif_path}")
        return

    print(f"Loading {fif_path}...")
    epochs = mne.read_epochs(fif_path, preload=True, verbose='error')
    
    # Set Average Ref (Important for comparison)
    epochs.set_eeg_reference('average', projection=False, verbose=False)
    
    # --- 1. GET GLOBAL BASELINE TRACE (Standard) ---
    # Just crop 0-5s. It was already baselined at -0.2 in preprocessing step.
    # If not, we baseline here once.
    epochs_global = epochs.copy().crop(tmin=0, tmax=5, include_tmax=False)
    # Optional: Re-apply global baseline to be sure
    # epochs_global.apply_baseline((None, 0)) 
    
    # Calculate ERP (Grand Average over all trials)
    # Shape: (n_chans, n_times)
    erp_global = epochs_global.get_data(copy=False).mean(axis=0)
    times_global = epochs_global.times
    
    # --- 2. GET LOCAL BASELINE TRACE (Piecewise) ---
    data_local, times_local = apply_piecewise_baseline_no_binning(epochs)
    # ERP
    erp_local = data_local.mean(axis=0)
    
    # --- PLOTTING ---
    # Pick a channel that shows visual/motor activity well (e.g., 'Oz' or 'C3' or 'Pz')
    # If we don't know index, pick a few representatives or Average of All Channels (GFP)
    
    # Let's plot the Global Field Power (GFP) - Standard deviation across channels
    gfp_global = np.std(erp_global, axis=0)
    gfp_local = np.std(erp_local, axis=0)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    
    # Plot 1: GFP Comparison
    ax1.plot(times_global, gfp_global, label='Global Baseline (Old)', color='grey', alpha=0.7)
    ax1.plot(times_local, gfp_local, label='Piecewise Baseline (New)', color='blue', linewidth=1.5)
    ax1.set_title(f"Effect of Piecewise Baselining (Subject {pair_id}, Player {player_num})\nGlobal Field Power (GFP)")
    ax1.set_ylabel("Amplitude (GFP)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add markers for cuts
    for t in [2.0, 4.0]:
        ax1.axvline(t, color='red', linestyle=':', label='Cut/Re-Baseline' if t==2.0 else "")

    # Plot 2: Single Channel Example (e.g. Channel 30 - likely Parietal/Occipital)
    # Trying to find a good visual channel. 
    ch_idx = 29 # Arbitrary channel index (approx Pz/Oz usually)
    ch_name = epochs.ch_names[ch_idx]
    
    trace_global = erp_global[ch_idx] * 1e6 # to microvolts
    trace_local = erp_local[ch_idx] * 1e6
    
    ax2.plot(times_global, trace_global, label=f'Global ({ch_name})', color='grey', alpha=0.7)
    ax2.plot(times_local, trace_local, label=f'Piecewise ({ch_name})', color='orange', linewidth=1.5)
    ax2.set_title(f"Single Channel ERP ({ch_name})")
    ax2.set_ylabel("Amplitude (µV)")
    ax2.set_xlabel("Time (s)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add markers
    for t in [2.0, 4.0]:
        ax2.axvline(t, color='red', linestyle=':')
        ax2.text(t, ax2.get_ylim()[1], " Re-Base", color='red', fontsize=8, ha='left', va='top')

    plt.tight_layout()
    out_file = out_dir / "check_baseline_effect.png"
    fig.savefig(out_file, dpi=300)
    print(f"[PLOT] Saved comparison to {out_file}")
    print("Check the plot: Do you see the orange line 'jumping' back to 0 at 2.0s and 4.0s?")

if __name__ == "__main__":
    run_baseline_check()