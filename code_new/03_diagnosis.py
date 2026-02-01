"""
03_diagnosis.py
----------------
PhD-Level Diagnostic Dashboard.
Generates high-resolution scientific plots to analyze:
1. Phase-locked activity (Butterfly + Phases)
2. Region-specific physiology (ROIs)
3. Spatial evolution (Topo Series)
4. Trial-by-trial consistency (ERP Image)
"""

import yaml
import numpy as np
import matplotlib.pyplot as plt
import mne
from pathlib import Path

# Set scientific plotting style
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16
})

def load_config():
    with open("code_new/config.yaml", 'r') as f:
        return yaml.safe_load(f)

CONFIG = load_config()

def apply_viz_filter(epochs):
    """Applies a temporary filter for visualization if configured."""
    filt_cfg = CONFIG['diagnosis']['visual_filter']
    if filt_cfg['apply']:
        print(f"    [VIZ] Applying temp filter ({filt_cfg['l_freq']}-{filt_cfg['h_freq']} Hz)...")
        # Return a copy to strictly avoid modifying original data
        return epochs.copy().filter(filt_cfg['l_freq'], filt_cfg['h_freq'], verbose='error')
    return epochs

def save_plot(fig, pair_id, player_id, plot_type):
    out_dir = Path(CONFIG['paths']['figures']) / "02_diagnosis"
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"pair-{pair_id:02d}_p{player_id}_{plot_type}.png"
    fig.savefig(out_dir / fname, dpi=300, bbox_inches='tight')
    print(f"    [SAVE] {fname}")
    plt.close(fig)

# --- PLOT 1: PHASE BUTTERFLY ---
def plot_phase_butterfly(evoked, pair_id, player_id):
    """
    Overlays all channels (Butterfly) and marks task phases.
    Helps identify if artifacts (e.g. movement) dominate specific phases.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot GFP (Global Field Power) - The "Energy" of the brain
    gfp = evoked.data.std(axis=0)
    times = evoked.times
    
    # Plot all channels in grey
    ax.plot(times, evoked.data.T * 1e6, color='grey', alpha=0.3, linewidth=0.5)
    
    # Plot GFP in black
    ax.plot(times, gfp * 1e6, color='black', linewidth=2, label='Global Field Power (GFP)')
    
    # Add Phase Markers
    phases = CONFIG['diagnosis']['phases']
    colors = {'Decision': 'green', 'Response': 'orange', 'Feedback': 'purple'}
    
    for phase, t_start in phases.items():
        ax.axvline(x=t_start, color=colors.get(phase, 'blue'), linestyle='--', alpha=0.8)
        ax.text(t_start + 0.05, ax.get_ylim()[1]*0.9, phase, 
                color=colors.get(phase, 'blue'), fontweight='bold', rotation=0)

    ax.set_title(f"Phase Analysis (Butterfly) - Pair {pair_id} Player {player_id}")
    ax.set_xlabel("Time (s) relative to Decision Start")
    ax.set_ylabel("Amplitude (µV)")
    ax.legend(loc='upper right')
    ax.grid(True, linestyle=':', alpha=0.5)
    
    save_plot(fig, pair_id, player_id, "01_phases_butterfly")

# --- PLOT 2: ROI TRACES ---
def plot_roi_traces(evoked, pair_id, player_id):
    """
    Plots average signals for specific anatomical regions (ROIs).
    Separates Vision (Occipital) from Movement (Motor) and Artifacts (Frontal).
    """
    rois = CONFIG['diagnosis']['regions_of_interest']
    n_rois = len(rois)
    
    fig, axes = plt.subplots(n_rois, 1, figsize=(10, 3*n_rois), sharex=True)
    if n_rois == 1: axes = [axes]
    
    for ax, (roi_name, channels) in zip(axes, rois.items()):
        # Check which channels exist
        valid_chans = [ch for ch in channels if ch in evoked.ch_names]
        if not valid_chans:
            continue
            
        # Get data for ROI
        roi_data = evoked.pick_channels(valid_chans).data * 1e6 # to µV
        mean_trace = roi_data.mean(axis=0)
        
        # Plot individual channels thin
        for trace, ch_name in zip(roi_data, valid_chans):
            ax.plot(evoked.times, trace, alpha=0.3, linewidth=1, label=ch_name)
            
        # Plot ROI mean thick
        ax.plot(evoked.times, mean_trace, color='black', linewidth=2, label='ROI Mean')
        
        # Add Phase Lines
        for t in CONFIG['diagnosis']['phases'].values():
            ax.axvline(x=t, color='red', linestyle=':', alpha=0.5)
            
        ax.set_title(f"ROI: {roi_name} ({len(valid_chans)} channels)")
        ax.set_ylabel("µV")
        ax.legend(loc='upper right', fontsize='small', ncol=2)
        ax.grid(True, alpha=0.3)
        
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"Regional Physiology - Pair {pair_id} Player {player_id}", y=1.02)
    
    save_plot(fig, pair_id, player_id, "02_roi_traces")

# --- PLOT 3: TOPO SERIES ---
def plot_topo_series(evoked, pair_id, player_id):
    """
    Plots topographic maps at defined timepoints.
    Shows spatial distribution of activity (e.g. is it visual? is it muscle?).
    """
    times = CONFIG['diagnosis']['topo_times']
    # Filter times that are within our data range
    valid_times = [t for t in times if evoked.times[0] <= t <= evoked.times[-1]]
    
    if not valid_times:
        print("    [WARN] No valid topo times found in data range.")
        return

    fig = evoked.plot_topomap(times=valid_times, ch_type='eeg', 
                              time_unit='s', show=False, nrows=2)
    
    # MNE plot_topomap returns a Figure, we just add a title
    fig.suptitle(f"Spatial Evolution - Pair {pair_id} Player {player_id}", fontsize=16)
    save_plot(fig, pair_id, player_id, "03_topo_series")

# --- PLOT 4: ERP IMAGE (TRIAL CONSISTENCY) ---
def plot_erp_image(epochs, pair_id, player_id):
    """
    Plots all trials for a representative channel (Oz).
    Shows if 'bad data' is consistent or sporadic.
    """
    # Pick Oz or first available channel
    pick = 'Oz' if 'Oz' in epochs.ch_names else epochs.ch_names[0]
    
    # Plot Image
    # Note: plot_image returns a list of figures
    figs = epochs.plot_image(picks=pick, combine='mean', 
                             title=f"Trial Consistency ({pick}) - Pair {pair_id} P{player_id}",
                             show=False)
    
    if figs:
        save_plot(figs[0], pair_id, player_id, "04_trial_consistency")


def run_diagnosis():
    # Setup Paths
    data_dir = Path(CONFIG['paths']['output_preprocess'])
    
    # Get subjects
    if CONFIG['subjects']['run_mode'] == 'single':
        pairs = [CONFIG['subjects']['single_pair_id']]
    else:
        all_p = range(CONFIG['subjects']['pair_range'][0], CONFIG['subjects']['pair_range'][1] + 1)
        pairs = [p for p in all_p if p not in CONFIG['subjects']['exclude_pairs']]

    print(f"--- STARTING DIAGNOSIS for {len(pairs)} Pair(s) ---")

    for pair_id in pairs:
        for player_num in [1, 2]:
            fname = data_dir / f"pair-{pair_id:02d}_player-{player_num}_clean_epo.fif"
            
            if not fname.exists():
                print(f"[SKIP] {fname} not found.")
                continue
                
            print(f"\nDiagnosing Pair {pair_id} Player {player_num}...")
            
            # 1. Load Data
            epochs = mne.read_epochs(fname, preload=True, verbose='error')
            
            # 2. Apply Diagnostic Filter (CRITICAL for interpretation)
            # This allows us to see "Brain" signal hidden under drift/noise
            epochs_viz = apply_viz_filter(epochs)
            evoked = epochs_viz.average()
            
            # 3. Generate Plots
            # A) Butterfly (Phases)
            plot_phase_butterfly(evoked, pair_id, player_num)
            
            # B) ROIs (Physiology)
            # Reload evoked to ensure we use correct channels for ROI picking if needed
            # (MNE methods work on copies usually, but safe is safe)
            plot_roi_traces(evoked, pair_id, player_num)
            
            # C) Topos (Space)
            plot_topo_series(evoked, pair_id, player_num)
            
            # D) ERP Image (Consistency)
            plot_erp_image(epochs_viz, pair_id, player_num)

if __name__ == "__main__":
    run_diagnosis()