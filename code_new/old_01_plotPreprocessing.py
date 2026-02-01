"""
01_plotPreprocessing.py
Scientific Visualization of Preprocessing Quality.
Compares Raw (Mapped) vs. Cleaned Data.
"""

import mne
import yaml
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Load Configuration
def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

CONFIG = load_config('code_new/config_preprocessing.yaml') 

# Same Mapping needed to make Raw comparable
BIOSEMI_TO_1020 = {
    'A1': 'Fp1', 'A2': 'AF7', 'A3': 'AF3', 'A4': 'F1', 'A15': 'T7', 'A31': 'Pz', 'A29': 'Oz', # (Subset for brevity in raw viz if needed, but we map all)
    # ... In plotting we usually trust the cleaned file, but for raw comparison:
}
# (Du kannst das volle Mapping aus script 01 hier reinkopieren wenn du exakten 1:1 Raw Vergleich willst,
# aber für den PSD Plot reicht es oft, die Kanäle einfach als 'EEG' zu behandeln).

def plot_preprocessing_report(pair_id, player_id):
    print(f"\n--- Generating Report for Pair {pair_id}, Player {player_id} ---")
    
    # Paths
    data_root = Path(CONFIG['paths']['data_root'])
    deriv_dir = Path(CONFIG['paths']['output_dir'])
    
    # 1. Load CLEANED Data
    clean_path = deriv_dir / f"pair-{pair_id:02d}_player-{player_id}_task-RPS_epo.fif"
    if not clean_path.exists():
        print(f"Cleaned file not found: {clean_path}")
        return

    epochs = mne.read_epochs(clean_path, verbose=False)
    evoked = epochs.average() # Create ERP (Event Related Potential)

    # 2. Plotting
    # We will create a figure with subplots
    # MNE's built-in plotting is interactive, but we can also save figures.
    
    # A) Power Spectral Density (Checking for Noise/Line Noise)
    # Note: Cleaned data is resampled (256Hz), so max freq is 128Hz.
    print("Plotting PSD (Frequency Domain)...")
    fig_psd = epochs.plot_psd(fmin=1, fmax=100, show=False)
    fig_psd.suptitle(f"PSD Spectrum (Cleaned) - Pair {pair_id} Player {player_id}")
    plt.show() # Remove if you just want to save
    
    # B) ERP Image (Variation across trials)
    # Shows if some trials are still noisy
    print("Plotting ERP Image (Trial Consistency)...")
    # Plotting 'Pz' or 'Oz' is usually good for visual tasks
    pick = 'Oz' if 'Oz' in epochs.ch_names else epochs.ch_names[0]
    fig_image = epochs.plot_image(picks=pick, combine='mean', show=False)
    # Note: plot_image returns list of figs
    plt.show()

    # C) Butterfly Plot with GFP (Global Field Power)
    # This is the "Medical/Scientific" view of the brain response
    print("Plotting Butterfly/GFP (Time Domain)...")
    fig_erp = evoked.plot_joint(times=[0.1, 0.2, 0.3], title=f"ERP - Pair {pair_id} P{player_id}", show=False)
    plt.show()

    print("Done. If you see clear peaks in the ERP (bottom plot) around 100-300ms, the preprocessing worked well.")

if __name__ == "__main__":
    # Check the pair defined in config
    pair = CONFIG['subjects']['single_pair_id']
    
    # Plot for Player 1
    plot_preprocessing_report(pair, 1)
    
    # Optional: Plot for Player 2
    # plot_preprocessing_report(pair, 2)