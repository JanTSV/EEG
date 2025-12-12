"""
Compare raw vs. preprocessed EEG data for a single player.

Loads raw BDF, applies minimal processing (epoching + downsampling only),
then plots alongside the fully preprocessed FIF epochs to visualize the
effect of filtering, ICA, re-referencing, etc.

Usage:
    python code/compare_preprocessing.py --pair 1 --player 1
"""

import argparse
import yaml
from pathlib import Path
import mne
import numpy as np
import matplotlib.pyplot as plt


def load_config(config_path=None):
    """Load preprocessing config."""
    default_path = Path(__file__).resolve().parent / "preprocess_config.yaml"
    cfg_path = Path(config_path) if config_path else default_path
    
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f), cfg_path


def resolve_path(path_str, base_dir):
    """Resolve path relative to base directory if not absolute."""
    path = Path(path_str)
    return path if path.is_absolute() else (base_dir / path).resolve()


def build_channel_mask(channel_names, include_patterns):
    """Return boolean mask for channels matching any pattern."""
    return np.array([
        any(pat in ch for pat in include_patterns) for ch in channel_names
    ], dtype=bool)


def load_raw_epochs(pair, player, cfg, cfg_path):
    """
    Load raw BDF and create minimal epochs (no filtering, just epoching + downsampling).
    
    Returns epochs with same structure as preprocessed but without signal processing.
    """
    base_dir = cfg_path.parent
    
    # Load raw BDF
    raw_path = resolve_path(
        cfg["paths"]["raw_template"].format(pair=pair), base_dir
    )
    events_path = resolve_path(
        cfg["paths"]["events_template"].format(pair=pair), base_dir
    )
    
    print(f"Loading raw data from {raw_path}")
    raw = mne.io.read_raw_bdf(str(raw_path), preload=True, verbose=False)
    
    # Load events
    import pandas as pd
    events_df = pd.read_csv(events_path, sep="\t")
    onset_sample = events_df["onset_sample"].values
    
    # Select player channels
    channel_names = np.array(raw.ch_names)
    include_patterns = (
        cfg["channel_masks"]["player1_includes"] if player == 1 
        else cfg["channel_masks"]["player2_includes"]
    )
    mask = build_channel_mask(channel_names, include_patterns)
    orig_labels = channel_names[mask].tolist()
    
    raw_player = raw.copy().pick(orig_labels)
    
    # Rename to standard BioSemi64
    biosemi_montage = mne.channels.make_standard_montage(cfg["processing"]["montage"])
    std_ch_names = biosemi_montage.ch_names
    expected = cfg["processing"]["expected_player_channels"]
    rename_map = {raw_player.ch_names[i]: std_ch_names[i] for i in range(expected)}
    raw_player.rename_channels(rename_map)
    raw_player.set_montage(biosemi_montage, match_case=False)
    
    # Create epochs (minimal processing)
    event_samples = onset_sample - 1
    mne_events = np.column_stack([
        event_samples, 
        np.zeros_like(event_samples), 
        np.ones_like(event_samples)
    ]).astype(int)
    
    prestim = cfg["timing"]["prestim_sec"]
    poststim = cfg["timing"]["poststim_sec"]
    
    epochs_raw = mne.Epochs(
        raw_player,
        mne_events,
        event_id={cfg["processing"]["event_id_label"]: cfg["processing"]["event_id_code"]},
        tmin=-prestim,
        tmax=poststim,
        baseline=None,  # No baseline correction for raw
        preload=True,
        detrend=None,
        verbose=False
    )
    
    # Downsample to match preprocessed data
    if cfg["processing"]["down_sample"]:
        target_rate = cfg["processing"]["downsample_rate_hz"]
        print(f"Downsampling raw epochs to {target_rate} Hz")
        epochs_raw = epochs_raw.copy().resample(target_rate, verbose=False)
    
    return epochs_raw


def load_preprocessed_epochs(pair, player, cfg, cfg_path):
    """Load preprocessed FIF epochs."""
    base_dir = cfg_path.parent
    deriv_dir = resolve_path(cfg["paths"]["derivatives_dir"], base_dir)
    pattern = cfg["paths"]["output_pattern"]
    fif_path = deriv_dir / pattern.format(pair=pair, player=player)
    
    if not fif_path.exists():
        raise FileNotFoundError(
            f"Preprocessed file not found: {fif_path}\n"
            f"Run preprocessing.py first to generate this file."
        )
    
    print(f"Loading preprocessed data from {fif_path}")
    return mne.read_epochs(str(fif_path), preload=True, verbose=False)


def plot_comparison(epochs_raw, epochs_preprocessed, pair, player, channels=None):
    """
    Plot raw vs. preprocessed data for visual comparison.
    
    Opens MNE's interactive browser with both datasets loaded for direct comparison.
    Also shows overlaid traces for selected channels.
    """
    if channels is None:
        # Select representative channels (frontal, central, parietal, occipital)
        channels = ["Fz", "Cz", "Pz", "Oz"]
    
    # Ensure channels exist in both datasets
    available_raw = set(epochs_raw.ch_names)
    available_prep = set(epochs_preprocessed.ch_names)
    channels = [ch for ch in channels if ch in available_raw and ch in available_prep]
    
    if not channels:
        print("Warning: No common channels found. Using first 4 channels from preprocessed data.")
        channels = epochs_preprocessed.ch_names[:4]
    
    n_channels = len(channels)
    
    # Average across epochs for cleaner visualization
    evoked_raw = epochs_raw.average(picks=channels)
    evoked_prep = epochs_preprocessed.average(picks=channels)
    
    # Create figure with subplots for each channel (overlaid comparison)
    fig, axes = plt.subplots(n_channels, 1, figsize=(14, 3 * n_channels), sharex=True)
    if n_channels == 1:
        axes = [axes]
    
    fig.suptitle(f"Raw vs. Preprocessed Comparison: Pair {pair:02d}, Player {player}", 
                 fontsize=16, fontweight="bold")
    
    times = evoked_raw.times
    
    # Plot each channel with both traces overlaid
    for i, ch in enumerate(channels):
        ax = axes[i]
        
        # Get data for this channel
        idx_raw = evoked_raw.ch_names.index(ch)
        idx_prep = evoked_prep.ch_names.index(ch)
        data_raw = evoked_raw.data[idx_raw, :] * 1e6  # Convert to µV
        data_prep = evoked_prep.data[idx_prep, :] * 1e6  # Convert to µV
        
        # Plot both traces overlaid with distinct colors
        ax.plot(times, data_raw, color='#1f77b4', linewidth=2, alpha=0.7, 
                label='Raw (epoched + downsampled)')
        ax.plot(times, data_prep, color='#ff7f0e', linewidth=2, alpha=0.8, 
                label='Preprocessed (filtered + ICA + reref + baseline)')
        
        # Add stimulus onset marker
        ax.axvline(0, color="red", linestyle="--", linewidth=2, alpha=0.7)
        ax.axhline(0, color="gray", linestyle=":", linewidth=1, alpha=0.5)
        
        # Labels and styling
        ax.set_ylabel(f"{ch}\nAmplitude (µV)", fontsize=11, fontweight='bold')
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Add y-axis range info
        y_min, y_max = ax.get_ylim()
        ax.text(0.02, 0.95, f"Range: [{y_min:.1f}, {y_max:.1f}] µV", 
                transform=ax.transAxes, fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    axes[-1].set_xlabel("Time (sec)", fontsize=12)
    
    plt.tight_layout()
    plt.show(block=False)
    
    # Open interactive MNE browsers for detailed inspection
    print("\n" + "="*60)
    print("Opening MNE interactive browsers...")
    print("Close the browser windows to continue.")
    print("="*60 + "\n")
    
    # Create a copy with modified description for clarity
    epochs_raw_copy = epochs_raw.copy()
    epochs_prep_copy = epochs_preprocessed.copy()
    
    # Set browser theme
    mne.set_config('MNE_BROWSER_THEME', 'light', set_env=True)
    
    print("Opening RAW epochs browser (blue in static plot)...")
    epochs_raw_copy.plot(n_epochs=10, n_channels=32, scalings="auto", 
                          title=f"RAW: Pair {pair:02d}, Player {player}", block=False)
    
    print("Opening PREPROCESSED epochs browser (orange in static plot)...")
    epochs_prep_copy.plot(n_epochs=10, n_channels=32, scalings="auto", 
                           title=f"PREPROCESSED: Pair {pair:02d}, Player {player}", block=True)
    
    # Also create butterfly plots for overall comparison
    fig2, axes2 = plt.subplots(1, 2, figsize=(16, 6))
    fig2.suptitle(f"All Channels Butterfly Plot: Pair {pair:02d}, Player {player}", 
                  fontsize=16, fontweight="bold")
    
    # Raw butterfly
    evoked_raw.plot(axes=axes2[0], show=False, titles=dict(eeg="Raw Data (All Channels)"))
    axes2[0].set_title("Raw Data (All Channels)", fontsize=14, fontweight='bold')
    
    # Preprocessed butterfly
    evoked_prep.plot(axes=axes2[1], show=False, titles=dict(eeg="Preprocessed Data (All Channels)"))
    axes2[1].set_title("Preprocessed Data (All Channels)", fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Compare raw vs. preprocessed EEG data."
    )
    parser.add_argument("--pair", type=int, required=True, help="Pair ID (e.g., 1)")
    parser.add_argument("--player", type=int, required=True, choices=[1, 2], 
                        help="Player number (1 or 2)")
    parser.add_argument("--config", type=str, default=None, 
                        help="Path to config file (default: code/preprocess_config.yaml)")
    parser.add_argument("--channels", type=str, nargs="+", default=None,
                        help="Specific channels to plot (default: Fz Cz Pz Oz)")
    
    args = parser.parse_args()
    
    # Load config
    cfg, cfg_path = load_config(args.config)
    
    print(f"\n{'='*60}")
    print(f"Comparing Pair {args.pair:02d}, Player {args.player}")
    print(f"{'='*60}\n")
    
    # Load raw and preprocessed epochs
    try:
        epochs_raw = load_raw_epochs(args.pair, args.player, cfg, cfg_path)
        epochs_preprocessed = load_preprocessed_epochs(args.pair, args.player, cfg, cfg_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # Print summary statistics
    print(f"\n{'='*60}")
    print("Summary Statistics:")
    print(f"{'='*60}")
    print(f"Raw epochs:          {len(epochs_raw)} trials")
    print(f"Preprocessed epochs: {len(epochs_preprocessed)} trials")
    print(f"Sampling rate:       {epochs_preprocessed.info['sfreq']} Hz")
    print(f"Number of channels:  {len(epochs_preprocessed.ch_names)}")
    print(f"Time window:         [{epochs_preprocessed.tmin:.2f}, {epochs_preprocessed.tmax:.2f}] sec")
    
    # Compute RMS (root-mean-square) to compare signal amplitude
    rms_raw = np.sqrt(np.mean(epochs_raw.get_data() ** 2))
    rms_prep = np.sqrt(np.mean(epochs_preprocessed.get_data() ** 2))
    print(f"\nRMS amplitude:")
    print(f"  Raw:          {rms_raw * 1e6:.2f} µV")
    print(f"  Preprocessed: {rms_prep * 1e6:.2f} µV")
    print(f"  Reduction:    {(1 - rms_prep / rms_raw) * 100:.1f}%")
    print(f"{'='*60}\n")
    
    # Plot comparison
    plot_comparison(epochs_raw, epochs_preprocessed, args.pair, args.player, args.channels)


if __name__ == "__main__":
    main()
