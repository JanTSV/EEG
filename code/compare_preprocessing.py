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


def load_raw_epochs(pair, player, cfg, cfg_path, show_mapping=False):
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
    if show_mapping:
        print("\nChannel mapping (original -> BioSemi64 standard):")
        for i in range(expected):
            print(f"  {raw_player.ch_names[i]:<12} -> {std_ch_names[i]}")
        print("")
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


def plot_comparison(epochs_raw, epochs_preprocessed, pair, player, channels=None, baseline_raw=True, avg_ref_raw=True):
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

    # Optionally baseline-correct raw epochs for plotting to remove DC offset
    raw_for_plot = epochs_raw.copy()
    if baseline_raw:
        try:
            raw_for_plot.apply_baseline((raw_for_plot.tmin, 0.0))
        except Exception:
            # Fallback: subtract mean over pre-stim window if baseline fails
            prestim_mask = (raw_for_plot.times >= raw_for_plot.tmin) & (raw_for_plot.times <= 0.0)
            data = raw_for_plot.get_data()
            baseline_mean = data[:, :, prestim_mask].mean(axis=2, keepdims=True)
            raw_for_plot._data = data - baseline_mean
    if avg_ref_raw:
        raw_for_plot.set_eeg_reference('average', verbose=False)
    
    # Average across epochs for cleaner visualization
    evoked_raw = raw_for_plot.average(picks=channels)
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
    
    # ===== NEW: Power spectrum comparison (all channels) =====
    # Compute Welch PSD for raw vs preprocessed, average over epochs but plot per-channel curves
    print("Computing power spectra (Welch) ...")
    sfreq = epochs_preprocessed.info["sfreq"]
    nperseg = int(sfreq * 2)  # 2-second segments
    # Get data arrays: shape (n_epochs, n_channels, n_times)
    data_raw = raw_for_plot.get_data()
    data_prep = epochs_preprocessed.get_data()
    # Average over epochs for PSD stability
    raw_mean = data_raw.mean(axis=0)   # (n_channels, n_times)
    prep_mean = data_prep.mean(axis=0) # (n_channels, n_times)
    # Compute PSD per channel using scipy.signal.welch (via MNE helper)
    from scipy.signal import welch
    freqs, psd_raw = welch(raw_mean, fs=sfreq, nperseg=nperseg, axis=1)
    _, psd_prep = welch(prep_mean, fs=sfreq, nperseg=nperseg, axis=1)

    # Plot all-channel PSD overlays: raw vs preprocessed
    fig_psd, (ax_psd_raw, ax_psd_prep) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    fig_psd.suptitle(f"Power Spectrum (All Channels): Pair {pair:02d}, Player {player}", fontsize=16, fontweight="bold")
    # Show frequencies up to 128 Hz to visualize filter rolloff (cutoff at 128 Hz)
    fmask = freqs <= 128.0
    for ch_idx in range(psd_raw.shape[0]):
        ax_psd_raw.plot(freqs[fmask], 10*np.log10(psd_raw[ch_idx, fmask]), color='#1f77b4', alpha=0.35, linewidth=1)
        ax_psd_prep.plot(freqs[fmask], 10*np.log10(psd_prep[ch_idx, fmask]), color='#ff7f0e', alpha=0.35, linewidth=1)
    ax_psd_raw.set_title("RAW (epoched + downsampled)", fontsize=13)
    ax_psd_prep.set_title("PREPROCESSED (filtered + ICA)", fontsize=13)
    for axx in (ax_psd_raw, ax_psd_prep):
        axx.set_xlabel("Frequency (Hz)")
        axx.grid(True, alpha=0.3)
    ax_psd_raw.set_ylabel("Power (dB)")
    plt.tight_layout()
    plt.show(block=False)

    # ===== NEW: Single-epoch overlay for selected channels =====
    print("Plotting single-epoch overlays for selected channels ...")
    # Pick the first epoch for demonstration
    epoch_idx = 0
    raw_epoch = data_raw[epoch_idx]   # (n_channels, n_times)
    prep_epoch = data_prep[epoch_idx] # (n_channels, n_times)

    # Choose up to 3 channels from the requested list (fallback to first 3 available)
    overlay_channels = channels[:3] if len(channels) >= 3 else channels
    if len(overlay_channels) < 3:
        # Ensure at least 3 channels if possible
        for ch in epochs_preprocessed.ch_names:
            if ch not in overlay_channels:
                overlay_channels.append(ch)
            if len(overlay_channels) == 3:
                break

    fig_epoch, ax_epoch = plt.subplots(1, 1, figsize=(12, 5))
    ax_epoch.set_title(f"Single Epoch Overlay (Raw vs Preprocessed): Pair {pair:02d}, Player {player}", fontsize=14, fontweight="bold")
    t = epochs_preprocessed.times
    colors = ["#1f77b4", "#2ca02c", "#d62728"]
    for i, ch in enumerate(overlay_channels):
        if ch in epochs_preprocessed.ch_names and ch in epochs_raw.ch_names:
            idx_p = epochs_preprocessed.ch_names.index(ch)
            idx_r = epochs_raw.ch_names.index(ch)
            ax_epoch.plot(t, raw_epoch[idx_r]*1e6, color=colors[i], alpha=0.5, linestyle='-', label=f"{ch} RAW")
            ax_epoch.plot(t, prep_epoch[idx_p]*1e6, color=colors[i], alpha=0.9, linestyle='--', label=f"{ch} PREP")
    ax_epoch.axvline(0, color="red", linestyle="--", linewidth=2, alpha=0.7)
    ax_epoch.set_xlabel("Time (sec)")
    ax_epoch.set_ylabel("Amplitude (µV)")
    ax_epoch.grid(True, alpha=0.3)
    ax_epoch.legend(loc="upper right", ncol=2, fontsize=9)
    plt.tight_layout()
    plt.show(block=False)
    
    # Also create butterfly plots for overall comparison
    # IMPORTANT: Use ALL channels here (not the 4-channel subset above)
    evoked_raw_all = raw_for_plot.average()  # average over epochs, all channels
    evoked_prep_all = epochs_preprocessed.average()

    fig2, axes2 = plt.subplots(1, 2, figsize=(16, 6))
    fig2.suptitle(f"All Channels Butterfly Plot: Pair {pair:02d}, Player {player}", 
                  fontsize=16, fontweight="bold")

    # Raw butterfly (all channels)
    evoked_raw_all.plot(axes=axes2[0], show=False, titles=dict(eeg="Raw Data (All Channels)"))
    axes2[0].set_title("Raw Data (All Channels)", fontsize=14, fontweight='bold')

    # Preprocessed butterfly (all channels)
    evoked_prep_all.plot(axes=axes2[1], show=False, titles=dict(eeg="Preprocessed Data (All Channels)"))
    axes2[1].set_title("Preprocessed Data (All Channels)", fontsize=14, fontweight='bold')

    # Compute shared y-limits dynamically based on both datasets
    raw_data_uv = evoked_raw_all.data * 1e6
    prep_data_uv = evoked_prep_all.data * 1e6
    data_min = min(raw_data_uv.min(), prep_data_uv.min())
    data_max = max(raw_data_uv.max(), prep_data_uv.max())
    # Add 15% padding and make symmetric for clarity
    abs_max = max(abs(data_min), abs(data_max)) * 1.15
    y_min, y_max = -abs_max, abs_max
    axes2[0].set_ylim(y_min, y_max)
    axes2[1].set_ylim(y_min, y_max)
    axes2[0].set_ylabel("µV")
    axes2[1].set_ylabel("µV")

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
    parser.add_argument("--show-mapping", action="store_true",
                        help="Print original -> standard channel name mapping and exit")
    
    args = parser.parse_args()
    
    # Load config
    cfg, cfg_path = load_config(args.config)
    
    print(f"\n{'='*60}")
    print(f"Comparing Pair {args.pair:02d}, Player {args.player}")
    print(f"{'='*60}\n")
    
    # Load raw and preprocessed epochs
    try:
        epochs_raw = load_raw_epochs(args.pair, args.player, cfg, cfg_path, show_mapping=args.show_mapping)
        epochs_preprocessed = load_preprocessed_epochs(args.pair, args.player, cfg, cfg_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    if args.show_mapping:
        print("Mapping shown. Exiting as requested (--show-mapping).")
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
    raw_for_metrics = epochs_raw.copy()
    try:
        raw_for_metrics.apply_baseline((raw_for_metrics.tmin, 0.0))
    except Exception:
        pass
    raw_for_metrics.set_eeg_reference('average', verbose=False)
    rms_raw = np.sqrt(np.mean(raw_for_metrics.get_data() ** 2))
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
