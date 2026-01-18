# ----- SKETCH -----
# EEG preprocessing pipeline for dual-player RPS task:
# 1. Load raw BDF files (BioSemi 64-channel system, 2048 Hz)
# 2. Select player-specific channels (masking by channel labels)
# 3. Standardize to canonical montage (BioSemi64)
# 4. Re-reference to average (neutral reference for group analysis)
# 5. Filter: notch (line noise) + band-pass (signal of interest)
# 6. Interpolate bad channels (marked in participants.tsv) - BEFORE ICA for clean input
# 7. Downsample to 256 Hz (speeds up ICA; sufficient for ERP analysis)
# 8. ICA (Independent Component Analysis) to remove eye/muscle artifacts
# 9. Epoch around stimulus onset with baseline correction
# 10. Save as FIF (Functional Image File Format, MNE-native)
#
# Methods:
# - Re-referencing: subtracts average of all channels from each channel
# - Notch filter: narrow band-stop at 50 Hz (AC mains artifact)
# - Band-pass (FIR): linear-phase filtering preserves temporal dynamics
# - ICA (fastICA): blind source separation; identifies independent components (eye blinks, muscle)
# - Baseline correction: subtracts mean voltage from -200 to 0 ms pre-stimulus
# - Epoching: extracts [-0.2, 5.0] sec windows around trial onset

# ----- IMPORTS -----
import pandas as pd
import mne
from mne.preprocessing import ICA
import numpy as np
from pathlib import Path
from typing import Optional, Sequence, Union
import matplotlib
matplotlib.use("QtAgg")
import warnings
import yaml
import argparse

warnings.filterwarnings("ignore", message="QObject::connect")

# ----- CODE -----
def load_config(config_path: Optional[Union[str, Path]] = None) -> tuple[dict, Path]:
    """Load preprocessing configuration from YAML and return config plus path."""
    default_path = Path(__file__).resolve().parent / "preprocess_config.yaml"
    cfg_path = Path(config_path) if config_path else default_path

    if not cfg_path.exists():
        raise FileNotFoundError(
            f"Config file not found at {cfg_path}. Update the path or create it."
        )

    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f), cfg_path


def build_channel_mask(channel_names: np.ndarray, include_patterns: Sequence[str]) -> np.ndarray:
    """Return boolean mask selecting channels that contain any allowed pattern."""
    return np.array([
        any(pat in ch for pat in include_patterns) for ch in channel_names
    ], dtype=bool)


def pipeline(output_dir="derivatives", config_path: Optional[Union[str, Path]] = None):
    # Step 1: read demographics
    print("starting pipeline ...")
    cfg, cfg_path = load_config(config_path)
    base_dir = cfg_path.parent

    def resolve_path(path_str: str) -> Path:
        path = Path(path_str)
        return path if path.is_absolute() else (base_dir / path)

    paths_cfg = cfg["paths"]
    
    # Update derivatives_dir to use specified output directory
    # base_dir is code/, so ../data/ goes to EEG/data/
    paths_cfg["derivatives_dir"] = f"../data/{output_dir}"
    processing_cfg = cfg["processing"]
    data_cfg = cfg["data"]
    timing_cfg = cfg["timing"]
    bad_cfg = cfg["bad_channels"]
    channel_masks_cfg = cfg["channel_masks"]
    plot_cfg = cfg.get("plot", {})

    demographics_path = resolve_path(paths_cfg["participants_tsv"])
    demographics = read_table(
        path_to_file=str(demographics_path),
        description="demographics",
    )
    
    # step 2: plot data
    if processing_cfg["plot_data"]:
        plot_raw_path = paths_cfg.get("plot_raw_path")
        plot_data(str(resolve_path(plot_raw_path)) if plot_raw_path else None, plot_cfg)
          
    # step 3: loop over pairs
    valid_pair_ids_list = extract_pair_ids(data_cfg["pair_ranges"])
    target_pairs = data_cfg.get("target_pairs")
    pairs_to_process = (
        valid_pair_ids_list if not target_pairs else [p for p in valid_pair_ids_list if p in target_pairs]
    )
    
    for pair in pairs_to_process:
        print(f'loading pair {pair}')

        # load trigger times
        print("loading trigger times ...")
        events_filename = resolve_path(paths_cfg["events_template"].format(pair=pair))
        events_df = read_table(path_to_file=str(events_filename), description="events")
        onset_sample = np.array(events_df.onset_sample, dtype=int)

        # epoch timing
        prestim = timing_cfg["prestim_sec"]
        poststim = timing_cfg["poststim_sec"]

        # load raw once per pair
        print("reading channels ...")
        raw_filename = resolve_path(paths_cfg["raw_template"].format(pair=pair))
        raw = mne.io.read_raw_bdf(str(raw_filename), preload=False)
        channel_names = np.array(raw.ch_names)

        for player in [1, 2]:
            process_pair_player(
                pair=pair,
                player=player,
                raw=raw,
                channel_names=channel_names,
                onset_sample=onset_sample,
                prestim=prestim,
                poststim=poststim,
                processing_cfg=processing_cfg,
                channel_masks_cfg=channel_masks_cfg,
                demographics=demographics,
                bad_cfg=bad_cfg,
                out_dir=resolve_path(paths_cfg["derivatives_dir"]),
                output_pattern=paths_cfg["output_pattern"],
            )

        print("\n")

    print("done")
    
    
def extract_pair_ids(pair_ids: Sequence[Sequence[int]]) -> list[int]:
    pair_ids_list = []
    for pair in pair_ids:
        pair_ids_list+= list(range(pair[0], pair[1]+1))
    
    return pair_ids_list


def process_pair_player(
    pair: int,
    player: int,
    raw: mne.io.BaseRaw,
    channel_names: np.ndarray,
    onset_sample: np.ndarray,
    prestim: float,
    poststim: float,
    processing_cfg: dict,
    channel_masks_cfg: dict,
    demographics: pd.DataFrame,
    bad_cfg: dict,
    out_dir: Path,
    output_pattern: str,
) -> None:
    print(f"  processing player {player} ...")

    # ===== STEP 1: CHANNEL SELECTION & MONTAGE STANDARDIZATION =====
    # Rationale: Dual-player recordings mix two players' channels (labeled "1-A", "2-A", "2-B", etc).
    # Extract one player's channels using pattern matching, then map to standard BioSemi64 names.
    include_patterns = (
        channel_masks_cfg["player1_includes"] if player == 1 else channel_masks_cfg["player2_includes"]
    )
    mask = build_channel_mask(channel_names, include_patterns)
    orig_labels_list = channel_names[mask].tolist()
    print(f"    found {len(orig_labels_list)} channels for player {player}")

    # Select and copy one player's channels
    raw_player = raw.copy().pick(orig_labels_list)
    
    # Map to standard BioSemi64 montage: ensures electrode positions are consistent across subjects
    # Why: enables group-level 3D visualization, source localization, and statistical tests
    biosemi_montage = mne.channels.make_standard_montage(processing_cfg["montage"])
    std_ch_names = biosemi_montage.ch_names
    expected_channels = processing_cfg["expected_player_channels"]
    if len(raw_player.ch_names) < expected_channels:
        raise RuntimeError(
            f"    Player {player}: expected at least {expected_channels} channels, got {len(raw_player.ch_names)}"
        )

    rename_map = {raw_player.ch_names[i]: std_ch_names[i] for i in range(expected_channels)}
    raw_player.rename_channels(rename_map)
    raw_player.set_montage(biosemi_montage, match_case=False)

    # ===== STEP 2: LOAD DATA & RE-REFERENCE =====
    # Load data into memory (required for reference and filtering operations)
    raw_player.load_data()

    # Average re-reference: subtracts mean of all EEG channels from each channel
    # Why: BioSemi uses active reference (CMS/DRL) during recording; average reference provides
    # a neutral baseline suitable for group-level ERP analysis and frequency-domain studies
    # Method: ref_channels="average" computes mean across all EEG channels, then subtracts from each
    if processing_cfg.get("reref_to_average", False):
        print("    re-referencing to average ...")
        raw_player.set_eeg_reference(ref_channels="average", projection=False)

    # ===== STEP 3: ARTIFACT REMOVAL - FILTERING (on continuous raw data) =====
    # Why filter raw (not epochs): avoids edge artifacts; FIR kernel is appropriate for long signals
    
    # Notch filter: narrow band-stop filter at line frequency (50 Hz in EU/AU, 60 Hz in US)
    # Purpose: removes AC mains artifact (hum), which is spatially widespread and difficult to remove later
    # Method: band-stop IIR filter centered at f0 Hz with narrow bandwidth (default ±1 Hz)
    if processing_cfg.get("notch_filter_enabled", False):
        f0 = processing_cfg.get("notch_freq_hz", 50.0)
        print(f"    applying notch filter at {f0} Hz on raw ...")
        raw_player.notch_filter(freqs=[f0], picks="eeg")

    # Band-pass filter: extracts signal of interest, removes slow drift (high-pass) and noise (low-pass)
    # Method: FIR (Finite Impulse Response) with linear phase
    # Why FIR: linear phase = no phase distortion; critical for ERP timing and coherence analyses
    # Why on raw: kernel length is not problematic on long continuous signals
    if processing_cfg.get("output_filter_enabled", False):
        l_out = processing_cfg.get("output_filter_low_hz")
        h_out = processing_cfg.get("output_filter_high_hz")
        method = processing_cfg.get("output_filter_method", "fir")
        print(f"    filtering raw (l={l_out}, h={h_out}, method={method}) ...")
        raw_player.filter(l_freq=l_out, h_freq=h_out, picks="eeg", method=method)

    # ===== STEP 4: BAD CHANNEL INTERPOLATION (OPTIONAL) =====
    # Interpolate channels marked as bad (noisy, high impedance, etc.) using spherical spline
    # Why: clean data improves ICA decomposition; removes obvious noise sources before ICA
    # Method: spherical spline (accurate) - fits surface on good channels, extrapolates to bad channels
    if processing_cfg["interpolate_bad_channels"]:
        sub_id = f"sub-{pair:02d}"
        chan_to_fix = demographics.loc[demographics["participant_id"] == sub_id]
        bad_str = chan_to_fix.iloc[0, bad_cfg["player1_column_index"]] if player == 1 else chan_to_fix.iloc[0, bad_cfg["player2_column_index"]]
        if isinstance(bad_str, str) and bad_str.strip():
            bads = [ch.strip() for ch in bad_str.split(",")]
            print(f"    interpolating bad channels: {bads}")
            raw_player.info["bads"] = bads
            raw_player = raw_player.interpolate_bads(reset_bads=True, mode="accurate")
        else:
            print("    no bad channels listed for this participant.")

    # ===== STEP 5: DOWNSAMPLING (BEFORE ICA) =====
    # Downsample before ICA to speed up computation without losing accuracy
    # Why here: ICA is computationally expensive; 256 Hz sufficient for component separation
    # Note: Event sample indices must be adjusted after resampling!
    original_sfreq = raw_player.info['sfreq']
    if processing_cfg["down_sample"]:
        target_sfreq = processing_cfg["downsample_rate_hz"]
        print(f"    downsampling to {target_sfreq} Hz (before ICA) ...")
        raw_player.resample(target_sfreq)
        
        # CRITICAL: Adjust event sample indices to match new sampling rate
        # Formula: new_sample = old_sample * (new_sfreq / old_sfreq)
        sfreq_ratio = target_sfreq / original_sfreq
        onset_sample = (onset_sample * sfreq_ratio).astype(int)
        print(f"    adjusted event sample indices for new sampling rate (ratio={sfreq_ratio:.4f})")
    
    # ===== STEP 6: ARTIFACT REMOVAL - ICA (INDEPENDENT COMPONENT ANALYSIS) =====
    # Goal: separate brain activity from physiological artifacts (eye blinks, muscle, heartbeat)
    # Method: fastICA finds statistically independent components; auto-detects EOG (eye) artifacts
    # Why after filtering & downsampling: filtered signal has better SNR; lower sfreq speeds up ICA
    # Typical: 15-25 components sufficient for 64-channel EEG; here using 20
    if processing_cfg.get("ica_enabled", False):
        print("    running ICA for artifact detection ...")
        n_components = processing_cfg.get("ica_n_components", 20)  # If int: components, if float: variance (picard only)
        method = processing_cfg.get("ica_method", "fastica")
        random_state = processing_cfg.get("ica_random_state", 42)
        max_iter = processing_cfg.get("ica_max_iter", "auto")
        plot = processing_cfg.get("ica_plot", False)
        
        ica = ICA(n_components=n_components, 
                  method=method,
                  random_state=random_state,
                  max_iter=max_iter)
        ica.fit(raw_player, picks="eeg")
        
        # Auto-detect EOG artifacts: uses frontal channels as proxy for eye activity
        # Threshold=3.0: standard; higher = stricter (fewer components excluded)
        try:
            # Use frontal EEG channels as proxies for EOG
            frontal_chs = [ch for ch in ['Fp1', 'Fp2', 'AF7', 'AF8'] if ch in raw_player.ch_names]

            # Find bad EOG components
            eog_indices, eog_scores = ica.find_bads_eog(raw_player, ch_name=frontal_chs, threshold=3.0)
            ica.exclude = eog_indices
            if eog_indices:
                print(f"    ICA detected {len(eog_indices)} EOG component(s): {eog_indices}")
        except RuntimeError:
            print("    No EOG channels found; skipping automatic EOG detection")

        if plot:
            # Plot ICA components for manual inspection
            ica.plot_components(inst=raw_player)

            # Plot properties for first 20 components
            ica.plot_properties(raw_player,
                                picks=range(min(ica.n_components_, 20)),
                                psd_args=dict(fmax=40))
        
        # Apply ICA: removes marked artifact components from raw signal (in-place)
        if ica.exclude:
            ica.apply(raw_player)
            print(f"    ICA applied, removed {len(ica.exclude)} component(s)")
        else:
            print("    No artifacts detected by ICA")

    # ===== STEP 6: EPOCHING WITH BASELINE CORRECTION =====
    # Extract [-0.2, 5.0] second windows around each stimulus onset
    # Why: creates time-synchronized trial structure (epochs) for averaging and statistical analysis
    # Format: MNE expects events as [sample_index, 0, event_id] (sample is 0-indexed)
    event_samples = onset_sample - 1
    mne_events = np.column_stack([event_samples, np.zeros_like(event_samples), np.ones_like(event_samples)]).astype(int)

    # Baseline correction: subtracts mean voltage from -0.2 to 0 ms (pre-stimulus period)
    # Why: removes slow DC drifts and session-level offsets, making neural deflections more interpretable
    # Method: computed per-channel, per-epoch; applied during epochs construction
    baseline_window = processing_cfg.get("baseline_window_sec")
    epochs = mne.Epochs(
        raw_player,
        mne_events,
        event_id={processing_cfg["event_id_label"]: processing_cfg["event_id_code"]},
        tmin=-prestim,
        tmax=poststim,
        baseline=tuple(baseline_window) if baseline_window is not None else processing_cfg["baseline"],
        preload=True,
        detrend=processing_cfg["detrend"],
    )

    # ===== STEP 7: VISUAL INSPECTION (OPTIONAL) =====
    # Plot filtered epochs for manual bad-channel marking
    # Temporary filter applied here (not saved) to improve visualization (higher frequency cutoff helps see components)
    # User can visually mark bad channels, which MNE will flag for later interpolation
    if processing_cfg["identify_bad_channels"]:
        print("    filtering for visual inspection ...")
        l_freq = processing_cfg.get("inspect_filter_low_hz")
        h_freq = processing_cfg.get("inspect_filter_high_hz")
        epochs_filt = epochs.copy().filter(l_freq=l_freq, h_freq=h_freq, picks="eeg")
        print("    opening plot window (mark bad channels manually)...")
        epochs_filt.plot(n_epochs=10, n_channels=32, scalings="auto")

    # ===== STEP 9: AMPLITUDE-BASED EPOCH REJECTION (OPTIONAL) =====
    # Flag and remove epochs with abnormally high voltages (e.g., muscle artifacts, electrode pops)
    # Method: peak-to-peak amplitude check; compares each epoch to user-defined threshold
    # Why: removes outlier trials contaminated by non-brain artifacts after ICA
    # Note: Typical EEG 200-500 µV; threshold set in config (200 µV is reasonable)
    if processing_cfg.get("amplitude_reject_enabled", False):
        threshold_uv = processing_cfg.get("amplitude_reject_threshold_uv", 200.0)
        threshold_v = threshold_uv * 1e-6  # convert µV to V
        n_before = len(epochs)
        epochs.drop_bad(reject=dict(eeg=threshold_v))
        n_after = len(epochs)
        n_dropped = n_before - n_after
        if n_dropped > 0:
            print(f"    amplitude rejection: dropped {n_dropped}/{n_before} epochs (threshold={threshold_uv} µV)")
        else:
            print(f"    amplitude rejection: no epochs dropped (threshold={threshold_uv} µV)")

    # ===== STEP 10: SAVE TO DISK =====
    # Save preprocessed epochs to FIF format (MNE-native binary format)
    # Why FIF: compact; preserves all metadata (channel info, montage, coordinate frame, sampling rate)
    # Enables: fast reload for further analysis, visualization, group-level averaging
    out_dir.mkdir(exist_ok=True, parents=True)
    out_fname = out_dir / output_pattern.format(pair=pair, player=player)
    print(f"    saving to {out_fname}")
    epochs.save(out_fname, overwrite=True)

    del epochs, raw_player
    

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

def plot_data(raw_path: Optional[str], plot_cfg: dict):
    """Plot a raw BDF file for manual inspection."""
    if not raw_path:
        raise ValueError("Set 'plot_raw_path' in preprocess_config.yaml to enable plotting.")

    print("plotting data ...")
    theme = plot_cfg.get("theme", "light")
    mne.set_config('MNE_BROWSER_THEME', theme, set_env=True)
    raw = mne.io.read_raw_bdf(raw_path, preload=False)
    raw.plot(block=True)  # will use the Matplotlib-based viewer
    
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EEG preprocessing pipeline")
    parser.add_argument("--out", type=str, default="derivatives",
                       help="Output directory (relative to data/); default: data/derivatives")
    args = parser.parse_args()
    
    pipeline(output_dir=args.out)
    
    
    

