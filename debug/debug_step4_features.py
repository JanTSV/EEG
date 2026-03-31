"""
debug_step4_features_granular.py
---------------------------------
Granular validation of the feature-extraction pipeline (time-binning).

Compares Python-derived ERP bin averages against the MATLAB ground-truth
feature matrix for all three trial epochs:

  Part A – Decision window  : −0.2 s  to  2.0 s  (8 bins × 0.25 s)
  Part B – Response window  : 1.8 s   to  4.0 s  (8 bins × 0.25 s)
  Part C – Feedback window  : 3.8 s   to  5.0 s  (4 bins × 0.25 s)

CRITICAL FIX applied here:
  MNE stores epoch data in Volts. All data are scaled to µV (× 1e6) before
  any further computation to match the MATLAB output.

For Parts B and C, both a baselined and a non-baselined variant are computed
and compared against the ground truth so that the exact MATLAB baseline
convention can be identified.
"""

import mne
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────────────
fif_path          = Path("/Volumes/HardDiskYF/ACADEMIA/EEG/derivatives_new2/"
                         "sub-01_task-RPS_desc-preproc_eeg.fif")
mat_features_path = Path("originalCode/debug_step2_features.mat")
# ─────────────────────────────────────────────────────────────────────────────


def get_bins(data_arr, start_offset_s=-0.2, bin_width=0.25, n_bins=8, fs=256):
    """
    Average epoch data into equal-width time bins.

    Parameters
    ----------
    data_arr      : np.ndarray, shape (n_trials, n_channels, n_times)
    start_offset_s: float
        Time (in seconds) of the first sample in data_arr relative to the
        trial event (negative = pre-stimulus baseline).
    bin_width     : float
        Width of each bin in seconds.
    n_bins        : int
        Number of bins to compute.
    fs            : float
        Sampling frequency in Hz.

    Returns
    -------
    np.ndarray, shape (n_trials, n_channels, n_bins)
        Mean amplitude within each bin.
    """
    zero_idx    = int(-start_offset_s * fs)   # sample index of t = 0
    bin_samples = int(bin_width * fs)
    binned_data = []

    for b in range(n_bins):
        idx_start = zero_idx + int(b * bin_samples)
        idx_end   = zero_idx + int((b + 1) * bin_samples)

        # Guard against bins that extend beyond the array boundary
        if idx_end > data_arr.shape[2]:
            print(f"  WARNING: bin {b} end index ({idx_end}) exceeds data "
                  f"length ({data_arr.shape[2]}) — truncating.")
            idx_end = data_arr.shape[2]

        chunk = data_arr[:, :, idx_start:idx_end]
        binned_data.append(np.mean(chunk, axis=2))

    return np.stack(binned_data, axis=2)


def run_debug():
    print("─── START: debug_step4 (FEATURE EXTRACTION — GRANULAR) ───\n")

    # ── 1. Load MATLAB ground-truth feature matrix ────────────────────────────
    # Expected layout: (n_trials, n_channels, n_bins_total)
    # Bins 0–7  → Part A (decision),  8–15 → Part B (response),  16–19 → Part C (feedback)
    try:
        mat         = scipy.io.loadmat(mat_features_path, squeeze_me=True)
        gt_features = mat['feature_matrix']   # (trials, channels, bins)
    except Exception as e:
        print(f"ERROR – could not load .mat file: {e}")
        return

    # ── 2. Load preprocessed epochs ───────────────────────────────────────────
    epochs = mne.read_epochs(fif_path, preload=True, verbose='error')

    # CRITICAL FIX: convert from Volts (MNE default) to µV
    print("  Scaling data to µV (× 1e6) ...")
    data_uv = epochs.get_data() * 1e6

    times = epochs.times
    fs    = epochs.info['sfreq']

    # ── 3. Common Average Reference (manual, array-level) ────────────────────
    # Applied manually because we are working on the raw numpy array rather
    # than a live MNE object after scaling.
    print("  Applying CAR (manual, array-level) ...")
    car      = np.mean(data_uv, axis=1, keepdims=True)   # mean across channels
    data_uv -= car

    # ── 4. Drop block-start trials ────────────────────────────────────────────
    # Every 40th trial (indices 0, 40, 80, …) is a block-start marker trial
    # that must be excluded before feature extraction.
    print("  Dropping block-start trials ...")
    drop_indices = list(range(0, 480, 40))
    keep_indices = [i for i in range(len(epochs)) if i not in drop_indices]
    data         = data_uv[keep_indices]   # → (468, n_channels, n_times)

    # ── 5. Part A — Decision window (−0.2 s to 2.0 s, 8 bins) ───────────────
    print("\n─── Part A: Decision window ───")
    mask_a  = (times >= -0.2) & (times <= 2.0)
    data_a  = data[:, :, mask_a].copy()

    # Baseline: mean amplitude during [−0.2, 0] s, computed on the full array
    # and subtracted from the Part A segment via broadcasting.
    base_mask_a    = (times >= -0.2) & (times <= 0)
    baseline_val_a = np.mean(data[:, :, base_mask_a], axis=2, keepdims=True)
    data_a        -= baseline_val_a   # (trials, channels, 1) broadcasts correctly

    bins_a = get_bins(data_a, start_offset_s=-0.2, n_bins=8, fs=fs)
    gt_a   = gt_features[:, :, 0:8]
    mae_a  = np.mean(np.abs(bins_a - gt_a))
    print(f"  MAE Part A : {mae_a:.5f} µV")

    # ── 6. Part B — Response window (1.8 s to 4.0 s, 8 bins) ────────────────
    print("\n─── Part B: Response window ───")
    mask_b  = (times >= 1.8) & (times <= 4.0)
    data_b  = data[:, :, mask_b].copy()

    # Two variants tested to identify which baseline convention MATLAB uses:
    #   • baselined : first 0.2 s of the Part B window used as baseline
    #   • raw       : no baseline correction
    base_samples   = int(0.2 * fs)
    baseline_val_b = np.mean(data_b[:, :, :base_samples], axis=2, keepdims=True)

    bins_b_base = get_bins(data_b - baseline_val_b, start_offset_s=-0.2, n_bins=8, fs=fs)
    bins_b_raw  = get_bins(data_b,                  start_offset_s=-0.2, n_bins=8, fs=fs)

    gt_b        = gt_features[:, :, 8:16]
    mae_b_base  = np.mean(np.abs(bins_b_base - gt_b))
    mae_b_raw   = np.mean(np.abs(bins_b_raw  - gt_b))
    print(f"  MAE Part B (baselined) : {mae_b_base:.5f} µV")
    print(f"  MAE Part B (raw)       : {mae_b_raw:.5f} µV")

    # ── 7. Part C — Feedback window (3.8 s to 5.0 s, 4 bins) ────────────────
    print("\n─── Part C: Feedback window ───")
    mask_c  = (times >= 3.8) & (times <= 5.0)
    data_c  = data[:, :, mask_c].copy()

    baseline_val_c = np.mean(data_c[:, :, :base_samples], axis=2, keepdims=True)

    bins_c_base = get_bins(data_c - baseline_val_c, start_offset_s=-0.2, n_bins=4, fs=fs)
    bins_c_raw  = get_bins(data_c,                  start_offset_s=-0.2, n_bins=4, fs=fs)

    gt_c        = gt_features[:, :, 16:20]
    mae_c_base  = np.mean(np.abs(bins_c_base - gt_c))
    mae_c_raw   = np.mean(np.abs(bins_c_raw  - gt_c))
    print(f"  MAE Part C (baselined) : {mae_c_base:.5f} µV")
    print(f"  MAE Part C (raw)       : {mae_c_raw:.5f} µV")

    # ── 8. Diagnostic plot ────────────────────────────────────────────────────
    # Show bin-by-bin traces for trial 0, channel 0 to inspect offset / shape.
    trial_idx = 0
    chan_idx   = 0

    fig, axes = plt.subplots(3, 1, figsize=(10, 12))

    # Panel 1 – Part A bin averages
    ax = axes[0]
    ax.set_title(f"Part A (Decision) — trial {trial_idx}, channel {chan_idx}")
    ax.plot(bins_a[trial_idx, chan_idx, :], 'b.-', label='Python')
    ax.plot(gt_a[trial_idx,   chan_idx, :], 'r.--', label='MATLAB')
    ax.set_ylabel('Amplitude (µV)')
    ax.legend()
    ax.grid(True)

    # Panel 2 – Part B: baselined vs. raw vs. MATLAB (baseline convention check)
    ax = axes[1]
    ax.set_title(f"Part B (Response) — baseline convention check  "
                 f"(trial {trial_idx}, channel {chan_idx})")
    ax.plot(bins_b_base[trial_idx, chan_idx, :], 'b.-',  label='Python (baselined)')
    ax.plot(bins_b_raw[trial_idx,  chan_idx, :], 'c.-',  label='Python (raw)')
    ax.plot(gt_b[trial_idx,        chan_idx, :], 'r.--', label='MATLAB')
    ax.set_ylabel('Amplitude (µV)')
    ax.legend()
    ax.grid(True)

    # Panel 3 – Part C: baselined vs. raw vs. MATLAB
    ax = axes[2]
    ax.set_title(f"Part C (Feedback) — baseline convention check  "
                 f"(trial {trial_idx}, channel {chan_idx})")
    ax.plot(bins_c_base[trial_idx, chan_idx, :], 'b.-',  label='Python (baselined)')
    ax.plot(bins_c_raw[trial_idx,  chan_idx, :], 'c.-',  label='Python (raw)')
    ax.plot(gt_c[trial_idx,        chan_idx, :], 'r.--', label='MATLAB')
    ax.set_ylabel('Amplitude (µV)')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    out_file = "debug_step4_granular_v2.png"
    plt.savefig(out_file)
    print(f"\nPlot saved: {out_file}")
    plt.show()


if __name__ == "__main__":
    run_debug()
