"""
debug_step2_resample_padding.py  (BRUTE-FORCE PADDING SEARCH)
--------------------------------------------------------------
Validates the resampling step by systematically testing which edge-padding
strategy best reproduces MATLAB's built-in 'resample' function.

MATLAB's 'resample' uses a polyphase filter with a Kaiser window (beta=5.0)
and a specific boundary treatment that is not directly documented. This script
iterates over all padding modes available in scipy.signal.resample_poly,
selects the one with the lowest MAE against the MATLAB ground truth, and
produces a three-panel diagnostic plot for the winning configuration.
"""

import mne
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from scipy.signal import resample_poly
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────────────
raw_bdf_path    = Path("/Volumes/HardDiskYF/ACADEMIA/EEG/ds006761-download/"
                       "sub-01/eeg/sub-01_task-RPS_eeg.bdf")
mat_epoch_path  = Path("originalCode/debug_step1_epoched.mat")  # used for TRL info
mat_resamp_path = Path("originalCode/debug_step2_resamp.mat")   # MATLAB Step 2 target

TARGET_FS = 256    # target sampling frequency in Hz
ORIG_FS   = 2048   # original sampling frequency of the BDF file
# ─────────────────────────────────────────────────────────────────────────────


def run_debug():
    print("─── START: debug_step2 (BRUTE-FORCE PADDING SEARCH) ───\n")

    # ── 1. Load MATLAB ground truth (resampled) ───────────────────────────────
    try:
        mat_res   = scipy.io.loadmat(str(mat_resamp_path), squeeze_me=True)
        gt_resamp = mat_res['debug_resamp']
        print(f"  MATLAB resampled shape : {gt_resamp.shape}")

        # Load trial definition to extract the same epoch boundaries as Step 1
        mat_trl          = scipy.io.loadmat(
            str(Path("originalCode/debug_step1_trl.mat")), squeeze_me=True
        )
        trl              = mat_trl['TRL']
        start_sample_mat = trl[0, 0]   # 1-based (MATLAB convention)
        end_sample_mat   = trl[0, 1]

    except Exception as e:
        print(f"ERROR – could not load .mat files: {e}")
        return

    # ── 2. Load and prepare raw Python data ───────────────────────────────────
    raw = mne.io.read_raw_bdf(raw_bdf_path, preload=True, verbose='error')

    # Select Player 1 channels (same criterion as Step 1)
    py_picks = [ch for ch in raw.ch_names if ('2-A' in ch) or ('2-B' in ch)]
    raw.pick_channels(py_picks)

    # Extract the trial segment (convert to 0-based indices)
    idx_start = int(start_sample_mat) - 1   # inclusive start
    idx_end   = int(end_sample_mat)          # exclusive stop
    data_raw  = raw.get_data(start=idx_start, stop=idx_end)
    print(f"  Python raw input shape : {data_raw.shape}")

    # Convert from Volts (MNE default) to µV to match MATLAB output
    data_raw *= 1e6

    # ── 3. Brute-force padding search ────────────────────────────────────────
    # Polyphase resampling ratio: 2048 → 256 Hz  ⟹  up=1, down=8
    up   = 1
    down = int(ORIG_FS / TARGET_FS)

    # All boundary-handling modes supported by scipy.signal.resample_poly
    pad_options = ['constant', 'line', 'mean', 'reflect', 'edge']

    best_pad  = None
    best_mae  = np.inf
    best_max  = np.inf
    best_data = None

    print(f"\n─── Padding comparison  "
          f"(factor {down}, window = Kaiser β=5.0) ───")
    print(f"{'PADDING':<10} | {'MAE (µV)':<12} | {'MAX (µV)':<12}")
    print("─" * 40)

    for pad in pad_options:
        try:
            # Polyphase filter with Kaiser window — matches MATLAB's default filter
            curr_resamp = resample_poly(
                data_raw, up, down, axis=1,
                padtype=pad, window=('kaiser', 5.0)
            )

            # Truncate to the shorter length if a ±1-sample rounding difference exists
            if curr_resamp.shape[1] != gt_resamp.shape[1]:
                min_len     = min(curr_resamp.shape[1], gt_resamp.shape[1])
                curr_resamp = curr_resamp[:, :min_len]
                curr_gt     = gt_resamp[:, :min_len]
            else:
                curr_gt = gt_resamp

            diff    = curr_resamp - curr_gt
            mae     = np.mean(np.abs(diff))
            max_err = np.max(np.abs(diff))

            print(f"{pad:<10} | {mae:.5e}  | {max_err:.5e}")

            # Track the best-performing padding mode
            if mae < best_mae:
                best_mae  = mae
                best_max  = max_err
                best_pad  = pad
                best_data = curr_resamp

        except Exception as e:
            print(f"{pad:<10} | ERROR: {e}")

    print("─" * 40)
    print(f">>> WINNER: '{best_pad}'  (MAE = {best_mae:.5e},  MAX = {best_max:.5e})")

    # ── 4. Diagnostic plot (winning padding mode only) ────────────────────────
    if best_data is None:
        print("No successful resampling — aborting plot.")
        return

    # Recompute residuals for the winning configuration
    min_len   = min(best_data.shape[1], gt_resamp.shape[1])
    best_data = best_data[:, :min_len]
    gt_plot   = gt_resamp[:, :min_len]
    diff      = best_data - gt_plot

    # Identify the worst-case channel
    max_err_per_ch = np.max(np.abs(diff), axis=1)
    worst_idx      = np.argmax(max_err_per_ch)
    worst_ch       = py_picks[worst_idx]

    zoom = 50   # number of samples shown in the start / end zoom panels

    fig, axes = plt.subplots(3, 1, figsize=(10, 8))

    # Panel 1 – zoom into the first N samples (check for onset artefacts)
    ax = axes[0]
    ax.set_title(f"Start (first {zoom} samples) – {worst_ch},  "
                 f"padding = '{best_pad}'")
    ax.plot(best_data[worst_idx, :zoom], '.-', label=f"Python ({best_pad})")
    ax.plot(gt_plot[worst_idx,   :zoom], 'x--', label='MATLAB')
    ax.set_ylabel('Amplitude (µV)')
    ax.legend()
    ax.grid(True)

    # Panel 2 – zoom into the last N samples (check for tail artefacts)
    ax = axes[1]
    ax.set_title(f"End (last {zoom} samples) – {worst_ch}")
    ax.plot(best_data[worst_idx, -zoom:], '.-', label=f"Python ({best_pad})")
    ax.plot(gt_plot[worst_idx,   -zoom:], 'x--', label='MATLAB')
    ax.set_ylabel('Amplitude (µV)')
    ax.legend()
    ax.grid(True)

    # Panel 3 – full residual trace for the worst channel
    ax = axes[2]
    ax.set_title(f"Residual (Python − MATLAB) – {worst_ch}  "
                 f"(MAE = {best_mae:.2e})")
    ax.plot(diff[worst_idx, :], color='red')
    ax.set_ylabel('Difference (µV)')
    ax.grid(True)

    plt.tight_layout()
    out_file = "debug_step2_padding_test.png"
    plt.savefig(out_file)
    print(f"\nPlot saved: {out_file}")
    plt.show()


if __name__ == "__main__":
    run_debug()
