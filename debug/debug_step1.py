"""
debug_step1.py (ALL CHANNELS)
------------------------------
Validates all channels: Python (MNE) vs. MATLAB output.

Loads the first trial from a BDF file using MNE and compares it
sample-by-sample against the ground-truth epoch exported from MATLAB.
Produces a three-panel diagnostic plot and prints a numerical summary.
"""

import mne
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────────────
# Test case: Pair 1, Player 1 (mirrors the MATLAB reference)
raw_bdf_path   = Path("/Volumes/HardDiskYF/ACADEMIA/EEG/ds006761-download/"
                      "sub-01/eeg/sub-01_task-RPS_eeg.bdf")
mat_epoch_path = Path("originalCode/debug_step1_epoched.mat")  # MATLAB Step 1 epoch
mat_trl_path   = Path("originalCode/debug_step1_trl.mat")      # MATLAB Step 1 trial definition
# ─────────────────────────────────────────────────────────────────────────────


def run_debug():
    print("─── START: debug_step1 (ALL CHANNELS) ───\n")

    # ── 1. Load MATLAB reference data ────────────────────────────────────────
    print("Loading MATLAB data ...")
    try:
        mat_data = scipy.io.loadmat(str(mat_epoch_path), squeeze_me=True)
        mat_trl  = scipy.io.loadmat(str(mat_trl_path),   squeeze_me=True)

        gt_trial_1       = mat_data['debug_data']   # ground-truth epoch (channels × samples)
        trl              = mat_trl['TRL']
        start_sample_mat = trl[0, 0]                # 1-based sample index (MATLAB convention)
        end_sample_mat   = trl[0, 1]

        print(f"  MATLAB shape : {gt_trial_1.shape}")

    except Exception as e:
        print(f"ERROR – could not load .mat files: {e}")
        return

    # ── 2. Load raw BDF via MNE ───────────────────────────────────────────────
    print("\nLoading Python / MNE data ...")
    raw = mne.io.read_raw_bdf(raw_bdf_path, preload=True, verbose='error')

    # ── 3. Channel selection (Player 1) ──────────────────────────────────────
    # Keep only channels whose names contain '2-A' or '2-B' (Player 1 electrodes).
    # NOTE: order is preserved as it appears in the file; MNE does not re-sort here.
    py_picks = [ch for ch in raw.ch_names if ('2-A' in ch) or ('2-B' in ch)]
    raw.pick_channels(py_picks)
    print(f"  Python channels selected : {len(py_picks)}")

    # ── 4. Extract the matching time segment ─────────────────────────────────
    # Convert from 1-based MATLAB indices to 0-based Python indices.
    idx_start = int(start_sample_mat) - 1   # inclusive start (0-based)
    idx_end   = int(end_sample_mat)          # exclusive stop  (0-based)

    py_data_trial_1, times = raw.get_data(
        start=idx_start, stop=idx_end, return_times=True
    )

    # ── 5. Shape alignment ───────────────────────────────────────────────────
    # If MATLAB exported (samples × channels) instead of (channels × samples),
    # transpose so both arrays share the same layout.
    if py_data_trial_1.shape != gt_trial_1.shape:
        if py_data_trial_1.shape == gt_trial_1.T.shape:
            gt_trial_1 = gt_trial_1.T
            print("  INFO: MATLAB array transposed to match Python layout.")
        else:
            print(f"CRITICAL – shape mismatch! "
                  f"Python: {py_data_trial_1.shape}  |  MATLAB: {gt_trial_1.shape}")
            return

    # ── 6. Unit / scaling check ───────────────────────────────────────────────
    # MNE stores data in Volts; MATLAB often uses µV.
    # Auto-detect and correct a factor-of-1e6 discrepancy.
    mean_py  = np.mean(np.abs(py_data_trial_1))
    mean_mat = np.mean(np.abs(gt_trial_1))
    ratio    = mean_py / mean_mat if mean_mat != 0 else 0

    if abs(ratio - 1e-6) < 1e-8:
        print(">>> AUTO-FIX: scaling Python data × 1e6  (V → µV)")
        py_data_trial_1 *= 1e6
    elif abs(ratio - 1e6) < 1e-1:
        print(">>> AUTO-FIX: scaling MATLAB data × 1e6  (V → µV)")
        gt_trial_1 *= 1e6

    # ── 7. Numerical comparison ───────────────────────────────────────────────
    diff_matrix = py_data_trial_1 - gt_trial_1                    # residual matrix

    max_err_per_ch = np.max(np.abs(diff_matrix), axis=1)          # max |error| per channel
    worst_ch_idx   = np.argmax(max_err_per_ch)                     # channel with largest error
    worst_ch_name  = py_picks[worst_ch_idx]
    worst_err      = max_err_per_ch[worst_ch_idx]
    global_mae     = np.mean(np.abs(diff_matrix))

    print(f"\n─── RESULTS ───────────────────────────────────")
    print(f"  Global MAE (all channels) : {global_mae:.5e}")
    print(f"  Worst channel             : {worst_ch_name}  (index {worst_ch_idx})")
    print(f"  Max absolute error there  : {worst_err:.5e}")

    if worst_err < 1e-10:
        print(">>> SUCCESS: Perfect agreement across all channels.")
    else:
        print(">>> CHECK: Deviations detected – inspect the plot below.")

    # ── 8. Diagnostic plot ────────────────────────────────────────────────────
    print("\n─── PLOT ───────────────────────────────────────")
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))

    # Panel 1 – Butterfly plot of residuals (all channels)
    # A channel that deviates strongly will stand out immediately.
    ax = axes[0]
    ax.set_title(f'Residuals (Python − MATLAB) — all {len(py_picks)} channels')
    ax.plot(times, diff_matrix.T, color='black', alpha=0.3, linewidth=0.5)
    ax.set_ylabel('Difference (µV)')
    ax.grid(True)

    # Panel 2 – Overlay of worst-case channel
    ax = axes[1]
    ax.set_title(f'Worst channel overlay: {worst_ch_name}  '
                 f'(max error = {worst_err:.2e})')
    ax.plot(times, py_data_trial_1[worst_ch_idx, :],
            label='Python', color='blue',   alpha=0.8)
    ax.plot(times, gt_trial_1[worst_ch_idx, :],
            label='MATLAB',  color='orange', linestyle='--', alpha=0.8)
    ax.legend(loc='upper right')
    ax.grid(True)

    # Panel 3 – Zoom into the first N samples of the worst channel
    # Useful for detecting sample-index shifts or off-by-one errors.
    ax = axes[2]
    zoom = 50
    ax.set_title(f'Zoom: first {zoom} samples — {worst_ch_name}')
    ax.plot(times[:zoom], py_data_trial_1[worst_ch_idx, :zoom],
            '.-', label='Python', color='blue')
    ax.plot(times[:zoom], gt_trial_1[worst_ch_idx, :zoom],
            'x--', label='MATLAB',  color='orange')
    ax.legend(loc='upper right')
    ax.grid(True)

    plt.tight_layout()
    out_file = "debug_step1_all_channels.png"
    plt.savefig(out_file)
    print(f"Plot saved: {out_file}")
    plt.show()


if __name__ == "__main__":
    run_debug()
