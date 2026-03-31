"""
debug_step2_resample.py
-----------------------
Validates the resampling step (2048 Hz → 256 Hz).

Uses Pair 1, Player 1 (no channel interpolation) to isolate the resampling
from any other processing. The Python/MNE output is compared sample-by-sample
against the MATLAB ground-truth exported after Step 2.

NOTE on method differences:
  MNE uses FFT-based resampling; FieldTrip typically uses a polyphase filter
  ('resample'). Residuals may therefore be non-zero even for a correct pipeline.
"""

import mne
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────────────
raw_bdf_path    = Path("/Volumes/HardDiskYF/ACADEMIA/EEG/ds006761-download/"
                       "sub-01/eeg/sub-01_task-RPS_eeg.bdf")
mat_epoch_path  = Path("originalCode/debug_step1_epoched.mat")  # used for TRL info
mat_resamp_path = Path("originalCode/debug_step2_resamp.mat")   # MATLAB Step 2 target

TARGET_FS = 256   # target sampling frequency in Hz
# ─────────────────────────────────────────────────────────────────────────────


def run_debug():
    print("─── START: debug_step2 (RESAMPLING) ───\n")

    # ── 1. Load MATLAB ground truth (resampled) ───────────────────────────────
    try:
        mat_res   = scipy.io.loadmat(str(mat_resamp_path), squeeze_me=True)
        gt_resamp = mat_res['debug_resamp']
        print(f"  MATLAB resampled shape : {gt_resamp.shape}")

        # Also load the trial definition to replicate the same epoch boundaries
        mat_trl          = scipy.io.loadmat(
            str(Path("originalCode/debug_step1_trl.mat")), squeeze_me=True
        )
        trl              = mat_trl['TRL']
        start_sample_mat = trl[0, 0]   # 1-based sample index (MATLAB convention)
        end_sample_mat   = trl[0, 1]

    except Exception as e:
        print(f"ERROR – could not load .mat files: {e}")
        return

    # ── 2. Reconstruct the Python pipeline ───────────────────────────────────
    print("Running Python pipeline ...")
    raw = mne.io.read_raw_bdf(raw_bdf_path, preload=True, verbose='error')

    # Select Player 1 channels (same criterion as Step 1)
    py_picks = [ch for ch in raw.ch_names if ('2-A' in ch) or ('2-B' in ch)]
    raw.pick_channels(py_picks)

    # Extract the trial segment (0-based indexing)
    idx_start = int(start_sample_mat) - 1   # inclusive start
    idx_end   = int(end_sample_mat)          # exclusive stop

    # Retrieve raw data for this single trial: shape (n_channels, n_times)
    data_raw = raw.get_data(start=idx_start, stop=idx_end)
    info     = raw.info

    # Wrap in an MNE EpochsArray (required shape: n_epochs × n_channels × n_times).
    # We use a single fake epoch so MNE's resampling routine can be applied cleanly.
    data_3d = data_raw[np.newaxis, :, :]       # add epoch dimension
    events  = np.array([[0, 0, 1]])            # minimal dummy event
    epochs  = mne.EpochsArray(
        data_3d, info, events=events, tmin=0, verbose='error'
    )

    # ── 3. Resample ───────────────────────────────────────────────────────────
    # MNE uses FFT-based resampling; FieldTrip uses a polyphase filter, so
    # small numerical differences between the two outputs are expected.
    print(f"  Resampling to {TARGET_FS} Hz ...")
    epochs.resample(TARGET_FS)

    py_resamp = epochs.get_data()[0]   # back to (n_channels, n_times)

    # ── 4. Unit correction ───────────────────────────────────────────────────
    # MNE stores data in Volts; MATLAB often exports µV.
    ratio = np.mean(np.abs(py_resamp)) / np.mean(np.abs(gt_resamp))
    if abs(ratio - 1e-6) < 1e-8:
        py_resamp *= 1e6
        print("  AUTO-FIX: Python data scaled × 1e6  (V → µV)")

    # ── 5. Shape alignment ───────────────────────────────────────────────────
    # Downsampling can introduce a ±1 sample difference due to rounding.
    # Truncate both arrays to the shorter length before computing residuals.
    if py_resamp.shape != gt_resamp.shape:
        print(f"  WARNING: shape mismatch – "
              f"Python {py_resamp.shape} vs. MATLAB {gt_resamp.shape}. "
              f"Truncating to shorter length.")
        min_len   = min(py_resamp.shape[1], gt_resamp.shape[1])
        py_resamp = py_resamp[:, :min_len]
        gt_resamp = gt_resamp[:, :min_len]

    # ── 6. Numerical comparison ───────────────────────────────────────────────
    diff = py_resamp - gt_resamp
    mae  = np.mean(np.abs(diff))
    print(f"\n─── RESULTS ───────────────────────────────────")
    print(f"  Global MAE (all channels) : {mae:.5e}")

    # Identify the worst-case channel for the diagnostic plot
    max_err_per_ch = np.max(np.abs(diff), axis=1)
    worst_idx      = np.argmax(max_err_per_ch)
    worst_ch       = py_picks[worst_idx]
    print(f"  Worst channel             : {worst_ch}  (index {worst_idx})")
    print(f"  Max absolute error there  : {max_err_per_ch[worst_idx]:.5e}")

    # ── 7. Diagnostic plot ────────────────────────────────────────────────────
    zoom = 50   # number of samples shown in the overlay panel

    plt.figure(figsize=(10, 6))

    # Panel 1 – overlay of the first N resampled samples (worst channel)
    plt.subplot(2, 1, 1)
    plt.title(f"Resampling comparison – {worst_ch}  (worst-case channel, "
              f"first {zoom} samples)")
    plt.plot(py_resamp[worst_idx, :zoom], '.-', label='Python')
    plt.plot(gt_resamp[worst_idx, :zoom], 'x--', label='MATLAB')
    plt.ylabel('Amplitude (µV)')
    plt.legend()
    plt.grid(True)

    # Panel 2 – full residual trace for the worst channel
    plt.subplot(2, 1, 2)
    plt.title(f"Residual (Python − MATLAB) – {worst_ch}")
    plt.plot(diff[worst_idx, :], color='red')
    plt.ylabel('Difference (µV)')
    plt.grid(True)

    plt.tight_layout()
    out_file = "debug_step2_resample.png"
    plt.savefig(out_file)
    print(f"\nPlot saved: {out_file}")
    plt.show()


if __name__ == "__main__":
    run_debug()
