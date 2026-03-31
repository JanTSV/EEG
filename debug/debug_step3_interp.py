"""
debug_step3_interp.py
---------------------
Validates the channel interpolation step for sub-02.

CRITICAL FIX applied here: BioSemi hardware channel names (e.g. '2-A1') are
remapped to standard 10-20 labels using the label list exported from MATLAB/
FieldTrip. Only after this renaming can MNE locate electrodes on the standard
montage and perform distance-based interpolation correctly.

NOTE on method differences:
  MNE uses spherical spline interpolation; FieldTrip uses weighted neighbour
  averaging. The two outputs will therefore be highly correlated but not
  numerically identical. Both 'raw MAE' and 'centered MAE' (DC-offset removed)
  are reported to separate shape errors from offset errors.
"""

import mne
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────────────
raw_bdf_path    = Path("/Volumes/HardDiskYF/ACADEMIA/EEG/ds006761-download/"
                       "sub-02/eeg/sub-02_task-RPS_eeg.bdf")
mat_interp_path = Path("originalCode/debug_step3_interp.mat")

# Target bad channels in 10-20 nomenclature (as reported by FieldTrip)
BAD_CHANNELS_TARGET = ['FC5', 'T7', 'POz', 'P2']
# ─────────────────────────────────────────────────────────────────────────────


def run_debug():
    print("─── START: debug_step3 (INTERPOLATION WITH CHANNEL MAPPING) ───\n")

    # ── 1. Load MATLAB ground truth, labels, and trial definition ─────────────
    try:
        mat_data  = scipy.io.loadmat(str(mat_interp_path), squeeze_me=True)
        gt_interp = mat_data['debug_interp']
        trl       = mat_data['TRL']

        start_sample = trl[0, 0]
        end_sample   = trl[0, 1]

        # Clean up labels exported from a MATLAB cell array (strip whitespace/artefacts)
        gt_labels = [str(l).strip() for l in mat_data['labels']]
        print(f"  MATLAB labels loaded : {len(gt_labels)} channels")

    except Exception as e:
        print(f"ERROR – could not load .mat files: {e}")
        print("  Hint: ensure 'labels' and 'TRL' are exported in the MATLAB script.")
        return

    # ── 2. Load raw BDF via MNE ───────────────────────────────────────────────
    print(f"  Loading raw file : {raw_bdf_path.name}")
    raw = mne.io.read_raw_bdf(raw_bdf_path, preload=True, verbose='error')

    # Select Player 2 channels.
    # The channel order in the BDF is assumed to match MATLAB's order before renaming.
    py_picks = [ch for ch in raw.ch_names if ('2-A' in ch) or ('2-B' in ch)]
    raw.pick_channels(py_picks)

    # ── 3. CRITICAL FIX: remap hardware names → 10-20 labels ─────────────────
    # Without this step MNE cannot assign electrode positions from the standard
    # montage, and the spline interpolation will fail or use wrong geometry.
    if len(raw.ch_names) != len(gt_labels):
        print(f"FATAL – channel count mismatch: "
              f"Python {len(raw.ch_names)} vs. MATLAB {len(gt_labels)}")
        return

    print("  Applying channel mapping (hardware names → 10-20 labels) ...")
    rename_map = {old: new for old, new in zip(raw.ch_names, gt_labels)}
    raw.rename_channels(rename_map)

    # Assign standard electrode positions (now possible because names are standard)
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage, on_missing='ignore')

    # ── 4. Extract and epoch the trial segment ────────────────────────────────
    idx_start = int(start_sample) - 1   # convert to 0-based inclusive start
    idx_end   = int(end_sample)          # 0-based exclusive stop

    # Scale from Volts (MNE default) to µV to match MATLAB output
    data_raw = raw.get_data(start=idx_start, stop=idx_end) * 1e6

    info   = raw.info.copy()
    epochs = mne.EpochsArray(
        data_raw[np.newaxis, :, :], info, tmin=0, verbose='error'
    )

    # ── 5. Interpolation ──────────────────────────────────────────────────────
    # MNE: spherical spline interpolation.
    # FieldTrip: weighted neighbour averaging (IDW).
    # → Results will be highly correlated but not numerically identical.
    epochs.info['bads'] = BAD_CHANNELS_TARGET
    print(f"  Interpolating bad channels : {epochs.info['bads']}")
    epochs.interpolate_bads(reset_bads=False, verbose='error')

    py_interp = epochs.get_data()[0]   # reduce to (n_channels, n_times)

    # ── 6. Numerical comparison (bad channels only) ───────────────────────────
    bad_indices = [epochs.ch_names.index(ch)
                   for ch in BAD_CHANNELS_TARGET if ch in epochs.ch_names]

    print("\n─── RESULTS (MNE spline vs. FieldTrip neighbour) ───────────────")
    print(f"{'Channel':<8} | {'Corr':<8} | {'Raw MAE':<10} | "
          f"{'Centered MAE':<14} | {'PTP':<8}")
    print("─" * 60)

    for idx, name in zip(bad_indices, BAD_CHANNELS_TARGET):
        py_trace = py_interp[idx, :]
        gt_trace = gt_interp[idx, :]

        # Raw error (includes any DC offset between methods)
        raw_mae = np.mean(np.abs(py_trace - gt_trace))

        # Centered error (DC offset removed → pure waveform shape comparison)
        py_centered  = py_trace - np.mean(py_trace)
        gt_centered  = gt_trace - np.mean(gt_trace)
        center_mae   = np.mean(np.abs(py_centered - gt_centered))

        corr = np.corrcoef(py_trace, gt_trace)[0, 1]
        ptp  = np.ptp(gt_trace)

        print(f"{name:<8} | {corr:.4f}   | {raw_mae:<10.1f} | "
              f"{center_mae:<14.2f} | {ptp:.1f}")

    # ── 7. Diagnostic plot (worst-correlation bad channel) ────────────────────
    if not bad_indices:
        return

    # Select the channel with the lowest correlation for the most informative plot
    corrs          = [np.corrcoef(py_interp[i], gt_interp[i])[0, 1]
                      for i in bad_indices]
    worst_corr_pos = np.argmin(corrs)
    plot_idx       = bad_indices[worst_corr_pos]
    plot_name      = BAD_CHANNELS_TARGET[worst_corr_pos]

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # Panel 1 – raw interpolated signals overlay
    ax = axes[0]
    ax.set_title(f"Interpolation comparison – {plot_name}  "
                 f"(corr = {corrs[worst_corr_pos]:.4f})")
    ax.plot(py_interp[plot_idx, :], label='Python (spline)',
            color='blue',   alpha=0.8)
    ax.plot(gt_interp[plot_idx, :], '--', label='MATLAB (FT neighbour)',
            color='orange', alpha=0.8)
    ax.set_ylabel('Amplitude (µV)')
    ax.legend()
    ax.grid(True)

    # Panel 2 – residual (reflects method difference, not a pipeline error)
    ax = axes[1]
    ax.set_title(f"Residual (Python − MATLAB) – {plot_name}  "
                 f"[method difference: spline vs. neighbour]")
    ax.plot(py_interp[plot_idx, :] - gt_interp[plot_idx, :],
            color='red', alpha=0.6)
    ax.set_ylabel('Difference (µV)')
    ax.grid(True)

    plt.tight_layout()
    out_file = "debug_step3_interp_fixed.png"
    plt.savefig(out_file)
    print(f"\nPlot saved: {out_file}")
    plt.show()


if __name__ == "__main__":
    run_debug()
