"""
debug_step3_exact_neighbors.py
-------------------------------
Validates channel interpolation using the exact neighbour sets reported by
FieldTrip's ft_prepare_neighbours.

Rather than re-running neighbour detection (which differs between FieldTrip's
radius-based search and MNE's distance-based approach), the neighbours for each
bad channel are hard-coded from the MATLAB log. This isolates the interpolation
algorithm itself from any neighbour-definition mismatch.

Interpolation method: Inverse Distance Weighting (IDW), w = 1 / d,
normalised so weights sum to 1 — the same scheme used by FieldTrip.
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

# Channels to be interpolated (Player 2 bad channels for this subject)
BAD_CHANNELS_TARGET = ['FC5', 'T7', 'POz', 'P2']

# Neighbour sets taken verbatim from the MATLAB/FieldTrip log.
# NOTE: FieldTrip selected fewer neighbours for T7 and P2 (4 instead of 5)
# because those electrodes have fewer physical neighbours within the search radius.
MATLAB_NEIGHBORS = {
    'FC5': ['F5', 'F7', 'FT7', 'FC3', 'C5'],
    'T7':  ['FT7', 'C5', 'TP7', 'CP5'],        # only 4 neighbours in MATLAB
    'POz': ['P1', 'PO3', 'Oz', 'Pz', 'PO4'],
    'P2':  ['Pz', 'CP2', 'P4', 'PO4'],          # only 4 neighbours in MATLAB
}
# ─────────────────────────────────────────────────────────────────────────────


def interpolate_fieldtrip_exact(epochs, bad_channels):
    """
    Interpolate bad channels using IDW with the fixed neighbour sets defined
    in MATLAB_NEIGHBORS. Operates on a copy of the epoch data array.

    Parameters
    ----------
    epochs       : mne.EpochsArray
    bad_channels : list of str

    Returns
    -------
    data : np.ndarray, shape (n_epochs, n_channels, n_times)
        Data array with bad channels replaced by their IDW estimates.
    """
    data     = epochs.get_data().copy()
    info     = epochs.info
    ch_names = info['ch_names']

    # Electrode positions (first 3 values of the 'loc' vector = x, y, z)
    positions = np.array([ch['loc'][:3] for ch in info['chs']])

    print("\n─── Custom Interpolation (exact FieldTrip neighbours) ───")

    for bad in bad_channels:
        if bad not in ch_names:
            continue

        if bad not in MATLAB_NEIGHBORS:
            print(f"  WARNING: no neighbours defined for '{bad}' — skipping.")
            continue

        neighbor_names   = MATLAB_NEIGHBORS[bad]
        bad_idx          = ch_names.index(bad)
        bad_pos          = positions[bad_idx]

        # Resolve neighbour names to array indices
        neighbor_indices = [ch_names.index(n) for n in neighbor_names
                            if n in ch_names]

        if len(neighbor_indices) != len(neighbor_names):
            print(f"  WARNING: not all neighbours found for '{bad}'.")

        print(f"  Interpolating '{bad}': "
              f"{len(neighbor_indices)} neighbours {neighbor_names}")

        # Inverse Distance Weighting: w_i = 1 / d_i,  Σw_i = 1
        neighbor_pos = positions[neighbor_indices]
        dists        = np.linalg.norm(neighbor_pos - bad_pos, axis=1)
        weights      = 1.0 / (dists + 1e-12)   # small epsilon avoids division by zero
        weights     /= np.sum(weights)

        # Weighted sum across the neighbour channels
        neighbor_data    = data[:, neighbor_indices, :]          # (epochs, neighbours, time)
        weights_bc       = weights[np.newaxis, :, np.newaxis]    # broadcast shape
        interpolated     = np.sum(neighbor_data * weights_bc, axis=1)

        data[:, bad_idx, :] = interpolated

    return data


def run_debug():
    print("─── START: debug_step3 (EXACT NEIGHBOUR INTERPOLATION) ───\n")

    # ── 1. Load MATLAB ground truth ───────────────────────────────────────────
    try:
        mat_data  = scipy.io.loadmat(str(mat_interp_path), squeeze_me=True)
        gt_interp = mat_data['debug_interp']
        gt_labels = [str(l).strip() for l in mat_data['labels']]
        trl       = mat_data['TRL']
        start_sample, end_sample = trl[0, 0], trl[0, 1]
    except Exception as e:
        print(f"ERROR – could not load .mat files: {e}")
        return

    # ── 2. Load raw BDF via MNE ───────────────────────────────────────────────
    raw = mne.io.read_raw_bdf(raw_bdf_path, preload=True, verbose='error')

    # Select Player 2 channels
    py_picks = [ch for ch in raw.ch_names if ('2-A' in ch) or ('2-B' in ch)]
    raw.pick_channels(py_picks)

    # ── 3. Rename channels and apply standard montage ─────────────────────────
    # Map generic BDF channel names to the 10-20 labels from MATLAB so that
    # neighbour lookups and montage positions are consistent.
    rename_map = {old: new for old, new in zip(raw.ch_names, gt_labels)}
    raw.rename_channels(rename_map)
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
    py_interp_data = interpolate_fieldtrip_exact(epochs, BAD_CHANNELS_TARGET)
    py_interp      = py_interp_data[0]   # reduce to (n_channels, n_times)

    # ── 6. Numerical comparison (bad channels only) ───────────────────────────
    bad_indices = [epochs.ch_names.index(ch)
                   for ch in BAD_CHANNELS_TARGET if ch in epochs.ch_names]

    print("\n─── RESULTS (exact neighbours) ─────────────────────────────")
    print(f"{'Channel':<8} | {'Corr':<8} | {'Raw MAE':<10} | {'Centered MAE':<12}")
    print("─" * 55)

    for idx, name in zip(bad_indices, BAD_CHANNELS_TARGET):
        py_trace = py_interp[idx, :]
        gt_trace = gt_interp[idx, :]

        raw_mae    = np.mean(np.abs(py_trace - gt_trace))
        py_c       = py_trace - np.mean(py_trace)
        gt_c       = gt_trace - np.mean(gt_trace)
        center_mae = np.mean(np.abs(py_c - gt_c))
        corr       = np.corrcoef(py_trace, gt_trace)[0, 1]

        print(f"{name:<8} | {corr:.5f}  | {raw_mae:<10.1f} | {center_mae:.3f}")

    # ── 7. Diagnostic plot (channel T7 — largest DC offset) ──────────────────
    # T7 typically shows the largest raw offset between Python and MATLAB,
    # making it the most informative channel to inspect visually.
    plot_ch   = 'T7'
    plot_idx  = epochs.ch_names.index(plot_ch)
    plot_corr = np.corrcoef(py_interp[plot_idx, :], gt_interp[plot_idx, :])[0, 1]

    py_c_plot = py_interp[plot_idx, :] - np.mean(py_interp[plot_idx, :])
    gt_c_plot = gt_interp[plot_idx, :] - np.mean(gt_interp[plot_idx, :])

    fig, axes = plt.subplots(3, 1, figsize=(10, 8))

    # Panel 1 – raw interpolated signals (may show a DC offset)
    ax = axes[0]
    ax.set_title(f"Interpolation – raw signals: {plot_ch}  "
                 f"(corr = {plot_corr:.4f})")
    ax.plot(py_interp[plot_idx, :], label='Python (IDW)', color='blue',   alpha=0.7)
    ax.plot(gt_interp[plot_idx, :], '--', label='MATLAB (FT)', color='orange', alpha=0.7)
    ax.set_ylabel('Amplitude (µV)')
    ax.legend()
    ax.grid(True)

    # Panel 2 – mean-centred signals (isolates waveform shape from DC offset)
    ax = axes[1]
    ax.set_title(f"Mean-centred signals: {plot_ch}  (shape comparison)")
    ax.plot(py_c_plot, label='Python', color='blue')
    ax.plot(gt_c_plot, '--', label='MATLAB', color='orange')
    ax.set_ylabel('Amplitude (µV)')
    ax.legend()
    ax.grid(True)

    # Panel 3 – residual of the centred signals
    ax = axes[2]
    ax.set_title(f"Residual of centred signals: {plot_ch}")
    ax.plot(py_c_plot - gt_c_plot, color='red', alpha=0.6)
    ax.set_ylabel('Difference (µV)')
    ax.grid(True)

    plt.tight_layout()
    out_file = "debug_step3_exact_plot.png"
    plt.savefig(out_file)
    print(f"\nPlot saved: {out_file}")
    plt.show()


if __name__ == "__main__":
    run_debug()
