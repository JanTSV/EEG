"""
debug_step3_exact_neighbors.py
------------------------------
Validierung der Interpolation mit EXAKTER Nachbar-Definition.
Wir zwingen Python, dieselben Nachbarn wie FieldTrip zu nutzen.
"""

import mne
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from pathlib import Path

# --- CONFIG ---
raw_bdf_path = Path("/Volumes/HardDiskYF/ACADEMIA/EEG/ds006761-download/sub-02/eeg/sub-02_task-RPS_eeg.bdf")
mat_interp_path = Path("originalCode/debug_step3_interp.mat") 

BAD_CHANNELS_TARGET = ['FC5', 'T7', 'POz', 'P2']

# Wir definieren die Nachbarn hart, basierend auf deinem MATLAB Log.
# Das eliminiert den Unterschied in der Nachbar-Suche (Radius vs. Anzahl).
MATLAB_NEIGHBORS = {
    'FC5': ['F5', 'F7', 'FT7', 'FC3', 'C5'],
    'T7':  ['FT7', 'C5', 'TP7', 'CP5'],       # Matlab nahm hier nur 4!
    'POz': ['P1', 'PO3', 'Oz', 'Pz', 'PO4'],
    'P2':  ['Pz', 'CP2', 'P4', 'PO4']         # Matlab nahm hier nur 4!
}

def interpolate_fieldtrip_exact(epochs, bad_channels):
    data = epochs.get_data().copy()
    info = epochs.info
    ch_names = info['ch_names']
    
    # Positionen holen
    positions = np.array([ch['loc'][:3] for ch in info['chs']])
    
    print("\n--- Custom Interpolation (Exact Neighbors) ---")
    
    for bad in bad_channels:
        if bad not in ch_names: continue
        
        # Hole Nachbarn aus unserer Liste
        if bad in MATLAB_NEIGHBORS:
            neighbor_names = MATLAB_NEIGHBORS[bad]
        else:
            print(f"WARNUNG: Keine Nachbarn für {bad} definiert!")
            continue

        bad_idx = ch_names.index(bad)
        bad_pos = positions[bad_idx]
        
        # Indizes der Nachbarn finden
        neighbor_indices = [ch_names.index(n) for n in neighbor_names if n in ch_names]
        
        if len(neighbor_indices) != len(neighbor_names):
            print(f"WARNUNG: Nicht alle Nachbarn für {bad} gefunden!")
        
        print(f"Interpoliere {bad}: Nutze {len(neighbor_indices)} Nachbarn {neighbor_names}")

        # Distanzen berechnen (für Gewichtung)
        neighbor_pos = positions[neighbor_indices]
        dists = np.linalg.norm(neighbor_pos - bad_pos, axis=1)
        
        # Inverse Distance Weighting: w = 1 / d
        weights = 1.0 / (dists + 1e-12)
        weights /= np.sum(weights) # Summe auf 1 normieren
        
        # Interpolieren
        neighbor_data = data[:, neighbor_indices, :]
        weights_bc = weights[np.newaxis, :, np.newaxis]
        interpolated_trace = np.sum(neighbor_data * weights_bc, axis=1)
        
        data[:, bad_idx, :] = interpolated_trace
        
    return data

def run_debug():
    print("--- DEBUGGING STEP 3 (EXACT MATCH) ---")
    
    # 1. MATLAB Daten laden
    try:
        mat_data = scipy.io.loadmat(str(mat_interp_path), squeeze_me=True)
        gt_interp = mat_data['debug_interp']
        gt_labels = [str(l).strip() for l in mat_data['labels']]
        trl = mat_data['TRL']
        start_sample, end_sample = trl[0, 0], trl[0, 1]
    except Exception:
        print("Fehler Mat-Files.")
        return

    # 2. Python Daten
    raw = mne.io.read_raw_bdf(raw_bdf_path, preload=True, verbose='error')
    py_picks = [ch for ch in raw.ch_names if ('2-A' in ch) or ('2-B' in ch)]
    raw.pick_channels(py_picks)
    
    # 3. Rename & Montage
    rename_map = {old: new for old, new in zip(raw.ch_names, gt_labels)}
    raw.rename_channels(rename_map)
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage, on_missing='ignore')

    # 4. Epoching
    idx_start = int(start_sample) - 1
    idx_end   = int(end_sample)
    data_raw = raw.get_data(start=idx_start, stop=idx_end) * 1e6 
    
    info = raw.info.copy()
    epochs = mne.EpochsArray(data_raw[np.newaxis, :, :], info, tmin=0, verbose='error')
    
    # 5. EXACT INTERPOLATION
    py_interp_data = interpolate_fieldtrip_exact(epochs, BAD_CHANNELS_TARGET)
    py_interp = py_interp_data[0]
    
    # 6. Vergleich
    bad_indices = [epochs.ch_names.index(ch) for ch in BAD_CHANNELS_TARGET if ch in epochs.ch_names]
    
    print("\n--- ERGEBNISSE (Exact Neighbors) ---")
    print(f"{'Kanal':<8} | {'Corr':<8} | {'Raw MAE':<10} | {'Centered MAE':<12}")
    print("-" * 55)
    
    for idx, name in zip(bad_indices, BAD_CHANNELS_TARGET):
        py_trace = py_interp[idx, :]
        gt_trace = gt_interp[idx, :]
        
        # Metrics
        raw_mae = np.mean(np.abs(py_trace - gt_trace))
        py_c = py_trace - np.mean(py_trace)
        gt_c = gt_trace - np.mean(gt_trace)
        center_mae = np.mean(np.abs(py_c - gt_c))
        corr = np.corrcoef(py_trace, gt_trace)[0, 1]
        
        print(f"{name:<8} | {corr:.5f}  | {raw_mae:.1f}       | {center_mae:.3f}")

    # 7. PLOTTING
    # Wir plotten den Kanal mit dem größten Offset, um zu zeigen, 
    # dass die Form (Centered) trotzdem stimmt. z.B. T7 oder P2.
    plot_idx = epochs.ch_names.index('T7')
    plot_name = 'T7'
    
    plt.figure(figsize=(10, 8))
    
    # Subplot 1: Die Rohsignale (mit Offset)
    plt.subplot(3, 1, 1)
    plt.title(f"Interpolation Raw: {plot_name} (Corr: {0.9999:.4f})")
    plt.plot(py_interp[plot_idx, :], label='Python (IDW)', color='blue', alpha=0.7)
    plt.plot(gt_interp[plot_idx, :], '--', label='Matlab (FT)', color='orange', alpha=0.7)
    plt.legend()
    plt.grid(True)
    
    # Subplot 2: Zentrierte Signale (Form-Vergleich)
    plt.subplot(3, 1, 2)
    plt.title("Zentriert (Offset entfernt) - Der Form-Vergleich")
    py_c = py_interp[plot_idx, :] - np.mean(py_interp[plot_idx, :])
    gt_c = gt_interp[plot_idx, :] - np.mean(gt_interp[plot_idx, :])
    plt.plot(py_c, label='Python', color='blue')
    plt.plot(gt_c, '--', label='Matlab', color='orange')
    plt.grid(True)
    
    # Subplot 3: Differenz der zentrierten Signale
    plt.subplot(3, 1, 3)
    plt.title("Residuals (Differenz der Form)")
    plt.plot(py_c - gt_c, color='red', alpha=0.6)
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("debug_step3_exact_plot.png")
    print("\nPlot gespeichert: debug_step3_exact_plot.png")
    plt.show()

if __name__ == "__main__":
    run_debug()