"""
debug_step3_interp.py
---------------------
Validierung der Interpolation (sub-02).
CRITICAL FIX: Kanal-Mapping (BioSemi -> 10-20) basierend auf MATLAB Labels.
"""

import mne
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from pathlib import Path

# --- CONFIG ---
# Pfade (sub-02!)
raw_bdf_path = Path("/Volumes/HardDiskYF/ACADEMIA/EEG/ds006761-download/sub-02/eeg/sub-02_task-RPS_eeg.bdf")
mat_interp_path = Path("originalCode/debug_step3_interp.mat") 

# Bad Channels (Ziel-Namen im 10-20 System)
BAD_CHANNELS_TARGET = ['FC5', 'T7', 'POz', 'P2']

def run_debug():
    print("--- START DEBUGGING STEP 3 (INTERPOLATION WITH MAPPING) ---")
    
    # 1. Laden der MATLAB Ground Truth & Labels
    try:
        mat_data = scipy.io.loadmat(str(mat_interp_path), squeeze_me=True)
        gt_interp = mat_data['debug_interp']
        gt_labels = mat_data['labels'] # WICHTIG: Die 10-20 Namen aus FieldTrip
        trl = mat_data['TRL']
        
        start_sample = trl[0, 0] 
        end_sample   = trl[0, 1]
        
        # Labels säubern (Matlab Cell-Array Artefakte entfernen)
        gt_labels = [str(l).strip() for l in gt_labels]
        print(f"MATLAB Labels geladen: {len(gt_labels)} Kanäle.")
        
    except Exception as e:
        print(f"FEHLER beim Laden der .mat Files: {e}")
        print("Hinweis: Hast du 'labels' und 'TRL' im MATLAB-Skript exportiert?")
        return

    # 2. Python Daten laden
    print(f"Lade Raw: {raw_bdf_path.name}")
    raw = mne.io.read_raw_bdf(raw_bdf_path, preload=True, verbose='error')
    
    # Filter auf Player 1 (2-A... 2-B...)
    # Wir verlassen uns darauf, dass die Reihenfolge in der BDF dieselbe ist wie in MATLAB vor dem Rename.
    py_picks = [ch for ch in raw.ch_names if ('2-A' in ch) or ('2-B' in ch)]
    raw.pick_channels(py_picks)
    
    # --- CRITICAL FIX: MAPPING ---
    if len(raw.ch_names) != len(gt_labels):
        print(f"FATAL: Kanalanzahl ungleich! Py: {len(raw.ch_names)} vs Mat: {len(gt_labels)}")
        return

    print("Wende Kanal-Mapping an (Hardware -> 10-20)...")
    # Erstelle Dictionary: {'2-A1': 'Fp1', ...}
    rename_map = {old: new for old, new in zip(raw.ch_names, gt_labels)}
    raw.rename_channels(rename_map)
    
    # Montage setzen (Jetzt klappt es, weil die Namen Standard sind!)
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage, on_missing='ignore')
    # -----------------------------

    # 3. Epoching (Exakt wie MATLAB)
    idx_start = int(start_sample) - 1
    idx_end   = int(end_sample)
    
    # Rohdaten holen und auf uV skalieren
    data_raw = raw.get_data(start=idx_start, stop=idx_end) * 1e6
    
    # 4. Interpolation
    info = raw.info.copy()
    epochs = mne.EpochsArray(data_raw[np.newaxis, :, :], info, tmin=0, verbose='error')
    
    # Bad Channels setzen
    epochs.info['bads'] = BAD_CHANNELS_TARGET
    print(f"Interpoliere Bads: {epochs.info['bads']}")
    
    # Interpolieren (MNE nutzt Splines, FieldTrip Neighbors -> Ergebnisse werden korrelieren, aber nicht identisch sein)
    epochs.interpolate_bads(reset_bads=False, verbose='error')
    
    py_interp = epochs.get_data()[0]
    
    # 5. Vergleich & Statistik (MIT OFFSET-KORREKTUR)
    bad_indices = [epochs.ch_names.index(ch) for ch in BAD_CHANNELS_TARGET if ch in epochs.ch_names]
    
    print("\n--- ERGEBNISSE (MNE Spline vs. FieldTrip Neighbor) ---")
    print(f"{'Kanal':<8} | {'Corr':<8} | {'Raw MAE':<10} | {'Centered MAE':<12} | {'PTP':<8}")
    print("-" * 60)
    
    for idx, name in zip(bad_indices, BAD_CHANNELS_TARGET):
        py_trace = py_interp[idx, :]
        gt_trace = gt_interp[idx, :]
        
        # 1. Raw Fehler (inkl. Offset)
        raw_diff = py_trace - gt_trace
        raw_mae = np.mean(np.abs(raw_diff))
        
        # 2. Centered Fehler (Offset entfernt -> Form-Vergleich)
        py_centered = py_trace - np.mean(py_trace)
        gt_centered = gt_trace - np.mean(gt_trace)
        
        center_diff = py_centered - gt_centered
        center_mae = np.mean(np.abs(center_diff))
        
        corr = np.corrcoef(py_trace, gt_trace)[0, 1]
        ptp = np.ptp(gt_trace)
        
        print(f"{name:<8} | {corr:.4f}   | {raw_mae:.1f}       | {center_mae:.2f}         | {ptp:.1f}")

    # 6. Plotting (Worst Bad Channel)
    if bad_indices:
        # Wir suchen den mit der schlechtesten Korrelation zum Anzeigen
        corrs = [np.corrcoef(py_interp[i], gt_interp[i])[0,1] for i in bad_indices]
        worst_corr_idx = np.argmin(corrs)
        
        plot_idx = bad_indices[worst_corr_idx]
        plot_name = BAD_CHANNELS_TARGET[worst_corr_idx]
        
        plt.figure(figsize=(10, 8))
        
        plt.subplot(2, 1, 1)
        plt.title(f"Interpolation Vergleich: {plot_name} (Corr: {corrs[worst_corr_idx]:.4f})")
        plt.plot(py_interp[plot_idx, :], label='Python (Spline)', color='blue', alpha=0.8)
        plt.plot(gt_interp[plot_idx, :], '--', label='Matlab (FT)', color='orange', alpha=0.8)
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        plt.title("Differenz (Methoden-Unterschied)")
        plt.plot(py_interp[plot_idx, :] - gt_interp[plot_idx, :], color='red', alpha=0.6)
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig("debug_step3_interp_fixed.png")
        print("\nPlot gespeichert: debug_step3_interp_fixed.png")
        plt.show()

if __name__ == "__main__":
    run_debug()