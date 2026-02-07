"""
debug_step1.py (ALL CHANNELS)
----------------
Validierung aller Kanäle: Python vs Matlab
Identifiziert automatisch den Kanal mit der größten Abweichung.
"""

import mne
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from pathlib import Path

# --- CONFIG ---
# Bitte hier exakte Pfade eintragen
# Wir testen Pair 1, Player 1 (wie im MATLAB Code festgelegt)
raw_bdf_path = Path("/Volumes/HardDiskYF/ACADEMIA/EEG/ds006761-download/sub-01/eeg/sub-01_task-RPS_eeg.bdf")
mat_epoch_path = Path("originalCode/debug_step1_epoched.mat") # Pfad zur .mat Datei von Schritt 1
mat_trl_path   = Path("originalCode/debug_step1_trl.mat")     # Pfad zur .mat Datei von Schritt 1

def run_debug():
    print("--- START DEBUGGING STEP 1 (ALL CHANNELS) ---")
    
    # 1. Laden der MATLAB Daten
    print(f"Lade MATLAB Daten...")
    try:
        mat_data = scipy.io.loadmat(str(mat_epoch_path), squeeze_me=True)
        mat_trl = scipy.io.loadmat(str(mat_trl_path), squeeze_me=True)
        
        gt_trial_1 = mat_data['debug_data'] 
        trl = mat_trl['TRL']
        start_sample_mat = trl[0, 0] 
        end_sample_mat   = trl[0, 1]
        
        print(f"  -> Matlab Shape: {gt_trial_1.shape}")
        
    except Exception as e:
        print(f"FEHLER beim Laden der .mat Files: {e}")
        return

    # 2. Laden der Python Daten
    print(f"\nLade Python (MNE)...")
    raw = mne.io.read_raw_bdf(raw_bdf_path, preload=True, verbose='error')
    
    # 3. Kanal-Auswahl (Player 1)
    py_picks = [ch for ch in raw.ch_names if ('2-A' in ch) or ('2-B' in ch)]
    # WICHTIG: Sortierung sicherstellen! Manchmal sortiert MNE alphabetisch, Matlab nicht.
    # Wir verlassen uns hier darauf, dass die 'picks' die Reihenfolge aus dem File behalten.
    raw.pick_channels(py_picks)
    print(f"  -> Python: {len(py_picks)} Kanäle gewählt.")

    # 4. Daten Extraktion
    idx_start = int(start_sample_mat) - 1  
    idx_end   = int(end_sample_mat)        
    
    py_data_trial_1, times = raw.get_data(start=idx_start, stop=idx_end, return_times=True)
    
    # Shape Check & Transpose
    if py_data_trial_1.shape != gt_trial_1.shape:
        if py_data_trial_1.shape == gt_trial_1.T.shape:
            gt_trial_1 = gt_trial_1.T
            print("  -> Matlab Daten transponiert angepasst.")
        else:
            print(f"CRITICAL: Shape Mismatch! Py: {py_data_trial_1.shape}, Mat: {gt_trial_1.shape}")
            return

    # 5. Numerischer Vergleich (ALLE KANÄLE)
    
    # Globale Skalierung prüfen (anhand des Mittelwerts aller Daten)
    mean_py = np.mean(np.abs(py_data_trial_1))
    mean_mat = np.mean(np.abs(gt_trial_1))
    ratio = mean_py / mean_mat if mean_mat != 0 else 0
    
    if abs(ratio - 1e-6) < 1e-8:
        print(">>> AUTO-FIX: Skalierung Python * 1e6 (uV Anpassung)")
        py_data_trial_1 *= 1e6
    elif abs(ratio - 1e6) < 1e-1:
        print(">>> AUTO-FIX: Skalierung Matlab * 1e6 (uV Anpassung)")
        gt_trial_1 *= 1e6

    # Differenzmatrix berechnen
    diff_matrix = py_data_trial_1 - gt_trial_1
    
    # Fehler pro Kanal berechnen (Max Absolute Error per Channel)
    max_err_per_ch = np.max(np.abs(diff_matrix), axis=1)
    
    # Den schlechtesten Kanal finden
    worst_ch_idx = np.argmax(max_err_per_ch)
    worst_ch_name = py_picks[worst_ch_idx]
    worst_err = max_err_per_ch[worst_ch_idx]
    
    global_mae = np.mean(np.abs(diff_matrix))
    
    print(f"\n--- ERGEBNISSE ---")
    print(f"Globaler MAE (über alle Kanäle): {global_mae:.5e}")
    print(f"Schlechtester Kanal: {worst_ch_name} (Index {worst_ch_idx})")
    print(f"Maximaler Fehler dort: {worst_err:.5e}")
    
    if worst_err < 1e-10:
        print(">>> SUCCESS: Perfekte Übereinstimmung auf allen Kanälen!")
    else:
        print(">>> CHECK: Es gibt Abweichungen. Siehe Plot.")

    # 6. PLOTTING
    print("\n--- PLOT ---")
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    
    # Plot 1: Butterfly Plot der Differenzen (Alle Kanäle)
    # Zeigt sofort, ob ein Kanal aus der Reihe tanzt
    ax = axes[0]
    ax.set_title(f'Residuals (Differenz) aller {len(py_picks)} Kanäle')
    # Wir plotten die Transponierte Matrix, damit Matplotlib eine Linie pro Kanal malt
    ax.plot(times, diff_matrix.T, color='black', alpha=0.3, linewidth=0.5)
    ax.set_ylabel('Diff (uV)')
    ax.grid(True)
    
    # Plot 2: Der "Worst Case" Kanal (Overlay)
    ax = axes[1]
    ax.set_title(f'Vergleich "Worst Channel": {worst_ch_name} (Max Err: {worst_err:.2e})')
    ax.plot(times, py_data_trial_1[worst_ch_idx, :], label='Python', color='blue', alpha=0.8)
    ax.plot(times, gt_trial_1[worst_ch_idx, :], label='Matlab', color='orange', linestyle='--', alpha=0.8)
    ax.legend(loc='upper right')
    ax.grid(True)
    
    # Plot 3: Zoom auf den Start des "Worst Case" Kanals (Check auf Shift)
    ax = axes[2]
    zoom = 50
    ax.set_title(f'Zoom Start ({worst_ch_name}, erste {zoom} Samples)')
    ax.plot(times[:zoom], py_data_trial_1[worst_ch_idx, :zoom], '.-', label='Python', color='blue')
    ax.plot(times[:zoom], gt_trial_1[worst_ch_idx, :zoom], 'x--', label='Matlab', color='orange')
    ax.grid(True)
    
    plt.tight_layout()
    out_file = "debug_step1_all_channels.png"
    plt.savefig(out_file)
    print(f"Plot gespeichert: {out_file}")
    plt.show()

if __name__ == "__main__":
    run_debug()