"""
debug_step2_resample.py (BRUTE FORCE PADDING)
-----------------------
Validierung des Resamplings: Polyphasen-Filter + Padding-Suche.
Testet systematisch, welches Rand-Verhalten (Padding) MATLABs 'resample' nutzt.
"""

import mne
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from scipy.signal import resample_poly
from pathlib import Path

# --- CONFIG ---
# Pfade wie in deinem Snippet angepasst
raw_bdf_path = Path("/Volumes/HardDiskYF/ACADEMIA/EEG/ds006761-download/sub-01/eeg/sub-01_task-RPS_eeg.bdf")
mat_epoch_path = Path("originalCode/debug_step1_epoched.mat") 
mat_resamp_path = Path("originalCode/debug_step2_resamp.mat") 

TARGET_FS = 256
ORIG_FS = 2048

def run_debug():
    print("--- START DEBUGGING STEP 2 (PADDDING SEARCH) ---")
    
    # 1. Laden der MATLAB Ground Truth
    try:
        mat_res = scipy.io.loadmat(str(mat_resamp_path), squeeze_me=True)
        gt_resamp = mat_res['debug_resamp']
        print(f"MATLAB Resampled Shape: {gt_resamp.shape}")
        
        # TRL Info laden
        mat_trl = scipy.io.loadmat(str(Path("originalCode/debug_step1_trl.mat")), squeeze_me=True)
        trl = mat_trl['TRL']
        start_sample_mat = trl[0, 0] 
        end_sample_mat   = trl[0, 1]
    except Exception as e:
        print(f"Fehler beim Laden der Mat-Files: {e}")
        return

    # 2. Python Daten laden
    raw = mne.io.read_raw_bdf(raw_bdf_path, preload=True, verbose='error')
    py_picks = [ch for ch in raw.ch_names if ('2-A' in ch) or ('2-B' in ch)]
    raw.pick_channels(py_picks)
    
    idx_start = int(start_sample_mat) - 1
    idx_end   = int(end_sample_mat)
    
    # Rohe Daten holen
    data_raw = raw.get_data(start=idx_start, stop=idx_end)
    print(f"Python Raw Input Shape: {data_raw.shape}")
    
    # Skalierungskorrektur (WICHTIG!)
    data_raw *= 1e6 
    
    # 3. PADDING TEST LOOP
    up = 1
    down = int(ORIG_FS / TARGET_FS)
    
    # Optionen für Randbehandlung in scipy.signal.resample_poly
    pad_options = ['constant', 'line', 'mean', 'reflect', 'edge']
    
    best_pad = None
    best_mae = np.inf
    best_max = np.inf
    best_data = None
    
    print(f"\n--- Starte Padding-Vergleich (Faktor {down}, Window=Kaiser-5.0) ---")
    print(f"{'PADDING':<10} | {'MAE (uV)':<12} | {'MAX (uV)':<12}")
    print("-" * 40)
    
    for pad in pad_options:
        try:
            # Polyphasen-Resampling mit explizitem Kaiser-Fenster (wie MATLAB Standard)
            curr_resamp = resample_poly(data_raw, up, down, axis=1, padtype=pad, window=('kaiser', 5.0))
            
            # Länge angleichen (falls Rundungsdifferenz)
            if curr_resamp.shape[1] != gt_resamp.shape[1]:
                min_len = min(curr_resamp.shape[1], gt_resamp.shape[1])
                curr_resamp = curr_resamp[:, :min_len]
                # Wir vergleichen gegen gekürztes GT, aber verändern das Original GT nicht global
                curr_gt = gt_resamp[:, :min_len]
            else:
                curr_gt = gt_resamp

            # Fehler berechnen
            diff = curr_resamp - curr_gt
            mae = np.mean(np.abs(diff))
            max_err = np.max(np.abs(diff))
            
            print(f"{pad:<10} | {mae:.5e}  | {max_err:.5e}")
            
            # Ist das der neue Beste?
            if mae < best_mae:
                best_mae = mae
                best_max = max_err
                best_pad = pad
                best_data = curr_resamp # Speichern für Plot
                
        except Exception as e:
            print(f"{pad:<10} | Error: {e}")

    print("-" * 40)
    print(f">>> GEWINNER: '{best_pad}'")
    
    # 4. PLOTTING (Nur mit dem Gewinner)
    if best_data is None:
        print("Kein erfolgreiches Resampling.")
        return

    # Differenz neu berechnen für Plot
    min_len = min(best_data.shape[1], gt_resamp.shape[1])
    best_data = best_data[:, :min_len]
    gt_plot = gt_resamp[:, :min_len]
    diff = best_data - gt_plot
    
    # Worst Channel finden
    max_err_per_ch = np.max(np.abs(diff), axis=1)
    worst_idx = np.argmax(max_err_per_ch)
    worst_ch = py_picks[worst_idx]
    
    plt.figure(figsize=(10, 8))
    
    # Zoom ANFANG
    plt.subplot(3,1,1)
    plt.title(f"Start (0-50 samples) - {worst_ch} using '{best_pad}'")
    plt.plot(best_data[worst_idx,:50], '.-', label=f'Py ({best_pad})')
    plt.plot(gt_plot[worst_idx,:50], 'x--', label='Mat')
    plt.legend()
    plt.grid(True)
    
    # Zoom ENDE
    plt.subplot(3,1,2)
    plt.title(f"Ende (letzte 50 samples) - {worst_ch}")
    plt.plot(best_data[worst_idx,-50:], '.-', label='Py')
    plt.plot(gt_plot[worst_idx,-50:], 'x--', label='Mat')
    plt.grid(True)
    
    # Differenz
    plt.subplot(3,1,3)
    plt.title(f"Differenz (MAE: {best_mae:.2e})")
    plt.plot(diff[worst_idx,:], color='red')
    plt.grid(True)
    
    plt.tight_layout()
    out_file = "debug_step2_padding_test.png"
    plt.savefig(out_file)
    print(f"Plot gespeichert: {out_file}")
    plt.show()

if __name__ == "__main__":
    run_debug()