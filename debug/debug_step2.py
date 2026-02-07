"""
debug_step2_resample.py
-----------------------
Validierung des Resamplings (2048 Hz -> 256 Hz).
Wir nutzen Pair 1, Player 1 (keine Interpolation), um nur das Resampling zu testen.
"""

import mne
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from pathlib import Path

# --- CONFIG ---
raw_bdf_path = Path("/Volumes/HardDiskYF/ACADEMIA/EEG/ds006761-download/sub-01/eeg/sub-01_task-RPS_eeg.bdf")
mat_epoch_path = Path("originalCode/debug_step1_epoched.mat") # Brauchen wir für TRL Info
mat_resamp_path = Path("originalCode/debug_step2_resamp.mat") # Das Ziel

TARGET_FS = 256

def run_debug():
    print("--- START DEBUGGING STEP 2 (RESAMPLING) ---")
    
    # 1. Laden der MATLAB Ground Truth (Resampled)
    try:
        mat_res = scipy.io.loadmat(str(mat_resamp_path), squeeze_me=True)
        gt_resamp = mat_res['debug_resamp']
        print(f"MATLAB Resampled Shape: {gt_resamp.shape}")
        
        # Lade TRL Info für korrektes Epoching
        mat_trl = scipy.io.loadmat(str(Path("originalCode/debug_step1_trl.mat")), squeeze_me=True)
        trl = mat_trl['TRL']
        start_sample_mat = trl[0, 0] 
        end_sample_mat   = trl[0, 1]
    except Exception as e:
        print(f"Fehler beim Laden der Mat-Files: {e}")
        return

    # 2. Python Pipeline nachbauen
    print("Python Pipeline läuft...")
    raw = mne.io.read_raw_bdf(raw_bdf_path, preload=True, verbose='error')
    
    # Channels wählen
    py_picks = [ch for ch in raw.ch_names if ('2-A' in ch) or ('2-B' in ch)]
    raw.pick_channels(py_picks)
    
    # Epoching (wie in Step 1 validiert)
    idx_start = int(start_sample_mat) - 1
    idx_end   = int(end_sample_mat)
    
    # WICHTIG: Wir müssen ein Epochs Object erstellen für korrektes Resampling in MNE
    # Oder wir resampeln das Raw Array manuell. MNE Epochs.resample ist am sichersten.
    # Wir erstellen ein "Fake" Epochs Array nur mit diesem einen Trial
    data_raw = raw.get_data(start=idx_start, stop=idx_end) # (Chan, Time)
    info = raw.info
    
    # Array in MNE Epochs Container packen (1 Trial)
    # Shape muss (n_epochs, n_channels, n_times) sein
    data_3d = data_raw[np.newaxis, :, :] 
    events = np.array([[0, 0, 1]]) # Fake Event
    
    # Epochs Array erstellen
    epochs = mne.EpochsArray(data_3d, info, events=events, tmin=0, verbose='error')
    
    # 3. RESAMPLING DURCHFÜHREN
    print(f"Resampling auf {TARGET_FS} Hz...")
    # MNE nutzt standardmäßig FFT-Resampling. 
    # FieldTrip nutzt oft 'resample' (Polyphase). Das wird spannend.
    epochs.resample(TARGET_FS)
    
    py_resamp = epochs.get_data()[0] # Zurück zu (Chan, Time)
    
    # 4. Vergleich
    # Skalierungskorrektur aus Step 1 anwenden (uV vs V)
    # Wir prüfen kurz die Ratio
    ratio = np.mean(np.abs(py_resamp)) / np.mean(np.abs(gt_resamp))
    if abs(ratio - 1e-6) < 1e-8:
        py_resamp *= 1e6
        print("-> Skalierung korrigiert (Python * 1e6)")
    
    # Shape Check (Rundungsfehler beim Downsampling können zu +/- 1 Sample führen)
    if py_resamp.shape != gt_resamp.shape:
        print(f"WARNUNG: Shape Mismatch! Py:{py_resamp.shape}, Mat:{gt_resamp.shape}")
        # Wir schneiden auf die kürzere Länge zu
        min_len = min(py_resamp.shape[1], gt_resamp.shape[1])
        py_resamp = py_resamp[:, :min_len]
        gt_resamp = gt_resamp[:, :min_len]

    # Differenz
    diff = py_resamp - gt_resamp
    mae = np.mean(np.abs(diff))
    
    print(f"Mean Absolute Error: {mae:.5e}")
    
    # Plotting (Worst Channel)
    max_err_per_ch = np.max(np.abs(diff), axis=1)
    worst_idx = np.argmax(max_err_per_ch)
    worst_ch = py_picks[worst_idx]
    
    plt.figure(figsize=(10, 6))
    plt.subplot(2,1,1)
    plt.title(f"Resampling Vergleich: {worst_ch} (Worst Case)")
    plt.plot(py_resamp[worst_idx,:50], '.-', label='Py')
    plt.plot(gt_resamp[worst_idx,:50], 'x--', label='Mat')
    plt.legend()
    
    plt.subplot(2,1,2)
    plt.title("Differenz")
    plt.plot(diff[worst_idx,:], color='red')
    plt.tight_layout()
    plt.savefig("debug_step2_resample.png")
    plt.show()

if __name__ == "__main__":
    run_debug()