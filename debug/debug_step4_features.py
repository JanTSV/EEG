import mne
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from pathlib import Path

# --- CONFIG ---
fif_path = Path("/Volumes/HardDiskYF/ACADEMIA/EEG/derivatives_new2/sub-01_task-RPS_desc-preproc_eeg.fif")
mat_features_path = Path("originalCode/debug_step2_features.mat")

def get_bins(data_arr, start_offset_s=-0.2, bin_width=0.25, n_bins=8, fs=256):
    """Hilfsfunktion für Time-Binning"""
    zero_idx = int(-start_offset_s * fs)
    bin_samples = int(bin_width * fs)
    binned_data = []
    for b in range(n_bins):
        idx_start = zero_idx + int(b * bin_samples)
        idx_end = zero_idx + int((b+1) * bin_samples)
        # Safety check for indices
        if idx_end > data_arr.shape[2]:
            print(f"Warnung: Bin {b} (idx {idx_end}) außerhalb der Daten ({data_arr.shape[2]})")
            idx_end = data_arr.shape[2]
            
        chunk = data_arr[:, :, idx_start:idx_end]
        binned_data.append(np.mean(chunk, axis=2))
    return np.stack(binned_data, axis=2)

def run_debug():
    print("--- START DEBUGGING STEP 2 (GRANULAR - uV FIX) ---")
    
    # 1. Load MATLAB Ground Truth
    try:
        mat = scipy.io.loadmat(mat_features_path, squeeze_me=True)
        gt_features = mat['feature_matrix'] # (Trials, Chans, Bins)
    except Exception as e:
        print(f"Fehler Mat-File: {e}"); return

    # 2. Load Python Data
    epochs = mne.read_epochs(fif_path, preload=True, verbose='error')
    
    # --- CRITICAL FIX: Convert to uV ---
    print("Scaling to uV (x 1e6)...")
    # Wir holen die Daten als Array und arbeiten damit weiter
    # MNE 'get_data()' gibt Volts zurück.
    data_volts = epochs.get_data()
    data_uv = data_volts * 1e6
    
    # Metadaten
    times = epochs.times
    fs = epochs.info['sfreq']
    
    # 3. Preprocessing Steps (CAR & Selection) - Manuell auf Array Ebene
    # Da wir nun ein numpy array haben, machen wir CAR manuell
    print("Applying CAR (Manual on Array)...")
    # CAR: Mittelwert über alle Kanäle (axis 1) abziehen
    car = np.mean(data_uv, axis=1, keepdims=True)
    data_uv -= car
    
    # Dropping block starts
    print("Dropping block starts...")
    drop_indices = list(range(0, 480, 40))
    keep_indices = [i for i in range(len(epochs)) if i not in drop_indices]
    
    data = data_uv[keep_indices] # (468, 64, 1332)
    
    # --- PART A Analysis (-0.2 to 2.0) ---
    print("\n--- Analysiere PART A (Decision) ---")
    mask_a = (times >= -0.2) & (times <= 2.0)
    data_a = data[:, :, mask_a].copy()
    
    # Baseline A: -0.2 bis 0
    # Indizes relativ zum Start von Part A (der bei -0.2 beginnt)
    # Da times auch bei -0.2 beginnt, passt die Maske direkt.
    base_mask_a = (times >= -0.2) & (times <= 0)
    
    # Wir müssen aufpassen: base_mask_a gilt für das volle array.
    # Für data_a müssen wir die entsprechenden Spalten nehmen.
    # Da data_a exakt dem mask_a entspricht, suchen wir den Overlap.
    # Einfacher: Wir berechnen Baseline am vollen Array und ziehen ab.
    
    baseline_val_a = np.mean(data[:, :, base_mask_a], axis=2, keepdims=True)
    data_a -= baseline_val_a # Broadcasting (Trials, Chans, 1)
    
    bins_a = get_bins(data_a, start_offset_s=-0.2, n_bins=8, fs=fs)
    gt_a = gt_features[:, :, 0:8]
    
    mae_a = np.mean(np.abs(bins_a - gt_a))
    print(f"MAE Part A: {mae_a:.5f} uV")

    # --- PART B Analysis (Response) ---
    print("\n--- Analysiere PART B (Response) ---")
    mask_b = (times >= 1.8) & (times <= 4.0)
    data_b = data[:, :, mask_b].copy()
    
    # Variante 1: Mit Baseline (die ersten 0.2s von Part B)
    base_samples = int(0.2 * fs)
    baseline_val_b = np.mean(data_b[:, :, :base_samples], axis=2, keepdims=True)
    
    # Bins
    bins_b_base = get_bins(data_b - baseline_val_b, start_offset_s=-0.2, n_bins=8, fs=fs)
    bins_b_raw  = get_bins(data_b, start_offset_s=-0.2, n_bins=8, fs=fs)
    
    gt_b = gt_features[:, :, 8:16]
    
    mae_b_base = np.mean(np.abs(bins_b_base - gt_b))
    mae_b_raw  = np.mean(np.abs(bins_b_raw - gt_b))
    
    print(f"MAE Part B (mit Baseline): {mae_b_base:.5f} uV")
    print(f"MAE Part B (OHNE Baseline): {mae_b_raw:.5f} uV")

    # --- PART C Analysis (Feedback) ---
    print("\n--- Analysiere PART C (Feedback) ---")
    mask_c = (times >= 3.8) & (times <= 5.0)
    data_c = data[:, :, mask_c].copy()
    
    baseline_val_c = np.mean(data_c[:, :, :base_samples], axis=2, keepdims=True)
    
    bins_c_base = get_bins(data_c - baseline_val_c, start_offset_s=-0.2, n_bins=4, fs=fs)
    bins_c_raw  = get_bins(data_c, start_offset_s=-0.2, n_bins=4, fs=fs)
    
    gt_c = gt_features[:, :, 16:20]
    
    mae_c_base = np.mean(np.abs(bins_c_base - gt_c))
    mae_c_raw  = np.mean(np.abs(bins_c_raw - gt_c))
    
    print(f"MAE Part C (mit Baseline): {mae_c_base:.5f} uV")
    print(f"MAE Part C (OHNE Baseline): {mae_c_raw:.5f} uV")

    # --- PLOTTING ---
    # Wir plotten die Differenz, um zu sehen, ob es ein Offset ist
    trial_idx = 0
    chan_idx = 0 
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    
    # Plot A
    axes[0].set_title(f"Part A (Corrected Scale) - Trial {trial_idx}, Ch {chan_idx}")
    axes[0].plot(bins_a[trial_idx, chan_idx, :], 'b.-', label='Py')
    axes[0].plot(gt_a[trial_idx, chan_idx, :], 'r--.', label='Mat')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot B (Vergleich Base vs Raw)
    axes[1].set_title(f"Part B (Baseline Check)")
    axes[1].plot(bins_b_base[trial_idx, chan_idx, :], 'b.-', label='Py (Baselined)')
    axes[1].plot(bins_b_raw[trial_idx, chan_idx, :], 'c.-', label='Py (Raw)')
    axes[1].plot(gt_b[trial_idx, chan_idx, :], 'r--.', label='Mat')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig("debug_step4_granular_v2.png")
    print("Plot gespeichert.")
    plt.show()

if __name__ == "__main__":
    run_debug()