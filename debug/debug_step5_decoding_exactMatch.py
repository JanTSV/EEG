import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score

# --- CONFIG ---
mat_supertrials_path = Path("originalCode/debug_step2c_supertrials.mat") 
mat_results_path = Path("originalCode/debug_step2b_results.mat") 

def run_exact_match():
    print("--- START DEBUGGING STEP 2c (EXACT CLASSIFIER MATCH) ---")
    
    # 1. Load MATLAB Super-Trials (Input)
    try:
        mat = scipy.io.loadmat(mat_supertrials_path, squeeze_me=True)
        X = mat['super_trials']   # (n_samples, n_features*n_time) ?? 
        # Cosmo flattet oft Time und Channels. 
        # Wir müssen checken, ob Cosmo Searchlight pro Timepoint gemacht hat.
        # Im Matlab Code steht: nh = cosmo_interval_neighborhood(..., 'radius', 0)
        # Und dann cosmo_searchlight. 
        # Das Averaging passiert VORHER auf dem vollen Dataset.
        # D.h. X hat wahrscheinlich Dimension (Samples, Chans * TimeBins).
        
        y = mat['super_targets']
        chunks = mat['super_chunks']
        print(f"Loaded Super-Trials: {X.shape}")
    except Exception as e:
        print(f"Fehler Input: {e}"); return

    # 2. Load MATLAB Results (Ground Truth Output)
    try:
        mat_res = scipy.io.loadmat(mat_results_path, squeeze_me=True)
        dec_struct = mat_res['decoding_accuracy']
        if dec_struct.shape == (1, 4) or dec_struct.shape == (4,): res = dec_struct[0]
        else: res = dec_struct
        gt_acc = np.squeeze(res['samples'].item())
        print(f"Loaded Ground Truth Accuracy: {gt_acc.shape}")
    except Exception: return

    # 3. Reconstruct Feature Dimensions
    # Wir wissen aus Step 2a: 64 Chans, 20 Bins.
    n_samples, n_features_total = X.shape
    n_chans = 64
    n_bins = n_features_total // n_chans
    
    if n_features_total % n_chans != 0:
        print(f"WARNUNG: Feature Dimension {n_features_total} nicht durch 64 teilbar!")
        # Fallback logic falls Cosmo channels flattet oder so
    
    print(f"Reshaping to ({n_samples}, {n_chans}, {n_bins})")
    # Cosmo flatten order ist meistens (Chans, Time) -> Fortran Order?
    # Oder (Time, Chans)? Cosmo ist da eigen.
    # Wir testen einfach das Decoding pro Time-Slice.
    # Wenn wir Searchlight replizieren wollen, müssen wir wissen, welche Spalten zu welchem Timebin gehören.
    # Annahme: Cosmo sortiert Feature 1..64 (Time 1), 65..128 (Time 2)...
    
    # Classifier Setup
    clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    
    py_accuracies = []
    
    # Loop over Time Bins (Searchlight logic)
    for b in range(n_bins):
        # Slice Features for this Bin
        # Annahme: Features sind blockweise sortiert [Chans_Bin1, Chans_Bin2, ...]
        idx_start = b * n_chans
        idx_end = (b + 1) * n_chans
        X_bin = X[:, idx_start:idx_end]
        
        # Cross-Validation (Leave-One-Chunk-Out)
        # Cosmo partitions based on 'chunks'
        unique_chunks = np.unique(chunks)
        scores = []
        
        for test_chunk in unique_chunks:
            test_mask = (chunks == test_chunk)
            train_mask = ~test_mask
            
            X_train, X_test = X_bin[train_mask], X_bin[test_mask]
            y_train, y_test = y[train_mask], y[test_mask]
            
            clf.fit(X_train, y_train)
            pred = clf.predict(X_test)
            scores.append(accuracy_score(y_test, pred))
            
        py_accuracies.append(np.mean(scores))
        
    py_accuracies = np.array(py_accuracies)
    
    # 4. Metrics & Plot
    corr = np.corrcoef(gt_acc, py_accuracies)[0,1]
    rmse = np.sqrt(np.mean((gt_acc - py_accuracies)**2))
    
    print(f"\n--- Results ---")
    print(f"Correlation: {corr:.5f}")
    print(f"RMSE:        {rmse:.5f}")
    
    plt.figure()
    plt.plot(gt_acc, 'r-o', label='Matlab')
    plt.plot(py_accuracies, 'b--x', label='Python (Exact Input)')
    plt.title(f"Classifier Identity Check (Corr: {corr:.4f})")
    plt.legend()
    plt.grid()
    plt.savefig("debug_step5_exact_match.png")
    plt.show()

if __name__ == "__main__":
    run_exact_match()