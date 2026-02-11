import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

# --- CONFIG ---
mat_features_path = Path("originalCode/debug_step2_features.mat") 
mat_results_path = Path("originalCode/debug_step2b_results.mat") 

def cosmo_like_averaging(X, y, n_average=4, n_repeats=20, seed=42):
    """
    Repliziert 'cosmo_average_samples'.
    Erstellt 'Super-Trials' durch Mittelung von n_average Trials der gleichen Klasse.
    
    Returns:
        X_avg, y_avg
    """
    rng = np.random.RandomState(seed)
    classes = np.unique(y)
    
    X_new = []
    y_new = []
    
    # Cosmo macht das oft pro Chunk, aber wir simulieren es hier global 
    # oder innerhalb der CV-Schleife. Um es einfach zu halten und Cosmo's
    # "Data Augmentation" Effekt zu sehen, machen wir es hier global pro Klasse.
    
    for cls in classes:
        # Indizes für diese Klasse finden
        cls_indices = np.where(y == cls)[0]
        n_available = len(cls_indices)
        
        # Wir generieren n_repeats * (n_available / n_average) Samples?
        # Nein, Cosmo generiert exakt 'n_repeats' pro Chunk pro Label.
        # Da wir hier keine Chunks simulieren wie Cosmo, generieren wir einfach
        # eine feste Anzahl an Super-Trials, um die Statistik zu verbessern.
        # Sagen wir: Wir wollen so viele Samples wie vorher, aber geglättet.
        
        # Wir erzeugen 'n_repeats' Super-Trials pro Klasse (ähnlich wie Cosmo pro Chunk macht)
        # Wenn Cosmo 10 Chunks hat und 20 Repeats -> 200 Samples pro Klasse.
        num_super_trials = 200 
        
        for _ in range(num_super_trials):
            # Ziehe n_average zufällige Indizes (mit Zurücklegen erlaubt bei Cosmo?)
            # Cosmo zieht meist ohne Zurücklegen innerhalb eines 'Repeats'.
            # Wir machen Random Choice für Robustheit.
            picks = rng.choice(cls_indices, size=n_average, replace=True)
            
            # Mitteln
            avg_trial = np.mean(X[picks], axis=0)
            X_new.append(avg_trial)
            y_new.append(cls)
            
    return np.array(X_new), np.array(y_new)


def run_decoding_averaged():
    print("--- START DEBUGGING STEP 2b (WITH AVERAGING) ---")
    
    # 1. Load Data
    try:
        mat_in = scipy.io.loadmat(mat_features_path, squeeze_me=True)
        features = mat_in['feature_matrix'] # (Trials, Chans, Bins)
        targets_all = mat_in['targets']
    except Exception: return

    # 2. Load MATLAB Result
    try:
        mat_res = scipy.io.loadmat(mat_results_path, squeeze_me=True)
        dec_struct = mat_res['decoding_accuracy']
        if dec_struct.shape == (1, 4) or dec_struct.shape == (4,): res_self = dec_struct[0]
        else: res_self = dec_struct
        gt_acc = np.squeeze(res_self['samples'].item())
    except Exception: return

    # 3. Setup Python Decoding
    y_full = targets_all[:, 0]
    valid_mask = ~np.isnan(y_full)
    y_full = y_full[valid_mask]
    X_full = features[valid_mask] 
    
    n_bins = X_full.shape[2]
    py_accuracies = []
    
    print(f"Running Decoding on {n_bins} Bins with Cosmo-like Averaging (4 trials)...")
    
    # Classifier
    clf = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))
    
    # Loop Time Bins
    for b in range(n_bins):
        X_bin = X_full[:, :, b]
        
        # --- HIER PASSIERT DIE MAGIE ---
        # Wir müssen Cross-Validation machen.
        # WICHTIG: Averaging darf NICHT Train und Test mischen (Data Leakage)!
        # Wir nutzen StratifiedKFold auf den ROHEN Daten, und mitteln DANN im Loop.
        
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        scores = []
        
        for train_idx, test_idx in cv.split(X_bin, y_full):
            # 1. Split Raw Data
            X_train_raw, X_test_raw = X_bin[train_idx], X_bin[test_idx]
            y_train_raw, y_test_raw = y_full[train_idx], y_full[test_idx]
            
            # 2. Apply Averaging (Super-Trials) NUR auf Training Data
            # Damit der Classifier robuste Muster lernt
            X_train_avg, y_train_avg = cosmo_like_averaging(X_train_raw, y_train_raw, n_average=4, seed=b)
            
            # 3. Test Data auch mitteln? 
            # Cosmo macht das: Test data sind auch gemittelte Chunks.
            # Wenn wir auf gemittelten Daten testen, reduziert sich das Rauschen im Test-Set massiv -> höhere Accuracy.
            X_test_avg, y_test_avg = cosmo_like_averaging(X_test_raw, y_test_raw, n_average=4, seed=b+100)
            
            # 4. Fit & Predict
            clf.fit(X_train_avg, y_train_avg)
            pred = clf.predict(X_test_avg)
            scores.append(accuracy_score(y_test_avg, pred))
            
        py_accuracies.append(np.mean(scores))
        if b % 5 == 0: print(f"Bin {b} done. Acc: {np.mean(scores):.2f}")
    
    py_accuracies = np.array(py_accuracies)
    
    # 5. Metrics & Plot
    correlation = np.corrcoef(gt_acc, py_accuracies)[0,1]
    rmse = np.sqrt(np.mean((gt_acc - py_accuracies)**2))
    
    print(f"\n--- Comparison Metrics ---")
    print(f"Correlation: {correlation:.4f}")
    print(f"RMSE:        {rmse:.4f}")

    plt.figure(figsize=(10, 6))
    plt.axhline(1/3, color='gray', linestyle=':', label='Chance')
    plt.plot(gt_acc, 'r-o', label='Matlab (Cosmo)', linewidth=2, alpha=0.7)
    plt.plot(py_accuracies, 'b--x', label=f'Python (Averaged, Corr={correlation:.2f})', linewidth=2, alpha=0.7)
    plt.title(f"Decoding with Averaging (Corr: {correlation:.3f})")
    plt.xlabel("Time Bins")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("debug_step5_averaging.png")
    plt.show()

if __name__ == "__main__":
    run_decoding_averaged()