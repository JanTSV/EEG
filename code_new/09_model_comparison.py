"""
09_model_comparison.py
----------------------
Extension 1: Compare Decoding Algorithms (SVM, MLP).
Optimized for Speed (Parallel Processing).
"""

import yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from joblib import Parallel, delayed

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

def load_config():
    with open("code_new/config.yaml", 'r') as f:
        return yaml.safe_load(f)

CONFIG = load_config()

def create_pseudo_trials(X, y, n_avg=4):
    """Re-used from 05_decoding.py for consistency."""
    classes = np.unique(y)
    X_pseudo, y_pseudo = [], []
    for cls in classes:
        idx = np.where(y == cls)[0]
        np.random.shuffle(idx)
        n_keep = (len(idx) // n_avg) * n_avg
        if n_keep == 0: continue
        idx = idx[:n_keep]
        curr_X = X[idx].reshape(-1, n_avg, X.shape[1]).mean(axis=1)
        curr_y = np.full(curr_X.shape[0], cls)
        X_pseudo.append(curr_X)
        y_pseudo.append(curr_y)
    if not X_pseudo: return None, None
    return np.concatenate(X_pseudo), np.concatenate(y_pseudo)

def get_classifier(name):
    """Factory for classifiers."""
    if name == "SVM":
        return make_pipeline(
            StandardScaler(),
            SVC(kernel=CONFIG['extensions']['models']['svm_kernel'], probability=False)
        )
    elif name == "MLP":
        return make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=tuple(CONFIG['extensions']['models']['mlp_hidden_layers']),
                max_iter=CONFIG['extensions']['models']['mlp_max_iter'],
                early_stopping=True, # Critical for speed/overfitting
                random_state=42
            )
        )
    else:
        raise ValueError(f"Unknown model: {name}")

def process_fold(f, train_idx, test_idx, X_all, y_all, model_name, n_bins, n_pseudo):
    """
    Helper function for Parallel execution.
    Computes accuracy for ONE fold across ALL time bins.
    """
    acc_fold = np.zeros(n_bins)
    
    X_tr, y_tr = X_all[train_idx], y_all[train_idx]
    X_te, y_te = X_all[test_idx], y_all[test_idx]
    
    for t in range(n_bins):
        Xt_tr, Xt_te = X_tr[:,:,t], X_te[:,:,t]
        
        # Pseudo-Trials inside the loop to avoid leakage
        X_tr_p, y_tr_p = create_pseudo_trials(Xt_tr, y_tr, n_pseudo)
        X_te_p, y_te_p = create_pseudo_trials(Xt_te, y_te, n_pseudo)
        
        if X_tr_p is None or X_te_p is None:
            acc_fold[t] = np.nan
            continue
            
        clf = get_classifier(model_name)
        clf.fit(X_tr_p, y_tr_p)
        acc_fold[t] = clf.score(X_te_p, y_te_p)
        
    return acc_fold

def run_model_comparison():
    cfg = CONFIG
    feat_dir = Path("/Volumes/HardDiskYF/ACADEMIA/EEG/ds006761-download/derivatives_new/02_features")
    res_dir = Path("/Volumes/HardDiskYF/ACADEMIA/EEG/ds006761-download/derivatives_new/03_decoding_results")
    res_dir.mkdir(parents=True, exist_ok=True)
    
    models = cfg['extensions']['models']['types']
    targets = cfg['decoding']['targets']
    
    # Get Pairs
    if cfg['subjects']['run_mode'] == 'single':
        pairs = [cfg['subjects']['single_pair_id']]
    else:
        all_p = range(cfg['subjects']['pair_range'][0], cfg['subjects']['pair_range'][1] + 1)
        pairs = [p for p in all_p if p not in cfg['subjects']['exclude_pairs']]
    
    n_jobs = cfg['extensions']['n_jobs']
    print(f"--- STARTING MODEL COMPARISON ({models}) ---")
    print(f"--- Using {n_jobs} cores (Parallel Processing) ---")

    for pair_id in pairs:
        print(f"\nProcessing Pair {pair_id}...")
        for player in [1, 2]:
            for tgt in targets:
                prefix = f"pair-{pair_id:02d}_player-{player}_target-{tgt['name']}"
                
                # Load Data
                try:
                    X_all = np.load(feat_dir / f"{prefix}_X.npy")
                    y_all = np.load(feat_dir / f"{prefix}_y.npy")
                except FileNotFoundError:
                    continue

                # Check Balance
                _, counts = np.unique(y_all, return_counts=True)
                if len(counts) < 2 or any(counts < cfg['decoding']['n_pseudo_avg']):
                    continue
                
                n_bins = X_all.shape[2]
                n_repeats = cfg['decoding']['n_repeats']
                n_folds = cfg['decoding']['n_folds']

                # Loop Models
                for model_name in models:
                    out_file = res_dir / f"{prefix}_acc_{model_name}.npy"
                    if out_file.exists():
                        print(f"  [SKIP] {model_name} for {tgt['name']}")
                        continue
                        
                    print(f"  Running {model_name} on {tgt['name']}...", end='\r')
                    
                    # --- PARALLEL EXECUTION ---
                    # We parallelize the Repeats * Folds loop
                    # Total jobs = 10 * 10 = 100 tasks per subject/target
                    tasks = []
                    for r in range(n_repeats):
                        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42+r)
                        for f, (train_idx, test_idx) in enumerate(cv.split(X_all[:,0,0], y_all)):
                            tasks.append((f, train_idx, test_idx))
                    
                    # Execute all folds in parallel
                    results = Parallel(n_jobs=n_jobs)(
                        delayed(process_fold)(
                            f, tr, te, X_all, y_all, model_name, n_bins, cfg['decoding']['n_pseudo_avg']
                        ) for (f, tr, te) in tasks
                    )
                    
                    # Aggregate: Results is list of (n_bins) arrays. 
                    # Convert to (n_repeats*n_folds, n_bins)
                    res_matrix = np.array(results)
                    
                    # Mean over all folds/repeats
                    mean_acc = np.nanmean(res_matrix, axis=0) * 100
                    
                    # Save
                    np.save(out_file, mean_acc)
                    print(f"  [DONE] {model_name} ({tgt['name']})     ")

if __name__ == "__main__":
    run_model_comparison()