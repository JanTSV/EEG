"""
11_time_gen.py
--------------
Extension 3: Temporal Generalization (Time-Gen).
Trains on time t1, Tests on time t2.
Output: 20x20 Accuracy Matrix per subject/target.
"""

import yaml
import numpy as np
from pathlib import Path
from joblib import Parallel, delayed

from sklearn.model_selection import StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

def load_config():
    with open("code_new/config.yaml", 'r') as f:
        return yaml.safe_load(f)

CONFIG = load_config()

def create_pseudo_trials(X, y, n_avg=4):
    """Creates pseudo-trials by averaging n_avg single trials of same class."""
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

def process_fold_tgm(f, train_idx, test_idx, X_all, y_all, n_bins, n_pseudo, shrinkage):
    """
    Computes TGM for one fold.
    Returns: Matrix (n_bins_train, n_bins_test)
    """
    tgm_fold = np.zeros((n_bins, n_bins))
    
    # Slice Data
    X_tr, y_tr = X_all[train_idx], y_all[train_idx]
    X_te, y_te = X_all[test_idx], y_all[test_idx]
    
    # 1. Pre-calculate Pseudo-Trials for ALL time bins first
    # This avoids re-shuffling inside the nested loop
    # We store them in a list/dict for fast access
    
    # WARNING: Pseudo-trial generation involves shuffling. 
    # To correspond to "Training on t1", we must form groups based on trial indices.
    # The groups (which trials form a pseudo-trial) must be consistent across time?
    # Actually, in our previous code, we re-shuffled for every time bin t. 
    # For TGM, it is scientifically cleaner if the "Identity" of a pseudo-trial 
    # is stable across time (i.e. Pseudo-Trial 1 is always Trial 4+8+12+1 averaged).
    
    # Let's generate the indices for grouping ONCE per fold
    classes = np.unique(y_tr)
    tr_groups = [] # List of list of indices
    tr_labels = []
    
    for cls in classes:
        idx = np.where(y_tr == cls)[0]
        np.random.shuffle(idx)
        n_keep = (len(idx) // n_pseudo) * n_pseudo
        if n_keep == 0: continue
        idx = idx[:n_keep]
        # Reshape to (n_groups, n_pseudo)
        groups = idx.reshape(-1, n_pseudo)
        for g in groups:
            tr_groups.append(g)
            tr_labels.append(cls)
            
    if not tr_groups: return np.full((n_bins, n_bins), np.nan)
    
    # Do the same for Test set
    te_groups = []
    te_labels = []
    for cls in classes: # Assumes same classes in test
        idx = np.where(y_te == cls)[0]
        np.random.shuffle(idx)
        n_keep = (len(idx) // n_pseudo) * n_pseudo
        if n_keep == 0: continue
        idx = idx[:n_keep]
        groups = idx.reshape(-1, n_pseudo)
        for g in groups:
            te_groups.append(g)
            te_labels.append(cls)
            
    if not te_groups: return np.full((n_bins, n_bins), np.nan)
    
    y_tr_p = np.array(tr_labels)
    y_te_p = np.array(te_labels)
    
    # 2. Loop Time (Train)
    for t_train in range(n_bins):
        # Build Training Data for t_train using the pre-defined groups
        # X_tr shape: (n_trials, n_chans, n_bins)
        # We need to average X_tr[group_indices, :, t_train]
        X_tr_p = np.array([X_tr[g, :, t_train].mean(axis=0) for g in tr_groups])
        
        # Train Classifier
        clf = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver='lsqr', shrinkage=shrinkage))
        clf.fit(X_tr_p, y_tr_p)
        
        # 3. Loop Time (Test)
        for t_test in range(n_bins):
            # Build Test Data for t_test
            X_te_p = np.array([X_te[g, :, t_test].mean(axis=0) for g in te_groups])
            
            # Score
            tgm_fold[t_train, t_test] = clf.score(X_te_p, y_te_p)
            
    return tgm_fold

def run_time_gen():
    cfg = CONFIG
    feat_dir = Path("/Volumes/HardDiskYF/ACADEMIA/EEG/ds006761-download/derivatives_new/02_features")
    res_dir = Path("/Volumes/HardDiskYF/ACADEMIA/EEG/ds006761-download/derivatives_new/03_decoding_results")
    res_dir.mkdir(parents=True, exist_ok=True)
    
    targets = cfg['decoding']['targets']
    # Filter targets? Time Gen is heavy. Let's run for 'own_current' and 'own_prev' first.
    # Or just run all if you have time.
    
    if cfg['subjects']['run_mode'] == 'single':
        pairs = [cfg['subjects']['single_pair_id']]
    else:
        all_p = range(cfg['subjects']['pair_range'][0], cfg['subjects']['pair_range'][1] + 1)
        pairs = [p for p in all_p if p not in cfg['subjects']['exclude_pairs']]
        
    n_jobs = cfg['extensions']['n_jobs']
    print(f"--- STARTING TIME GENERALIZATION ---")
    
    for pair_id in pairs:
        print(f"\nProcessing Pair {pair_id}...")
        for player in [1, 2]:
            for tgt in targets:
                prefix = f"pair-{pair_id:02d}_player-{player}_target-{tgt['name']}"
                out_file = res_dir / f"{prefix}_tgm.npy"
                
                if out_file.exists():
                    print(f"  [SKIP] TGM exists for {tgt['name']}")
                    continue
                
                try:
                    X_all = np.load(feat_dir / f"{prefix}_X.npy")
                    y_all = np.load(feat_dir / f"{prefix}_y.npy")
                except: continue

                # Check Balance
                _, counts = np.unique(y_all, return_counts=True)
                if len(counts) < 2 or any(counts < cfg['decoding']['n_pseudo_avg']): continue
                
                print(f"  Calc TGM: {tgt['name']}...", end='\r')
                
                n_bins = X_all.shape[2]
                n_repeats = cfg['decoding']['n_repeats']
                n_folds = cfg['decoding']['n_folds']
                
                # Prepare parallel tasks
                tasks = []
                for r in range(n_repeats):
                    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42+r)
                    for f, (train_idx, test_idx) in enumerate(cv.split(X_all[:,0,0], y_all)):
                        tasks.append((f, train_idx, test_idx))
                
                # Run Parallel
                # Note: We need to pass shrinkage from config
                shrinkage = cfg['decoding']['shrinkage']
                results = Parallel(n_jobs=n_jobs)(
                    delayed(process_fold_tgm)(
                        f, tr, te, X_all, y_all, n_bins, cfg['decoding']['n_pseudo_avg'], shrinkage
                    ) for (f, tr, te) in tasks
                )
                
                # Average (n_tasks, n_bins, n_bins) -> (n_bins, n_bins)
                tgm_avg = np.nanmean(np.array(results), axis=0) * 100
                
                np.save(out_file, tgm_avg)
                print(f"  [DONE] {tgt['name']} (20x20 Matrix saved)     ")

if __name__ == "__main__":
    run_time_gen()