"""
05_decoding.py (UPGRADED)
-------------------------
Step 4: Decoding Analysis.
Loops over all defined targets (Own, Opp, Prev, etc.).
"""
# ... Imports same as before ...
import yaml
import numpy as np
import matplotlib.pyplot as plt # Keep imports
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

def load_config():
    with open("code_new/config.yaml", 'r') as f:
        return yaml.safe_load(f)

CONFIG = load_config()

def create_pseudo_trials(X, y, n_avg=4):
    # Same code as before
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

def run_decoding_for_pair(pair_id, player_num):
    feat_dir = Path("/Volumes/HardDiskYF/ACADEMIA/EEG/ds006761-download/derivatives_new/02_features")
    res_dir = Path("/Volumes/HardDiskYF/ACADEMIA/EEG/ds006761-download/derivatives_new/03_decoding_results")
    res_dir.mkdir(parents=True, exist_ok=True)
    
    # LOOP OVER TARGETS defined in config
    targets = CONFIG['decoding']['targets']
    
    for tgt in targets:
        tgt_name = tgt['name']
        prefix = f"pair-{pair_id:02d}_player-{player_num}_target-{tgt_name}"
        
        if (res_dir / f"{prefix}_acc.npy").exists():
            print(f"  [SKIP] Exists: {tgt_name}")
            continue

        print(f"  Decoding Target: {tgt_name} ...", end='\r')
        
        try:
            X_all = np.load(feat_dir / f"{prefix}_X.npy")
            y_all = np.load(feat_dir / f"{prefix}_y.npy")
        except FileNotFoundError:
            continue

        # Check Balance (Fix for Player 2 issue)
        _, counts = np.unique(y_all, return_counts=True)
        if len(counts) < 2 or any(counts < CONFIG['decoding']['n_pseudo_avg']):
            print(f"  [WARN] Not enough data for {tgt_name}. Skipping.")
            continue

        # ... (Same Decoding Logic: Repeats -> CV -> Time) ...
        n_bins = X_all.shape[2]
        accuracies = np.zeros((CONFIG['decoding']['n_repeats'], CONFIG['decoding']['n_folds'], n_bins))
        
        for r in range(CONFIG['decoding']['n_repeats']):
            cv = StratifiedKFold(n_splits=CONFIG['decoding']['n_folds'], shuffle=True, random_state=42+r)
            for f, (train_idx, test_idx) in enumerate(cv.split(X_all[:,0,0], y_all)):
                X_tr, y_tr = X_all[train_idx], y_all[train_idx]
                X_te, y_te = X_all[test_idx], y_all[test_idx]
                
                for t in range(n_bins):
                    Xt_tr, Xt_te = X_tr[:,:,t], X_te[:,:,t]
                    X_tr_p, y_tr_p = create_pseudo_trials(Xt_tr, y_tr, CONFIG['decoding']['n_pseudo_avg'])
                    X_te_p, y_te_p = create_pseudo_trials(Xt_te, y_te, CONFIG['decoding']['n_pseudo_avg'])
                    
                    if X_tr_p is None or X_te_p is None:
                        accuracies[r, f, t] = np.nan
                        continue
                        
                    clf = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver='lsqr', shrinkage=CONFIG['decoding']['shrinkage']))
                    clf.fit(X_tr_p, y_tr_p)
                    accuracies[r, f, t] = clf.score(X_te_p, y_te_p)

        # Save
        mean_acc = np.nanmean(accuracies, axis=(0, 1)) * 100
        np.save(res_dir / f"{prefix}_acc.npy", mean_acc)
        print(f"  [DONE] {tgt_name}")

if __name__ == "__main__":
    cfg = load_config()
    pairs = range(cfg['subjects']['pair_range'][0], cfg['subjects']['pair_range'][1] + 1)
    if cfg['subjects']['run_mode'] == 'single': pairs = [cfg['subjects']['single_pair_id']]
    pairs = [p for p in pairs if p not in cfg['subjects']['exclude_pairs']]
    
    for p in pairs:
        print(f"\nProcessing Pair {p}...")
        for player in [1, 2]:
            run_decoding_for_pair(p, player)