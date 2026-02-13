import yaml
import mne
import numpy as np
import pandas as pd
from pathlib import Path

# ML Imports
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

# =============================================================================
# CLASS: Feature Extractor (Unchanged logic, cleaner implementation)
# =============================================================================
class FeatureExtractor:
    def __init__(self, config):
        self.cfg = config['params']
        self.fs = 256 

    def _get_bins(self, data, start_offset_s, n_bins):
        zero_idx = int(-start_offset_s * self.fs)
        bin_samples = int(self.cfg['bin_width'] * self.fs)
        binned_data = []
        for b in range(n_bins):
            idx_start = zero_idx + int(b * bin_samples)
            idx_end = zero_idx + int((b+1) * bin_samples)
            if idx_end > data.shape[2]: idx_end = data.shape[2]
            chunk = data[:, :, idx_start:idx_end]
            binned_data.append(np.mean(chunk, axis=2))
        return np.stack(binned_data, axis=2)

    def transform(self, epochs):
        # 1. Scale to uV
        data_uv = epochs.get_data() * 1e6
        times = epochs.times
        
        # 2. CAR
        data_uv -= np.mean(data_uv, axis=1, keepdims=True)
        
        # 3. Drop Block Starts (Indices 0, 40, 80...)
        n_total = len(epochs)
        drop_indices = list(range(0, 480, 40))
        keep_indices = [i for i in range(n_total) if i not in drop_indices]
        data_uv = data_uv[keep_indices]
        
        # 4. Slicing logic (A, B, C)
        base_samples = int(self.cfg['baseline_window'] * self.fs)
        
        # Part A
        mask_a = (times >= self.cfg['part_a_tmin']) & (times <= self.cfg['part_a_tmax'])
        data_a = data_uv[:, :, mask_a].copy()
        data_a -= np.mean(data_a[:, :, :base_samples], axis=2, keepdims=True)
        bins_a = self._get_bins(data_a, self.cfg['part_a_tmin'], 8)
        
        # Part B
        mask_b = (times >= self.cfg['part_b_tmin']) & (times <= self.cfg['part_b_tmax'])
        data_b = data_uv[:, :, mask_b].copy()
        data_b -= np.mean(data_b[:, :, :base_samples], axis=2, keepdims=True)
        bins_b = self._get_bins(data_b, -0.2, 8)
        
        # Part C
        mask_c = (times >= self.cfg['part_c_tmin']) & (times <= self.cfg['part_c_tmax'])
        data_c = data_uv[:, :, mask_c].copy()
        data_c -= np.mean(data_c[:, :, :base_samples], axis=2, keepdims=True)
        bins_c = self._get_bins(data_c, -0.2, 4)
        
        X = np.concatenate([bins_a, bins_b, bins_c], axis=2)
        return X, keep_indices

# =============================================================================
# CLASS: Target Manager (Handles Prev Trials & Winner Status)
# =============================================================================
class TargetManager:
    def __init__(self, events_df, config):
        self.events = events_df
        self.cfg = config
        
    def get_winner_status(self):
        """Determines if current player (P1) is Winner or Loser."""
        # Outcome: 1=Draw, 2=P1 Win, 3=P2 Win
        outcomes = self.events.iloc[:, 8].values # Column 9 (index 8) is Outcome
        p1_wins = np.sum(outcomes == 2)
        p2_wins = np.sum(outcomes == 3)
        
        if p1_wins > p2_wins: return "Winner"
        elif p2_wins > p1_wins: return "Loser"
        else: return "Draw"

    def get_target(self, target_cfg, keep_indices):
        """Returns y vector for specific target config."""
        
        # 1. Base Extraction
        if 'source' in target_cfg:
            # It's a derived target (Previous Trial)
            # Find source column
            src_name = target_cfg['source']
            src_cfg = next(t for t in self.cfg['targets'] if t['name'] == src_name)
            col_idx = src_cfg['column_index']
            # We assume column indices 4, 6, 8 (P1, P2, Outcome) from loading logic below
            # But here we work on full events DF before dropping!
            # Matlab script: Col 5 (P1), Col 7 (P2), Col 9 (Outcome) -> Indices 4, 6, 8
            # Mapping config index 0->4, 1->6, 2->8
            map_idx = {0: 4, 1: 6, 2: 8}
            raw_y = self.events.iloc[:, map_idx[col_idx]].values
            
            # Apply Shift
            shift = target_cfg.get('shift', 0)
            if shift > 0:
                raw_y = np.roll(raw_y, shift)
                # First trial of array is now garbage (last trial wrapped around)
                # But we drop first trials of blocks later anyway!
                # IMPORTANT: Block boundaries!
                # Matlab script logic handles block boundaries explicitly.
                # Since we drop index 0, 40, 80... later, the 'previous' of trial 1 is trial 0.
                # This is valid within a block.
                # 'Previous' of trial 40 (start of block 2) is trial 39. 
                # But trial 40 is dropped later! So we are safe.
                pass 
        else:
            # Direct extraction
            col_idx = target_cfg['column_index']
            map_idx = {0: 4, 1: 6, 2: 8}
            raw_y = self.events.iloc[:, map_idx[col_idx]].values

        # 2. Filter to kept indices (Dropped Block Starts)
        y_filtered = raw_y[keep_indices]
        return y_filtered

# =============================================================================
# CLASS: Decoder (With Safety Checks)
# =============================================================================
class Decoder:
    def __init__(self, config):
        self.n_repeats = config['params']['n_repeats']
        self.n_avg = config['params']['n_super_trials']
        self.n_folds = config['params']['n_folds']
        self.clean_cfg = config['cleaning']
        
    def _create_super_trials(self, X, y, seed):
        rng = np.random.RandomState(seed)
        classes = np.unique(y)
        X_new, y_new = [], []
        
        for cls in classes:
            cls_idx = np.where(y == cls)[0]
            n_per_class = len(cls_idx)
            rng.shuffle(cls_idx)
            n_super = n_per_class // self.n_avg
            
            for i in range(n_super):
                indices = cls_idx[i*self.n_avg : (i+1)*self.n_avg]
                X_new.append(np.mean(X[indices], axis=0))
                y_new.append(cls)
                
        return np.array(X_new), np.array(y_new)

    def run(self, X, y):
        # --- 1. DATA CLEANING (The Fix) ---
        # Remove '0' (No Response) and NaNs
        invalid_mask = np.isin(y, self.clean_cfg['drop_values']) | np.isnan(y)
        valid_mask = ~invalid_mask
        
        X_clean = X[valid_mask]
        y_clean = y[valid_mask]
        
        # Check validity
        unique, counts = np.unique(y_clean, return_counts=True)
        # We need at least 4 trials per class to make ONE super trial
        if len(unique) < 2:
            print(f"      [SKIP] Only 1 class found: {unique}")
            return None
        if min(counts) < self.n_avg * 2: # Min 2 super trials recommended
             print(f"      [SKIP] Not enough raw trials for averaging. Counts: {counts}")
             return None

        results = []
        n_bins = X.shape[2]
        clf = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))
        
        # --- 2. DECODING LOOP ---
        for r in range(self.n_repeats):
            # Super Trials
            X_avg, y_avg = self._create_super_trials(X_clean, y_clean, seed=r)
            
            # Safety Check for Cross-Validation
            u_avg, c_avg = np.unique(y_avg, return_counts=True)
            if len(c_avg) < 2 or min(c_avg) < self.n_folds:
                if r == 0: 
                    print(f"      [WARN] Not enough super-trials for CV. Counts: {dict(zip(u_avg, c_avg))}")
                continue # Skip repeat
            
            cv = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=r)
            
            for b in range(n_bins):
                X_bin = X_avg[:, :, b]
                fold_accs = []
                
                for train, test in cv.split(X_bin, y_avg):
                    clf.fit(X_bin[train], y_avg[train])
                    pred = clf.predict(X_bin[test])
                    fold_accs.append(accuracy_score(y_avg[test], pred))
                
                # Store Mean over folds (to save space in CSV) or per fold?
                # User requested detailed CSV. We store per fold.
                for f_idx, acc in enumerate(fold_accs):
                    results.append({
                        'repeat': r,
                        'fold': f_idx,
                        'time_bin': b,
                        'accuracy': acc
                    })
                    
        if not results: return None
        return pd.DataFrame(results)

# =============================================================================
# MAIN
# =============================================================================
def run_pipeline():
    with open("code_new2/config_decoding.yaml", 'r') as f: cfg = yaml.safe_load(f)
    
    deriv_root = Path(cfg['paths']['deriv_root'])
    bids_root = Path(cfg['paths']['bids_root'])
    results_dir = Path(cfg['paths']['results_dir'])
    results_dir.mkdir(parents=True, exist_ok=True)
    
    feature_extractor = FeatureExtractor(cfg)
    decoder = Decoder(cfg)
    
    for sub in cfg['subjects']['include']:
        sub_str = f"sub-{sub:02d}"
        print(f"\nProcessing {sub_str}...")
        
        # Load
        fif_path = deriv_root / f"{sub_str}_task-RPS_desc-preproc_eeg.fif"
        tsv_path = bids_root / sub_str / "eeg" / f"{sub_str}_task-RPS_events.tsv"
        
        if not fif_path.exists(): continue
        
        epochs = mne.read_epochs(fif_path, preload=True, verbose='error')
        events_df = pd.read_csv(tsv_path, sep='\t')
        
        # Targets Helper
        tm = TargetManager(events_df, cfg)
        winner_status = tm.get_winner_status()
        print(f"   Status: {winner_status}")
        
        # Features
        X, keep_indices = feature_extractor.transform(epochs)
        
        all_results = []
        
        # Loop Targets (Now 5 targets!)
        for target_cfg in cfg['targets']:
            t_name = target_cfg['name']
            print(f"   Target: {t_name}")
            
            y = tm.get_target(target_cfg, keep_indices)
            
            df_res = decoder.run(X, y)
            
            if df_res is not None:
                df_res['target'] = t_name
                df_res['subject'] = sub_str
                df_res['player_status'] = winner_status # Add Winner/Loser Info!
                all_results.append(df_res)
        
        if all_results:
            final_df = pd.concat(all_results, ignore_index=True)
            out_csv = results_dir / f"{sub_str}_decoding_results.csv"
            final_df.to_csv(out_csv, index=False)
            print(f"   Saved: {out_csv}")

if __name__ == "__main__":
    run_pipeline()