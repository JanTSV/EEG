import yaml
import mne
import numpy as np
import pandas as pd
from pathlib import Path

# ML Imports
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.base import clone
from joblib import Parallel, delayed

# PyTorch Imports
import torch
import torch.nn as nn
import torch.optim as optim

# =============================================================================
# PyTorch Estimator (Simple MLP to compare with linear models)
# =============================================================================
class TorchMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, n_classes)
        )
    def forward(self, x):
        # Flatten if input is 3D
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.net(x)

class PyTorchEstimator(BaseEstimator, ClassifierMixin):
    def __init__(self, hidden_dim=64, lr=0.001, epochs=50, batch_size=32):
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.model_ = None
        self.classes_ = None
        self.label_map_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        
        X_t = torch.FloatTensor(X)
        
        # Map labels to 0-indexed for PyTorch CrossEntropyLoss
        self.label_map_ = {val: idx for idx, val in enumerate(self.classes_)}
        y_mapped = np.array([self.label_map_[val] for val in y])
        y_t = torch.LongTensor(y_mapped)
        
        channels = X.shape[1]
        samples = X.shape[2] if len(X.shape) > 2 else 1
        input_dim = channels * samples
        
        self.model_ = TorchMLP(input_dim, self.hidden_dim, n_classes)
            
        optimizer = optim.Adam(self.model_.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        
        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.model_.train()
        for _ in range(self.epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                outputs = self.model_(xb)
                loss = criterion(outputs, yb)
                loss.backward()
                optimizer.step()
        return self

    def predict(self, X):
        self.model_.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X)
            outputs = self.model_(X_t)
            _, predicted_mapped = torch.max(outputs.data, 1)
        
        # Map back to original labels
        predicted_mapped = predicted_mapped.numpy()
        reverse_map = {idx: val for val, idx in self.label_map_.items()}
        return np.array([reverse_map[idx] for idx in predicted_mapped])

    def score(self, X, y):
        preds = self.predict(X)
        return np.mean(preds == y)

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
        outcomes = self.events.iloc[:, 8].values
        p1_wins = np.sum(outcomes == 2)
        p2_wins = np.sum(outcomes == 3)
        
        if p1_wins > p2_wins: return "Winner"
        elif p2_wins > p1_wins: return "Loser"
        else: return "Draw"

    def get_target(self, target_cfg, keep_indices):
        """Returns y vector for specific target config."""
        if 'source' in target_cfg:
            src_name = target_cfg['source']
            src_cfg = next(t for t in self.cfg['targets'] if t['name'] == src_name)
            col_idx = src_cfg['column_index']
            map_idx = {0: 4, 1: 6, 2: 8}
            raw_y = self.events.iloc[:, map_idx[col_idx]].values
            
            shift = target_cfg.get('shift', 0)
            if shift > 0:
                raw_y = np.roll(raw_y, shift)
        else:
            col_idx = target_cfg['column_index']
            map_idx = {0: 4, 1: 6, 2: 8}
            raw_y = self.events.iloc[:, map_idx[col_idx]].values

        y_filtered = raw_y[keep_indices]
        return y_filtered

# =============================================================================
# CLASS: Decoder (With Safety Checks & Multi-Model Support)
# =============================================================================
class Decoder:
    def __init__(self, config):
        self.n_repeats = config['params']['n_repeats']
        self.n_avg = config['params']['n_super_trials']
        self.n_folds = config['params']['n_folds']
        self.clean_cfg = config['cleaning']
        self.models_cfg = config.get('models', {'lda': {'enabled': True}})
        
        self.pipelines = {}
        
        if self.models_cfg.get('lda', {}).get('enabled', False):
            self.pipelines['lda'] = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))
            
        if self.models_cfg.get('svm', {}).get('enabled', False):
            m_cfg = self.models_cfg['svm']
            svm_kernel = m_cfg.get('kernel', 'linear')
            self.pipelines['svm'] = make_pipeline(StandardScaler(), SVC(C=m_cfg.get('C', 1.0), kernel=svm_kernel))
            if svm_kernel != 'linear':
                print(f"[WARN] SVM kernel='{svm_kernel}' has no coef_; Haufe patterns will not be computed for model 'svm'.")
            
        if self.models_cfg.get('logreg', {}).get('enabled', False):
            m_cfg = self.models_cfg['logreg']
            self.pipelines['logreg'] = make_pipeline(StandardScaler(), LogisticRegression(C=m_cfg.get('C', 1.0), solver='lbfgs', max_iter=1000))

        if self.models_cfg.get('ridge', {}).get('enabled', False):
            m_cfg = self.models_cfg['ridge']
            self.pipelines['ridge'] = make_pipeline(StandardScaler(), RidgeClassifier(alpha=m_cfg.get('alpha', 1.0)))

        if self.models_cfg.get('torch_mlp', {}).get('enabled', False):
            m_cfg = self.models_cfg['torch_mlp']
            self.pipelines['torch_mlp'] = make_pipeline(StandardScaler(), PyTorchEstimator(
                hidden_dim=m_cfg.get('hidden_dim', 64), 
                epochs=m_cfg.get('epochs', 50), 
                lr=m_cfg.get('learning_rate', 0.001),
                batch_size=m_cfg.get('batch_size', 32)
            ))

        if not self.pipelines:
            print("[WARN] No models enabled in config! Defaulting to LDA.")
            self.pipelines['lda'] = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))

    def _extract_haufe_patterns(self, fitted_estimator, X_train):
        """Compute Haufe activation patterns for fitted linear estimators.

        Returns
        -------
        patterns : ndarray | None
            Shape (n_pattern_rows, n_features) or None if unavailable.
        row_labels : list[str]
            Labels for pattern rows/classes.
        """
        estimator = fitted_estimator
        scaler = None

        # Handle sklearn pipelines from make_pipeline(StandardScaler(), clf)
        if hasattr(fitted_estimator, 'named_steps'):
            steps = list(fitted_estimator.named_steps.values())
            if not steps:
                return None, []
            estimator = steps[-1]
            scaler = fitted_estimator.named_steps.get('standardscaler', None)

        if not hasattr(estimator, 'coef_'):
            return None, []

        coef = estimator.coef_
        if coef is None:
            return None, []

        coef = np.atleast_2d(coef).astype(float, copy=False)  # (n_rows, n_features)

        # Convert weights from standardized-feature space back to raw-feature space
        if (scaler is not None) and hasattr(scaler, 'scale_'):
            scale = np.asarray(scaler.scale_, dtype=float).copy()
            scale[scale == 0.0] = 1.0
            coef = coef / scale[np.newaxis, :]

        # Covariance of training data in raw feature space
        Xc = X_train - np.mean(X_train, axis=0, keepdims=True)
        if Xc.shape[0] < 2:
            return None, []

        cov_x = np.atleast_2d(np.cov(Xc, rowvar=False)).astype(float, copy=False)
        cov_x = np.nan_to_num(cov_x, nan=0.0, posinf=0.0, neginf=0.0)

        # Full Haufe transform: A = Sigma_X * W * Sigma_S^{-1}, with S = XW^T
        # (Haufe et al., 2014). This is exact for multi-output linear decoders.
        S = Xc @ coef.T  # (n_samples, n_rows)
        cov_s = np.atleast_2d(np.cov(S, rowvar=False)).astype(float, copy=False)
        cov_s = np.nan_to_num(cov_s, nan=0.0, posinf=0.0, neginf=0.0)

        eps = 1e-9
        cov_s_reg = cov_s + eps * np.eye(cov_s.shape[0], dtype=float)
        cov_s_inv = np.linalg.pinv(cov_s_reg)

        patterns = (cov_x @ coef.T @ cov_s_inv).T  # (n_rows, n_features)

        classes = getattr(estimator, 'classes_', None)
        if classes is not None and len(classes) == patterns.shape[0]:
            row_labels = [str(c) for c in classes]
        elif classes is not None and len(classes) == 2 and patterns.shape[0] == 1:
            row_labels = [f"{classes[1]}_vs_{classes[0]}"]
        else:
            row_labels = [f"component_{i}" for i in range(patterns.shape[0])]

        return patterns, row_labels

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

    def _run_single_repeat(self, r, X_clean, y_clean, clf, model_name, n_bins, ch_names):
        import torch
        torch.set_num_threads(1)
        results = []
        patterns = []
        X_avg, y_avg = self._create_super_trials(X_clean, y_clean, seed=r)
        
        u_avg, c_avg = np.unique(y_avg, return_counts=True)
        if len(c_avg) < 2 or min(c_avg) < self.n_folds:
            if r == 0: 
                print(f"        [WARN] Not enough super-trials for CV. Counts: {dict(zip(u_avg, c_avg))}")
            return results, patterns
        
        cv = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=r)
        for b in range(n_bins):
            X_bin = X_avg[:, :, b]
            fold_accs = []
            
            for f_idx, (train, test) in enumerate(cv.split(X_bin, y_avg)):
                my_clf = clone(clf)
                my_clf.fit(X_bin[train], y_avg[train])
                pred = my_clf.predict(X_bin[test])
                fold_accs.append(accuracy_score(y_avg[test], pred))

                patt, row_labels = self._extract_haufe_patterns(my_clf, X_bin[train])
                if patt is not None:
                    for row_idx in range(patt.shape[0]):
                        for ch_idx, ch_name in enumerate(ch_names):
                            patterns.append({
                                'model': model_name,
                                'repeat': r,
                                'fold': f_idx,
                                'time_bin': b,
                                'pattern_row': row_idx,
                                'pattern_label': row_labels[row_idx],
                                'channel_idx': ch_idx,
                                'channel': ch_name,
                                'pattern_value': float(patt[row_idx, ch_idx])
                            })
            
            for f_idx, acc in enumerate(fold_accs):
                results.append({
                    'model': model_name,
                    'repeat': r,
                    'fold': f_idx,
                    'time_bin': b,
                    'accuracy': acc
                })
        return results, patterns

    def run(self, X, y, ch_names):
        # --- 1. DATA CLEANING (The Fix) ---
        invalid_mask = np.isin(y, self.clean_cfg['drop_values']) | np.isnan(y)
        valid_mask = ~invalid_mask
        
        X_clean = X[valid_mask]
        y_clean = y[valid_mask]
        
        unique, counts = np.unique(y_clean, return_counts=True)
        if len(unique) < 2:
            print(f"      [SKIP] Only 1 class found: {unique}")
            return None, None
        if min(counts) < self.n_avg * 2:
             print(f"      [SKIP] Not enough raw trials for averaging. Counts: {counts}")
             return None, None

        all_results = []
        all_patterns = []
        n_bins = X.shape[2]
        
        # --- 2. DECODING LOOP ---
        for model_name, clf in self.pipelines.items():
            print(f"      Running model: {model_name}")
            
            repeat_results = Parallel(n_jobs=-1)(
                delayed(self._run_single_repeat)(r, X_clean, y_clean, clf, model_name, n_bins, ch_names)
                for r in range(self.n_repeats)
            )
            for res_list, pat_list in repeat_results:
                all_results.extend(res_list)
                all_patterns.extend(pat_list)
                        
        if not all_results:
            return None, None

        df_results = pd.DataFrame(all_results)
        df_patterns = pd.DataFrame(all_patterns) if all_patterns else None
        return df_results, df_patterns

# =============================================================================
# MAIN
# =============================================================================
def run_pipeline():
    with open("code/config_decoding.yaml", 'r') as f: cfg = yaml.safe_load(f)
    
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
        all_patterns = []
        
        # Loop Targets
        for target_cfg in cfg['targets']:
            t_name = target_cfg['name']
            print(f"   Target: {t_name}")
            
            y = tm.get_target(target_cfg, keep_indices)
            
            df_res, df_pat = decoder.run(X, y, epochs.ch_names)
            
            if df_res is not None:
                df_res['target'] = t_name
                df_res['subject'] = sub_str
                df_res['player_status'] = winner_status
                all_results.append(df_res)

            if df_pat is not None:
                df_pat['target'] = t_name
                df_pat['subject'] = sub_str
                df_pat['player_status'] = winner_status
                all_patterns.append(df_pat)
        
        if all_results:
            final_df = pd.concat(all_results, ignore_index=True)
            out_csv = results_dir / f"{sub_str}_decoding_results.csv"
            final_df.to_csv(out_csv, index=False)
            print(f"   Saved: {out_csv}")

        if all_patterns:
            final_patterns = pd.concat(all_patterns, ignore_index=True)
            # Save averaged Haufe patterns across CV folds/repeats for cleaner topoplots
            agg_cols = ['subject', 'target', 'player_status', 'model', 'time_bin',
                        'pattern_row', 'pattern_label', 'channel_idx', 'channel']
            final_patterns = (
                final_patterns
                .groupby(agg_cols, as_index=False)['pattern_value']
                .mean()
            )
            out_pat = results_dir / f"{sub_str}_haufe_patterns.csv"
            final_patterns.to_csv(out_pat, index=False)
            print(f"   Saved: {out_pat}")

if __name__ == "__main__":
    run_pipeline()
