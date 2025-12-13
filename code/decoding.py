# ----- SKETCH -----
# Decoding Pipeline
# 1. Setup & Data Loading
# 2. TARGET LOOP: Iterate over Current/Previous Self/Other
# 3. Data Matching with "Shift" logic (for previous trials)
# 4. Training & Stats
# 5. Save results in subfolders

import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
import warnings
from scipy.stats import ttest_1samp
import shutil

# Machine Learning
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from mne.decoding import SlidingEstimator, Vectorizer

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim

warnings.filterwarnings("ignore")

# --- CONFIG LOADER ---
def load_config():
    base_dir = Path(__file__).resolve().parent.parent 
    cfg_path = Path(__file__).resolve().parent / "decoding_config.yaml"
    with cfg_path.open("r") as f:
        return yaml.safe_load(f), base_dir

def resolve_path(path_str, base):
    p = Path(path_str)
    return p if p.is_absolute() else (base / p).resolve()

# --- PYTORCH MODEL WRAPPER ---
class SimpleEEGNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, n_classes)
        )
    def forward(self, x):
        return self.net(x)

class PyTorchEstimator(BaseEstimator, ClassifierMixin):
    def __init__(self, hidden_dim=64, lr=0.001, epochs=10, batch_size=32):
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.model_ = None
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        input_dim = X.shape[1]
        n_classes = len(self.classes_)
        
        X_t = torch.FloatTensor(X)
        y_t = torch.LongTensor(y)
        
        self.model_ = SimpleEEGNet(input_dim, self.hidden_dim, n_classes)
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
            _, predicted = torch.max(outputs.data, 1)
        return predicted.numpy()

    def score(self, X, y):
        preds = self.predict(X)
        return np.mean(preds == y)

# --- DATA PREP (With Shift Logic) ---
def average_trials_by_class(X, y, n_avg, n_repeats=20, random_state=42):
    """Average random subsets of trials within each class (matches cosmo_average_samples).
    
    Parameters
    ----------
    X : np.ndarray
        Shape (n_trials, n_channels, n_times).
    y : np.ndarray
        Class labels (n_trials,).
    n_avg : int
        Number of trials to average together per sample.
    n_repeats : int
        Number of averaged samples to create per class.
    random_state : int
        Seed for reproducibility.
    
    Returns
    -------
    X_averaged : np.ndarray
        Shape (n_samples, n_channels, n_times) where n_samples = n_classes * n_repeats.
    y_averaged : np.ndarray
        Corresponding labels.
    """
    rng = np.random.RandomState(random_state)
    classes = np.unique(y)
    X_avg_list = []
    y_avg_list = []
    
    for cls in classes:
        idx = np.where(y == cls)[0]
        
        if len(idx) < n_avg:
            print(f"    ! Warning: Class {cls} has only {len(idx)} trials, need {n_avg}. Skipping.")
            continue
        
        for rep in range(n_repeats):
            # Random sample without replacement
            sample_idx = rng.choice(idx, size=n_avg, replace=False)
            X_avg_list.append(X[sample_idx].mean(axis=0))
            y_avg_list.append(cls)
    
    return np.array(X_avg_list), np.array(y_avg_list)


def bin_data(data, sfreq, times, bin_cfg):
    """Bin continuous time samples into fixed windows (e.g., 250 ms).

    Parameters
    ----------
    data : np.ndarray
        Shape (n_epochs, n_channels, n_times).
    sfreq : float
        Sampling frequency in Hz.
    times : np.ndarray
        Time vector matching the last dimension of data.
    bin_cfg : dict
        Configuration with keys: enabled (bool), tmin, tmax, bin_size_sec, reduction.

    Returns
    -------
    data_binned : np.ndarray
        Shape (n_epochs, n_channels, n_bins).
    times_binned : np.ndarray
        Bin-centered time points.
    """
    if not bin_cfg.get("enabled", False):
        return data, times

    tmin = bin_cfg.get("tmin", times[0])
    tmax = bin_cfg.get("tmax", times[-1])
    bin_size = bin_cfg.get("bin_size_sec", 0.25)
    reduction = bin_cfg.get("reduction", "mean")  # "mean", "center", "trimmed_mean"
    trim_frac = bin_cfg.get("trim_fraction", 0.1)   # used when reduction == "trimmed_mean"

    # Crop to desired window
    mask = (times >= tmin) & (times <= tmax + 1e-9)
    data = data[..., mask]
    times = times[mask]

    bin_samples = int(round(bin_size * sfreq))
    total_samples = data.shape[-1]
    n_bins = total_samples // bin_samples
    if n_bins < 1:
        raise ValueError(f"Not enough samples for binning: bin_size={bin_size}s, available={total_samples/sfreq:.3f}s")

    # Trim to full bins
    data = data[..., : n_bins * bin_samples]
    data = data.reshape(data.shape[0], data.shape[1], n_bins, bin_samples)

    if reduction == "center":
        center_idx = bin_samples // 2
        data_binned = data[:, :, :, center_idx]
    elif reduction == "trimmed_mean":
        # Trim lowest and highest fractions along the bin-sample axis before averaging
        # reshape already (epochs, channels, bins, samples)
        k = int(np.floor(trim_frac * bin_samples))
        if k > 0:
            data_sorted = np.sort(data, axis=-1)
            data_trimmed = data_sorted[..., k:-k]
        else:
            data_trimmed = data
        data_binned = data_trimmed.mean(axis=-1)
    else:
        data_binned = data.mean(axis=-1)

    times_binned = np.arange(n_bins) * bin_size + tmin + (bin_size / 2.0)
    return data_binned, times_binned


def prepare_data(epochs, cfg, base_dir, target_name, target_config, bin_cfg):
    """
    Loads behavioral labels and aligns them with EEG.
    Handles 'shift' for previous trial decoding and optional binning.
    """
    # 1. Load TSV
    tsv_path = resolve_path(cfg["paths"]["events_file"], base_dir)
    is_dummy = False
    
    if not tsv_path.exists():
        if cfg["experiment"]["allow_dummy_data"]:
             print("    ! Events file missing. Using DUMMY data.")
             y = np.zeros(len(epochs))
             y[len(epochs)//2:] = 1
             return epochs.get_data(), y, epochs.times, True
        raise FileNotFoundError(f"Missing: {tsv_path}")
    
    events_df = pd.read_csv(tsv_path, sep='\t')
    
    # Trim if needed
    if len(epochs) != len(events_df):
        min_len = min(len(epochs), len(events_df))
        events_df = events_df.iloc[:min_len]
        epochs = epochs[:min_len]

    # 2. Remove first trial of each block (no history for previous trial decoding)
    # Paper uses 40-trial blocks; first trial is 0, 40, 80, ...
    block_size = 40
    first_of_block_mask = np.array([(i % block_size) == 0 for i in range(len(epochs))])
    epochs = epochs[~first_of_block_mask]
    events_df = events_df[~first_of_block_mask].reset_index(drop=True)
    print(f"    > Removed {first_of_block_mask.sum()} first-of-block trials")
    
    # 3. Extract and Shift Labels
    col_name = target_config["column"]
    shift = target_config.get("shift", 0)
    
    # Get raw values
    y_raw = events_df[col_name].values.astype(float)
    
    # Apply Shift (e.g., for "previous_self")
    if shift > 0:
        # Shift array to right (introduce NaNs at start)
        y_raw = np.roll(y_raw, shift)
        # Set first 'shift' elements to NaN (as they have no history)
        y_raw[:shift] = np.nan
        
    # 4. Filter Invalid Labels (NaNs from shift or missing responses)
    # Valid = Not NaN AND Not 0 (assuming 0 is 'no response')
    valid_mask = (~np.isnan(y_raw)) & (y_raw > 0)
    
    epochs_clean = epochs[valid_mask]
    y_clean = y_raw[valid_mask]
    
    # 5. Remap Classes (to 0, 1, 2)
    unique_classes = np.unique(y_clean)
    
    if len(unique_classes) < 2:
        if cfg["experiment"]["allow_dummy_data"]:
            print("    ! Not enough classes. Using DUMMY data.")
            y_clean = np.zeros(len(epochs_clean))
            y_clean[len(epochs_clean)//2:] = 1
            is_dummy = True
        else:
            raise ValueError(f"Not enough classes for {target_name}: {unique_classes}")
    else:
        class_map = {k: v for v, k in enumerate(unique_classes)}
        y_clean = np.array([class_map[k] for k in y_clean])

    print(f"    > Target: {target_name} | Shift: {shift} | Valid Trials: {len(y_clean)}")
    data = epochs_clean.get_data()
    times = epochs_clean.times
    data, times = bin_data(data, epochs_clean.info["sfreq"], times, bin_cfg)
    
    # NOTE: Sample averaging is now done INSIDE cross-validation to prevent data leakage
    # (previously averaged before CV, causing same trials to appear in train and test)

    return data, y_clean, times, is_dummy

# --- CUSTOM CV WITH PER-FOLD AVERAGING ---
def cross_val_with_averaging(estimator, X, y, cv, avg_cfg, random_state=42):
    """
    Custom cross-validation that does sample averaging WITHIN each fold.
    This prevents data leakage (same trials in train and test).
    
    Parameters
    ----------
    estimator : sklearn estimator or SlidingEstimator
        The model to train and evaluate.
    X : np.ndarray
        Shape (n_trials, n_channels, n_times) for time-resolved,
        or (n_trials, n_features) for static.
    y : np.ndarray
        Labels (n_trials,).
    cv : cross-validator
        e.g., StratifiedKFold.
    avg_cfg : dict
        Sample averaging config with keys: enabled, n_trials_per_average, n_repeats.
    random_state : int
        Seed for reproducibility.
    
    Returns
    -------
    scores : np.ndarray
        Shape (n_folds, n_times) for time-resolved, or (n_folds,) for static.
    """
    scores_list = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        # Split data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Average within each fold separately (prevents leakage)
        if avg_cfg.get("enabled", False):
            n_avg = avg_cfg.get("n_trials_per_average", 4)
            n_repeats = avg_cfg.get("n_repeats", 20)
            
            # Use different random seed per fold for diversity
            train_seed = random_state + fold_idx * 100
            test_seed = random_state + fold_idx * 100 + 50
            
            X_train, y_train = average_trials_by_class(X_train, y_train, n_avg, n_repeats, train_seed)
            X_test, y_test = average_trials_by_class(X_test, y_test, n_avg, n_repeats, test_seed)
        
        # Train and score
        estimator.fit(X_train, y_train)
        score = estimator.score(X_test, y_test)
        scores_list.append(score)
    
    return np.array(scores_list)

# --- STATS ---
def calculate_stats(model_name, scores, times, chance_level, alpha=0.05):
    n_folds, n_times = scores.shape
    p_values = []
    for t in range(n_times):
        t_stat, p_val = ttest_1samp(scores[:, t], chance_level)
        p_values.append(p_val)
    
    p_values = np.array(p_values)
    significant = p_values < alpha
    
    return pd.DataFrame({
        "time": times,
        "model": model_name,
        "mean_accuracy": np.mean(scores, axis=0),
        "std_dev": np.std(scores, axis=0),
        "p_value": p_values,
        "significant": significant
    })

# --- PLOTTING ---
def save_plots(all_results, times, cfg, out_dir, title_prefix=""):
    chance = cfg["stats"]["chance_level"]
    
    # Joint Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axhline(chance, color='k', linestyle='--', label=f'Chance ({chance:.2f})')
    ax.axvline(0, color='r', linestyle=':')
    
    for name, df in all_results.items():
        color = cfg["plotting"]["colors"].get(name, "gray")
        ax.plot(times, df["mean_accuracy"], label=name.upper(), color=color, linewidth=2)
        if cfg["plotting"]["show_shadow"]:
            ax.fill_between(times, 
                            df["mean_accuracy"] - df["std_dev"]/2, 
                            df["mean_accuracy"] + df["std_dev"]/2, 
                            color=color, alpha=0.15)
            
    ax.set_title(f"Decoding: {title_prefix}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Accuracy")
    ax.set_xlim(times[0], times[-1])
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    
    fig.savefig(out_dir / "comparison_plot.png", dpi=300)
    plt.close(fig) # Close to save memory
    
    # Individual Plots
    for name, df in all_results.items():
        fig, ax = plt.subplots(figsize=(10, 5))
        color = cfg["plotting"]["colors"].get(name, "blue")
        
        ax.plot(times, df["mean_accuracy"], color=color, label="Accuracy")
        ax.axhline(chance, color='k', linestyle='--')
        ax.axvline(0, color='r', linestyle=':')
        ax.fill_between(times, df["mean_accuracy"] - df["std_dev"], df["mean_accuracy"] + df["std_dev"], color=color, alpha=0.2)
        
        sig_times = df[df["significant"]]["time"]
        if len(sig_times) > 0:
            ax.scatter(sig_times, [chance - 0.02]*len(sig_times), marker='.', color='black', s=10)

        ax.set_title(f"{name.upper()} - {title_prefix}")
        fig.savefig(out_dir / f"result_{name}.png")
        plt.close(fig)

# --- RUNNERS ---
def run_models(X, y, times, cfg, out_dir, title_prefix):
    cv = StratifiedKFold(n_splits=cfg["experiment"]["n_folds"])
    stats_collection = {}
    
    # Get averaging config
    avg_cfg = cfg.get("experiment", {}).get("sample_averaging", {})
    rand_state = cfg.get("experiment", {}).get("random_state", 42)
    
    # LDA
    if cfg["models"]["lda"]["enabled"]:
        print("    > LDA (with per-fold averaging)..." if avg_cfg.get("enabled") else "    > LDA...")
        clf = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))
        slider = SlidingEstimator(clf, scoring='accuracy', n_jobs=-1, verbose=False)
        scores = cross_val_with_averaging(slider, X, y, cv, avg_cfg, rand_state)
        stats_collection["lda"] = calculate_stats("lda", scores, times, cfg["stats"]["chance_level"])

    # MLP
    if cfg["models"]["mlp"]["enabled"]:
        print("    > MLP (with per-fold averaging)..." if avg_cfg.get("enabled") else "    > MLP...")
        m_cfg = cfg["models"]["mlp"]
        clf = make_pipeline(StandardScaler(), MLPClassifier(
            hidden_layer_sizes=tuple(m_cfg["hidden_layer_sizes"]), 
            max_iter=m_cfg["max_iter"]))
        slider = SlidingEstimator(clf, scoring='accuracy', n_jobs=-1, verbose=False)
        scores = cross_val_with_averaging(slider, X, y, cv, avg_cfg, rand_state)
        stats_collection["mlp"] = calculate_stats("mlp", scores, times, cfg["stats"]["chance_level"])
    
    # SVM
    if cfg["models"]["svm"]["enabled"]:
        print("    > SVM (with per-fold averaging)..." if avg_cfg.get("enabled") else "    > SVM...")
        m_cfg = cfg["models"]["svm"]
        clf = make_pipeline(StandardScaler(), SVC(
            C=m_cfg["C"], 
            kernel=m_cfg["kernel"],
            gamma=m_cfg["gamma"]))
        slider = SlidingEstimator(clf, scoring='accuracy', n_jobs=-1, verbose=False)
        scores = cross_val_with_averaging(slider, X, y, cv, avg_cfg, rand_state)
        stats_collection["svm"] = calculate_stats("svm", scores, times, cfg["stats"]["chance_level"])
        
    # PyTorch
    if cfg["models"]["eegnet_slidingWindow"]["enabled"]:
        print("    > EEGNet_slidingWindow (with per-fold averaging)..." if avg_cfg.get("enabled") else "    > EEGNet_slidingWindow...")
        p_cfg = cfg["models"]["eegnet_slidingWindow"]
        clf = make_pipeline(StandardScaler(), PyTorchEstimator(
            hidden_dim=p_cfg["hidden_dim"], epochs=p_cfg["epochs"], lr=p_cfg["learning_rate"]))
        slider = SlidingEstimator(clf, scoring='accuracy', n_jobs=1, verbose=False)
        scores = cross_val_with_averaging(slider, X, y, cv, avg_cfg, rand_state)
        stats_collection["eegnet_slidingWindow"] = calculate_stats("eegnet_slidingWindow", scores, times, cfg["stats"]["chance_level"])

    # LogReg (Whole Trial)
    if cfg["models"]["log_reg"]["enabled"]:
        print("    > LogReg (Static, with per-fold averaging)..." if avg_cfg.get("enabled") else "    > LogReg (Static)...")
        m_cfg = cfg["models"]["log_reg"]
        clf = make_pipeline(Vectorizer(), StandardScaler(), LogisticRegression(C=m_cfg["C"], max_iter=m_cfg["max_iter"]))
        
        # Manual CV for static model with averaging
        scores_list = []
        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            if avg_cfg.get("enabled", False):
                n_avg = avg_cfg.get("n_trials_per_average", 4)
                n_repeats = avg_cfg.get("n_repeats", 20)
                train_seed = rand_state + fold_idx * 100
                test_seed = rand_state + fold_idx * 100 + 50
                X_train, y_train = average_trials_by_class(X_train, y_train, n_avg, n_repeats, train_seed)
                X_test, y_test = average_trials_by_class(X_test, y_test, n_avg, n_repeats, test_seed)
            
            clf.fit(X_train, y_train)
            scores_list.append(clf.score(X_test, y_test))
        
        scores = np.array(scores_list)
        # Add to text report only
        with open(out_dir / "logreg_stats.txt", "w") as f:
            f.write(f"Mean Accuracy: {np.mean(scores):.4f}\nStd: {np.std(scores):.4f}")

    # Save Results
    if stats_collection:
        save_plots(stats_collection, times, cfg, out_dir, title_prefix)
        full_report = pd.concat(stats_collection.values(), ignore_index=True)
        full_report.to_csv(out_dir / "evaluation_report.csv", index=False)


# --- MAIN ---
def main():
    print("=== 4D DECODING PIPELINE (Self/Other x Current/Previous) ===")
    cfg, base_dir = load_config()
    
    results_base = resolve_path(cfg["paths"]["results_dir_base"], base_dir)
    
    # Get pair and players from config
    pair = cfg["pair_player"]["pair"]
    players = cfg["pair_player"]["players"]
    
    print(f"\nProcessing Pair {pair:02d}, Players {players}")
    print("="*60)
    
    # LOOP OVER PLAYERS
    for player in players:
        print(f"\n{'='*60}")
        print(f"PLAYER {player}")
        print("="*60)
        
        # Build paths for this player
        input_file = resolve_path(
            cfg["paths"]["input_template"].format(pair=pair, player=player),
            base_dir
        )
        events_file = resolve_path(
            cfg["paths"]["events_template"].format(pair=pair),
            base_dir
        )
        
        # Update config with current events file (for prepare_data)
        cfg["paths"]["events_file"] = str(events_file.relative_to(base_dir))
        
        if not input_file.exists():
            print(f"ERROR: Input file missing: {input_file}")
            continue
        if not events_file.exists():
            print(f"ERROR: Events file missing: {events_file}")
            continue
            
        # 1. Load EEG for this player
        print(f"  > Loading EEG for pair {pair:02d}, player {player}...")
        epochs_base = mne.read_epochs(str(input_file), preload=True, verbose=False)
        
        # Create player-specific results directory
        player_dir = results_base / f"pair-{pair:02d}_player-{player}"
        
        # 2. LOOP OVER TARGETS
        targets = cfg["experiment"]["targets"]
        
        for target_name, target_config in targets.items():
            print(f"\n--- Processing Target: {target_name.upper()} ---")
            
            # Prepare Data for this specific target (shift logic happens here)
            X, y, times, is_dummy = prepare_data(epochs_base.copy(), cfg, base_dir, target_name, target_config, cfg.get("binning", {}))
            
            # Create Subfolder: pair-XX_player-Y / target_name
            prefix = "DUMMY_" if is_dummy else ""
            out_dir = player_dir / f"{prefix}{target_name}"
            if out_dir.exists(): shutil.rmtree(out_dir) # Clean start
            out_dir.mkdir(parents=True, exist_ok=True)
            
            # Run
            title = f"pair-{pair:02d}_player-{player} {target_name} {prefix}({target_config['column']})"
            run_models(X, y, times, cfg, out_dir, title)
        
    print("\n" + "="*60)
    print("=== ALL DONE. Check results folders. ===")
    print("="*60)

if __name__ == "__main__":
    main()