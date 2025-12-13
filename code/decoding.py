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
from sklearn.model_selection import StratifiedKFold, cross_val_score
from mne.decoding import SlidingEstimator, Vectorizer, cross_val_multiscore

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
def prepare_data(epochs, cfg, base_dir, target_name, target_config):
    """
    Loads behavioral labels and aligns them with EEG.
    Handles 'shift' for previous trial decoding.
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

    # 2. Extract and Shift Labels
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
        
    # 3. Filter Invalid Labels (NaNs from shift or missing responses)
    # Valid = Not NaN AND Not 0 (assuming 0 is 'no response')
    valid_mask = (~np.isnan(y_raw)) & (y_raw > 0)
    
    epochs_clean = epochs[valid_mask]
    y_clean = y_raw[valid_mask]
    
    # 4. Remap Classes (to 0, 1, 2)
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
    return epochs_clean.get_data(), y_clean, epochs_clean.times, is_dummy

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
    ax.set_xlim(-0.2, 1.0)
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
    
    # LDA
    if cfg["models"]["lda"]["enabled"]:
        print("    > LDA...")
        clf = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))
        slider = SlidingEstimator(clf, scoring='accuracy', n_jobs=-1, verbose=False)
        scores = cross_val_multiscore(slider, X, y, cv=cv, n_jobs=-1)
        stats_collection["lda"] = calculate_stats("lda", scores, times, cfg["stats"]["chance_level"])

    # MLP
    if cfg["models"]["mlp"]["enabled"]:
        print("    > MLP...")
        m_cfg = cfg["models"]["mlp"]
        clf = make_pipeline(StandardScaler(), MLPClassifier(
            hidden_layer_sizes=tuple(m_cfg["hidden_layer_sizes"]), 
            max_iter=m_cfg["max_iter"]))
        slider = SlidingEstimator(clf, scoring='accuracy', n_jobs=-1, verbose=False)
        scores = cross_val_multiscore(slider, X, y, cv=cv, n_jobs=-1)
        stats_collection["mlp"] = calculate_stats("mlp", scores, times, cfg["stats"]["chance_level"])
        
    # PyTorch
    if cfg["models"]["eegnet_slidingWindow"]["enabled"]:
        print("    > EEGNet_slidingWindow...")
        p_cfg = cfg["models"]["eegnet_slidingWindow"]
        clf = make_pipeline(StandardScaler(), PyTorchEstimator(
            hidden_dim=p_cfg["hidden_dim"], epochs=p_cfg["epochs"], lr=p_cfg["learning_rate"]))
        slider = SlidingEstimator(clf, scoring='accuracy', n_jobs=1, verbose=False)
        scores = cross_val_multiscore(slider, X, y, cv=cv, n_jobs=1)
        stats_collection["eegnet_slidingWindow"] = calculate_stats("eegnet_slidingWindow", scores, times, cfg["stats"]["chance_level"])

    # LogReg (Whole Trial)
    if cfg["models"]["log_reg"]["enabled"]:
        print("    > LogReg (Static)...")
        m_cfg = cfg["models"]["log_reg"]
        clf = make_pipeline(Vectorizer(), StandardScaler(), LogisticRegression(C=m_cfg["C"], max_iter=m_cfg["max_iter"]))
        scores = cross_val_score(clf, X, y, cv=cv, n_jobs=-1)
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
    
    input_file = resolve_path(cfg["paths"]["input_file"], base_dir)
    results_base = resolve_path(cfg["paths"]["results_dir_base"], base_dir)
    
    if not input_file.exists():
        print("ERROR: Input file missing.")
        return
        
    # 1. Load EEG Base
    print("  > Loading EEG...")
    epochs_base = mne.read_epochs(input_file, preload=True, verbose=False)
    epochs_base.resample(4.0) # Paper binning
    epochs_base.crop(-0.2, 1.0)
    
    # 2. LOOP OVER TARGETS
    targets = cfg["experiment"]["targets"]
    
    for target_name, target_config in targets.items():
        print(f"\n--- Processing Target: {target_name.upper()} ---")
        
        # Prepare Data for this specific target (shift logic happens here)
        X, y, times, is_dummy = prepare_data(epochs_base.copy(), cfg, base_dir, target_name, target_config)
        
        # Create Subfolder
        prefix = "DUMMY_" if is_dummy else ""
        out_dir = results_base / f"{prefix}{target_name}"
        if out_dir.exists(): shutil.rmtree(out_dir) # Clean start
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Run
        title = f"{target_name} {prefix}({target_config['column']})"
        run_models(X, y, times, cfg, out_dir, title)
        
    print("\n=== ALL DONE. Check results folders. ===")

if __name__ == "__main__":
    main()