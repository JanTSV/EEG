# ----- SKETCH -----
# Decoding Pipeline
# 1. Setup: Load Config & Data
# 2. Pre-Check: Handle Labels (Real vs. Dummy)
# 3. Model Zoo: Define pipelines for LDA, LogReg, and MLP (NN)
# 4. Execution Loop: Run enabled models
# 5. Reporting: Save CSV metrics and PNG plots for each model

# ----- IMPORTS -----
import mne
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Union
import matplotlib.pyplot as plt
import yaml
import json
import warnings

# Sklearn / Machine Learning
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from mne.decoding import SlidingEstimator, Vectorizer

# Ignore some MNE warnings for cleaner output
warnings.filterwarnings("ignore", category=RuntimeWarning) 

# ----- HELPER FUNCTIONS -----

def load_config(config_path: Optional[Union[str, Path]] = None) -> tuple[dict, Path]:
    default_path = Path(__file__).resolve().parent / "decoding_config.yaml"
    cfg_path = Path(config_path) if config_path else default_path
    
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found at {cfg_path}")
        
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f), cfg_path

def prepare_data(epochs, cfg):
    """Handles label extraction and dummy data generation."""
    events = epochs.events
    y = events[:, 2]
    
    classes = np.unique(y)
    n_classes = len(classes)
    
    print(f"  > Found {n_classes} classes: {classes}")
    
    # DUMMY DATA LOGIC
    if n_classes < 2:
        if cfg["experiment"]["allow_dummy_data"]:
            print("  ! WARNING: Only 1 class found. Generating DUMMY labels (Split 50/50).")
            print("  ! NOTE: Results show temporal drift, not classification!")
            n_trials = len(y)
            # Create fake labels: 0 for first half, 1 for second half
            y_dummy = np.zeros(n_trials)
            y_dummy[n_trials//2:] = 1
            return epochs.get_data(), y_dummy
        else:
            raise ValueError("Not enough classes for decoding and dummy data is disabled.")
            
    return epochs.get_data(), y

# ----- MODEL RUNNERS -----

def run_lda_time_resolved(X, y, cfg, out_dir, times):
    """Model 1: Sliding Window LDA"""
    print("\n[1/3] Running LDA (Time-Resolved)...")
    
    model_cfg = cfg["models"]["lda"]
    cv = StratifiedKFold(n_splits=cfg["experiment"]["n_folds"])
    
    clf = make_pipeline(
        StandardScaler(),
        LinearDiscriminantAnalysis(solver=model_cfg["solver"], shrinkage=model_cfg["shrinkage"])
    )
    
    # The SlidingEstimator wraps the classifier to run on each time point
    slider = SlidingEstimator(clf, scoring=model_cfg["metric"], n_jobs=-1, verbose=False)
    
    # Run Cross-Validation
    scores = mne.decoding.cross_val_multiscore(slider, X, y, cv=cv, n_jobs=-1)
    mean_scores = np.mean(scores, axis=0)
    
    # Save & Plot
    np.save(out_dir / "lda_scores.npy", scores)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(times, mean_scores, label="LDA Accuracy")
    ax.axhline(0.5, color='k', linestyle='--', label="Chance")
    ax.axvline(0, color='r', linestyle=':')
    ax.set_title("LDA: Time-Resolved Accuracy")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.grid(True)
    
    fig.savefig(out_dir / "plot_lda_time_course.png")
    print(f"  -> Max Accuracy: {np.max(mean_scores):.2f} at {times[np.argmax(mean_scores)]:.2f}s")

def run_logreg_whole_trial(X, y, cfg, out_dir):
    """Model 2: Logistic Regression (Whole Trial)"""
    print("\n[2/3] Running Logistic Regression (Whole Trial)...")
    
    model_cfg = cfg["models"]["log_reg"]
    cv = StratifiedKFold(n_splits=cfg["experiment"]["n_folds"])
    
    # Vectorizer flattens (Channels x Time) into one long feature vector
    clf = make_pipeline(
        Vectorizer(),
        StandardScaler(),
        LogisticRegression(C=model_cfg["C"], max_iter=model_cfg["max_iter"])
    )
    
    scores = cross_val_score(clf, X, y, cv=cv, n_jobs=-1)
    
    print(f"  -> Mean Accuracy: {np.mean(scores):.2%}")
    
    # Save simple text report
    with open(out_dir / "logreg_results.txt", "w") as f:
        f.write(f"Mean Accuracy: {np.mean(scores):.4f}\n")
        f.write(f"Std Dev: {np.std(scores):.4f}\n")
        f.write(f"Scores per fold: {scores}\n")

def run_mlp_neural_net(X, y, cfg, out_dir):
    """Model 3: Neural Network (MLP)"""
    print("\n[3/3] Running Neural Network (MLP)...")
    
    model_cfg = cfg["models"]["mlp"]
    cv = StratifiedKFold(n_splits=cfg["experiment"]["n_folds"])
    
    clf = make_pipeline(
        Vectorizer(), # Flattens input
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=tuple(model_cfg["hidden_layer_sizes"]),
            activation=model_cfg["activation"],
            solver=model_cfg["solver"],
            alpha=model_cfg["alpha"],
            max_iter=model_cfg["max_iter"],
            random_state=cfg["experiment"]["random_state"]
        )
    )
    
    scores = cross_val_score(clf, X, y, cv=cv, n_jobs=-1)
    
    print(f"  -> Mean Accuracy: {np.mean(scores):.2%}")
    
    with open(out_dir / "mlp_results.txt", "w") as f:
        f.write(f"Configuration: {model_cfg['hidden_layer_sizes']}\n")
        f.write(f"Mean Accuracy: {np.mean(scores):.4f}\n")

# ----- MAIN PIPELINE -----

def main_pipeline():
    print("=== STARTING DECODING PIPELINE ===")
    
    # 1. Setup
    cfg, cfg_path = load_config()
    base_dir = cfg_path.parent
    input_file = Path(base_dir) / cfg["paths"]["input_file"]
    results_dir = Path(base_dir) / cfg["paths"]["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Data Loading
    print(f"Loading: {input_file}")
    if not input_file.exists():
        print("ERROR: Input file not found.")
        return

    epochs = mne.read_epochs(input_file, preload=True, verbose=False)
    X, y = prepare_data(epochs, cfg)
    
    # 3. Run Models
    if cfg["models"]["lda"]["enabled"]:
        run_lda_time_resolved(X, y, cfg, results_dir, epochs.times)
        
    if cfg["models"]["log_reg"]["enabled"]:
        run_logreg_whole_trial(X, y, cfg, results_dir)
        
    if cfg["models"]["mlp"]["enabled"]:
        run_mlp_neural_net(X, y, cfg, results_dir)
        
    print(f"\n=== DONE. Results saved to {results_dir} ===")

if __name__ == "__main__":
    main_pipeline()