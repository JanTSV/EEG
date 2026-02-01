"""
13_research_plots.py (ULTIMATE EDITION)
---------------------------------------
High-End Visualization for Research Papers.
Combines Time-Course, Peak Statistics, and TGM.

Outputs 3 Panels:
A) Time Generalization Matrix (with stats)
B) Model Comparison over Time (Line Plot)
C) Model Peak Performance (Boxplot + Stats)
"""

import yaml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path

# Scientific Style
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'figure.figsize': (18, 6) # Wide format for 3 panels
})

def load_config():
    with open("code_new/config.yaml", 'r') as f:
        return yaml.safe_load(f)

CONFIG = load_config()

def load_all_subjects_data(metric_type, target, model_suffix=""):
    """Loads raw data for all subjects."""
    res_dir = Path("/Volumes/HardDiskYF/ACADEMIA/EEG/ds006761-download/derivatives_new/03_decoding_results")
    data_list = []
    
    if CONFIG['subjects']['run_mode'] == 'single':
        pairs = [CONFIG['subjects']['single_pair_id']]
    else:
        all_p = range(CONFIG['subjects']['pair_range'][0], CONFIG['subjects']['pair_range'][1] + 1)
        pairs = [p for p in all_p if p not in CONFIG['subjects']['exclude_pairs']]
        
    for pair_id in pairs:
        for player in [1, 2]:
            fname = f"pair-{pair_id:02d}_player-{player}_target-{target}_{metric_type}{model_suffix}.npy"
            fpath = res_dir / fname
            if fpath.exists():
                d = np.load(fpath)
                if not np.isnan(d).all():
                    data_list.append(d)
    
    if not data_list: return None
    return np.array(data_list)

# --- PLOT 1: TGM with Stats ---
def plot_tgm_research(ax, target):
    data = load_all_subjects_data('tgm', target)
    if data is None: return

    # Statistics (Uncorrected p < 0.01)
    t_vals, p_vals = stats.ttest_1samp(data, popmean=33.333, axis=0)
    sig_mask = p_vals < 0.01
    mean_tgm = np.mean(data, axis=0)
    
    im = ax.imshow(mean_tgm, origin='lower', cmap='viridis', 
                   extent=[0, 5, 0, 5], vmin=30, vmax=40)
    
    # Contours
    x = np.linspace(0, 5, data.shape[1])
    X, Y = np.meshgrid(x, x)
    ax.contour(X, Y, sig_mask, levels=[0.5], colors='white', linewidths=1, linestyles='--')
    
    # Phase Grid
    for p in [2.0, 4.0]:
        ax.axvline(p, color='white', alpha=0.5, lw=1)
        ax.axhline(p, color='white', alpha=0.5, lw=1)
    ax.plot([0, 5], [0, 5], 'k:', lw=1)
    
    ax.set_title(f"A) Time Generalization\n(White contour: p<0.01)", fontweight='bold', loc='left')
    ax.set_xlabel("Testing Time (s)")
    ax.set_ylabel("Training Time (s)")
    plt.colorbar(im, ax=ax, label="Accuracy (%)")

# --- PLOT 2: Model Time Course ---
def plot_model_timecourse(ax, target):
    models = {'LDA': '', 'SVM': '_SVM', 'MLP': '_MLP'}
    colors = {'LDA': 'purple', 'SVM': 'orange', 'MLP': 'teal'}
    
    for name, suffix in models.items():
        data = load_all_subjects_data('acc', target, suffix)
        if data is None: continue
        
        mean = np.mean(data, axis=0)
        sem = np.std(data, axis=0) / np.sqrt(len(data))
        times = np.linspace(0, 5, len(mean))
        
        ax.plot(times, mean, label=name, color=colors[name], linewidth=2)
        ax.fill_between(times, mean-sem, mean+sem, color=colors[name], alpha=0.1)

    # Phases Background
    ax.axvspan(0, 2, color='grey', alpha=0.05)
    ax.axvspan(2, 4, color='grey', alpha=0.1)
    ax.axhline(33.33, ls='--', c='k')
    
    ax.set_title(f"B) Model Comparison (Time Course)", fontweight='bold', loc='left')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Accuracy (%)")
    ax.legend(loc='upper right')
    ax.set_ylim(25, 40)

# --- PLOT 3: Peak Performance Boxplot ---
def plot_model_peaks(ax, target):
    models = ['LDA', 'SVM', 'MLP']
    suffixes = ['', '_SVM', '_MLP']
    peak_accs, labels = [], []
    
    # Create DataFrame
    import pandas as pd
    df_rows = []
    
    # Load data for pairing
    # We need to ensure subject order is preserved for paired t-tests
    # load_all_subjects_data returns (n_subs, n_times) in consistent order
    
    data_map = {}
    for model, suffix in zip(models, suffixes):
        d = load_all_subjects_data('acc', target, suffix)
        if d is not None:
            data_map[model] = np.max(d, axis=1) # Max over time
            
    if not data_map: return

    # Transform to long format for Seaborn
    n_subs = len(data_map['LDA'])
    for i in range(n_subs):
        for m in models:
            if m in data_map:
                df_rows.append({'Subject': i, 'Model': m, 'Peak Accuracy (%)': data_map[m][i]})
    
    df = pd.DataFrame(df_rows)
    
    # Plot
    sns.boxplot(x='Model', y='Peak Accuracy (%)', data=df, ax=ax, width=0.5, palette="pastel")
    sns.stripplot(x='Model', y='Peak Accuracy (%)', data=df, ax=ax, color='black', alpha=0.4, jitter=True)
    
    # Stats: MLP vs LDA
    if 'MLP' in data_map and 'LDA' in data_map:
        t, p = stats.ttest_rel(data_map['MLP'], data_map['LDA'])
        signif = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "ns"
        
        # Draw bracket
        y_max = df['Peak Accuracy (%)'].max() + 0.5
        ax.plot([0, 0, 2, 2], [y_max, y_max+0.5, y_max+0.5, y_max], lw=1, c='k')
        ax.text(1, y_max+0.7, f"MLP vs LDA\n{signif} (p={p:.3f})", ha='center', va='bottom', fontsize=10)

    ax.set_title(f"C) Peak Performance Stats", fontweight='bold', loc='left')
    ax.grid(axis='y', ls=':', alpha=0.5)

def run_research_plots():
    out_dir = Path("figures/06_extensions")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Create 3-Panel Figure
    fig = plt.figure(figsize=(20, 6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.2, 0.8])
    
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    
    target = 'own_current' # Main focus
    
    plot_tgm_research(ax1, target)
    plot_model_timecourse(ax2, target)
    plot_model_peaks(ax3, target)
    
    plt.tight_layout()
    fig.savefig(out_dir / "fig4_research_extensions_full.png", dpi=300)
    print(f"[PLOT] Saved {out_dir}/fig4_research_extensions_full.png")

if __name__ == "__main__":
    run_research_plots()