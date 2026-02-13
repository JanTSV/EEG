import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import seaborn as sns
from scipy import stats
from pathlib import Path

# --- CONFIG ---
CHANCE_LEVEL = 33.33

# Defined Phases with distinct pastel colors for separation
# Format: (StartBin, EndBin, Label, Color)
PHASES = [
    (0, 7.5, "Decision", "#fff3e0"),  # Light Orange
    (7.5, 15.5, "Response", "#e3f2fd"), # Light Blue
    (15.5, 19.5, "Feedback", "#f3e5f5") # Light Purple
]

COLORS = {
    'main': '#1f77b4',       # Standard Blue
    'winner': '#009688',     # Teal
    'loser': '#CD5C5C',      # IndianRed
    'chance': '#444444',
    'heatmap': ['#ffffff', '#fff9c4', '#ffcc80', '#ef5350'] # White -> Yellow -> Orange -> Red
}

PLOT_TARGETS = ['Self_Response', 'Other_Response', 'Outcome', 'Prev_Self_Response']

def load_config():
    with open("code_new2/config_decoding.yaml", 'r') as f: return yaml.safe_load(f)

def aggregate_data(results_dir):
    all_files = list(results_dir.glob("*_decoding_results.csv"))
    if not all_files: raise FileNotFoundError("No CSVs found.")
    df_list = [pd.read_csv(f) for f in all_files]
    df = pd.concat(df_list, ignore_index=True)
    df['accuracy'] *= 100
    # Aggregate to Subject Level first
    return df.groupby(['subject', 'target', 'time_bin', 'player_status'])['accuracy'].mean().reset_index()

def get_p_values(df, target, bins, mode='one_sample'):
    p_vals = []
    for t in bins:
        d = df[(df['target'] == target) & (df['time_bin'] == t)]
        if mode == 'one_sample':
            _, p = stats.ttest_1samp(d['accuracy'], CHANCE_LEVEL, alternative='greater')
        else: # two_sample
            w = d[d['player_status'] == 'Winner']['accuracy']
            l = d[d['player_status'] == 'Loser']['accuracy']
            _, p = stats.ttest_ind(w, l) if (len(w)>1 and len(l)>1) else (0, 1.0)
        p_vals.append(p)
    return p_vals

def plot_heatmap(ax, p_vals, n_bins):
    """Plots the significance strip."""
    map_vals = np.zeros((1, n_bins))
    for i, p in enumerate(p_vals):
        if p < 0.001: v = 3
        elif p < 0.01: v = 2
        elif p < 0.05: v = 1
        else: v = 0
        map_vals[0, i] = v
        
    cmap = mcolors.ListedColormap(COLORS['heatmap'])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    
    ax.imshow(map_vals, aspect='auto', cmap=cmap, norm=norm, extent=[-0.5, 19.5, 0, 1])
    ax.set_yticks([])
    ax.set_xlim(-0.5, 19.5)
    
    # Add vertical phase dividers
    for (_, end, _, _) in PHASES:
        ax.axvline(end, color='white', linewidth=2)

def style_phase_background(ax):
    """Applies colored phase backgrounds and labels."""
    for (start, end, label, color) in PHASES:
        ax.axvspan(start, end, color=color, alpha=0.6, lw=0, zorder=0)
        # Vertical divider line
        ax.axvline(end, color='white', linewidth=2, zorder=1)
        # Label (only if it's a main plot)
        if ax.get_ylabel(): 
            ax.text((start+end)/2, ax.get_ylim()[1]*0.98, label, 
                   ha='center', va='top', fontsize=9, fontweight='bold', color='#666')

def plot_fig2(df, out_dir):
    print("Plotting Figure 2 (Polished)...")
    targets = [t for t in PLOT_TARGETS if t in df['target'].unique()]
    fig = plt.figure(figsize=(9, 3.5 * len(targets)))
    gs = gridspec.GridSpec(len(targets)*2, 1, height_ratios=[3, 0.25]*len(targets), hspace=0.35)
    
    bins = np.sort(df['time_bin'].unique())
    
    for i, tgt in enumerate(targets):
        df_t = df[df['target'] == tgt]
        
        # Main
        ax = plt.subplot(gs[2*i])
        ax.set_ylim(31, 40) # Set limits BEFORE styling to place text correctly
        ax.set_ylabel("Accuracy (%)")
        style_phase_background(ax)
        
        ax.axhline(CHANCE_LEVEL, color=COLORS['chance'], linestyle='--', lw=1.5, zorder=2)
        sns.lineplot(data=df_t, x='time_bin', y='accuracy', ax=ax, color=COLORS['main'], 
                     linewidth=2.5, marker='o', markersize=6, errorbar=('ci', 95), zorder=3)
        
        ax.set_title(f"{chr(65+i)}) {tgt.replace('_', ' ')}", loc='left', fontweight='bold', pad=10)
        ax.set_xlim(-0.5, 19.5)
        ax.set_xticklabels([])
        
        # Strip
        ax_s = plt.subplot(gs[2*i+1])
        plot_heatmap(ax_s, get_p_values(df_t, tgt, bins), len(bins))
        if i == len(targets)-1: ax_s.set_xlabel("Time Bins")
        else: ax_s.set_xticks([])

    plt.tight_layout()
    plt.savefig(out_dir / "Figure2_GrandAverage_Polished.png", dpi=300)

def plot_fig3(df, out_dir):
    print("Plotting Figure 3 (Polished Split)...")
    if len(df['player_status'].unique()) < 2: return
    targets = [t for t in PLOT_TARGETS if t in df['target'].unique()]
    
    fig = plt.figure(figsize=(9, 5 * len(targets)))
    # Ratios: Plot(3), StripW(0.2), StripL(0.2), StripDiff(0.2)
    gs = gridspec.GridSpec(len(targets)*4, 1, height_ratios=[3, 0.2, 0.2, 0.2]*len(targets), hspace=0.4)
    bins = np.sort(df['time_bin'].unique())
    
    for i, tgt in enumerate(targets):
        df_t = df[df['target'] == tgt]
        base = 4*i
        
        # Main
        ax = plt.subplot(gs[base])
        ax.set_ylim(31, 40)
        ax.set_ylabel("Accuracy (%)")
        style_phase_background(ax)
        ax.axhline(CHANCE_LEVEL, color=COLORS['chance'], linestyle='--', zorder=2)
        
        sns.lineplot(data=df_t[df_t['player_status']=='Winner'], x='time_bin', y='accuracy', ax=ax,
                     color=COLORS['winner'], lw=2, marker='o', errorbar=('ci', 95), label='Winner', zorder=3)
        sns.lineplot(data=df_t[df_t['player_status']=='Loser'], x='time_bin', y='accuracy', ax=ax,
                     color=COLORS['loser'], lw=2, marker='o', errorbar=('ci', 95), label='Loser', zorder=3)
        
        ax.set_title(f"{chr(65+i)}) {tgt.replace('_', ' ')}", loc='left', fontweight='bold')
        ax.set_xlim(-0.5, 19.5)
        ax.set_xticklabels([])
        ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
        
        # Strips
        for j, (mode, label) in enumerate([('Winner', 'Win'), ('Loser', 'Lose')]):
            ax_s = plt.subplot(gs[base+1+j])
            p = get_p_values(df_t[df_t['player_status']==mode], tgt, bins, 'one_sample')
            plot_heatmap(ax_s, p, len(bins))
            ax_s.set_ylabel(label, rotation=0, ha='right', va='center', fontsize=8)
            ax_s.set_xticks([])
            
        # Diff Strip
        ax_d = plt.subplot(gs[base+3])
        p = get_p_values(df_t, tgt, bins, 'two_sample')
        plot_heatmap(ax_d, p, len(bins))
        ax_d.set_ylabel("Diff", rotation=0, ha='right', va='center', fontsize=8)
        if i == len(targets)-1: ax_d.set_xlabel("Time Bins")
        else: ax_d.set_xticks([])

    plt.tight_layout()
    plt.savefig(out_dir / "Figure3_WinnerLoser_Polished.png", dpi=300)

def run():
    cfg = load_config()
    res_dir = Path(cfg['paths']['results_dir'])
    fig_dir = Path(cfg['paths']['figures_dir'])
    try:
        df = aggregate_data(res_dir)
        plot_fig2(df, fig_dir)
        plot_fig3(df, fig_dir)
    except Exception as e: print(e)

if __name__ == "__main__":
    run()