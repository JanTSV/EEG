"""
07_plot_paper_fig1.py (VISUAL POLISH + LOGIC FIX)
-------------------------------------------------
Replication of Figure 1 (Behavioral Results).
Professional "Raincloud"-style visualization.

FIX: Correct mapping of outcomes based on MATLAB script:
1 = Draw
2 = Player 1 Win
3 = Player 2 Win
"""

import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Scientific Style Standards
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.figsize': (15, 12)
})

# Professional Color Palette (Seaborn-ish)
COLORS = {
    'win': '#4c72b0',   # Blue
    'lose': '#c44e52',  # Red
    'draw': '#55a868',  # Green
    'gold': '#bcbd22',  # Gold/Olive
    'orange': '#ff7f0e',# Orange
    'brown': '#8c564b', # Brown
    'purple': '#8172b3',# Purple
    'grey': '#7f7f7f'   # Grey
}

def load_config():
    with open("code_new/config.yaml", 'r') as f:
        return yaml.safe_load(f)

CONFIG = load_config()

# --- Data Loading ---
def get_pair_behavior(pair_id, data_root):
    sub_str = f"sub-{pair_id:02d}"
    tsv_path = data_root / sub_str / 'eeg' / f"{sub_str}_task-RPS_events.tsv"
    if not tsv_path.exists(): return None
    try:
        df = pd.read_csv(tsv_path, sep='\t')
        
        # --- FIX 1: Mapping Outcome (1=Draw, 2=P1, 3=P2) ---
        p1_wins = np.sum(df['outcome'] == 2)
        p2_wins = np.sum(df['outcome'] == 3)
        draws = np.sum(df['outcome'] == 1)
        
        winner_player = 1 if p1_wins > p2_wins else (2 if p2_wins > p1_wins else 0)

        n_games = len(df)
        
        if winner_player == 1: 
            win_wins = p1_wins; lose_wins = p2_wins
        elif winner_player == 2: 
            win_wins = p2_wins; lose_wins = p1_wins
        else: 
            win_wins = p1_wins; lose_wins = p2_wins # Tie case

        metrics_c = {
            'Winner wins': (win_wins / n_games) * 100,
            'Winner loses': (lose_wins / n_games) * 100,
            'Draw': (draws / n_games) * 100
        }

        player_stats = []; change_stats = {'Win': [], 'Loss': [], 'Draw': []}
        for p in [1, 2]:
            col_resp = f"player{p}_resp"
            moves = pd.to_numeric(df[col_resp], errors='coerce').values
            outcomes = df['outcome'].values
            valid_mask = (moves != 0) & (~np.isnan(moves))
            clean_moves = moves[valid_mask]
            
            if len(clean_moves) > 0:
                counts = pd.Series(clean_moves).value_counts(normalize=True) * 100
                sorted_counts = counts.sort_values(ascending=False).values
                if len(sorted_counts) < 3: sorted_counts = np.pad(sorted_counts, (0, 3-len(sorted_counts)))
            else: sorted_counts = [0, 0, 0]
            player_stats.append({'Most': sorted_counts[0], 'Mid': sorted_counts[1], 'Least': sorted_counts[2]})
            
            for t in range(1, len(moves)):
                if not valid_mask[t] or not valid_mask[t-1]: continue
                did_switch = 1 if moves[t] != moves[t-1] else 0
                prev_out = outcomes[t-1]
                
                # --- FIX 2: Strategy Condition Logic ---
                if prev_out == 1:
                    condition = 'Draw'
                elif (prev_out == 2 and p == 1) or (prev_out == 3 and p == 2):
                    condition = 'Win'
                else:
                    condition = 'Loss'
                
                change_stats[condition].append(did_switch)

        metrics_e = {
            'After win': np.mean(change_stats['Win']) * 100 if change_stats['Win'] else np.nan,
            'After loss': np.mean(change_stats['Loss']) * 100 if change_stats['Loss'] else np.nan,
            'After draw': np.mean(change_stats['Draw']) * 100 if change_stats['Draw'] else np.nan
        }
        return metrics_c, metrics_e, player_stats
    except Exception as e: print(f"Error Pair {pair_id}: {e}"); return None

# --- Polished Plotting Function ---
def plot_raincloud_polished(ax, data, title, color, label_bottom):
    """High-quality Raincloud simulation."""
    data = [d for d in data if not np.isnan(d)]
    if not data: return
    
    # 1. Violin (Density cloud)
    parts = ax.violinplot(data, vert=True, showmeans=False, showmedians=False, showextrema=False, widths=0.8)
    for pc in parts['bodies']:
        pc.set_facecolor(color)
        pc.set_edgecolor(color)
        pc.set_alpha(0.3) # Lighter density

    # 2. Jittered Scatter (Rain)
    x_jitter = np.random.normal(1, 0.05, size=len(data))
    ax.scatter(x_jitter, data, color=color, edgecolor='white', linewidth=0.5, alpha=0.7, s=25, zorder=2)
    
    # 3. Boxplot (Summary)
    bp = ax.boxplot(data, vert=True, widths=0.15, patch_artist=True, 
                    boxprops=dict(facecolor='white', color=color, linewidth=1.5),
                    medianprops=dict(color='black', linewidth=2),
                    whiskerprops=dict(color=color, linewidth=1),
                    capprops=dict(color=color, linewidth=1),
                    showfliers=False, zorder=3)

    # Layout styling
    ax.set_title(title, fontweight='bold', pad=10)
    ax.set_xticks([1])
    ax.set_xticklabels([label_bottom], fontweight='bold')
    ax.yaxis.grid(True, linestyle=':', alpha=0.6) # Subtle grid
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylabel("Percentage (%)")

def run_fig1_replication():
    data_root = Path(CONFIG['paths']['data_root'])
    if CONFIG['subjects']['run_mode'] == 'single': pairs = [CONFIG['subjects']['single_pair_id']]
    else:
        all_p = range(CONFIG['subjects']['pair_range'][0], CONFIG['subjects']['pair_range'][1] + 1)
        pairs = [p for p in all_p if p not in CONFIG['subjects']['exclude_pairs']]
    
    data_1c = {'Winner wins': [], 'Winner loses': [], 'Draw': []}
    data_1d = {'Most': [], 'Mid': [], 'Least': []}
    data_1e = {'After win': [], 'After loss': [], 'After draw': []}
    
    print(f"Collecting Behavior for {len(pairs)} pairs...")
    for pair_id in pairs:
        res = get_pair_behavior(pair_id, data_root)
        if res:
            m_c, m_e, p_stats = res
            for k, v in m_c.items(): data_1c[k].append(v)
            for k, v in m_e.items(): data_1e[k].append(v)
            for p in p_stats:
                data_1d['Most'].append(p['Most']); data_1d['Mid'].append(p['Mid']); data_1d['Least'].append(p['Least'])

    # --- PLOTTING ---
    fig, axes = plt.subplots(3, 3, figsize=(14, 14))
    fig.subplots_adjust(hspace=0.4, wspace=0.3)
    
    # Row 1: Fig 1C (Outcomes)
    plot_raincloud_polished(axes[0,0], data_1c['Winner wins'], "A) Game Outcome", COLORS['win'], "Winner Wins")
    plot_raincloud_polished(axes[0,1], data_1c['Winner loses'], "", COLORS['lose'], "Winner Loses")
    plot_raincloud_polished(axes[0,2], data_1c['Draw'], "", COLORS['draw'], "Draw")
    for ax in axes[0,:]: ax.axhline(33.3, color='grey', linestyle='--', alpha=0.5, linewidth=1)
    
    # Row 2: Fig 1D (Bias)
    plot_raincloud_polished(axes[1,0], data_1d['Most'], "B) Response Bias", COLORS['gold'], "Most Played")
    plot_raincloud_polished(axes[1,1], data_1d['Mid'], "", COLORS['orange'], "Mid Played")
    plot_raincloud_polished(axes[1,2], data_1d['Least'], "", COLORS['brown'], "Least Played")
    for ax in axes[1,:]: ax.axhline(33.3, color='grey', linestyle='--', alpha=0.5, linewidth=1)
    
    # Row 3: Fig 1E (Strategy Switch)
    plot_raincloud_polished(axes[2,0], data_1e['After win'], "C) Strategy Change", COLORS['purple'], "Switch after Win")
    plot_raincloud_polished(axes[2,1], data_1e['After loss'], "", COLORS['lose'], "Switch after Loss")
    plot_raincloud_polished(axes[2,2], data_1e['After draw'], "", COLORS['grey'], "Switch after Draw")
    for ax in axes[2,:]: ax.set_ylim(0, 100) # Change is 0-100%

    fig.suptitle("Behavioral Results", fontsize=16, fontweight='bold', y=0.98)
    
    out_dir = Path("figures/paper_replication")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "fig1_behavior.png", dpi=300, bbox_inches='tight')
    print(f"[PLOT] Saved {out_dir}/fig1_behavior.png")

if __name__ == "__main__":
    run_fig1_replication()