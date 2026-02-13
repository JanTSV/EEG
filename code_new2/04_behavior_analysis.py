import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn as sns
from pathlib import Path

# --- STYLE CONFIG ---
plt.rcParams.update({'font.size': 12, 'figure.dpi': 300})
COLORS = {
    'outcomes': ['#4c72b0', '#c44e52', '#808080'], # Winner(Blue), Loser(Red), Draw(Grey)
    'ranks':    ['#bcbd22', '#ff7f0e', '#17becf'], # Gold, Orange, Cyan
    'switch':   ['#2ca02c', '#d62728', '#7f7f7f'], # Win(Green), Lose(Red), Draw(Grey)
    'rps':      ['#666666', '#aaaaaa', '#dddddd']  # Rock, Paper, Scissors (Greys/Blacks)
}

def load_config():
    with open("code_new2/config_decoding.yaml", 'r') as f: return yaml.safe_load(f)

def analyze_behavior(events_df, sub_id):
    """Extracts stats including WHICH option was rank 1/2/3."""
    valid = events_df[(events_df['player1_resp'] != 0) & (events_df['player2_resp'] != 0)].copy()
    n_trials = len(valid)
    stats = []

    # Determine Pair Winner
    p1_wins = np.sum(valid['outcome'] == 2)
    p2_wins = np.sum(valid['outcome'] == 3)
    p1_is_winner = p1_wins > p2_wins

    for player_num in [1, 2]:
        prefix = f"player{player_num}"
        # Win/Loss counts
        win_code = 2 if player_num == 1 else 3
        lose_code = 3 if player_num == 1 else 2
        
        n_wins = np.sum(valid['outcome'] == win_code)
        n_loss = np.sum(valid['outcome'] == lose_code)
        n_draw = np.sum(valid['outcome'] == 1)
        status = "Winner" if (p1_is_winner and player_num==1) or (not p1_is_winner and player_num==2) else "Loser"

        # Strategy (Counts of R(1), P(2), S(3))
        counts = valid[f'{prefix}_resp'].value_counts()
        for c in [1, 2, 3]: 
            if c not in counts: counts[c] = 0
        
        # Sort: Index 0 is Most played key (1, 2 or 3)
        sorted_keys = counts.sort_values(ascending=False).index.tolist() # e.g. [1, 3, 2] (Rock, Scissors, Paper)
        sorted_vals = counts.sort_values(ascending=False).values / n_trials * 100
        
        # Switch Strategy
        curr = valid[f'{prefix}_resp'].values[1:]
        prev = valid[f'{prefix}_resp'].values[:-1]
        prev_out = valid['outcome'].values[:-1]
        switched = (curr != prev)
        
        mask_win = (prev_out == win_code)
        mask_loss = (prev_out == lose_code)
        mask_draw = (prev_out == 1)

        row = {
            'subject': f"sub-{sub_id:02d}_p{player_num}",
            'status': status,
            'pct_win': (n_wins/n_trials)*100,
            'pct_loss': (n_loss/n_trials)*100,
            'pct_draw': (n_draw/n_trials)*100,
            # Strategy magnitude
            'most_pct': sorted_vals[0],
            'mid_pct': sorted_vals[1],
            'least_pct': sorted_vals[2],
            # Strategy Identity (1=Rock, 2=Paper, 3=Scissors)
            'most_id': sorted_keys[0],
            'mid_id': sorted_keys[1],
            'least_id': sorted_keys[2],
            # Switching
            'switch_win': np.mean(switched[mask_win])*100 if np.sum(mask_win)>0 else np.nan,
            'switch_loss': np.mean(switched[mask_loss])*100 if np.sum(mask_loss)>0 else np.nan,
            'switch_draw': np.mean(switched[mask_draw])*100 if np.sum(mask_draw)>0 else np.nan
        }
        stats.append(row)
    return stats

def add_pie_charts(ax, df):
    """Adds small pie charts above the violins showing R/P/S distribution."""
    # Positions for Most, Mid, Least
    positions = [0, 1, 2] 
    ranks = ['most_id', 'mid_id', 'least_id']
    
    for i, rank_col in enumerate(ranks):
        # Count how often Rock(1), Paper(2), Scissors(3) was the Rank X choice
        counts = df[rank_col].value_counts()
        # Ensure order R, P, S
        pie_data = [counts.get(1, 0), counts.get(2, 0), counts.get(3, 0)]
        
        # Inset Axes
        # bbox_to_anchor controls position (x, y, width, height) in axes coords
        ax_ins = inset_axes(ax, width="30%", height="30%", loc='upper center', 
                           bbox_to_anchor=(i/3, 0.75, 0.33, 0.33), 
                           bbox_transform=ax.transAxes)
        
        ax_ins.pie(pie_data, colors=COLORS['rps'], startangle=90, 
                  wedgeprops={'edgecolor':'k', 'linewidth': 0.5})
        ax_ins.set_aspect('equal')

def plot_behavior_polished(df, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    
    # --- A: Outcomes ---
    # Construct Long Format for Seaborn
    w = df[df['status']=='Winner']; l = df[df['status']=='Loser']
    d_out = pd.DataFrame({
        'Type': ['Winner Wins']*len(w) + ['Loser Wins']*len(l) + ['Draws']*len(df),
        'Pct': np.concatenate([w['pct_win'], l['pct_win'], df['pct_draw']])
    })
    sns.violinplot(data=d_out, x='Type', y='Pct', ax=axes[0], palette=COLORS['outcomes'], inner='quartile')
    axes[0].set_title("Game Outcomes", fontweight='bold')
    axes[0].set_ylabel("Percentage (%)")
    axes[0].set_xlabel("")
    
    # --- B: Response Bias (With Pies!) ---
    d_bias = pd.DataFrame({
        'Rank': ['Most']*len(df) + ['Mid']*len(df) + ['Least']*len(df),
        'Pct': np.concatenate([df['most_pct'], df['mid_pct'], df['least_pct']])
    })
    sns.violinplot(data=d_bias, x='Rank', y='Pct', ax=axes[1], palette=COLORS['ranks'], inner='quartile')
    add_pie_charts(axes[1], df) # <--- The Magic
    axes[1].set_title("Response Bias", fontweight='bold')
    axes[1].set_ylabel("")
    axes[1].set_xlabel("Frequency Rank")
    
    # Add Legend for Pies
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor=c, label=l, markersize=8) 
                       for c, l in zip(COLORS['rps'], ['Rock', 'Paper', 'Scissors'])]
    axes[1].legend(handles=legend_elements, loc='upper right', fontsize=8, title="Pie Legend")

    # --- C: Switching ---
    d_switch = pd.DataFrame({
        'Cond': ['After Win']*len(df) + ['After Loss']*len(df) + ['After Draw']*len(df),
        'Pct': np.concatenate([df['switch_win'], df['switch_loss'], df['switch_draw']])
    })
    sns.violinplot(data=d_switch, x='Cond', y='Pct', ax=axes[2], palette=COLORS['switch'], inner='quartile')
    axes[2].set_title("Switching Strategy", fontweight='bold')
    axes[2].set_ylabel("")
    axes[2].set_xlabel("Previous Outcome")

    # Polish
    for ax in axes:
        ax.grid(axis='y', linestyle=':', alpha=0.5)
        ax.set_ylim(-5, 105)
        sns.despine(ax=ax)

    plt.tight_layout()
    out_file = out_dir / "Figure1_Behavior_Polished.png"
    plt.savefig(out_file, dpi=300)
    print(f"Saved: {out_file}")

def run():
    cfg = load_config()
    bids_root = Path(cfg['paths']['bids_root'])
    out_dir = Path(cfg['paths']['figures_dir'])
    
    all_stats = []
    for sub_id in cfg['subjects']['include']:
        sub_str = f"sub-{sub_id:02d}"
        tsv = bids_root / sub_str / "eeg" / f"{sub_str}_task-RPS_events.tsv"
        if tsv.exists():
            try:
                all_stats.extend(analyze_behavior(pd.read_csv(tsv, sep='\t'), sub_id))
            except: pass
            
    plot_behavior_polished(pd.DataFrame(all_stats), out_dir)

if __name__ == "__main__":
    run()