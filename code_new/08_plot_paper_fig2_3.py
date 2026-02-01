"""
08_plot_paper_fig2_3.py (STYLED + CI FIX + LOGIC FIX)
-----------------------------------------------------
Replication of Figure 2 & 3.
Fig 2: Colored phases style.
Fig 3: Winners vs Losers with clear Confidence Intervals.

FIX: Correct mapping of outcomes:
1 = Draw (ignored for win count)
2 = Player 1 Win
3 = Player 2 Win
"""

import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Scientific Style
plt.rcParams.update({'font.size': 12, 'figure.figsize': (12, 10)})

def load_config():
    if Path("code_new/config.yaml").exists():
        with open("code_new/config.yaml", 'r') as f: return yaml.safe_load(f)

CONFIG = load_config()

def determine_winner_loser(pair_id, data_root):
    sub_str = f"sub-{pair_id:02d}"
    tsv_path = data_root / sub_str / 'eeg' / f"{sub_str}_task-RPS_events.tsv"
    if not tsv_path.exists(): return 0, 0
    df = pd.read_csv(tsv_path, sep='\t')
    
    # --- FIX MAPPING HERE ---
    # 1=Draw, 2=P1 Win, 3=P2 Win
    p1_wins = np.sum(df['outcome'] == 2)
    p2_wins = np.sum(df['outcome'] == 3)
    
    if p1_wins > p2_wins: return 1, 2
    if p2_wins > p1_wins: return 2, 1
    return 0, 0 

def load_accuracies_for_target(target_name):
    data_root = Path(CONFIG['paths']['data_root'])
    # Ensure path works regardless of where script is run
    if Path("/Volumes/HardDiskYF/ACADEMIA/EEG/ds006761-download/derivatives_new/03_decoding_results").exists():
        res_dir = Path("/Volumes/HardDiskYF/ACADEMIA/EEG/ds006761-download/derivatives_new/03_decoding_results")
    else:
        res_dir = Path("/Volumes/HardDiskYF/ACADEMIA/EEG/ds006761-download/derivatives_new/03_decoding_results")

    all_accs, win_accs, lose_accs = [], [], []
    
    if CONFIG['subjects']['run_mode'] == 'single': 
        pairs = [CONFIG['subjects']['single_pair_id']]
    else:
        all_p = range(CONFIG['subjects']['pair_range'][0], CONFIG['subjects']['pair_range'][1] + 1)
        pairs = [p for p in all_p if p not in CONFIG['subjects']['exclude_pairs']]
        
    for pair_id in pairs:
        win_p, lose_p = determine_winner_loser(pair_id, data_root)
        for player in [1, 2]:
            file_path = res_dir / f"pair-{pair_id:02d}_player-{player}_target-{target_name}_acc.npy"
            if file_path.exists():
                acc = np.load(file_path)
                if np.isnan(acc).all(): continue
                all_accs.append(acc)
                if player == win_p: win_accs.append(acc)
                elif player == lose_p: lose_accs.append(acc)
    return np.array(all_accs), np.array(win_accs), np.array(lose_accs)

def plot_styled_curve(ax, data):
    """Plots data split into 3 colored phases for Fig 2."""
    if len(data) == 0: return
    mean = np.mean(data, axis=0); sem = np.std(data, axis=0) / np.sqrt(len(data))
    times = np.linspace(0, 5, len(mean))
    indices = {'Decision': (0, 9, '#F4A460'), 'Response': (8, 17, '#CD5C5C'), 'Feedback': (16, 20, '#9370DB')}
    for _, (start, end, color) in indices.items():
        if start >= len(times): continue
        safe_end = min(end, len(times))
        t_seg = times[start:safe_end]; m_seg = mean[start:safe_end]; s_seg = sem[start:safe_end]
        ax.plot(t_seg, m_seg, color=color, linewidth=2, marker='o', markersize=4)
        ax.fill_between(t_seg, m_seg - s_seg, m_seg + s_seg, color=color, alpha=0.2)

def setup_styled_ax(ax, title):
    ax.axvspan(0, 2, color='#F4A460', alpha=0.05); ax.text(1, 44, 'Decision', ha='center', va='top', fontsize=9, fontweight='bold')
    ax.axvspan(2, 4, color='#CD5C5C', alpha=0.05); ax.text(3, 44, 'Response', ha='center', va='top', fontsize=9, fontweight='bold')
    ax.axvspan(4, 5, color='#9370DB', alpha=0.05); ax.text(4.5, 44, 'Feedback', ha='center', va='top', fontsize=9, fontweight='bold')
    ax.axhline(33.33, color='black', linestyle='--', linewidth=1)
    ax.set_title(title, loc='left', fontsize=12, fontweight='bold')
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Decoding accuracy (%)")
    ax.set_ylim(31, 40); ax.set_xlim(0, 5) # Adjusted ylim to fit data better

def run_full_plots():
    targets = ['own_current', 'opp_current', 'own_prev', 'opp_prev']
    titles = ['A) Own response', 'B) Opponent\'s response', 'C) Own previous response', 'D) Opponent\'s previous response']
    
    out_dir = Path("figures/paper_replication")
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- FIG 2: OVERALL (STYLED) ---
    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10))
    axes2 = axes2.flatten()
    print("Generating Figure 2 (Styled)...")
    for i, tgt in enumerate(targets):
        all_d, _, _ = load_accuracies_for_target(tgt)
        setup_styled_ax(axes2[i], titles[i])
        if len(all_d) > 0: plot_styled_curve(axes2[i], all_d)
    fig2.tight_layout(); fig2.savefig(out_dir / "fig2_decoding.png", dpi=300); print(f"[PLOT] Saved {out_dir}/fig2_decoding.png")

    # --- FIG 3: SPLIT (WITH CI SHADING) ---
    fig3, axes3 = plt.subplots(2, 2, figsize=(12, 10))
    axes3 = axes3.flatten()
    print("Generating Figure 3 (Winners vs Losers with CI)...")
    for i, tgt in enumerate(targets):
        _, win_d, lose_d = load_accuracies_for_target(tgt)
        
        # Background setup (grey zones)
        axes3[i].axvspan(0, 2, color='grey', alpha=0.05)
        axes3[i].axvspan(2, 4, color='grey', alpha=0.1)
        axes3[i].axvspan(4, 5, color='grey', alpha=0.05)
        axes3[i].axhline(33.33, color='black', linestyle='--')
        axes3[i].set_title(titles[i], loc='left', fontweight='bold')
        axes3[i].set_ylim(31, 45); axes3[i].set_xlim(0, 5)
        
        # Plot Lines WITH SHADING (CI)
        if len(win_d) > 0:
            def plot_with_ci(ax, d, c, l):
                m = np.mean(d, axis=0)
                s = np.std(d, axis=0)/np.sqrt(len(d)) # SEM
                t = np.linspace(0, 5, len(m))
                ax.plot(t, m, color=c, label=l, linewidth=2)
                # HERE IS THE SHADING: Increased alpha for better visibility
                ax.fill_between(t, m-s, m+s, color=c, alpha=0.3) 
            
            plot_with_ci(axes3[i], win_d, 'blue', 'Winners')
            plot_with_ci(axes3[i], lose_d, 'green', 'Losers')
            
        if i == 0: axes3[i].legend(loc='upper left')

    fig3.tight_layout(); fig3.savefig(out_dir / "fig3_decoding_split.png", dpi=300); print(f"[PLOT] Saved {out_dir}/fig3_decoding_split.png")

if __name__ == "__main__":
    run_full_plots()