"""
10_continuous_analysis.py (LOGIC FIX)
-------------------------------------
Extension 2: Continuous Performance Analysis.
Correlates 'Win Rate' with 'Decoding Accuracy' across all subjects.

FIX: Correct mapping of outcomes:
1 = Draw
2 = Player 1 Win
3 = Player 2 Win
"""

import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from pathlib import Path

# Style
plt.rcParams.update({'font.size': 12, 'figure.figsize': (10, 6)})

def load_config():
    if Path("code_new/config.yaml").exists():
        with open("code_new/config.yaml", 'r') as f: return yaml.safe_load(f)

CONFIG = load_config()

def get_player_stats(pair_id, player_num, data_root):
    """Calculates Win Rate for a specific player."""
    sub_str = f"sub-{pair_id:02d}"
    tsv_path = data_root / sub_str / 'eeg' / f"{sub_str}_task-RPS_events.tsv"
    
    if not tsv_path.exists(): return None
    
    try:
        df = pd.read_csv(tsv_path, sep='\t')
        n_games = len(df)
        if n_games == 0: return 0.0
        
        # --- FIX MAPPING ---
        # 1=Draw, 2=P1 win, 3=P2 win
        target_outcome = 2 if player_num == 1 else 3
        
        wins = np.sum(df['outcome'] == target_outcome)
        win_rate = (wins / n_games) * 100
        return win_rate
    except:
        return None

def load_decoding_metric(pair_id, player_num, target, phase_bins):
    """
    Loads decoding result and averages over a specific time phase.
    """
    # Robust path handling
    if Path("/Volumes/HardDiskYF/ACADEMIA/EEG/ds006761-download/derivatives_new/03_decoding_results").exists():
        res_dir = Path("/Volumes/HardDiskYF/ACADEMIA/EEG/ds006761-download/derivatives_new/03_decoding_results")
    else:
        res_dir = Path("/Volumes/HardDiskYF/ACADEMIA/EEG/ds006761-download/derivatives_new/03_decoding_results")

    file_path = res_dir / f"pair-{pair_id:02d}_player-{player_num}_target-{target}_acc.npy"
    
    if not file_path.exists(): return None
    
    acc_curve = np.load(file_path) # Shape (20,)
    if np.isnan(acc_curve).all(): return None
    
    # Average over the phase
    start, end = phase_bins
    # Safety check
    if start >= len(acc_curve): return None
    end = min(end, len(acc_curve))
    
    return np.mean(acc_curve[start:end])

def run_correlation_analysis():
    cfg = CONFIG
    data_root = Path(cfg['paths']['data_root'])
    out_dir = Path("figures/05_continuous_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Define Analysis Parameters
    # 0.25s bins. 
    # Decision (0-2s) -> Bins 0-8
    # Response (2-4s) -> Bins 8-16
    phases = {
        'Decision': (0, 8),
        'Response': (8, 16)
    }
    
    targets = ['own_prev', 'opp_prev', 'own_current'] 
    
    # Collect Data
    data_records = []
    
    # Get Pairs
    if cfg['subjects']['run_mode'] == 'single':
        pairs = [cfg['subjects']['single_pair_id']]
    else:
        all_p = range(cfg['subjects']['pair_range'][0], cfg['subjects']['pair_range'][1] + 1)
        pairs = [p for p in all_p if p not in cfg['subjects']['exclude_pairs']]
        
    print(f"Collecting stats for {len(pairs)} pairs...")
    
    for pair_id in pairs:
        for player in [1, 2]:
            win_rate = get_player_stats(pair_id, player, data_root)
            if win_rate is None: continue
            
            record = {
                'pair': pair_id,
                'player': player,
                'win_rate': win_rate
            }
            
            # Load decoding for all targets/phases
            has_data = False
            for tgt in targets:
                for phase_name, bins in phases.items():
                    val = load_decoding_metric(pair_id, player, tgt, bins)
                    if val is not None:
                        record[f"{tgt}_{phase_name}"] = val
                        has_data = True
            
            if has_data:
                data_records.append(record)
                
    df = pd.DataFrame(data_records)
    print(f"Analyzed {len(df)} players.")
    
    # --- PLOTTING ---
    plot_targets = ['own_prev', 'opp_prev'] # Focus on these
    
    fig, axes = plt.subplots(len(plot_targets), len(phases), figsize=(12, 10))
    fig.subplots_adjust(hspace=0.4, wspace=0.3)
    
    for i, tgt in enumerate(plot_targets):
        for j, (phase, _) in enumerate(phases.items()):
            col_name = f"{tgt}_{phase}"
            if col_name not in df.columns: continue
            
            ax = axes[i, j]
            
            # Scatter Plot with Regression
            sns.regplot(data=df, x='win_rate', y=col_name, ax=ax, 
                        scatter_kws={'alpha':0.6, 'edgecolor':'white'},
                        line_kws={'color':'red'})
            
            # Calculate Correlation
            valid = df.dropna(subset=['win_rate', col_name])
            if len(valid) > 2:
                r, p = pearsonr(valid['win_rate'], valid[col_name])
                stats_text = f"r={r:.2f}, p={p:.3f}"
            else:
                stats_text = "N/A"
            
            # Styling
            readable_tgt = "Own Previous" if "own" in tgt else "Opponent Previous"
            ax.set_title(f"{readable_tgt} ({phase})\n{stats_text}")
            ax.set_xlabel("Win Rate (%)")
            ax.set_ylabel("Decoding Accuracy (%)")
            ax.grid(True, linestyle=':', alpha=0.5)
            
    fig.suptitle("Continuous Performance Analysis: Predictability vs. Success", fontsize=16, y=0.98)
    
    save_path = out_dir / "correlation_matrix.png"
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[PLOT] Saved {save_path}")
    
    # Optional: Check 'own_current' vs Win Rate
    plt.figure(figsize=(6, 5))
    col = 'own_current_Response'
    if col in df.columns:
        valid = df.dropna(subset=['win_rate', col])
        sns.regplot(data=valid, x='win_rate', y=col, line_kws={'color':'green'})
        r, p = pearsonr(valid['win_rate'], valid[col])
        plt.title(f"Current Move Decoding vs. Win Rate\nr={r:.2f}, p={p:.3f}")
        plt.xlabel("Win Rate (%)")
        plt.ylabel("Decoding Acc (Response Phase)")
        plt.savefig(out_dir / "correlation_current_move.png", dpi=300)
        print(f"[PLOT] Saved correlation_current_move.png")

if __name__ == "__main__":
    run_correlation_analysis()