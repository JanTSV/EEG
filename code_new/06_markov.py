"""
06_behavioral_markov.py
-----------------------
Step 4: Behavioral Analysis.
Replicates Figure 1F from the paper[cite: 144].
Calculates the predictability of player moves using a sliding-window Markov Chain.
"""

import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Scientific Style
plt.rcParams.update({'font.size': 12, 'figure.figsize': (10, 6)})

def load_config():
    with open("code_new/config.yaml", 'r') as f:
        return yaml.safe_load(f)

CONFIG = load_config()

def get_clean_moves(tsv_path, col_name):
    """Loads moves and removes NaNs/Zeros."""
    try:
        df = pd.read_csv(tsv_path, sep='\t')
        if col_name not in df.columns: return None
        
        moves = pd.to_numeric(df[col_name], errors='coerce')
        # Filter 0 (No Response) and NaN
        valid = (moves != 0) & (~np.isnan(moves))
        return moves[valid].values.astype(int)
    except Exception as e:
        print(f"Error: {e}")
        return None

def predict_markov(moves, window_size, n_choices=3):
    """
    Predicts the next move based on the transition probabilities 
    observed in the last `window_size` trials.
    
    Markov Order 1: State is (Move t-1).
    """
    n_trials = len(moves)
    correct_preds = 0
    total_preds = 0
    
    # We can only start predicting when we have enough history
    # Start at window_size + 1 (need 1 previous move for state + window for history)
    start_idx = window_size + 1
    
    for t in range(start_idx, n_trials):
        # 1. Define History Window
        # Look at data from t - window_size - 1 to t - 1
        history = moves[t - window_size - 1 : t - 1]
        
        # 2. Build Transition Matrix from History
        # Count transitions: Move(k) -> Move(k+1)
        counts = np.zeros((n_choices + 1, n_choices + 1)) # +1 to handle 1-based indexing easily
        
        # Simple loop over history (fast enough for 480 trials)
        for k in range(len(history) - 1):
            prev = history[k]
            curr = history[k+1]
            counts[prev, curr] += 1
            
        # 3. Identify Current State (Move at t-1)
        current_state = moves[t-1]
        
        # 4. Make Prediction
        # Look at the row for current_state in counts
        # Which next move has the highest count?
        row = counts[current_state, :]
        
        if np.sum(row) == 0:
            # If we've never seen this state in the window, guess random
            # (In this code, random guess = 1/3 chance, expected value adds 0.33)
            # To measure strict accuracy, we count it as 0 unless we implement random logic.
            # Let's assume strict prediction: if no data, we can't predict.
            predicted_move = -1 
        else:
            # Argmax gives the index of the highest count
            # Use random choice for ties to be scientifically accurate
            best_indices = np.where(row == row.max())[0]
            predicted_move = np.random.choice(best_indices)
            
        # 5. Verify
        actual_move = moves[t]
        
        if predicted_move == actual_move:
            correct_preds += 1
        
        total_preds += 1
        
    return (correct_preds / total_preds) * 100 if total_preds > 0 else 0

def run_markov_analysis():
    cfg = load_config()
    
    # Settings
    w_min = cfg['behavioral']['window_min']
    w_max = cfg['behavioral']['window_max']
    windows = range(w_min, w_max + 1)
    
    data_root = Path(cfg['paths']['data_root'])
    
    # Subjects
    if cfg['subjects']['run_mode'] == 'single':
        pairs = [cfg['subjects']['single_pair_id']]
    else:
        all_p = range(cfg['subjects']['pair_range'][0], cfg['subjects']['pair_range'][1] + 1)
        pairs = [p for p in all_p if p not in cfg['subjects']['exclude_pairs']]

    print(f"--- STARTING MARKOV ANALYSIS ({w_min}-{w_max} window) ---")
    
    results = {'windows': list(windows)}

    for pair_id in pairs:
        sub_str = f"sub-{pair_id:02d}"
        tsv_path = data_root / sub_str / 'eeg' / f"{sub_str}_task-RPS_events.tsv"
        
        if not tsv_path.exists(): continue
        
        print(f"\nAnalyzing Pair {pair_id}...")
        
        for player_num in [1, 2]:
            col = cfg['behavioral']['target_cols'][f"player_{player_num}"]
            moves = get_clean_moves(tsv_path, col)
            
            if moves is None or len(moves) < w_max:
                print(f"  [WARN] Not enough moves for Player {player_num}")
                continue
                
            accs = []
            # Loop over window sizes
            for w in windows:
                acc = predict_markov(moves, w)
                accs.append(acc)
            
            # Store results
            key = f"pair{pair_id}_p{player_num}"
            results[key] = accs
            print(f"  Player {player_num}: Mean Acc = {np.mean(accs):.1f}%")

    # Plotting (Replicating Figure 1F)
    plot_markov_results(results, windows)

def plot_markov_results(results, windows):
    fig, ax = plt.subplots()
    
    # Plot Chance Level
    ax.axhline(33.33, color='black', linestyle='--', label='Chance (33%)')
    
    # Collect all curves for average
    all_curves = []
    
    for key, accs in results.items():
        if key == 'windows': continue
        # Plot individual lines (thin, grey)
        ax.plot(windows, accs, color='grey', alpha=0.3, linewidth=1)
        all_curves.append(accs)
        
    # Plot Average (Blue, Thick)
    if all_curves:
        mean_curve = np.mean(all_curves, axis=0)
        ax.plot(windows, mean_curve, color='blue', linewidth=3, label='Group Average')
        
        # Optional: Confidence Interval
        std_curve = np.std(all_curves, axis=0)
        ax.fill_between(windows, mean_curve - std_curve, mean_curve + std_curve, color='blue', alpha=0.1)

    ax.set_title("Markov Chain Predictability (Replication of Fig 1F)")
    ax.set_xlabel("Window Size (N previous games)")
    ax.set_ylabel("Prediction Accuracy (%)")
    ax.set_ylim(20, 60)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    out_dir = Path("figures/04_behavior")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "markov_chain_accuracy.png", dpi=300)
    print(f"\n[PLOT] Saved to figures/04_behavior/markov_chain_accuracy.png")

if __name__ == "__main__":
    run_markov_analysis()