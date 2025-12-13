# ----- GROUP-LEVEL DECODING AGGREGATION -----
# Aggregate decoding results across pairs/players
# Computes mean, SEM, and statistical tests
#
# Usage:
#   python code/aggregate_decoding.py --pairs 1 2 3 --targets current_self previous_self

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import ttest_1samp
import yaml

def load_config():
    """Load decoding config to get paths and settings."""
    cfg_path = Path(__file__).resolve().parent / "decoding_config.yaml"
    with cfg_path.open("r") as f:
        return yaml.safe_load(f)

def load_results(results_dir, pair, player, target_name):
    """Load evaluation_report.csv for a given pair/player/target."""
    csv_path = results_dir / f"{target_name}" / "evaluation_report.csv"
    
    # Check if pair-player specific results exist (if you save per pair/player)
    # For now, assuming results are saved per target in a flat structure
    # Adjust if your structure is different
    
    if not csv_path.exists():
        return None
    
    df = pd.read_csv(csv_path)
    return df

def aggregate_across_subjects(pairs, players, cfg, target_name, model_name="lda"):
    """
    Aggregate decoding results across pairs and players.
    
    Parameters
    ----------
    pairs : list of int
        Pair IDs to include.
    players : list of int
        Player numbers (1, 2) to include.
    cfg : dict
        Config dictionary.
    target_name : str
        Target to aggregate (e.g., 'current_self').
    model_name : str
        Model name to filter results (e.g., 'lda').
    
    Returns
    -------
    times : np.ndarray
        Time vector.
    mean_acc : np.ndarray
        Mean accuracy across subjects.
    sem_acc : np.ndarray
        Standard error of mean.
    all_accs : np.ndarray
        All individual accuracy curves (n_subjects, n_timepoints).
    """
    base_dir = Path(__file__).resolve().parent.parent
    results_base = base_dir / cfg["paths"]["results_dir_base"]
    
    all_accuracies = []
    subject_ids = []
    
    for pair in pairs:
        for player in players:
            # Construct path: results_base / pair-XX_player-Y / target_name / evaluation_report.csv
            player_dir = results_base / f"pair-{pair:02d}_player-{player}"
            csv_path = player_dir / target_name / "evaluation_report.csv"
            
            if not csv_path.exists():
                print(f"  ! Warning: No results for pair {pair}, player {player}, target {target_name}")
                print(f"    Expected: {csv_path}")
                continue
            
            df = pd.read_csv(csv_path)
            
            # Filter by model
            df_model = df[df["model"] == model_name]
            
            if df_model.empty:
                print(f"  ! Warning: No {model_name} results for pair {pair}, player {player}")
                continue
            
            # Extract accuracy time course
            times = df_model["time"].values
            acc = df_model["mean_accuracy"].values
            
            all_accuracies.append(acc)
            subject_ids.append(f"pair{pair:02d}_player{player}")
    
    if len(all_accuracies) == 0:
        raise ValueError("No results found for specified pairs/players")
    
    # Convert to array
    all_accuracies = np.array(all_accuracies)  # shape: (n_subjects, n_timepoints)
    
    # Compute group statistics
    mean_acc = np.mean(all_accuracies, axis=0)
    sem_acc = np.std(all_accuracies, axis=0) / np.sqrt(len(all_accuracies))
    
    print(f"\n  > Aggregated {len(all_accuracies)} subjects for {target_name}")
    print(f"    Time points: {len(times)}")
    print(f"    Mean accuracy range: [{mean_acc.min():.3f}, {mean_acc.max():.3f}]")
    
    return times, mean_acc, sem_acc, all_accuracies, subject_ids

def statistical_test(all_accuracies, chance_level=0.333, alpha=0.05):
    """
    Perform one-sample t-test at each timepoint.
    
    Parameters
    ----------
    all_accuracies : np.ndarray
        Shape (n_subjects, n_timepoints).
    chance_level : float
        Chance level for comparison.
    alpha : float
        Significance threshold.
    
    Returns
    -------
    p_values : np.ndarray
        P-values at each timepoint.
    significant : np.ndarray
        Boolean mask of significant timepoints.
    """
    n_times = all_accuracies.shape[1]
    p_values = []
    
    for t in range(n_times):
        _, p_val = ttest_1samp(all_accuracies[:, t], chance_level)
        p_values.append(p_val)
    
    p_values = np.array(p_values)
    significant = p_values < alpha
    
    return p_values, significant

def plot_group_results(times, mean_acc, sem_acc, all_accuracies, target_name, 
                       model_name, cfg, out_path, show_individuals=False):
    """Plot group-level results with mean ± SEM."""
    chance = cfg["stats"]["chance_level"]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot individual subjects (optional)
    if show_individuals:
        for i, acc in enumerate(all_accuracies):
            ax.plot(times, acc, color='gray', alpha=0.2, linewidth=0.5)
    
    # Plot group mean
    color = cfg["plotting"]["colors"].get(model_name, "blue")
    ax.plot(times, mean_acc, color=color, linewidth=3, label=f'{model_name.upper()} (N={len(all_accuracies)})')
    ax.fill_between(times, mean_acc - sem_acc, mean_acc + sem_acc, color=color, alpha=0.3)
    
    # Statistical significance markers
    p_vals, sig = statistical_test(all_accuracies, chance)
    sig_times = times[sig]
    if len(sig_times) > 0:
        ax.scatter(sig_times, [chance - 0.015] * len(sig_times), 
                  marker='*', color='black', s=50, zorder=10, label='p < 0.05')
    
    # Reference lines
    ax.axhline(chance, color='k', linestyle='--', linewidth=1.5, label=f'Chance ({chance:.2f})')
    ax.axvline(0, color='r', linestyle=':', linewidth=1.5, alpha=0.7)
    
    # Styling
    ax.set_title(f"Group-Level Decoding: {target_name.upper()} ({model_name.upper()})", 
                fontsize=16, fontweight='bold')
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_xlim(times[0], times[-1])
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"  > Saved: {out_path}")

def plot_all_models(model_results, target_name, cfg, out_path, show_individuals=False):
    """Plot all models together for comparison."""
    chance = cfg["stats"]["chance_level"]
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Plot each model
    for model_name, results in model_results.items():
        times = results['times']
        mean_acc = results['mean_acc']
        sem_acc = results['sem_acc']
        all_accs = results['all_accs']
        sig = results['sig']
        
        color = cfg["plotting"]["colors"].get(model_name, "gray")
        
        # Plot individual subjects (optional, only for first model to avoid clutter)
        if show_individuals and model_name == list(model_results.keys())[0]:
            for i, acc in enumerate(all_accs):
                ax.plot(times, acc, color='gray', alpha=0.15, linewidth=0.5, zorder=1)
        
        # Plot group mean
        label = f'{model_name.upper()} (N={len(all_accs)})'
        ax.plot(times, mean_acc, color=color, linewidth=2.5, label=label, zorder=3)
        
        if cfg["plotting"]["show_shadow"]:
            ax.fill_between(times, mean_acc - sem_acc, mean_acc + sem_acc, 
                          color=color, alpha=0.2, zorder=2)
        
        # Significance markers (offset vertically by model)
        sig_times = times[sig]
        if len(sig_times) > 0:
            marker_y = chance - 0.015 - (list(model_results.keys()).index(model_name) * 0.01)
            ax.scatter(sig_times, [marker_y] * len(sig_times), 
                      marker='*', color=color, s=40, zorder=10, alpha=0.8)
    
    # Reference lines
    ax.axhline(chance, color='k', linestyle='--', linewidth=1.5, 
              label=f'Chance ({chance:.2f})', zorder=4)
    ax.axvline(0, color='r', linestyle=':', linewidth=1.5, alpha=0.7, zorder=4)
    
    # Styling
    ax.set_title(f"Group-Level Decoding: {target_name.upper()} (All Models)", 
                fontsize=16, fontweight='bold')
    ax.set_xlabel("Time (s)", fontsize=14)
    ax.set_ylabel("Accuracy", fontsize=14)
    ax.set_xlim(times[0], times[-1])
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"  > Saved combined plot: {out_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Aggregate decoding results across pairs/players."
    )
    parser.add_argument("--pairs", type=int, nargs="+", required=True,
                       help="Pair IDs to include (e.g., 1 2 3)")
    parser.add_argument("--players", type=int, nargs="+", default=[1, 2],
                       help="Player numbers (default: 1 2)")
    parser.add_argument("--targets", type=str, nargs="+", default=None,
                       help="Targets to aggregate (default: all in config)")
    parser.add_argument("--models", type=str, nargs="+", default=None,
                       help="Models to aggregate (default: all enabled in config)")
    parser.add_argument("--show-individuals", action="store_true",
                       help="Show individual subject curves in plots")
    
    args = parser.parse_args()
    
    # Load config
    cfg = load_config()
    
    # Get targets
    if args.targets is None:
        targets = list(cfg["experiment"]["targets"].keys())
    else:
        targets = args.targets
    
    # Get models (all enabled models by default)
    if args.models is None:
        models = [name for name, model_cfg in cfg["models"].items() 
                 if model_cfg.get("enabled", False)]
    else:
        models = args.models
    
    print("="*60)
    print(f"GROUP-LEVEL AGGREGATION")
    print(f"Pairs: {args.pairs}")
    print(f"Players: {args.players}")
    print(f"Targets: {targets}")
    print(f"Models: {models}")
    print("="*60)
    
    # Create output directory
    base_dir = Path(__file__).resolve().parent.parent
    results_base = base_dir / cfg["paths"]["results_dir_base"]
    group_dir = results_base / "group_analysis"
    group_dir.mkdir(exist_ok=True, parents=True)
    
    # Aggregate for each target and model
    for target_name in targets:
        print(f"\n--- Processing: {target_name} ---")
        
        # Store all model results for combined plot
        all_model_results = {}
        
        for model_name in models:
            try:
                # Aggregate
                times, mean_acc, sem_acc, all_accs, subject_ids = aggregate_across_subjects(
                    args.pairs, args.players, cfg, target_name, model_name
                )
                
                # Save aggregated data
                out_csv = group_dir / f"{target_name}_{model_name}_group.csv"
                p_vals, sig = statistical_test(all_accs, cfg["stats"]["chance_level"])
                
                df = pd.DataFrame({
                    "time": times,
                    "mean_accuracy": mean_acc,
                    "sem": sem_acc,
                    "p_value": p_vals,
                    "significant": sig,
                    "n_subjects": len(all_accs)
                })
                df.to_csv(out_csv, index=False)
                print(f"  > Saved: {out_csv}")
                
                # Store for combined plot
                all_model_results[model_name] = {
                    'times': times,
                    'mean_acc': mean_acc,
                    'sem_acc': sem_acc,
                    'all_accs': all_accs,
                    'p_vals': p_vals,
                    'sig': sig
                }
                
            except Exception as e:
                print(f"  ! Error processing {target_name}/{model_name}: {e}")
                continue
        
        # Create combined plot with all models
        if all_model_results:
            out_fig_combined = group_dir / f"{target_name}_all_models_group.png"
            plot_all_models(all_model_results, target_name, cfg, out_fig_combined, args.show_individuals)
    
    print("\n" + "="*60)
    print("DONE. Check results in: data/derivatives/decoding_results/group_analysis/")
    print("="*60)

if __name__ == "__main__":
    main()
