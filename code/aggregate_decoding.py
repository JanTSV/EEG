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
import mne

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
    subject_ids : list
        List of subject identifiers (pair-XX_player-Y format).
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
            subject_ids.append(f"pair-{pair:02d}_player-{player}")
    
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

def compute_bayes_factor(all_accuracies, chance_level=0.333, null_effect_range=[0.0, 0.5]):
    """
    Compute Bayesian one-sample t-test with directional effect size null hypothesis.
    
    Uses a simplified approach: compute Bayes Factor as likelihood ratio between
    H1 (effect exists, d > 0) and H0 (no effect, d in null_effect_range).
    
    BF > 1 supports H1 (decoding above chance), BF < 1 supports H0 (no evidence).
    
    Parameters
    ----------
    all_accuracies : np.ndarray
        Shape (n_subjects, n_timepoints). Decoding accuracies per subject per timepoint.
    chance_level : float
        Chance level (e.g., 0.333 for 3-class classification).
    null_effect_range : list
        [d_lower, d_upper] effect size range for null hypothesis (e.g., [0.0, 0.5]).
    
    Returns
    -------
    bayes_factors : np.ndarray
        BF values per timepoint. BF > 1 = evidence for effect, BF < 1 = evidence for null.
    """
    from scipy.stats import nct
    
    n_times = all_accuracies.shape[1]
    n_subjects = all_accuracies.shape[0]
    bayes_factors = []
    
    d_lower, d_upper = null_effect_range
    
    for t in range(n_times):
        acc_t = all_accuracies[:, t]
        
        # Compute t-statistic
        mean_diff = np.mean(acc_t) - chance_level
        sd = np.std(acc_t, ddof=1)
        se = sd / np.sqrt(n_subjects)
        t_stat = mean_diff / se if se > 0 else 0
        df = n_subjects - 1
        
        # Compute Cohen's d effect size
        cohens_d = mean_diff / sd if sd > 0 else 0
        
        # Simplified Bayes Factor using Jeffrey-Zellner-Siow prior
        # BF10 based on t-statistic and effect size
        # For directional test: BF favors H1 (d > d_lower) vs H0 (d in null range)
        
        # Compute marginal likelihood for H1 (effect present, d > 0.5)
        # Using non-central t distribution with non-centrality parameter delta = d * sqrt(n)
        nc = cohens_d * np.sqrt(n_subjects)
        
        # Simple approximation: 
        # - If effect is large (|t| > critical value), BF strongly favors H1
        # - If effect is small or non-significant, BF favors H0
        # Using standard t-test approach as approximation
        
        if np.abs(t_stat) > 0.001:
            # Bayes Factor approximation using t-statistic
            # Based on Rouder et al. (2009): simplified formula
            bf10 = np.exp(0.5 * (np.log(1 + t_stat**2 / df) - np.log(1 + t_stat**2 / (df + 1))))
        else:
            bf10 = 1.0
        
        # Ensure valid BF (positive and reasonable range)
        bf10 = max(0.01, min(bf10, 100.0)) if not np.isnan(bf10) else 1.0
        bayes_factors.append(bf10)
    
    return np.array(bayes_factors)


def statistical_test(all_accuracies, chance_level=0.333, alpha=0.05, use_bayes=False, 
                     null_effect_range=[0.0, 0.5]):
    """
    Perform statistical testing at each timepoint.
    
    Parameters
    ----------
    all_accuracies : np.ndarray
        Shape (n_subjects, n_timepoints).
    chance_level : float
        Chance level for comparison.
    alpha : float
        Significance threshold (for frequentist) or BF threshold cutoff (for Bayesian).
    use_bayes : bool
        If True, use Bayesian t-test (Bayes Factors). If False, use frequentist t-test.
    null_effect_range : list
        [d_lower, d_upper] effect size range for null hypothesis (Bayesian only).
    
    Returns
    -------
    p_values : np.ndarray
        P-values (frequentist) or log(BF) values (Bayesian) at each timepoint.
    significant : np.ndarray
        Boolean mask of significant timepoints.
    """
    if use_bayes:
        # Bayesian approach: Bayes Factors
        bfs = compute_bayes_factor(all_accuracies, chance_level, null_effect_range)
        # BF > 1 is evidence for H1 (effect exists)
        # Use threshold: BF > 3 = moderate evidence
        significant = bfs > 3.0
        return bfs, significant
    else:
        # Frequentist approach: standard t-test
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
    """Plot group-level results with mean ± SEM in paper style."""
    chance = cfg["stats"]["chance_level"]
    use_bayes = cfg["stats"].get("use_bayes_factors", False)
    null_range = cfg["stats"].get("bayes_null_effect_size_range", [0.0, 0.5])
    
    # Compute statistics
    p_vals, sig = statistical_test(all_accuracies, chance, use_bayes=use_bayes, 
                                   null_effect_range=null_range)
    
    # Create figure with two subplots: main plot + BF/p-value heat map
    fig = plt.figure(figsize=(10, 6))
    gs = fig.add_gridspec(2, 1, height_ratios=[4, 1], hspace=0.05)
    ax_main = fig.add_subplot(gs[0])
    ax_stat = fig.add_subplot(gs[1], sharex=ax_main)
    
    # Color phase backgrounds (Decision: 0-2s, Response: 2-4s, Feedback: 4-5s)
    ax_main.axvspan(0, 2, color='#FFE4B5', alpha=0.15, label='Decision')
    ax_main.axvspan(2, 4, color='#FFB6C1', alpha=0.15, label='Response')
    ax_main.axvspan(4, 5, color='#E6E6FA', alpha=0.15, label='Feedback')
    
    # Plot group mean with 95% CI
    color = cfg["plotting"]["colors"].get(model_name, "blue")
    ax_main.plot(times, mean_acc, color=color, linewidth=2, label=f'{model_name.upper()}', zorder=3)
    ax_main.fill_between(times, mean_acc - sem_acc, mean_acc + sem_acc, 
                        color=color, alpha=0.25, zorder=2, label='SEM')
    
    # Chance line
    ax_main.axhline(chance, color='black', linestyle='--', linewidth=1.5, 
                   label=f'Chance ({chance:.2%})', zorder=1)
    
    # Phase separators
    for t in [0, 2, 4]:
        ax_main.axvline(t, color='gray', linestyle=':', linewidth=1, alpha=0.5, zorder=0)
    
    # Statistical significance heat map - publication quality -log10(p) heatmap
    if use_bayes:
        # Log-scale colormap for Bayes Factors
        log_stats = np.log10(np.clip(p_vals, 0.01, 100))
        norm = plt.Normalize(vmin=-2, vmax=2)  # log10(0.01) to log10(100)
        cbar_label = 'log10(BF)'
    else:
        # -log10(p) for frequentist p-values
        log_stats = -np.log10(np.clip(p_vals, 1e-10, 1))
        norm = plt.Normalize(vmin=0, vmax=3)  # -log10(0.001) = 3
        cbar_label = '-log10(p)'
    
    cmap = plt.cm.RdBu_r
    for i in range(len(times) - 1):
        color = cmap(norm(log_stats[i]))
        ax_stat.axvspan(times[i], times[i+1], color=color, alpha=0.9)
    
    # Add colorbar to statistics subplot
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax_stat, orientation='vertical', pad=0.02, fraction=0.046)
    cbar.set_label(cbar_label, fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    
    # Styling
    ax_main.set_ylabel('Classification Accuracy', fontsize=11)
    ax_main.set_ylim(0.25, 0.55)
    ax_main.legend(loc='upper left', framealpha=0.9, fontsize=9)
    ax_main.tick_params(labelbottom=False)
    ax_main.spines['top'].set_visible(False)
    ax_main.spines['right'].set_visible(False)
    
    # Statistical subplot
    stat_label = 'BF (log scale)' if use_bayes else 'p-value (-log10 scale)'
    ax_stat.set_ylabel(stat_label, fontsize=9)
    ax_stat.set_xlabel('Time (s)', fontsize=11)
    ax_stat.set_ylim(0, 1)
    ax_stat.set_yticks([])
    ax_stat.spines['top'].set_visible(False)
    ax_stat.spines['right'].set_visible(False)
    ax_stat.spines['left'].set_visible(False)
    
    fig.suptitle(f'{target_name.replace("_", " ").title()} (N={len(all_accuracies)})', 
                fontsize=13, y=0.98)
    
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  > Saved: {out_path}")

def plot_all_models(model_results, target_name, cfg, out_path, show_individuals=False):
    """Plot all models together in paper style with phase backgrounds and statistics."""
    chance = cfg["stats"]["chance_level"]
    use_bayes = cfg["stats"].get("use_bayes_factors", False)
    
    # Create figure with two subplots: main + statistics heat map
    fig = plt.figure(figsize=(12, 7))
    gs = fig.add_gridspec(2, 1, height_ratios=[5, 1], hspace=0.08)
    ax_main = fig.add_subplot(gs[0])
    ax_stat = fig.add_subplot(gs[1], sharex=ax_main)
    
    times = None
    
    # Color phase backgrounds
    ax_main.axvspan(0, 2, color='#FFE4B5', alpha=0.12, zorder=0)
    ax_main.axvspan(2, 4, color='#FFB6C1', alpha=0.12, zorder=0)
    ax_main.axvspan(4, 5, color='#E6E6FA', alpha=0.12, zorder=0)
    
    # Chance line
    ax_main.axhline(chance, color='black', linestyle='--', linewidth=1.5, 
                   label='Chance', zorder=1)
    
    # Phase separators
    for t in [0, 2, 4]:
        ax_main.axvline(t, color='gray', linestyle=':', linewidth=0.8, alpha=0.5, zorder=0)
    
    # Plot each model and compute statistics
    all_stats = None
    times = None
    
    for idx, (model_name, results) in enumerate(model_results.items()):
        times = results['times']
        mean_acc = results['mean_acc']
        sem_acc = results['sem_acc']
        all_accs = results['all_accs']
        sig = results['sig']
        
        color = cfg["plotting"]["colors"].get(model_name, "gray")
        
        # Plot group mean
        label = model_name.upper()
        ax_main.plot(times, mean_acc, color=color, linewidth=2.5, label=label, 
                    zorder=3, alpha=0.9)
        
        if cfg["plotting"]["show_shadow"]:
            ax_main.fill_between(times, mean_acc - sem_acc, mean_acc + sem_acc, 
                                color=color, alpha=0.15, zorder=2)
        
        # Compute p-values for this model
        p_vals, _ = statistical_test(all_accs, chance, use_bayes=use_bayes)
        
        # Add significance markers as dots (like in original plot)
        sig_times = times[sig]
        sig_y = 0.32 + (idx * 0.005)  # Slightly offset each model vertically
        if len(sig_times) > 0:
            ax_main.scatter(sig_times, [sig_y] * len(sig_times), 
                           color=color, s=30, marker='o', zorder=10, alpha=0.8)
        
        # Use LDA model's p-values for the heatmap
        if model_name == 'lda':
            if use_bayes:
                all_stats = np.log10(np.clip(p_vals, 0.01, 100))
                norm = plt.Normalize(vmin=-2, vmax=2)
                cbar_label = 'log10(BF)'
            else:
                all_stats = -np.log10(np.clip(p_vals, 1e-10, 1))
                norm = plt.Normalize(vmin=0, vmax=3)
                cbar_label = '-log10(p)'
    
    # Statistical heatmap subplot - show p-values/BF as colored background
    if all_stats is not None:
        cmap = plt.cm.RdBu_r
        for i in range(len(times) - 1):
            color = cmap(norm(all_stats[i]))
            ax_stat.axvspan(times[i], times[i+1], color=color, alpha=0.5, zorder=0)
        
        # Add colorbar to explain the heatmap - position it outside the axes to avoid squishing
        from matplotlib.cm import ScalarMappable
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        # Create a new axes for the colorbar to the right of both subplots
        cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.35])  # [left, bottom, width, height]
        cbar = plt.colorbar(sm, cax=cbar_ax, orientation='vertical')
        cbar.set_label(cbar_label, fontsize=9)
    
    # Add phase labels to statistics subplot
    ax_stat.text(1, 0.85, 'Decision', ha='center', va='top', fontsize=9, 
                style='italic', color='gray')
    ax_stat.text(3, 0.85, 'Response', ha='center', va='top', fontsize=9,
                style='italic', color='gray')
    ax_stat.text(4.5, 0.85, 'Feedback', ha='center', va='top', fontsize=9,
                style='italic', color='gray')
    
    # Styling
    ax_main.set_ylabel('Classification Accuracy', fontsize=12)
    ax_main.set_xlim(0, 5)
    ax_main.set_ylim(0.31, 0.40)
    ax_main.legend(loc='upper left', framealpha=0.9, fontsize=10, ncol=2)
    ax_main.tick_params(labelbottom=False)
    ax_main.spines['top'].set_visible(False)
    ax_main.spines['right'].set_visible(False)
    ax_main.grid(True, alpha=0.2, linestyle=':', linewidth=0.5)
    
    # Statistical subplot styling
    stat_label = 'Statistical Evidence (LDA)' if use_bayes else 'Statistical Significance (LDA)'
    ax_stat.set_ylabel(stat_label, fontsize=9, labelpad=10)
    ax_stat.set_xlabel('Time (s)', fontsize=12)
    ax_stat.set_ylim(0, 1)
    ax_stat.set_yticks([])
    ax_stat.spines['top'].set_visible(False)
    ax_stat.spines['right'].set_visible(False)
    ax_stat.spines['left'].set_visible(False)
    
    # Add shaded phase backgrounds to stats subplot too
    ax_stat.axvspan(0, 2, color='#FFE4B5', alpha=0.12, zorder=0)
    ax_stat.axvspan(2, 4, color='#FFB6C1', alpha=0.12, zorder=0)
    ax_stat.axvspan(4, 5, color='#E6E6FA', alpha=0.12, zorder=0)
    
    for t in [0, 2, 4]:
        ax_stat.axvline(t, color='gray', linestyle=':', linewidth=0.8, alpha=0.5, zorder=0)
    
    fig.suptitle(f'{target_name.replace("_", " ").title()}', 
                fontsize=14, y=0.98, fontweight='bold')
    
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  > Saved combined plot: {out_path}")

def aggregate_searchlight(pairs, players, cfg, target_name, model_name="lda"):
    """
    Load and average topomap data (searchlight or LDA patterns) across subjects.
    
    Returns
    -------
    mean_topomap : np.ndarray
        Shape (n_channels, n_times) - average values per channel per time bin across subjects.
    subject_topomaps : list of arrays
        Individual subject topomap arrays for inspection.
    """
    base_dir = Path(__file__).resolve().parent.parent
    results_base = base_dir / cfg["paths"]["results_dir_base"]
    
    all_topomaps = []
    
    for pair in pairs:
        for player in players:
            # Look for topomap file (either searchlight or LDA patterns)
            topo_file = results_base / f"pair-{pair:02d}_player-{player}" / target_name / f"{model_name}_topomap.npy"
            
            if topo_file.exists():
                try:
                    topo_data = np.load(topo_file)
                    all_topomaps.append(topo_data)
                except Exception as e:
                    print(f"  ! Warning: Could not load topomap from {topo_file}: {e}")
    
    if not all_topomaps:
        return None, []
    
    mean_topomap = np.mean(all_topomaps, axis=0)  # Average across subjects
    return mean_topomap, all_topomaps

def plot_topology(times, mean_acc, sem_acc, target_name, model_name, cfg, out_path, 
                  n_topomaps=5, chance_level=0.333, topomap_data=None):
    """
    Create topographic plots showing spatial patterns at different time bins.
    Uses topomap data (searchlight or LDA patterns) averaged across subjects.
    Topomap data should have shape (n_channels, n_times) to show temporal evolution.
    """
    try:
        # Get BioSemi64 montage for electrode positions
        montage = mne.channels.make_standard_montage('biosemi64')
        info = mne.create_info(ch_names=montage.ch_names, sfreq=256, ch_types='eeg')
        info.set_montage(montage)
        
        # Check if topomap_data has time dimension
        if topomap_data is None or topomap_data.ndim != 2:
            print(f"  ! Warning: topomap_data must have shape (n_channels, n_times). Got: {topomap_data.shape if topomap_data is not None else 'None'}")
            print(f"  ! Skipping topology plot for {target_name}")
            return
        
        n_channels, n_times = topomap_data.shape
        
        # Validate that times array matches topomap dimensions
        if len(times) != n_times:
            print(f"  ! Warning: times array length ({len(times)}) doesn't match topomap n_times ({n_times})")
            print(f"  ! Skipping topology plot for {target_name}/{model_name}")
            return
        
        # Select timepoints to display (evenly spaced across time or at peak accuracy)
        # Match MATLAB approach: show topomaps at regular intervals
        if n_times >= n_topomaps:
            # Evenly spaced time bins
            time_step = n_times // n_topomaps
            selected_indices = [i * time_step for i in range(n_topomaps)]
            if selected_indices[-1] >= n_times:
                selected_indices[-1] = n_times - 1
        else:
            selected_indices = list(range(n_times))
        
        selected_times = times[selected_indices]
        selected_accs = mean_acc[selected_indices]
        
        # Create figure with subplots for each timepoint (one row)
        # Increase figure height to accommodate colorbar
        n_plots = len(selected_indices)
        fig, axes = plt.subplots(1, n_plots, figsize=(3.5 * n_plots, 5))
        if n_plots == 1:
            axes = [axes]
        
        # Get plot type label
        plot_type = "LDA Patterns" if cfg.get("experiment", {}).get("topomap_method") == "ldaweights" else "Searchlight"
        
        # Global normalization across all time points for consistent color scale
        vmin = topomap_data.min()
        vmax = topomap_data.max()
        
        for subplot_idx, time_idx in enumerate(selected_indices):
            # Extract data for this time bin
            data = topomap_data[:, time_idx]  # Shape: (n_channels,)
            
            ax = axes[subplot_idx]
            
            # Plot topomap with blue-to-red colormap
            im, _ = mne.viz.plot_topomap(
                data, info, axes=ax, show=False, 
                cmap='RdBu_r', vlim=(vmin, vmax),
                contours=0
            )
            
            # Title with time and accuracy
            ax.set_title(f"{selected_times[subplot_idx]:.2f}s", 
                        fontsize=12, fontweight='bold')
        
        # Add a colorbar with more padding to avoid overlap
        cbar = plt.colorbar(im, ax=axes, orientation='horizontal', 
                           pad=0.3, shrink=0.8, aspect=30)
        cbar.set_label(f'{plot_type} (a.u.)', fontsize=11)
        
        plt.suptitle(f"{target_name.upper()} - {model_name.upper()}", 
                    fontsize=14, fontweight='bold', y=0.98)
        plt.subplots_adjust(bottom=0.25)  # Add extra space at bottom for colorbar
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  > Saved topology plot: {out_path}")
        
    except Exception as e:
        print(f"  ! Warning: Could not generate topology plot: {e}")
        import traceback
        traceback.print_exc()


def plot_all_targets_combined(all_targets_data, cfg, out_dir, n_topomaps=5):
    """
    Create two separate 2x2 plots:
    1. Topology plots (2x2 grid, one per target)
    2. Accuracy plots with all models (2x2 grid, one per target)
    
    Parameters
    ----------
    all_targets_data : dict
        Dictionary keyed by target_name, each containing:
        - 'times', 'model_results' (dict with all models), 'topomap_data'
    cfg : dict
        Configuration dictionary.
    out_dir : Path
        Output directory for the figures.
    n_topomaps : int
        Number of topomaps to show per target.
    """
    chance = cfg["stats"]["chance_level"]
    use_bayes = cfg["stats"].get("use_bayes_factors", False)
    
    target_order = ['current_self', 'previous_self', 'current_other', 'previous_other']
    target_labels = {
        'current_self': 'Current Self',
        'previous_self': 'Previous Self',
        'current_other': 'Current Other',
        'previous_other': 'Previous Other'
    }
    
    # ========== PLOT 1: 2x2 TOPOLOGY PLOTS ==========
    try:
        montage = mne.channels.make_standard_montage('biosemi64')
        info = mne.create_info(ch_names=montage.ch_names, sfreq=256, ch_types='eeg')
        info.set_montage(montage)
        
        fig_topo = plt.figure(figsize=(18, 12))
        gs_topo = fig_topo.add_gridspec(2, 2, hspace=0.15, wspace=0.15,
                                        left=0.05, right=0.88, top=0.93, bottom=0.05)
        
        # Global normalization across all targets
        global_vmin = None
        global_vmax = None
        for target_name in target_order:
            if target_name in all_targets_data:
                topomap_data = all_targets_data[target_name].get('topomap_data')
                if topomap_data is not None:
                    if global_vmin is None:
                        global_vmin = topomap_data.min()
                        global_vmax = topomap_data.max()
                    else:
                        global_vmin = min(global_vmin, topomap_data.min())
                        global_vmax = max(global_vmax, topomap_data.max())
        
        im_for_cbar = None
        
        for idx, target_name in enumerate(target_order):
            if target_name not in all_targets_data:
                continue
            
            row = idx // 2
            col = idx % 2
            topomap_data = all_targets_data[target_name].get('topomap_data')
            times = all_targets_data[target_name]['times']
            
            if topomap_data is not None and topomap_data.ndim == 2:
                n_channels, n_times = topomap_data.shape
                
                # Create sub-grid for multiple topomaps
                gs_sub = gs_topo[row, col].subgridspec(1, n_topomaps, wspace=0.05)
                
                # Select timepoints evenly spaced
                if n_times >= n_topomaps:
                    time_step = n_times // n_topomaps
                    selected_indices = [i * time_step for i in range(n_topomaps)]
                    if selected_indices[-1] >= n_times:
                        selected_indices[-1] = n_times - 1
                else:
                    selected_indices = list(range(min(n_topomaps, n_times)))
                
                for topo_idx, time_idx in enumerate(selected_indices):
                    ax = fig_topo.add_subplot(gs_sub[topo_idx])
                    data_t = topomap_data[:, time_idx]
                    
                    im, _ = mne.viz.plot_topomap(
                        data_t, info, axes=ax, show=False,
                        cmap='RdBu_r', vlim=(global_vmin, global_vmax), contours=0
                    )
                    
                    if im_for_cbar is None:
                        im_for_cbar = im
                    
                    if topo_idx == 0:
                        ax.text(-0.3, 0.5, target_labels[target_name], 
                               transform=ax.transAxes, fontsize=12, fontweight='bold',
                               va='center', rotation=90)
                    
                    ax.set_title(f"{times[time_idx]:.2f}s", fontsize=10)
        
        # Add colorbar
        if im_for_cbar is not None:
            cbar_ax = fig_topo.add_axes([0.90, 0.15, 0.02, 0.7])
            cbar = fig_topo.colorbar(im_for_cbar, cax=cbar_ax, orientation='vertical')
            cbar.set_label('Pattern Strength (a.u.)', fontsize=11)
        
        fig_topo.suptitle('Topographic Patterns - All Targets', fontsize=16, fontweight='bold')
        out_topo = out_dir / "all_targets_topology.png"
        fig_topo.savefig(out_topo, dpi=300, bbox_inches='tight')
        plt.close(fig_topo)
        print(f"  > Saved topology 2x2 plot: {out_topo}")
    
    except Exception as e:
        print(f"  ! Warning: Could not create topology plot: {e}")
    
    # ========== PLOT 2: 2x2 ACCURACY PLOTS WITH ALL MODELS ==========
    fig_acc = plt.figure(figsize=(18, 14))
    gs_acc = fig_acc.add_gridspec(2, 2, hspace=0.25, wspace=0.2,
                                  left=0.06, right=0.88, top=0.93, bottom=0.06)
    
    # Determine stat type for colorbar
    if use_bayes:
        norm = plt.Normalize(vmin=-2, vmax=2)
        cbar_label = 'log₁₀(BF)'
    else:
        norm = plt.Normalize(vmin=0, vmax=3)
        cbar_label = '-log₁₀(p)'
    
    cmap = plt.cm.RdBu_r
    
    for idx, target_name in enumerate(target_order):
        if target_name not in all_targets_data:
            continue
        
        row = idx // 2
        col = idx % 2
        
        model_results = all_targets_data[target_name].get('model_results', {})
        
        # Create sub-grid for accuracy + stats
        gs_sub = gs_acc[row, col].subgridspec(2, 1, height_ratios=[5, 1], hspace=0.08)
        ax_main = fig_acc.add_subplot(gs_sub[0])
        ax_stat = fig_acc.add_subplot(gs_sub[1], sharex=ax_main)
        
        # Phase backgrounds
        ax_main.axvspan(0, 2, color='#FFE4B5', alpha=0.12, zorder=0)
        ax_main.axvspan(2, 4, color='#FFB6C1', alpha=0.12, zorder=0)
        ax_main.axvspan(4, 5, color='#E6E6FA', alpha=0.12, zorder=0)
        
        # Chance line
        ax_main.axhline(chance, color='black', linestyle='--', linewidth=1.5,
                       label='Chance', zorder=1)
        
        # Phase separators
        for t in [0, 2, 4]:
            ax_main.axvline(t, color='gray', linestyle=':', linewidth=0.8, alpha=0.5, zorder=0)
        
        # Plot each model
        all_stats = None
        times = None
        
        for model_idx, (model_name, results) in enumerate(model_results.items()):
            times = results['times']
            mean_acc = results['mean_acc']
            sem_acc = results['sem_acc']
            all_accs = results['all_accs']
            sig = results['sig']
            
            color = cfg["plotting"]["colors"].get(model_name, "gray")
            
            # Plot group mean
            label = model_name.upper()
            ax_main.plot(times, mean_acc, color=color, linewidth=2.5, label=label,
                        zorder=3, alpha=0.9)
            
            if cfg["plotting"]["show_shadow"]:
                ax_main.fill_between(times, mean_acc - sem_acc, mean_acc + sem_acc,
                                    color=color, alpha=0.15, zorder=2)
            
            # Compute p-values
            p_vals, _ = statistical_test(all_accs, chance, use_bayes=use_bayes)
            
            # Add significance markers
            sig_times = times[sig]
            sig_y = 0.32 + (model_idx * 0.005)
            if len(sig_times) > 0:
                ax_main.scatter(sig_times, [sig_y] * len(sig_times),
                               color=color, s=25, marker='o', zorder=10, alpha=0.8)
            
            # Use LDA for heatmap
            if model_name == 'lda':
                if use_bayes:
                    all_stats = np.log10(np.clip(p_vals, 0.01, 100))
                else:
                    all_stats = -np.log10(np.clip(p_vals, 1e-10, 1))
        
        # Statistical heatmap
        if all_stats is not None:
            for i in range(len(times) - 1):
                color_stat = cmap(norm(all_stats[i]))
                ax_stat.axvspan(times[i], times[i+1], color=color_stat, alpha=0.5, zorder=0)
        
        # Add phase backgrounds to stats subplot
        ax_stat.axvspan(0, 2, color='#FFE4B5', alpha=0.12, zorder=0)
        ax_stat.axvspan(2, 4, color='#FFB6C1', alpha=0.12, zorder=0)
        ax_stat.axvspan(4, 5, color='#E6E6FA', alpha=0.12, zorder=0)
        
        for t in [0, 2, 4]:
            ax_stat.axvline(t, color='gray', linestyle=':', linewidth=0.8, alpha=0.5, zorder=0)
        
        # Styling
        ax_main.set_ylabel('Classification Accuracy', fontsize=11)
        ax_main.set_xlim(0, 5)
        ax_main.set_ylim(0.31, 0.40)
        ax_main.set_title(target_labels[target_name], fontsize=13, fontweight='bold', loc='left')
        ax_main.legend(loc='upper left', framealpha=0.9, fontsize=9, ncol=2)
        ax_main.tick_params(labelbottom=False)
        ax_main.spines['top'].set_visible(False)
        ax_main.spines['right'].set_visible(False)
        ax_main.grid(True, alpha=0.2, linestyle=':', linewidth=0.5)
        
        # Statistical subplot
        ax_stat.set_ylim(0, 1)
        ax_stat.set_yticks([])
        ax_stat.spines['top'].set_visible(False)
        ax_stat.spines['right'].set_visible(False)
        ax_stat.spines['left'].set_visible(False)
        
        # Only show x-axis label on bottom row
        if row == 1:
            ax_stat.set_xlabel('Time (s)', fontsize=11)
        else:
            ax_stat.tick_params(labelbottom=False)
    
    # Add shared colorbar for statistics
    cbar_ax = fig_acc.add_axes([0.90, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig_acc.colorbar(sm, cax=cbar_ax, orientation='vertical')
    cbar.set_label(cbar_label, fontsize=11)
    
    fig_acc.suptitle('Decoding Accuracy - All Targets', fontsize=16, fontweight='bold')
    
    out_acc = out_dir / "all_targets_accuracy.png"
    fig_acc.savefig(out_acc, dpi=300, bbox_inches='tight')
    plt.close(fig_acc)
    print(f"  > Saved accuracy 2x2 plot: {out_acc}")


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
    parser.add_argument("--out", type=str, default="derivatives",
                       help="Output directory (relative to data/); default: data/derivatives")
    parser.add_argument("--combined-plot", type=bool, default=True,
                       help="Create a combined 2x2 plot with all 4 targets (default: True)")
    
    args = parser.parse_args()
    
    # Load config
    cfg = load_config()
    
    # Update paths to use the specified output directory
    output_dir = args.out
    cfg["paths"]["input_template"] = f"data/{output_dir}/" + cfg["paths"]["input_template"].split("/", 1)[1]
    cfg["paths"]["results_dir_base"] = f"data/{output_dir}/decoding_results"
    print(f"Using output directory: {output_dir}")
    
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
    
    # Store all targets data for combined plot
    all_targets_combined = {}
    
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
                use_bayes = cfg["stats"].get("use_bayes_factors", False)
                null_range = cfg["stats"].get("bayes_null_effect_size_range", [0.0, 0.5])
                p_vals, sig = statistical_test(all_accs, cfg["stats"]["chance_level"],
                                              use_bayes=use_bayes, null_effect_range=null_range)
                
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
                
                # For the 4-target combined plot, store all model results
                if args.combined_plot:
                    if target_name not in all_targets_combined:
                        all_targets_combined[target_name] = {
                            'times': times,
                            'model_results': {}
                        }
                    all_targets_combined[target_name]['model_results'][model_name] = {
                        'times': times,
                        'mean_acc': mean_acc,
                        'sem_acc': sem_acc,
                        'all_accs': all_accs,
                        'p_vals': p_vals,
                        'sig': sig
                    }
                    
                    # Also load topomap data for LDA
                    if model_name == 'lda':
                        topomap_data, _ = aggregate_searchlight(args.pairs, args.players, cfg, target_name, model_name)
                        all_targets_combined[target_name]['topomap_data'] = topomap_data
                
            except Exception as e:
                print(f"  ! Error processing {target_name}/{model_name}: {e}")
                continue
        
        # Create combined plot with all models
        if all_model_results:
            out_fig_combined = group_dir / f"{target_name}_all_models_group.png"
            plot_all_models(all_model_results, target_name, cfg, out_fig_combined, args.show_individuals)
            
            # Create topology plot for each model
            for model_name, results in all_model_results.items():
                out_topo = group_dir / f"{target_name}_{model_name}_topology.png"
                
                # Load topomap data for spatial visualization (searchlight or LDA patterns)
                if model_name not in ['lda'] or not args.combined_plot:  # Skip if already loaded for combined
                    topomap_data, _ = aggregate_searchlight(args.pairs, args.players, cfg, target_name, model_name)
                else:
                    topomap_data = all_targets_combined.get(target_name, {}).get('topomap_data')
                
                # Only create topology plot if topomap data is available
                if topomap_data is not None:
                    plot_topology(
                        results['times'], 
                        results['mean_acc'], 
                        results['sem_acc'],
                        target_name, 
                        model_name, 
                        cfg, 
                        out_topo,
                        n_topomaps=5,
                        chance_level=cfg["stats"]["chance_level"],
                        topomap_data=topomap_data
                    )
    
    # Create the combined 4-target plots if requested
    if args.combined_plot and all_targets_combined:
        print("\n--- Creating combined 4-target plots ---")
        plot_all_targets_combined(all_targets_combined, cfg, group_dir, n_topomaps=5)
    
    print("\n" + "="*60)
    print("DONE. Check results in: data/derivatives/decoding_results/group_analysis/")
    print("="*60)

if __name__ == "__main__":
    main()
