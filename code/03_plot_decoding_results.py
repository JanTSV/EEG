import yaml
import pandas as pd
import numpy as np
import mne
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.gridspec import GridSpecFromSubplotSpec
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
    (7.5, 15.5, "Response", "#e3f2fd"),  # Light Blue
    (15.5, 19.5, "Feedback", "#f3e5f5"),  # Light Purple
]

COLORS = {
    "main": "#1f77b4",  # Standard Blue
    "winner": "#009688",  # Teal
    "loser": "#CD5C5C",  # IndianRed
    "chance": "#444444",
    "heatmap": [
        "#ffffff",
        "#fff9c4",
        "#ffcc80",
        "#ef5350",
    ],  # White -> Yellow -> Orange -> Red
}

PLOT_TARGETS = ["Self_Response", "Other_Response", "Outcome", "Prev_Self_Response"]


def load_config():
    with open("code/config_decoding.yaml", "r") as f:
        return yaml.safe_load(f)


def aggregate_data(results_dir):
    all_files = list(results_dir.glob("*_decoding_results.csv"))
    if not all_files:
        raise FileNotFoundError("No CSVs found.")
    df_list = [pd.read_csv(f) for f in all_files]
    df = pd.concat(df_list, ignore_index=True)

    # Backward compatibility
    if "model" not in df.columns:
        df["model"] = "lda"

    df["accuracy"] *= 100
    # Aggregate to Subject Level first
    return (
        df.groupby(["subject", "target", "time_bin", "player_status", "model"])[
            "accuracy"
        ]
        .mean()
        .reset_index()
    )


def aggregate_haufe_data(results_dir):
    all_files = list(results_dir.glob("*_haufe_patterns.csv"))
    if not all_files:
        return None
    df_list = [pd.read_csv(f) for f in all_files]
    df = pd.concat(df_list, ignore_index=True)
    needed = {
        "subject",
        "target",
        "model",
        "time_bin",
        "pattern_row",
        "pattern_label",
        "channel",
        "channel_idx",
        "pattern_value",
    }
    if not needed.issubset(df.columns):
        return None

    # Subject-level average per bin/channel while preserving pattern identity
    return (
        df.groupby(
            [
                "subject",
                "target",
                "model",
                "time_bin",
                "pattern_row",
                "pattern_label",
                "channel",
                "channel_idx",
            ],
            as_index=False,
        )["pattern_value"]
        .mean()
    )


def _select_pattern_component(df_haufe_t):
    """Select one stable Haufe component for plotting (avoid mixing multiclass rows)."""
    if df_haufe_t is None or df_haufe_t.empty:
        return df_haufe_t, None

    if "pattern_row" not in df_haufe_t.columns:
        return df_haufe_t, "component_0"

    comp_strength = (
        df_haufe_t.groupby(["pattern_row", "pattern_label"], as_index=False)["pattern_value"]
        .apply(lambda s: np.mean(np.abs(s)))
        .rename(columns={"pattern_value": "abs_mean"})
        .sort_values("abs_mean", ascending=False)
    )

    best = comp_strength.iloc[0]
    df_sel = df_haufe_t[df_haufe_t["pattern_row"] == best["pattern_row"]]
    return df_sel, str(best["pattern_label"])


def get_p_values(df, target, bins, mode="one_sample"):
    p_vals = []
    for t in bins:
        d = df[(df["target"] == target) & (df["time_bin"] == t)]
        if mode == "one_sample":
            _, p = stats.ttest_1samp(d["accuracy"], CHANCE_LEVEL, alternative="greater")
        else:  # two_sample
            w = d[d["player_status"] == "Winner"]["accuracy"]
            l = d[d["player_status"] == "Loser"]["accuracy"]
            _, p = stats.ttest_ind(w, l) if (len(w) > 1 and len(l) > 1) else (0, 1.0)
        p_vals.append(p)
    return p_vals


def plot_heatmap(ax, p_vals, n_bins):
    """Plots the significance strip."""
    map_vals = np.zeros((1, n_bins))
    for i, p in enumerate(p_vals):
        if p < 0.001:
            v = 3
        elif p < 0.01:
            v = 2
        elif p < 0.05:
            v = 1
        else:
            v = 0
        map_vals[0, i] = v

    cmap = mcolors.ListedColormap(COLORS["heatmap"])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    ax.imshow(map_vals, aspect="auto", cmap=cmap, norm=norm, extent=[-0.5, 19.5, 0, 1])
    ax.set_yticks([])
    ax.set_xlim(-0.5, 19.5)

    # Add vertical phase dividers
    for _, end, _, _ in PHASES:
        ax.axvline(end, color="white", linewidth=2)


def style_phase_background(ax):
    """Applies colored phase backgrounds and labels."""
    for start, end, label, color in PHASES:
        ax.axvspan(start, end, color=color, alpha=0.6, lw=0, zorder=0)
        # Vertical divider line
        ax.axvline(end, color="white", linewidth=2, zorder=1)
        # Label (only if it's a main plot)
        if ax.get_ylabel():
            ax.text(
                (start + end) / 2,
                ax.get_ylim()[1] * 0.98,
                label,
                ha="center",
                va="top",
                fontsize=9,
                fontweight="bold",
                color="#666",
            )


def _plot_haufe_topomap_row(subspec, df_haufe_t, n_bins, bin_width_s, component_label=None):
    """Plot one topomap per 4 bins (1 second per topomap) with shared time axis."""
    if df_haufe_t is None or df_haufe_t.empty:
        ax_empty = plt.subplot(subspec)
        ax_empty.axis("off")
        ax_empty.text(0.5, 0.5, "No Haufe patterns found", ha="center", va="center", fontsize=9, color="#666")
        return

    group_size = 4
    bin_groups = [(start, min(start + group_size - 1, n_bins - 1)) for start in range(0, n_bins, group_size)]
    n_groups = len(bin_groups)

    ch_df = (
        df_haufe_t[["channel", "channel_idx"]]
        .drop_duplicates()
        .sort_values("channel_idx")
        .reset_index(drop=True)
    )
    ch_names = ch_df["channel"].tolist()

    info = mne.create_info(ch_names=ch_names, sfreq=256, ch_types="eeg")
    info.set_montage(mne.channels.make_standard_montage("standard_1020"), on_missing="warn")

    # Build one map per 1-second group (4 bins)
    maps = []
    for start_bin, end_bin in bin_groups:
        d = df_haufe_t[(df_haufe_t["time_bin"] >= start_bin) & (df_haufe_t["time_bin"] <= end_bin)]
        vec = (
            d.groupby(["channel", "channel_idx"], as_index=False)["pattern_value"]
            .mean()
            .sort_values("channel_idx")["pattern_value"]
            .values
        )
        # Ensure correct length/order
        if len(vec) != len(ch_names):
            merged = ch_df.merge(
                d.groupby(["channel", "channel_idx"], as_index=False)["pattern_value"].mean(),
                on=["channel", "channel_idx"],
                how="left",
            )
            vec = merged["pattern_value"].fillna(0.0).values
        maps.append(vec)

    vmax = max(np.max(np.abs(v)) for v in maps) if maps else 1.0
    vmax = max(vmax, 1e-12)

    inner = GridSpecFromSubplotSpec(2, n_groups, subplot_spec=subspec, height_ratios=[8, 1.2], hspace=0.15, wspace=0.25)

    for i, ((start_bin, end_bin), vec) in enumerate(zip(bin_groups, maps)):
        ax_map = plt.subplot(inner[0, i])
        mne.viz.plot_topomap(vec, info, axes=ax_map, show=False, cmap="RdBu_r", vlim=(-vmax, vmax), contours=0)
        t_start = start_bin * bin_width_s
        t_end = (end_bin + 1) * bin_width_s
        ax_map.set_title(f"{t_start:.1f}-{t_end:.1f}s", fontsize=8)

    ax_time = plt.subplot(inner[1, :])
    ax_time.set_xlim(0, float(n_groups))
    ax_time.set_ylim(0, 1)
    ax_time.set_yticks([])
    ax_time.set_xticks(np.arange(0, n_groups + 1, 1))
    ax_time.set_xlabel("Time (s)")
    for spine in ["left", "right", "top"]:
        ax_time.spines[spine].set_visible(False)
    for x in np.arange(0, n_groups + 1, 1):
        ax_time.axvline(x, color="#cccccc", linewidth=0.6, alpha=0.6)
    if component_label:
        ax_time.set_title(f"Haufe component: {component_label}", fontsize=8, pad=2)


def plot_fig2(df, out_dir, model_name, cfg, df_haufe=None):
    print(f"Plotting Figure 2 (Polished) for {model_name}...")
    
    y_min = cfg['plotting'].get('y_min', 25)
    y_max = cfg['plotting'].get('y_max', 45)
    bin_width_s = cfg['params'].get('bin_width', 0.25)
    
    targets = [t for t in PLOT_TARGETS if t in df["target"].unique()]
    fig = plt.figure(figsize=(9, 3.5 * len(targets)))
    gs = gridspec.GridSpec(
        len(targets) * 3, 1, height_ratios=[3, 0.25, 2.1] * len(targets), hspace=0.4
    )

    bins = np.sort(df["time_bin"].unique())

    for i, tgt in enumerate(targets):
        base = 3 * i
        df_t = df[df["target"] == tgt]

        # Main
        ax = plt.subplot(gs[base])
        ax.set_ylim(y_min, y_max)  # Set limits BEFORE styling to place text correctly
        ax.set_ylabel("Accuracy (%)")
        style_phase_background(ax)

        ax.axhline(
            CHANCE_LEVEL, color=COLORS["chance"], linestyle="--", lw=1.5, zorder=2
        )
        sns.lineplot(
            data=df_t,
            x="time_bin",
            y="accuracy",
            ax=ax,
            color=COLORS["main"],
            linewidth=2.5,
            marker="o",
            markersize=6,
            errorbar=("ci", 95),
            zorder=3,
        )

        ax.set_title(
            f"{chr(65 + i)}) {tgt.replace('_', ' ')}",
            loc="left",
            fontweight="bold",
            pad=10,
        )
        ax.set_xlim(-0.5, 19.5)
        ax.set_xticklabels([])

        # Strip
        ax_s = plt.subplot(gs[base + 1])
        plot_heatmap(ax_s, get_p_values(df_t, tgt, bins), len(bins))
        if i == len(targets) - 1:
            ax_s.set_xlabel("Time Bins")
        else:
            ax_s.set_xticks([])

        # Haufe topoplots (1 map per 4 bins = 1 second)
        ax_topo_spec = gs[base + 2]
        df_haufe_t = None
        component_label = None
        if df_haufe is not None:
            df_haufe_t = df_haufe[df_haufe["target"] == tgt]
            df_haufe_t, component_label = _select_pattern_component(df_haufe_t)
        _plot_haufe_topomap_row(
            ax_topo_spec,
            df_haufe_t,
            n_bins=len(bins),
            bin_width_s=bin_width_s,
            component_label=component_label,
        )

    plt.tight_layout()
    plt.savefig(out_dir / f"Figure2_GrandAverage_Polished_{model_name}.png", dpi=300)
    plt.close()


def plot_fig3(df, out_dir, model_name, cfg):
    print(f"Plotting Figure 3 (Polished Split) for {model_name}...")
    
    y_min = cfg['plotting'].get('y_min', 25)
    y_max = cfg['plotting'].get('y_max', 45)
    
    if len(df["player_status"].unique()) < 2:
        return
    targets = [t for t in PLOT_TARGETS if t in df["target"].unique()]

    fig = plt.figure(figsize=(9, 5 * len(targets)))
    # Ratios: Plot(3), StripW(0.2), StripL(0.2), StripDiff(0.2)
    gs = gridspec.GridSpec(
        len(targets) * 4, 1, height_ratios=[3, 0.2, 0.2, 0.2] * len(targets), hspace=0.4
    )
    bins = np.sort(df["time_bin"].unique())

    for i, tgt in enumerate(targets):
        df_t = df[df["target"] == tgt]
        base = 4 * i

        # Main
        ax = plt.subplot(gs[base])
        ax.set_ylim(y_min, y_max)
        ax.set_ylabel("Accuracy (%)")
        style_phase_background(ax)
        ax.axhline(CHANCE_LEVEL, color=COLORS["chance"], linestyle="--", zorder=2)

        sns.lineplot(
            data=df_t[df_t["player_status"] == "Winner"],
            x="time_bin",
            y="accuracy",
            ax=ax,
            color=COLORS["winner"],
            lw=2,
            marker="o",
            errorbar=("ci", 95),
            label="Winner",
            zorder=3,
        )
        sns.lineplot(
            data=df_t[df_t["player_status"] == "Loser"],
            x="time_bin",
            y="accuracy",
            ax=ax,
            color=COLORS["loser"],
            lw=2,
            marker="o",
            errorbar=("ci", 95),
            label="Loser",
            zorder=3,
        )

        ax.set_title(
            f"{chr(65 + i)}) {tgt.replace('_', ' ')}", loc="left", fontweight="bold"
        )
        ax.set_xlim(-0.5, 19.5)
        ax.set_xticklabels([])
        ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

        # Strips
        for j, (mode, label) in enumerate([("Winner", "Win"), ("Loser", "Lose")]):
            ax_s = plt.subplot(gs[base + 1 + j])
            p = get_p_values(
                df_t[df_t["player_status"] == mode], tgt, bins, "one_sample"
            )
            plot_heatmap(ax_s, p, len(bins))
            ax_s.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=8)
            ax_s.set_xticks([])

        # Diff Strip
        ax_d = plt.subplot(gs[base + 3])
        p = get_p_values(df_t, tgt, bins, "two_sample")
        plot_heatmap(ax_d, p, len(bins))
        ax_d.set_ylabel("Diff", rotation=0, ha="right", va="center", fontsize=8)
        if i == len(targets) - 1:
            ax_d.set_xlabel("Time Bins")
        else:
            ax_d.set_xticks([])

    plt.tight_layout()
    plt.savefig(out_dir / f"Figure3_WinnerLoser_Polished_{model_name}.png", dpi=300)
    plt.close()


def plot_model_comparison(df, out_dir, cfg):
    print("Plotting Model Comparison...")
    
    y_min = cfg['plotting'].get('y_min', 25)
    y_max = cfg['plotting'].get('y_max', 45)
    
    targets = [t for t in PLOT_TARGETS if t in df["target"].unique()]
    fig = plt.figure(figsize=(9, 3.5 * len(targets)))
    gs = gridspec.GridSpec(len(targets), 1, hspace=0.35)

    for i, tgt in enumerate(targets):
        df_t = df[df["target"] == tgt]
        
        ax = plt.subplot(gs[i])
        ax.set_ylim(y_min, y_max)
        ax.set_ylabel("Accuracy (%)")
        style_phase_background(ax)
        ax.axhline(CHANCE_LEVEL, color=COLORS["chance"], linestyle="--", lw=1.5, zorder=2)
        
        sns.lineplot(
            data=df_t,
            x="time_bin",
            y="accuracy",
            hue="model",
            ax=ax,
            linewidth=2.5,
            marker="o",
            markersize=6,
            errorbar=("ci", 95),
            zorder=3,
        )
        
        ax.set_title(f"{chr(65 + i)}) {tgt.replace('_', ' ')}", loc="left", fontweight="bold", pad=10)
        ax.set_xlim(-0.5, 19.5)
        if i == len(targets) - 1:
            ax.set_xlabel("Time Bins")
        else:
            ax.set_xlabel("")
            ax.set_xticklabels([])
            
        ax.legend(loc="upper right", fontsize=9, framealpha=0.9, title="Model")

    plt.tight_layout()
    plt.savefig(out_dir / "Figure_Model_Comparison.png", dpi=300)
    plt.close()


def run():
    cfg = load_config()
    res_dir = Path(cfg['paths']['results_dir'])
    fig_dir = Path(cfg['paths']['figures_dir'])
    fig_dir.mkdir(parents=True, exist_ok=True)
    try:
        df_all = aggregate_data(res_dir)
        df_haufe_all = aggregate_haufe_data(res_dir)
        
        # Plot grand average comparison of all models
        if len(df_all["model"].unique()) > 1:
            plot_model_comparison(df_all, fig_dir, cfg)
            
        models = df_all['model'].unique()
        for model in models:
            df_model = df_all[df_all['model'] == model]
            df_haufe_model = None
            if df_haufe_all is not None:
                df_haufe_model = df_haufe_all[df_haufe_all['model'] == model]
            plot_fig2(df_model, fig_dir, model, cfg, df_haufe=df_haufe_model)
            plot_fig3(df_model, fig_dir, model, cfg)
    except Exception as e: print(e)

if __name__ == "__main__":
    run()
