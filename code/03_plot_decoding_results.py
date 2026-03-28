import yaml
import pandas as pd
import numpy as np
import mne
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.gridspec import GridSpecFromSubplotSpec
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import seaborn as sns
from scipy import stats
from pathlib import Path

# --- CONFIG ---
CHANCE_LEVEL = 33.33

# Defined Phases with distinct pastel colors for separation
# Format: (StartBin, EndBin, Label, Color)
PHASES = [
    (0, 8.0, "Decision", "#fff3e0"),  # Light Orange
    (8.0, 16.0, "Response", "#e3f2fd"),  # Light Blue
    (16.0, 20.0, "Feedback", "#f3e5f5"),  # Light Purple
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

PLOT_TARGETS = ["Self_Response", "Other_Response", "Prev_Self_Response", "Prev_Other_Response"]
TOPO_SCALE_PERCENTILE = 90.0


def _compute_y_ticks(y_min, y_max, step=5):
    lo = int(np.floor(y_min / step) * step)
    hi = int(np.ceil(y_max / step) * step)
    return np.arange(lo, hi + step, step)


def _style_accuracy_axis(ax, y_min, y_max):
    ax.set_ylim(y_min, y_max)
    ax.set_yticks(np.arange(32, 40.1, 2))
    ax.grid(axis="y", color="#cfcfcf", linewidth=0.6, alpha=0.45)
    ax.grid(axis="x", visible=False)


def _strip_legend_handles():
    return [
        Patch(facecolor=COLORS["heatmap"][0], edgecolor="#999", label="ns"),
        Patch(facecolor=COLORS["heatmap"][1], edgecolor="#999", label="p < .05"),
        Patch(facecolor=COLORS["heatmap"][2], edgecolor="#999", label="p < .01"),
        Patch(facecolor=COLORS["heatmap"][3], edgecolor="#999", label="p < .001"),
    ]


def _add_split_legends(fig, accuracy_handles, significance_handles, y=1.01):
    """Add two separate figure legends for better readability."""
    leg_acc = fig.legend(
        handles=accuracy_handles,
        title="Accuracy",
        loc="upper left",
        bbox_to_anchor=(0.02, y),
        ncol=len(accuracy_handles),
        frameon=False,
        fontsize=8,
        title_fontsize=8,
        handlelength=2.2,
        columnspacing=1.2,
    )
    fig.add_artist(leg_acc)

    fig.legend(
        handles=significance_handles,
        title="Significance strip",
        loc="upper right",
        bbox_to_anchor=(0.98, y),
        ncol=len(significance_handles),
        frameon=False,
        fontsize=8,
        title_fontsize=8,
        handlelength=1.6,
        columnspacing=0.9,
    )


def _save_figure_all_formats(fig, out_file_png):
    fig.savefig(out_file_png, dpi=300)
    fig.savefig(out_file_png.with_suffix(".pdf"))
    fig.savefig(out_file_png.with_suffix(".svg"))


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
        return df_haufe_t

    if "pattern_row" not in df_haufe_t.columns:
        return df_haufe_t

    comp_strength = (
        df_haufe_t.groupby(["pattern_row", "pattern_label"], as_index=False)["pattern_value"]
        .apply(lambda s: np.mean(np.abs(s)))
        .rename(columns={"pattern_value": "abs_mean"})
        .sort_values("abs_mean", ascending=False)
    )

    best = comp_strength.iloc[0]
    df_sel = df_haufe_t[df_haufe_t["pattern_row"] == best["pattern_row"]]
    return df_sel


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


def plot_heatmap(ax, p_vals, n_bins, bin_width_s=0.25):
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

    t_max = n_bins * float(bin_width_s)
    ax.imshow(map_vals, aspect="auto", cmap=cmap, norm=norm, extent=[0.0, t_max, 0, 1])
    ax.set_yticks([])
    ax.set_xlim(0.0, t_max)

    # Add vertical phase dividers
    for _, end, _, _ in PHASES:
        ax.axvline(end * float(bin_width_s), color="white", linewidth=2)


def style_phase_background(ax, x_scale=1.0):
    """Applies colored phase backgrounds and labels."""
    for start, end, label, color in PHASES:
        start_x = start * x_scale
        end_x = end * x_scale
        ax.axvspan(start_x, end_x, color=color, alpha=0.6, lw=0, zorder=0)
        # Vertical divider line
        ax.axvline(end_x, color="white", linewidth=2, zorder=1)
        # Label (only if it's a main plot)
        if ax.get_ylabel():
            ax.text(
                (start_x + end_x) / 2,
                ax.get_ylim()[1] * 0.98,
                label,
                ha="center",
                va="top",
                fontsize=8,
                fontweight="semibold",
                color="#777",
            )


def _plot_haufe_topomap_row(
    subspec,
    df_haufe_t,
    n_bins,
    bin_width_s=0.25,
    show_time_axis=False,
    show_colorbar=True,
    fixed_vmax=None,
):
    """Plot one topomap per 4 bins (1 second per topomap)."""
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
    if fixed_vmax is not None:
        vmax = float(fixed_vmax)
    vmax = max(vmax, 1e-12)

    n_cols_inner = n_groups
    width_ratios = [1] * n_groups

    if show_time_axis:
        inner = GridSpecFromSubplotSpec(
            2,
            n_cols_inner,
            subplot_spec=subspec,
            width_ratios=width_ratios,
            height_ratios=[8, 1.2],
            hspace=0.1,
            wspace=0.25,
        )
    else:
        inner = GridSpecFromSubplotSpec(
            1,
            n_cols_inner,
            subplot_spec=subspec,
            width_ratios=width_ratios,
            hspace=0.0,
            wspace=0.25,
        )

    first_im = None
    last_map_ax = None
    for i, ((start_bin, end_bin), vec) in enumerate(zip(bin_groups, maps)):
        col_i = i
        ax_map = plt.subplot(inner[0, col_i])
        im, _ = mne.viz.plot_topomap(
            vec,
            info,
            axes=ax_map,
            show=False,
            cmap="RdBu_r",
            vlim=(-vmax, vmax),
            contours=0,
        )
        if first_im is None:
            first_im = im
        last_map_ax = ax_map

    if show_colorbar and first_im is not None and last_map_ax is not None:
        fig = plt.gcf()
        pos = last_map_ax.get_position()
        cbar_w = 0.008
        cbar_pad = 0.004
        cbar_x = min(pos.x1 + cbar_pad, 0.992 - cbar_w)
        cax = fig.add_axes([cbar_x, pos.y0, cbar_w, pos.height])
        cb = fig.colorbar(first_im, cax=cax, orientation="vertical")
        cb.ax.tick_params(labelsize=7, length=2)
        cb.set_label("Pattern", fontsize=7)

    if show_time_axis:
        ax_time = plt.subplot(inner[1, :])
        ax_time.set_xlim(0, float(n_groups))
        ax_time.set_ylim(0, 1)
        ax_time.set_yticks([])
        ax_time.set_xticks(np.arange(0, n_groups + 1, 1))
        group_duration_s = 4 * float(bin_width_s)
        tick_labels = [f"{x * group_duration_s:.0f}" for x in np.arange(0, n_groups + 1, 1)]
        ax_time.set_xticklabels(tick_labels)
        ax_time.set_xlabel("Time (s)")
        for spine in ["left", "right", "top"]:
            ax_time.spines[spine].set_visible(False)
        for x in np.arange(0, n_groups + 1, 1):
            ax_time.axvline(x, color="#cccccc", linewidth=0.6, alpha=0.6)


def plot_fig2(df, out_dir, model_name, cfg, df_haufe=None):
    print(f"Plotting Figure 2 (Polished) for {model_name}...")
    
    y_min = 31
    y_max = 40
    bin_width_s = cfg['params'].get('bin_width', 0.25)
    
    targets = [t for t in PLOT_TARGETS if t in df["target"].unique()]
    n_cols = 2 if len(targets) > 1 else 1
    n_rows = int(np.ceil(len(targets) / n_cols))

    fig = plt.figure(figsize=(12.5, 3.9 * n_rows))
    outer = gridspec.GridSpec(n_rows, n_cols, hspace=0.18, wspace=0.2)

    bins = np.sort(df["time_bin"].unique())
    t_max = len(bins) * bin_width_s

    selected_haufe_by_target = {}
    if df_haufe is not None:
        for tgt in targets:
            d = df_haufe[df_haufe["target"] == tgt]
            selected_haufe_by_target[tgt] = _select_pattern_component(d)

    global_topo_vmax = None
    if selected_haufe_by_target:
        vals = [
            np.abs(d["pattern_value"].values)
            for d in selected_haufe_by_target.values()
            if d is not None and not d.empty
        ]
        if vals:
            abs_all = np.concatenate(vals)
            # Robust shared scaling across all targets: avoid outlier-driven colorbar inflation
            global_topo_vmax = float(np.percentile(abs_all, TOPO_SCALE_PERCENTILE))

    for i, tgt in enumerate(targets):
        row = i // n_cols
        col = i % n_cols
        panel = GridSpecFromSubplotSpec(
            3,
            1,
            subplot_spec=outer[row, col],
            height_ratios=[3, 0.25, 2.1],
            hspace=0.14,
        )
        df_t = df[df["target"] == tgt]

        # Main
        ax = plt.subplot(panel[0])
        _style_accuracy_axis(ax, y_min, y_max)  # Set limits BEFORE styling to place text correctly
        ax.set_ylabel("Accuracy (%)")
        style_phase_background(ax, x_scale=bin_width_s)

        ax.axhline(
            CHANCE_LEVEL, color=COLORS["chance"], linestyle="--", lw=1.5, zorder=2
        )
        df_t_plot = df_t.copy()
        df_t_plot["time_s"] = (df_t_plot["time_bin"] + 0.5) * bin_width_s
        sns.lineplot(
            data=df_t_plot,
            x="time_s",
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
        ax.set_xlim(0.0, t_max)
        ax.set_xlabel("")
        ax.set_xticklabels([])

        # Strip
        ax_s = plt.subplot(panel[1])
        plot_heatmap(ax_s, get_p_values(df_t, tgt, bins), len(bins), bin_width_s=bin_width_s)
        ax_s.set_xticks([])

        # Haufe topoplots (1 map per 4 bins = 1 second)
        ax_topo_spec = panel[2]
        df_haufe_t = selected_haufe_by_target.get(tgt, None)
        _plot_haufe_topomap_row(
            ax_topo_spec,
            df_haufe_t,
            n_bins=len(bins),
            bin_width_s=bin_width_s,
            show_time_axis=(row == n_rows - 1),
            show_colorbar=(col == n_cols - 1),
            fixed_vmax=global_topo_vmax,
        )

    fig.text(0.012, 0.22, "Haufe patterns", rotation=90, ha="center", va="center", fontsize=8, color="#666")

    # Hide any unused grid cell(s)
    total_cells = n_rows * n_cols
    for j in range(len(targets), total_cells):
        row = j // n_cols
        col = j % n_cols
        ax_unused = plt.subplot(outer[row, col])
        ax_unused.axis("off")

    acc_handles = [
        Line2D([0], [0], color=COLORS["main"], marker="o", linewidth=2, label="Mean accuracy"),
        Line2D([0], [0], color=COLORS["chance"], linestyle="--", linewidth=1.5, label="Chance"),
    ]
    sig_handles = _strip_legend_handles()
    _add_split_legends(fig, acc_handles, sig_handles, y=1.01)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    _save_figure_all_formats(fig, out_dir / f"Figure2_GrandAverage_Polished_{model_name}.png")
    plt.close()


def plot_fig3(df, out_dir, model_name, cfg):
    print(f"Plotting Figure 3 (Polished Split) for {model_name}...")
    
    y_min = 31
    y_max = 40
    bin_width_s = cfg['params'].get('bin_width', 0.25)
    
    if len(df["player_status"].unique()) < 2:
        return
    targets = [t for t in PLOT_TARGETS if t in df["target"].unique()]

    n_cols = 2 if len(targets) > 1 else 1
    n_rows = int(np.ceil(len(targets) / n_cols))

    fig = plt.figure(figsize=(12.5, 4.0 * n_rows))
    outer = gridspec.GridSpec(n_rows, n_cols, hspace=0.2, wspace=0.2)

    bins = np.sort(df["time_bin"].unique())
    t_max = len(bins) * bin_width_s

    for i, tgt in enumerate(targets):
        df_t = df[df["target"] == tgt]
        row = i // n_cols
        col = i % n_cols
        panel = GridSpecFromSubplotSpec(
            4,
            1,
            subplot_spec=outer[row, col],
            height_ratios=[3, 0.2, 0.2, 0.2],
            hspace=0.12,
        )

        # Main
        ax = plt.subplot(panel[0])
        _style_accuracy_axis(ax, y_min, y_max)
        ax.set_ylabel("Accuracy (%)")
        style_phase_background(ax, x_scale=bin_width_s)
        ax.axhline(CHANCE_LEVEL, color=COLORS["chance"], linestyle="--", zorder=2)

        df_t_plot = df_t.copy()
        df_t_plot["time_s"] = (df_t_plot["time_bin"] + 0.5) * bin_width_s

        sns.lineplot(
            data=df_t_plot[df_t_plot["player_status"] == "Winner"],
            x="time_s",
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
            data=df_t_plot[df_t_plot["player_status"] == "Loser"],
            x="time_s",
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
        ax.set_xlim(0.0, t_max)
        ax.set_xlabel("")
        ax.set_xticklabels([])
        if ax.get_legend() is not None:
            ax.get_legend().remove()

        # Strips
        for j, (mode, label) in enumerate([("Winner", "Win"), ("Loser", "Lose")]):
            ax_s = plt.subplot(panel[1 + j])
            p = get_p_values(
                df_t[df_t["player_status"] == mode], tgt, bins, "one_sample"
            )
            plot_heatmap(ax_s, p, len(bins), bin_width_s=bin_width_s)
            ax_s.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=8)
            ax_s.set_xticks([])

        # Diff Strip
        ax_d = plt.subplot(panel[3])
        p = get_p_values(df_t, tgt, bins, "two_sample")
        plot_heatmap(ax_d, p, len(bins), bin_width_s=bin_width_s)
        ax_d.set_ylabel("Diff", rotation=0, ha="right", va="center", fontsize=8)
        ax_d.set_xticks([])

    # Hide any unused grid cell(s)
    total_cells = n_rows * n_cols
    for j in range(len(targets), total_cells):
        row = j // n_cols
        col = j % n_cols
        ax_unused = plt.subplot(outer[row, col])
        ax_unused.axis("off")

    acc_handles = [
        Line2D([0], [0], color=COLORS["winner"], marker="o", linewidth=2, label="Winner"),
        Line2D([0], [0], color=COLORS["loser"], marker="o", linewidth=2, label="Loser"),
        Line2D([0], [0], color=COLORS["chance"], linestyle="--", linewidth=1.5, label="Chance"),
    ]
    sig_handles = _strip_legend_handles()
    _add_split_legends(fig, acc_handles, sig_handles, y=1.01)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    _save_figure_all_formats(fig, out_dir / f"Figure3_WinnerLoser_Polished_{model_name}.png")
    plt.close()


def plot_model_comparison(df, out_dir, cfg):
    print("Plotting Model Comparison...")
    
    y_min = 31
    y_max = 40
    bin_width_s = cfg['params'].get('bin_width', 0.25)
    
    targets = [t for t in PLOT_TARGETS if t in df["target"].unique()]
    bins = np.sort(df["time_bin"].unique())
    model_order = sorted(df["model"].unique())
    model_palette = dict(zip(model_order, sns.color_palette(n_colors=len(model_order))))
    t_max = len(bins) * bin_width_s
    n_cols = 2 if len(targets) > 1 else 1
    n_rows = int(np.ceil(len(targets) / n_cols))
    fig = plt.figure(figsize=(12.5, 3.0 * n_rows))
    gs = gridspec.GridSpec(n_rows, n_cols, hspace=0.2, wspace=0.2)

    for i, tgt in enumerate(targets):
        df_t = df[df["target"] == tgt]
        row = i // n_cols
        col = i % n_cols
        
        ax = plt.subplot(gs[row, col])
        _style_accuracy_axis(ax, y_min, y_max)
        ax.set_ylabel("Accuracy (%)")
        style_phase_background(ax, x_scale=bin_width_s)
        ax.axhline(CHANCE_LEVEL, color=COLORS["chance"], linestyle="--", lw=1.5, zorder=2)

        df_t_plot = df_t.copy()
        df_t_plot["time_s"] = (df_t_plot["time_bin"] + 0.5) * bin_width_s
        
        sns.lineplot(
            data=df_t_plot,
            x="time_s",
            y="accuracy",
            hue="model",
            hue_order=model_order,
            palette=model_palette,
            ax=ax,
            linewidth=2.5,
            marker="o",
            markersize=6,
            errorbar=("ci", 95),
            zorder=3,
        )
        
        ax.set_title(f"{chr(65 + i)}) {tgt.replace('_', ' ')}", loc="left", fontweight="bold", pad=10)
        ax.set_xlim(0.0, t_max)
        if row == n_rows - 1:
            tick_pos = np.arange(0, t_max + 1e-9, 1.0)
            ax.set_xticks(tick_pos)
            ax.set_xticklabels([f"{p:.0f}" for p in tick_pos])
            ax.set_xlabel("Time (s)")
        else:
            ax.set_xlabel("")
            ax.set_xticklabels([])
            
        if ax.get_legend() is not None:
            ax.get_legend().remove()

    # Hide any unused grid cell(s)
    total_cells = n_rows * n_cols
    for j in range(len(targets), total_cells):
        row = j // n_cols
        col = j % n_cols
        ax_unused = plt.subplot(gs[row, col])
        ax_unused.axis("off")

    model_handles = [
        Line2D([0], [0], color=model_palette[m], marker="o", linewidth=2, label=m)
        for m in model_order
    ]
    handles = [
        *model_handles,
        Line2D([0], [0], color=COLORS["chance"], linestyle="--", linewidth=1.5, label="Chance"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=min(6, len(handles)), frameon=False, bbox_to_anchor=(0.5, 1.01), fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    _save_figure_all_formats(fig, out_dir / "Figure_Model_Comparison.png")
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
