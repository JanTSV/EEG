#!/usr/bin/env python3
"""
Rank decoding approaches across preprocessing runs.

This script scans run folders like:
  data/derivatives__no_ica/
  data/derivatives__0.1-100.0hz_notch50+100_no_ica/

For each run, it loads all CSVs in:
  <run>/results_decoding/*_decoding_results.csv

It then computes a robust per-model score:
  1) subject-target-time average accuracy
  2) area-over-chance (AOC) per subject+target+model
  3) average AOC across targets per subject+model
  4) trimmed mean across subjects per model (default trim=20%)

Output:
  CSV ranking with rows like model_run = "lda_no_ica".
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


def _normalize_run_label(run_dir: Path) -> str:
    name = run_dir.name
    for prefix in ("derivatives__", "derivatives_", "derivatives"):
        if name.startswith(prefix):
            label = name[len(prefix):].strip("_")
            return label if label else "default"
    return name


def _find_run_result_dirs(data_root: Path) -> list[tuple[str, Path]]:
    result = []
    for p in sorted(data_root.iterdir()):
        if not p.is_dir() or not p.name.startswith("derivatives"):
            continue
        res = p / "results_decoding"
        if res.is_dir():
            result.append((_normalize_run_label(p), res))
    return result


def _load_single_run(results_dir: Path) -> pd.DataFrame:
    files = sorted(results_dir.glob("*_decoding_results.csv"))
    if not files:
        return pd.DataFrame()

    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)

    if "model" not in df.columns:
        df["model"] = "lda"

    needed_cols = {"subject", "target", "time_bin", "accuracy", "model"}
    missing = needed_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {results_dir}: {sorted(missing)}")

    # normalize to percent scale
    if np.nanmax(df["accuracy"].to_numpy(dtype=float)) <= 1.5:
        df["accuracy"] = df["accuracy"] * 100.0

    # subject-level curve (collapse repeats/folds/player_status if present)
    df_subject = (
        df.groupby(["subject", "target", "time_bin", "model"], as_index=False)["accuracy"]
        .mean()
    )
    return df_subject


def _trimmed_mean(values: Iterable[float], proportion_to_cut: float) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    if arr.size < 3 or proportion_to_cut <= 0:
        return float(np.mean(arr))
    return float(stats.trim_mean(arr, proportiontocut=proportion_to_cut))


def rank_model_runs(
    data_root: Path,
    out_csv: Path,
    chance_level: float = 33.33,
    trim_proportion: float = 0.20,
) -> pd.DataFrame:
    run_dirs = _find_run_result_dirs(data_root)
    if not run_dirs:
        raise FileNotFoundError(f"No run folders with results_decoding found under: {data_root}")

    rows: list[dict] = []

    for run_label, results_dir in run_dirs:
        df = _load_single_run(results_dir)
        if df.empty:
            continue

        # area over chance per subject/target/model
        df = df.assign(aoc=df["accuracy"] - float(chance_level))
        subj_target_model = (
            df.groupby(["subject", "target", "model"], as_index=False)["aoc"]
            .mean()
            .rename(columns={"aoc": "aoc_target"})
        )

        # average over targets per subject/model
        subj_model = (
            subj_target_model.groupby(["subject", "model"], as_index=False)["aoc_target"]
            .mean()
            .rename(columns={"aoc_target": "aoc_subject"})
        )

        for model, dmod in subj_model.groupby("model", sort=True):
            vals = dmod["aoc_subject"].to_numpy(dtype=float)
            trimmed = _trimmed_mean(vals, trim_proportion)
            mean = float(np.nanmean(vals)) if len(vals) else np.nan
            median = float(np.nanmedian(vals)) if len(vals) else np.nan
            sd = float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else np.nan
            mad = float(np.nanmedian(np.abs(vals - np.nanmedian(vals)))) if len(vals) else np.nan

            rows.append(
                {
                    "model_run": f"{model}_{run_label}",
                    "model": model,
                    "run": run_label,
                    "score_trimmed_aoc": trimmed,
                    "mean_aoc": mean,
                    "median_aoc": median,
                    "subject_sd": sd,
                    "subject_mad": mad,
                    "n_subjects": int(dmod["subject"].nunique()),
                    "n_targets": int(subj_target_model[subj_target_model["model"] == model]["target"].nunique()),
                    "chance_level": float(chance_level),
                    "trim_proportion": float(trim_proportion),
                    "results_dir": str(results_dir),
                }
            )

    if not rows:
        raise RuntimeError("No ranking rows could be computed. Check your CSV content.")

    ranking = pd.DataFrame(rows)
    ranking = ranking.sort_values(
        ["score_trimmed_aoc", "median_aoc", "mean_aoc"],
        ascending=False,
    ).reset_index(drop=True)
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(out_csv, index=False)
    return ranking


def plot_top_model_runs(
    ranking: pd.DataFrame,
    out_path: Path,
    top_n: int = 10,
    score_col: str = "score_trimmed_aoc",
) -> Path:
    if score_col not in ranking.columns:
        raise ValueError(f"Column not found for plotting: {score_col}")

    d = ranking.copy()
    d = d[np.isfinite(d[score_col].to_numpy(dtype=float))]
    if d.empty:
        raise RuntimeError("No finite scores available to plot.")

    if top_n <= 0:
        raise ValueError("--top-n must be >= 1")

    d_top = d.nsmallest(top_n, columns=["rank"]).copy()
    d_top = d_top.sort_values(score_col, ascending=True)

    height = max(4.5, 0.55 * len(d_top))
    fig, ax = plt.subplots(figsize=(12.0, height))
    sns.barplot(
        data=d_top,
        x=score_col,
        y="model_run",
        hue="model_run",
        palette="viridis",
        legend=False,
        ax=ax,
        orient="h",
    )

    ax.axvline(0.0, color="#555", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Trimmed AOC score (percentage points above chance)")
    ax.set_ylabel("model_run")
    ax.set_title(f"Top {len(d_top)} model_run by trimmed AOC")
    ax.grid(axis="x", color="#d0d0d0", linewidth=0.6, alpha=0.6)
    ax.grid(axis="y", visible=False)

    # Numeric value labels inside bars
    x_min, x_max = ax.get_xlim()
    x_span = (x_max - x_min) if x_max > x_min else 1.0
    in_pad = 0.02 * x_span
    for p in ax.patches:
        w = float(p.get_width())
        y = p.get_y() + p.get_height() / 2.0
        if w >= 0:
            x = w - in_pad
            ha = "right"
        else:
            x = w + in_pad
            ha = "left"
        ax.text(
            x,
            y,
            f"{w:.2f}",
            va="center",
            ha=ha,
            fontsize=8,
            color="white",
            fontweight="semibold",
        )

    # Add rank labels at end of bars
    x_off = 0.01 * (x_max - x_min) if x_max > x_min else 0.05
    for _, row in d_top.iterrows():
        val = float(row[score_col])
        y_label = row["model_run"]
        ax.text(val + x_off, y_label, f"#{int(row['rank'])}", va="center", ha="left", fontsize=8)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    fig.savefig(out_path.with_suffix(".pdf"))
    fig.savefig(out_path.with_suffix(".svg"))
    plt.close(fig)
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank decoding model+run combinations (e.g., lda_no_ica) across derivatives runs."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Root containing derivatives* folders (default: data)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results_decoding/model_run_ranking.csv"),
        help="Output CSV path (default: results_decoding/model_run_ranking.csv)",
    )
    parser.add_argument(
        "--chance-level",
        type=float,
        default=33.33,
        help="Chance level in percent (default: 33.33)",
    )
    parser.add_argument(
        "--trim-proportion",
        type=float,
        default=0.20,
        help="Proportion to trim from each tail across subjects (default: 0.20)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Top N model_run rows to plot (default: 10)",
    )
    parser.add_argument(
        "--plot-out",
        type=Path,
        default=Path("results_decoding/model_run_ranking_topN.png"),
        help="Output path for top-N bar plot (PNG; PDF/SVG also saved)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable bar-plot generation",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    ranking = rank_model_runs(
        data_root=args.data_root,
        out_csv=args.out,
        chance_level=args.chance_level,
        trim_proportion=args.trim_proportion,
    )

    if not args.no_plot:
        plot_path = plot_top_model_runs(
            ranking=ranking,
            out_path=args.plot_out,
            top_n=args.top_n,
            score_col="score_trimmed_aoc",
        )
        print(f"Saved top-{args.top_n} bar plot to: {plot_path}")

    print(f"Saved ranking to: {args.out}")
    print(ranking.head(15).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
