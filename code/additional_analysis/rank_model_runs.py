"""
Rank decoding approaches across preprocessing runs.

---------------------
This script provides the model-comparison summary used in the final version of
the university project. It is designed to compare decoding performance across
multiple preprocessing variants (for example, different filtering or ICA
settings) while keeping the scoring procedure robust and easy to reproduce.

The main idea is to collapse the time-resolved decoding outputs into a single
subject-level score per model and preprocessing run, then rank these model-run
combinations using a robust group statistic.

Data layout
-----------
The script scans run folders such as:
    data/derivatives__no_ica/
    data/derivatives__0.1-100.0hz_notch50+100_no_ica/

For each run, it loads all CSV files in:
    <run>/results_decoding/*_decoding_results.csv

These CSVs are expected to come from the decoding pipeline and contain at
minimum the columns ``subject``, ``target``, ``time_bin``, ``accuracy``, and
``model``.

Scoring pipeline
---------------
The script computes a robust per-model score in four steps:

1. Subject-level averaging
     Decoding accuracy is first averaged across repeated folds and repetitions
     for each subject, target, time bin, and model.

2. Area over chance (AOC)
     Accuracy is converted into percentage points above the theoretical chance
     level (default: 33.33% for a 3-class task).

3. Subject-level summary
     AOC is averaged across time bins and targets (with optional target
     exclusions such as ``Outcome``), yielding one score per subject and model.

4. Robust group summary
     A trimmed mean across subjects (default: 20% per tail) is used as the final
     ranking score, reducing the influence of outlier subjects.

Outputs
-------
- A CSV ranking table with one row per model+run combination.
- An optional bar plot of the top-N combinations, saved as PNG/PDF/SVG.

Interpretation
-------------
The resulting ranking is a compact summary of decoding performance across
preprocessing pipelines. It is useful for comparing run variants, but it does
not replace the time-resolved decoding plots or significance analyses.
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


def _find_repo_root(start: Path) -> Path:
    """
    Find the repository root by walking upward until ``pyproject.toml`` is found.

    Parameters
    ----------
    start : pathlib.Path
        Starting location, usually the directory containing this script.

    Returns
    -------
    pathlib.Path
        Path to the repository root.

    Raises
    ------
    FileNotFoundError
        If no parent directory contains a ``pyproject.toml`` file.
    """
    for p in [start, *start.parents]:
        if (p / "pyproject.toml").exists():
            return p
    raise FileNotFoundError(
        f"Could not locate repository root from: {start} (missing pyproject.toml)"
    )


def _normalize_run_label(run_dir: Path) -> str:
    """
    Convert a derivatives folder name into a compact human-readable run label.

    Examples
    --------
    - ``derivatives__no_ica`` -> ``no_ica``
    - ``derivatives__0.1-100.0hz_notch50+100_no_ica`` ->
      ``0.1-100.0hz_notch50+100_no_ica``

    Parameters
    ----------
    run_dir : pathlib.Path
        Path to a derivatives directory.

    Returns
    -------
    str
        Normalized label that is appended to the model name in the ranking
        output.
    """
    name = run_dir.name
    for prefix in ("derivatives__", "derivatives_", "derivatives"):
        if name.startswith(prefix):
            label = name[len(prefix):].strip("_")
            return label if label else "default"
    return name


def _find_run_result_dirs(data_root: Path) -> list[tuple[str, Path]]:
    """
    Discover all result directories containing decoding CSV files.

    Parameters
    ----------
    data_root : pathlib.Path
        Root directory containing ``derivatives*`` folders.

    Returns
    -------
    list[tuple[str, pathlib.Path]]
        A list of ``(run_label, results_dir)`` tuples.
    """
    result = []
    for p in sorted(data_root.iterdir()):
        if not p.is_dir() or not p.name.startswith("derivatives"):
            continue
        res = p / "results_decoding"
        if res.is_dir():
            result.append((_normalize_run_label(p), res))
    return result


def _load_single_run(results_dir: Path) -> pd.DataFrame:
    """
    Load and collapse the decoding results for one preprocessing run.

    The function concatenates all matching decoding CSVs, validates the
    expected columns, converts accuracy values to percent if needed, and then
    averages repeated measurements across folds/repetitions/player-status
    effects to produce a subject-level time course.

    Parameters
    ----------
    results_dir : pathlib.Path
        Directory containing ``*_decoding_results.csv`` files.

    Returns
    -------
    pandas.DataFrame
        Data frame with one row per ``subject × target × time_bin × model`` and
        an ``accuracy`` column in percent.
    """
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
    """
    Compute a robust trimmed mean after removing non-finite values.

    Parameters
    ----------
    values : Iterable[float]
        Sequence of numeric values to summarize.
    proportion_to_cut : float
        Fraction to remove from each tail before averaging.

    Returns
    -------
    float
        Trimmed mean, or ``NaN`` if no finite values are available.
    """
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
    exclude_targets: tuple[str, ...] = ("Outcome",),
) -> pd.DataFrame:
    """
    Rank model-run combinations across all preprocessing variants.

    Parameters
    ----------
    data_root : pathlib.Path
        Root directory containing the preprocessing runs.
    out_csv : pathlib.Path
        Output location for the ranking CSV.
    chance_level : float, default=33.33
        Chance-level decoding accuracy in percent.
    trim_proportion : float, default=0.20
        Fraction to cut from each tail when computing the trimmed mean across
        subjects.
    exclude_targets : tuple[str, ...], default=("Outcome",)
        Targets that should be excluded from the ranking.

    Returns
    -------
    pandas.DataFrame
        Ranking table sorted from best to worst according to the trimmed AOC
        score.
    """
    run_dirs = _find_run_result_dirs(data_root)
    if not run_dirs:
        raise FileNotFoundError(f"No run folders with results_decoding found under: {data_root}")

    rows: list[dict] = []

    for run_label, results_dir in run_dirs:
        df = _load_single_run(results_dir)
        if df.empty:
            continue

        if exclude_targets:
            df = df[~df["target"].isin(exclude_targets)].copy()
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
                    "excluded_targets": ",".join(exclude_targets) if exclude_targets else "",
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
    """
    Plot the top-N ranked model-run combinations as a horizontal bar chart.

    Parameters
    ----------
    ranking : pandas.DataFrame
        Ranking table returned by :func:`rank_model_runs`.
    out_path : pathlib.Path
        Output path for the PNG figure. Matching PDF and SVG files are also
        written.
    top_n : int, default=10
        Number of top rows to visualize.
    score_col : str, default="score_trimmed_aoc"
        Column used to order and plot the bars.

    Returns
    -------
    pathlib.Path
        Path to the saved PNG figure.
    """
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
    """
    Parse command-line arguments for the ranking script.

    Returns
    -------
    argparse.Namespace
        Parsed CLI arguments.
    """
    repo_root = _find_repo_root(Path(__file__).resolve())

    parser = argparse.ArgumentParser(
        description="Rank decoding model+run combinations (e.g., lda_no_ica) across derivatives runs."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=repo_root / "data",
        help="Root containing derivatives* folders (default: data)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=repo_root / "results_decoding" / "model_run_ranking.csv",
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
        "--exclude-targets",
        nargs="*",
        default=["Outcome"],
        help="Targets to exclude from ranking (default: Outcome)",
    )
    parser.add_argument(
        "--top-n",
        type=str,
        default="10",
        help="Top N model_run rows to plot, or 'all' (default: 10)",
    )
    parser.add_argument(
        "--plot-out",
        type=Path,
        default=repo_root / "results_decoding" / "model_run_ranking_topN.png",
        help="Output path for top-N bar plot (PNG; PDF/SVG also saved)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable bar-plot generation",
    )
    return parser.parse_args()


def main() -> int:
    """
    Command-line entry point.

    The function runs the ranking analysis, optionally generates the top-N bar
    plot, and prints a compact textual summary of the strongest model-run
    combinations.

    Returns
    -------
    int
        Exit code for the process.
    """
    args = parse_args()

    ranking = rank_model_runs(
        data_root=args.data_root,
        out_csv=args.out,
        chance_level=args.chance_level,
        trim_proportion=args.trim_proportion,
        exclude_targets=tuple(args.exclude_targets),
    )

    if not args.no_plot:
        top_n_raw = str(args.top_n).strip().lower()
        if top_n_raw == "all":
            top_n = len(ranking)
        else:
            try:
                top_n = int(args.top_n)
            except ValueError as exc:
                raise ValueError("--top-n must be a positive integer or 'all'") from exc
            if top_n <= 0:
                raise ValueError("--top-n must be >= 1 or 'all'")

        plot_path = plot_top_model_runs(
            ranking=ranking,
            out_path=args.plot_out,
            top_n=top_n,
            score_col="score_trimmed_aoc",
        )
        print(f"Saved top-{top_n} bar plot to: {plot_path}")

    print(f"Saved ranking to: {args.out}")
    print(ranking.head(15).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
