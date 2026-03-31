"""
This script is only used for 1 plot in the final report.
The spectra per run can be found in the figures folder!

This script is intentionally isolated from existing derivatives/results.
All outputs are written into a new temporary folder that can be deleted safely.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import yaml


VARIANTS = [
    {
        "slug": "no_ica_no_freq_filter",
        "label": "No ICA | No frequency filtering",
        "highpass_hz": None,
        "lowpass_hz": None,
        "notch_freqs": [],
        "color": "#2ca02c",
    },
    {
        "slug": "no_ica_hp0.1_lp100_notch50+100",
        "label": "No ICA | HP 0.1 | LP 100 | Notch 50+100",
        "highpass_hz": 0.1,
        "lowpass_hz": 100.0,
        "notch_freqs": [50.0, 100.0],
        "color": "#1f77b4",
    },
    {
        "slug": "no_ica_hp0.3_lp40_notch50",
        "label": "No ICA | HP 0.3 | LP 40 | Notch 50",
        "highpass_hz": 0.3,
        "lowpass_hz": 40.0,
        "notch_freqs": [50.0],
        "color": "#ff7f0e",
    },
]


def _find_repo_root(start: Path) -> Path:
    """Find repository root by walking upwards for pyproject.toml."""
    for p in [start, *start.parents]:
        if (p / "pyproject.toml").exists():
            return p
    raise FileNotFoundError(
        f"Could not locate repository root from: {start} (missing pyproject.toml)"
    )


def _load_pipeline_class(preprocess_py: Path):
    spec = importlib.util.spec_from_file_location("preprocess_module", preprocess_py)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import preprocessing script: {preprocess_py}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "EEGPipeline"):
        raise RuntimeError(f"EEGPipeline not found in: {preprocess_py}")
    return mod.EEGPipeline


def _compute_mean_psd_db(epochs: mne.Epochs, fmin: float = 0.0, fmax: float = 120.0):
    nyquist = float(epochs.info["sfreq"]) / 2.0
    fmax = min(float(fmax), nyquist)

    psd = epochs.compute_psd(
        method="welch",
        picks="eeg",
        fmin=float(fmin),
        fmax=float(fmax),
        verbose="error",
    )
    data, freqs = psd.get_data(return_freqs=True)
    # data: (n_epochs, n_channels, n_freqs)
    power = np.mean(data, axis=(0, 1))
    db = 10.0 * np.log10(power + np.finfo(float).eps)
    return freqs, db


def _make_base_cfg(
    cfg: dict,
    subject: int,
    out_dir: Path,
    report_dir: Path,
    highpass_hz,
    lowpass_hz,
    notch_freqs,
) -> dict:
    c = copy.deepcopy(cfg)

    c.setdefault("subjects", {})
    c["subjects"]["include"] = [int(subject)]

    c.setdefault("guardrails", {})
    c["guardrails"]["allow_ica"] = False

    c.setdefault("paths", {})
    c["paths"]["output_dir"] = str(out_dir)
    c["paths"]["report_dir"] = str(report_dir)

    c.setdefault("params", {})
    c["params"]["highpass_hz"] = highpass_hz
    c["params"]["lowpass_hz"] = lowpass_hz
    c["params"]["notch_freqs"] = list(notch_freqs)
    c["params"]["notch_widths"] = None  # Use MNE defaults

    return c


def _subject_exists_in_bids(cfg: dict, subject: int) -> bool:
    bids_root = Path(cfg["paths"]["bids_root"])
    sub_str = f"sub-{int(subject):02d}"
    eeg_dir = bids_root / sub_str / "eeg"
    bdf_path = eeg_dir / f"{sub_str}_task-{cfg['params']['task_name']}_eeg.bdf"
    events_path = eeg_dir / f"{sub_str}_task-{cfg['params']['task_name']}_events.tsv"
    return eeg_dir.exists() and bdf_path.exists() and events_path.exists()


def _collect_spectra_from_existing(work_root: Path, subject: int, fmin: float, fmax: float):
    sub_str = f"sub-{int(subject):02d}"
    spectra = []
    for variant in VARIANTS:
        out_fif = work_root / variant["slug"] / "derivatives" / f"{sub_str}_task-RPS_desc-preproc_eeg.fif"
        if not out_fif.exists():
            raise FileNotFoundError(
                f"Missing preprocessed file for plot-only mode: {out_fif}\n"
                f"Run without --plot-only first, or use the correct --work-root."
            )
        epochs = mne.read_epochs(out_fif, preload=False, verbose="error")
        freqs, mean_db = _compute_mean_psd_db(epochs, fmin=fmin, fmax=fmax)
        spectra.append(
            {
                "label": variant["label"],
                "slug": variant["slug"],
                "color": variant["color"],
                "freqs": freqs,
                "mean_db": mean_db,
            }
        )
    return spectra


def _plot_spectra(
    spectra,
    sub_str: str,
    out_base: Path,
    layout: str = "panels",
    ymin: float | None = None,
    ymax: float | None = None,
):
    if layout == "overlay":
        fig, ax = plt.subplots(figsize=(11, 5))
        for s in spectra:
            ax.plot(s["freqs"], s["mean_db"], linewidth=1.8, label=s["label"], color=s["color"])
        ax.set_title(f"{sub_str} | Spectrum comparison across filter setups")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Mean PSD (dB)")
        if ymin is not None or ymax is not None:
            ax.set_ylim(bottom=ymin, top=ymax)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)
    else:
        fig, axes = plt.subplots(nrows=len(spectra), ncols=1, figsize=(11, 2.7 * len(spectra)), sharex=True, sharey=True)
        if len(spectra) == 1:
            axes = [axes]
        for ax, s in zip(axes, spectra):
            ax.plot(s["freqs"], s["mean_db"], linewidth=1.8, color=s["color"])
            ax.set_title(s["label"], loc="left", fontsize=10)
            ax.grid(True, alpha=0.3)
            if ymin is not None or ymax is not None:
                ax.set_ylim(bottom=ymin, top=ymax)
        axes[-1].set_xlabel("Frequency (Hz)")
        # Keep ylabel outside tick labels to avoid overlap
        fig.supylabel("Mean PSD (dB)", x=0.02)
        fig.suptitle(f"{sub_str} | Spectrum comparison across filter setups", y=0.995)

    plt.tight_layout(rect=[0.05, 0.02, 1.0, 0.98])
    fig.savefig(out_base.with_suffix(".png"), dpi=200)
    fig.savefig(out_base.with_suffix(".pdf"))
    fig.savefig(out_base.with_suffix(".svg"))
    plt.close(fig)


def main() -> int:
    repo_root = _find_repo_root(Path(__file__).resolve())

    parser = argparse.ArgumentParser(
        description="Run sub-03 preprocessing with three no-ICA filter setups and compare spectra."
    )
    parser.add_argument(
        "--subject",
        "--sub",
        dest="subject",
        type=int,
        default=3,
        help="Subject ID (default: 3). Alias: --sub",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=repo_root / "code" / "config_preprocessing.yaml",
        help="Path to base preprocessing config",
    )
    parser.add_argument(
        "--preprocess-script",
        type=Path,
        default=repo_root / "code" / "01_preprocess.py",
        help="Path to preprocessing Python script containing EEGPipeline",
    )
    parser.add_argument(
        "--work-root",
        type=Path,
        default=repo_root / "tmp" / f"sub-03_filter_compare_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        help="Temporary output folder (new folder will be created)",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Only regenerate plot/CSV from existing FIF files under --work-root",
    )
    parser.add_argument(
        "--plot-layout",
        choices=["panels", "overlay"],
        default="panels",
        help="Plot style: panels (separate subplots) or overlay (default: panels)",
    )
    parser.add_argument("--fmin", type=float, default=0.0, help="PSD lower bound in Hz")
    parser.add_argument("--fmax", type=float, default=120.0, help="PSD upper bound in Hz")
    parser.add_argument(
        "--ymin",
        type=float,
        default=None,
        help="Fixed lower y-limit in dB (default: auto)",
    )
    parser.add_argument(
        "--ymax",
        type=float,
        default=None,
        help="Fixed upper y-limit in dB (default: auto)",
    )
    args = parser.parse_args()

    if (args.ymin is not None) and (args.ymax is not None) and (args.ymin >= args.ymax):
        raise ValueError(f"Invalid y-limits: ymin ({args.ymin}) must be < ymax ({args.ymax})")

    if args.plot_only:
        if not args.work_root.exists():
            raise FileNotFoundError(f"--plot-only requires existing --work-root: {args.work_root}")
    else:
        if args.work_root.exists() and any(args.work_root.iterdir()):
            raise FileExistsError(
                f"Work folder already exists and is not empty: {args.work_root}\n"
                f"Use a new --work-root to keep this run isolated."
            )
        args.work_root.mkdir(parents=True, exist_ok=True)

    with args.config.open("r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    if not _subject_exists_in_bids(base_cfg, args.subject):
        print(
            f"[ERROR] Subject sub-{int(args.subject):02d} does not exist in BIDS input or is missing files.\n"
            f"        Expected files under: {Path(base_cfg['paths']['bids_root']) / f'sub-{int(args.subject):02d}' / 'eeg'}"
        )
        return 1

    EEGPipeline = _load_pipeline_class(args.preprocess_script)

    sub_str = f"sub-{int(args.subject):02d}"
    spectra = []

    if args.plot_only:
        print(f"[PLOT-ONLY] Loading existing FIFs from: {args.work_root}")
        spectra = _collect_spectra_from_existing(args.work_root, args.subject, args.fmin, args.fmax)
    else:
        for variant in VARIANTS:
            run_dir = args.work_root / variant["slug"]
            out_dir = run_dir / "derivatives"
            report_dir = run_dir / "report"
            out_dir.mkdir(parents=True, exist_ok=True)
            report_dir.mkdir(parents=True, exist_ok=True)

            cfg_this = _make_base_cfg(
                cfg=base_cfg,
                subject=args.subject,
                out_dir=out_dir,
                report_dir=report_dir,
                highpass_hz=variant["highpass_hz"],
                lowpass_hz=variant["lowpass_hz"],
                notch_freqs=variant["notch_freqs"],
            )

            cfg_path = run_dir / "config_preprocessing.yaml"
            with cfg_path.open("w", encoding="utf-8") as f:
                yaml.safe_dump(cfg_this, f, sort_keys=False, allow_unicode=True)

            print(f"\n[RUN] {variant['slug']}")
            print(f"      config: {cfg_path}")

            pipeline = EEGPipeline(str(cfg_path))
            pipeline.run_subject(int(args.subject))

            out_fif = out_dir / f"{sub_str}_task-RPS_desc-preproc_eeg.fif"
            if not out_fif.exists():
                raise FileNotFoundError(f"Expected output not found: {out_fif}")

            epochs = mne.read_epochs(out_fif, preload=False, verbose="error")
            freqs, mean_db = _compute_mean_psd_db(epochs, fmin=args.fmin, fmax=args.fmax)
            spectra.append(
                {
                    "label": variant["label"],
                    "slug": variant["slug"],
                    "color": variant["color"],
                    "freqs": freqs,
                    "mean_db": mean_db,
                }
            )

    out_base = args.work_root / f"{sub_str}_filter_spectra_comparison"
    _plot_spectra(
        spectra=spectra,
        sub_str=sub_str,
        out_base=out_base,
        layout=args.plot_layout,
        ymin=args.ymin,
        ymax=args.ymax,
    )

    # Also save long-format table for optional downstream plotting
    rows = []
    for s in spectra:
        for f, db in zip(s["freqs"], s["mean_db"]):
            rows.append({"variant": s["slug"], "label": s["label"], "freq_hz": float(f), "mean_psd_db": float(db)})
    pd.DataFrame(rows).to_csv(args.work_root / f"{sub_str}_filter_spectra_values.csv", index=False)

    print("\n[DONE]")
    print(f"Temporary folder: {args.work_root}")
    print(f"Combined plot: {out_base.with_suffix('.png')}")
    print("You can delete the whole folder manually when finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
