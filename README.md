# EEG decoding project

This repository contains the full EEG pipeline for the Rock-Paper-Scissors
project:

- preprocessing of the raw BioSemi recordings,
- time-resolved decoding with multiple classifiers,
- figure generation for the final report,
- and a helper script that ranks model/run combinations.

The code is meant to be run directly from the scripts in the `code/` folder.
A shell script is included to run multiple decoding and plotting tasks in one
go.

## Important

The repository does not ship the EEG data itself.
For the pipeline to work, you must place the BIDS input files in a local data/
folder at the project root.

## Getting the data

The easiest way to match the expected structure is to install the OpenNeuro
dataset into the project root and use data/ as the local dataset directory.
This keeps the BIDS layout identical to what the scripts expect while letting
DataLad fetch files on demand.

Example:

```bash
# from the repository root
datalad install -d data https://github.com/OpenNeuroDatasets/ds006761
cd data
datalad get .
```

If you already have the dataset elsewhere, you can also copy or symlink it into
data/ as long as the BIDS structure remains the same.

## Requirements

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) recommended, or pip
- Local EEG dataset in the expected BIDS layout inside `data/`
- OpenNeuro dataset ds006761 installed into `data/` or provided with the same
  BIDS structure

## Main scripts

- `code/01_preprocess.py` — preprocessing of the raw BDF recordings
- `code/02_decoding.py` — decoding analysis for all configured targets
- `code/03_plot_decoding_results.py` — decoding figures and Haufe plots
- `code/additional_analysis/rank_model_runs.py` — ranking of model/run setups

## Script arguments

The preprocessing, decoding, and plotting scripts are intended to be run
directly. Their behavior is controlled through the YAML configuration files in
`code/`, so they do not currently take extra command-line arguments.

The ranking helper does expose a small CLI for convenience:

- `--data-root` — root directory containing the `derivatives*` folders
- `--out` — output CSV path for the ranking table
- `--chance-level` — chance level in percent
- `--trim-proportion` — fraction trimmed from each tail across subjects
- `--exclude-targets` — targets to ignore when ranking
- `--top-n` — number of top rows to plot, or `all`
- `--plot-out` — output path for the top-N bar plot
- `--no-plot` — disable plot generation

## Quick start

```bash
# Install dependencies
uv sync

# Run preprocessing directly
uv run python code/01_preprocess.py

# Run decoding directly
uv run python code/02_decoding.py

# Create decoding figures directly
uv run python code/03_plot_decoding_results.py

# Rank all model/run combinations directly
uv run code/additional_analysis/rank_model_runs.py --top-n all
```

## Batch execution

The shell script is meant for running multiple decoding and plotting tasks in a
single pass, for example when comparing several preprocessing variants or when
recreating the final report figures.

Available arguments:

- `--figures-only` — regenerate only the figures for all derivative folders
  without re-running decoding
- `--override` — force a full re-run and overwrite existing `results_decoding`
  and figure outputs
- `-h`, `--help` — show the built-in usage help

```bash
bash run_decoding_all_derivatives.sh
bash run_decoding_all_derivatives.sh --figures-only
bash run_decoding_all_derivatives.sh --override
```

## Preprocessing

The preprocessing pipeline loads raw BioSemi BDF recordings, applies the
configured filtering and artifact-rejection steps, interpolates bad channels,
epochs the data, resamples it, and writes FIF files plus quality-control
figures.

Key points:

- the configuration lives in `code/config_preprocessing.yaml`,
- channel mapping and filtering are configuration-driven,
- ICA is optional and controlled by the config,
- the config path is relative, so the script can be run from the repository
  root.

Output example:

- `data/sub-01/sub-01_task-RPS_desc-preproc_eeg.fif`
- `data/report/sub-01_qc.png`

## Decoding

The decoding pipeline uses the preprocessed epochs and evaluates the enabled
classifiers with repeated stratified cross-validation and super-trial
averaging.

Current models:

- `lda`
- `svm`
- `logreg`
- `ridge`
- `mlp`

The model set and their hyperparameters are defined in
`code/config_decoding.yaml`.

The decoding script writes one CSV per subject with both accuracy results and,
for linear models, Haufe-pattern exports.

## Plotting

`code/03_plot_decoding_results.py` creates the final figures from the decoding
outputs:

- grand-average model comparison,
- per-model time-resolved decoding figures,
- winner-vs-loser figures,
- and Haufe topographies for models that provide them.

The plotting script reads the same configuration file as the decoding script so
that the figures stay aligned with the chosen analysis settings.

## Ranking analysis

`code/additional_analysis/rank_model_runs.py` summarizes the decoding results
across preprocessing variants and ranks the model/run combinations by a robust
area-over-chance score.

It can rank a limited number of top entries or all available combinations.

## Output folders

Typical outputs are stored under:

- `data/derivatives*` — preprocessing variants and intermediate outputs
- `data/results_decoding/` — decoding CSV files and ranking outputs
- `figures/` — final figures used in the report

## Troubleshooting

- If the scripts cannot find the input data, make sure the local `data/`
  folder exists and contains the BIDS files.
- If decoding is slow, reduce the number of repeats or disable expensive
  models in `code/config_decoding.yaml`.
- If preprocessing fails, verify that the raw file names and subject folders
  match the expected BIDS structure.

## Further reading

- MNE-Python: https://mne.tools/stable/index.html
- Configuration files: `code/config_preprocessing.yaml` and
  `code/config_decoding.yaml`

## Pipeline overview

```mermaid
flowchart TD
    A[Local data/ folder with BIDS EEG files] --> B[01_preprocess.py]
    B --> C[Preprocessed FIF epochs in data/]
    C --> D[02_decoding.py]
    D --> E[Decoding CSVs and Haufe patterns in results_decoding/]
    E --> F[03_plot_decoding_results.py]
    F --> G[Final figures in figures/]
    E --> H[rank_model_runs.py]
    H --> I[Model/run ranking CSV and top-N plots]
    J[run_decoding_all_derivatives.sh] --> D
    J --> F
```
