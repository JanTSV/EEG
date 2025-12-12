# EEG Preprocessing Pipeline

Dual-player EEG preprocessing for Rock-Paper-Scissors task data. Processes BioSemi 64-channel recordings through a configurable pipeline including filtering, ICA artifact removal, epoching, and downsampling.

## Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- BioSemi 64-channel EEG data in BIDS format

## Quick Start

```bash
# Install dependencies
uv sync

# Run preprocessing pipeline (processes pairs defined in config)
uv run python code/preprocessing.py
```

Output: Preprocessed epochs saved as FIF files in `data/derivatives/`

## Configuration

All preprocessing parameters are in **`code/preprocess_config.yaml`** (fully commented). Key sections:

### Paths

- `raw_template`: BioSemi BDF files (dual-player recordings)
- `events_template`: Event markers (stimulus onset times)
- `derivatives_dir`: Output location for preprocessed epochs

### Processing Steps (Enable/Disable as Needed)

**Filtering:**

- `notch_filter_enabled`: Remove 50/60 Hz AC mains hum (default: `true`, 50 Hz)
- `output_filter_enabled`: Band-pass filter (default: 0.1-100 Hz, FIR)

**Artifact Removal:**

- `reref_to_average`: Average re-reference (default: `true`)
- `ica_enabled`: Independent Component Analysis for eye/muscle artifacts (default: `false`)
- `amplitude_reject_enabled`: Reject epochs by voltage threshold (default: `false`)

**Data Processing:**

- `baseline_window_sec`: Baseline correction window (default: -0.2 to 0 sec)
- `down_sample`: Resample to 256 Hz (default: `true`, reduces file size 8×)
- `interpolate_bad_channels`: Interpolate noisy channels via spherical spline

### Target Data

- `target_pairs`: Which pair IDs to process (e.g., `[01]` for testing, `[1, 2, 3, ...]` for all)
- `pair_ranges`: Available pairs (some subjects missing: pairs 10, 23-24 excluded)

## Preprocessing Pipeline Overview

The pipeline (`code/preprocessing.py`) executes these steps per player:

1. **Channel Selection** – Extract player-specific channels from dual-player recording
2. **Montage Standardization** – Map to BioSemi64 standard positions
3. **Re-referencing** – Average reference (subtracts mean across channels)
4. **Filtering** – Notch (line noise) + band-pass (signal of interest)
5. **ICA** – Separate brain from physiological artifacts (optional)
6. **Epoching** – Extract [-0.2, 5.0] sec windows around stimulus onset
7. **Baseline Correction** – Subtract pre-stimulus mean (-0.2 to 0 sec)
8. **Bad Channel Interpolation** – Restore noisy channels via spline
9. **Amplitude Rejection** – Flag outlier epochs (optional)
10. **Downsampling** – 2048 Hz → 256 Hz (optional)
11. **Save** – Output as FIF format with metadata

See inline comments in `preprocessing.py` for detailed method explanations.

## Output Format

**FIF (Functional Image File Format):**

- MNE-native binary format
- Preserves metadata: channel info, montage, sampling rate, coordinate frames
- Filename: `pair-{pair:02d}_player-{player}_task-RPS_eeg-epo.fif`

**Load preprocessed data:**

```python
import mne
epochs = mne.read_epochs('data/derivatives/pair-01_player-1_task-RPS_eeg-epo.fif')
```

## Typical Workflow

1. **Initial test:** Set `target_pairs: [01]` in config → run pipeline on one pair
2. **Inspect output:** Check `data/derivatives/` for FIF files
3. **Compare results:** Use comparison script to visualize preprocessing effects
4. **Adjust parameters:** Enable ICA (`ica_enabled: true`) or change filter bounds
5. **Process all data:** Set `target_pairs` to full list or use `pair_ranges`

## Visualizing Preprocessing Effects

Compare raw vs. preprocessed data to see the impact of filtering, ICA, and other processing steps:

```bash
# Compare pair 1, player 1 (requires preprocessed FIF file to exist)
uv run python code/compare_preprocessing.py --pair 1 --player 1

# Compare with specific channels
uv run python code/compare_preprocessing.py --pair 1 --player 1 --channels Fz Cz Pz
```

**What it shows:**

- Top plot: Raw data (epoched + downsampled only)
- Bottom plot: Preprocessed data (filtered, ICA, re-referenced, baseline corrected)
- Butterfly plots: All channels overlaid to see global patterns
- RMS amplitude reduction: Quantifies noise removal

Use this to validate that preprocessing improves signal quality without distorting neural responses.

## ERP Visualization

Generate publication-ready ERP plots (Joint Plots and ROI Traces) based on the preprocessed data.

```bash
# Run plotting pipeline
uv run code/plot_results.py
```

## Configuration (code/plot_config.yaml)

- **Time Window:**  
  Crop data to relevant components (e.g., `-0.2` to `1.0` s).

- **Channel Selection:**  
  Set `picks`: `"eeg"` for a butterfly plot (all channels) or a list `["Fz", "Pz"]` for specific traces.

- **Variability:**  
  Toggle between `mean` (clean line) or `sem` (standard error shadow).

- **Output:**  
  Figures saved to `data/derivatives/figures/` (e.g., `erp_joint_plot.png`).

---

## Decoding Analysis

Run multivariate pattern analysis (MVPA) to decode decision-making processes from EEG signals.  
The pipeline supports multiple models and handles dummy data generation for testing technical workflows.

### Bash

```bash
# Run advanced decoding pipeline
uv run code/advanced_decoding.py
```


## Troubleshooting

**All epochs dropped:**

- `amplitude_reject_threshold_uv` too strict (try 300-500 µV or disable)

**Missing files:**

- Check `participants.tsv` for bad channel lists
- Verify raw BDF files exist at `raw_template` path

**Slow processing:**

- Disable `ica_enabled` for faster runs
- Reduce `ica_n_components` (default: 20)

## Alternative: Pip Installation

```bash
pip install -r requirements.txt
python code/preprocessing.py
```

## Further Reading

- **Original detailed docs:** See `README_OLD.md` for experiment details
- **MNE-Python:** https://mne.tools/stable/index.html
- **Config reference:** All options documented in `code/preprocess_config.yaml`
