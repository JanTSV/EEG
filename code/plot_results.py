# ----- SKETCH -----
# EEG Plotting Pipeline for single-subject analysis:
# 1. Load configuration from YAML
# 2. Load preprocessed Epochs (.fif file) from disk
# 3. Average Epochs to create "Evoked" objects (ERPs)
# 4. Generate plots (Butterfly plot or specific channels)
# 5. Save figures to disk for the presentation

# ----- IMPORTS -----
import mne
import numpy as np
from pathlib import Path
from typing import Optional, Union
import matplotlib.pyplot as plt
import yaml
import matplotlib

# Use QtAgg for interactive plots if configured
# matplotlib.use("QtAgg") 

# ----- CODE -----
def load_config(config_path: Optional[Union[str, Path]] = None) -> tuple[dict, Path]:
    """Load plotting configuration from YAML and return config plus path."""
    # Default to a config file in the same directory
    default_path = Path(__file__).resolve().parent / "plot_config.yaml"
    cfg_path = Path(config_path) if config_path else default_path

    if not cfg_path.exists():
        raise FileNotFoundError(
            f"Config file not found at {cfg_path}. Please create it."
        )

    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f), cfg_path

def plot_pipeline(config_path: Optional[Union[str, Path]] = None):
    print("starting plotting pipeline ...")
    
    # 1. Load Config
    cfg, cfg_path = load_config(config_path)
    base_dir = cfg_path.parent
    
    def resolve_path(path_str: str) -> Path:
        path = Path(path_str)
        return path if path.is_absolute() else (base_dir / path)

    input_file = resolve_path(cfg["paths"]["input_file"])
    figures_dir = resolve_path(cfg["paths"]["figures_dir"])
    plot_cfg = cfg["plot"]

    # 2. Load Data
    print(f"loading epochs from {input_file} ...")
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}. Wait for preprocessing to finish!")
        
    epochs = mne.read_epochs(input_file, preload=True)
    
    # Print some info for verification
    print(f"  Trials found: {len(epochs)}")
    print(f"  Sampling rate: {epochs.info['sfreq']} Hz")
    
    # Apply cropping if configured (Zooming in time)
    if "crop" in plot_cfg:
        tmin = plot_cfg["crop"].get("tmin", None)
        tmax = plot_cfg["crop"].get("tmax", None)
        print(f"cropping data to {tmin}s - {tmax}s ...")
        epochs.crop(tmin=tmin, tmax=tmax)

    # 3. Compute Evoked (Average across trials)
    print("averaging epochs to evoked ...")
    # Select specific conditions if configured
    evoked = epochs[plot_cfg["conditions"]].average()
    
    # 4. Plotting
    print("generating plots ...")
    
    # Setup MNE styling
    mne.viz.set_browser_backend("matplotlib")
    
    # Handle "picks" logic: 'eeg' string vs list of channel names
    picks = plot_cfg.get("picks", "eeg")
    
    # --- PLOT A: Joint Plot (Topography + Butterfly) ---
    # This automatically uses all channels if picks="eeg"
    # It shows the global field power and topographies at peak times.
    fig1 = evoked.plot_joint(
        times="peaks", 
        title=f"Joint Plot: {plot_cfg['title']}",
        show=False,
        picks=picks
    )
    
    # --- PLOT B: Traces (Detail View) ---
    # If "eeg" is selected, this makes a butterfly plot (all lines overlaid).
    # If specific channels are selected, it shows them individually.
    # Note: For error bands (shadows), we stick to the mean for clarity in this milestone.
    
    fig2 = evoked.plot(
        picks=picks,
        spatial_colors=True, # Cool feature: colors lines by head position
        gfp=True, # Show Global Field Power (overall strength)
        window_title=f"Traces: {plot_cfg['title']}",
        show=False
    )

    # 5. Save Results
    figures_dir.mkdir(exist_ok=True, parents=True)
    
    save_path_joint = figures_dir / "erp_joint_plot.png"
    save_path_traces = figures_dir / "erp_traces.png"
    
    print(f"saving figures to {figures_dir} ...")
    fig1.savefig(save_path_joint)
    fig2.savefig(save_path_traces)
    
    print("done. plots saved.")
    
    if plot_cfg.get("interactive", False):
        plt.show()

if __name__ == "__main__":
    plot_pipeline()