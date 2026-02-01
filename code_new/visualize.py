"""
visualize.py (COMPLETE FIX)
---------------------------
Helper functions for quality control plots.
Contains:
- plot_raw_quality: For Raw data (PSD)
- plot_epochs_quality: For Epoched data (ERP + Butterfly + Image)
Robust against missing montages (coordinates).
"""

import matplotlib.pyplot as plt
import numpy as np
import mne
from pathlib import Path

def get_out_dir():
    """Helper to find valid figure directory."""
    # Try standard path
    out_dir = Path("figures/00_quality_checks")
    # Fallback if run from code_new/
    if not out_dir.parent.exists() and Path("../figures").exists():
        out_dir = Path("../figures/00_quality_checks")
    
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir

def plot_psd_check(raw, pair_id, player_num, step_name, out_dir):
    """
    Plots PSD (Power Spectral Density) to check for line noise/alpha.
    """
    try:
        # Modern MNE syntax
        fig = raw.compute_psd(fmax=100).plot(average=True, spatial_colors=False, show=False)
        out_file = out_dir / f"pair-{pair_id:02d}_p{player_num}_{step_name}_psd.png"
        fig.savefig(out_file)
        plt.close(fig)
        print(f"    [PLOT] Saved: {out_file.name}")
    except Exception as e:
        print(f"    [WARN] PSD Plot failed: {e}")

def plot_raw_quality(raw, pair_id, player_num, step_name="raw"):
    """
    Wrapper to plot Raw QC (PSD).
    """
    out_dir = get_out_dir()
    plot_psd_check(raw, pair_id, player_num, step_name, out_dir)

def plot_epochs_quality(epochs, pair_id, player_num, step_name="clean"):
    """
    Plots ERP image and Joint Plot (Butterfly + Topo).
    Robust against missing montage (skips Topo if no coords).
    """
    out_dir = get_out_dir()
    
    # 1. ERP / Joint Plot
    evoked = epochs.average()
    times = np.arange(0.1, 0.5, 0.1) # Time points for topomaps
    
    try:
        # Versuche normalen Plot mit Topos
        fig_erp = evoked.plot_joint(times=times, title=f"ERP ({step_name}) - Pair {pair_id} P{player_num}", show=False)
        out_file = out_dir / f"pair-{pair_id:02d}_p{player_num}_{step_name}_erp.png"
        fig_erp.savefig(out_file)
        plt.close(fig_erp)
        print(f"    [PLOT] Saved: {out_file.name}")
        
    except RuntimeError as e:
        if "digitization points" in str(e) or "Channel locations" in str(e):
            print("    [WARN] No channel locations found. Plotting ERP traces only (no Topomaps).")
            # Fallback: Plot only the butterfly trace without Topos
            try:
                fig_simple = evoked.plot(spatial_colors=False, show=False)
                out_file = out_dir / f"pair-{pair_id:02d}_p{player_num}_{step_name}_erp_trace.png"
                fig_simple.savefig(out_file)
                plt.close(fig_simple)
                print(f"    [PLOT] Saved (Simple): {out_file.name}")
            except:
                print("    [WARN] Even simple ERP plot failed.")
        else:
            print(f"    [WARN] ERP Plot failed: {e}")
            
    except Exception as e:
        print(f"    [WARN] ERP Plot failed: {e}")

    # 2. Trial Consistency (Image Plot)
    try:
        # Pick a channel that likely exists
        pick = epochs.ch_names[0]
        preferred = ['A1', 'A32', 'Oz', 'Pz', '1-A1', '1-A32']
        for p in preferred:
            if p in epochs.ch_names:
                pick = p
                break
                
        fig_im = epochs.plot_image(picks=pick, combine=None, title=f"Trial Consistency ({pick})", show=False)
        if isinstance(fig_im, list): fig_im = fig_im[0]
        
        out_file_im = out_dir / f"pair-{pair_id:02d}_p{player_num}_{step_name}_trials.png"
        fig_im.savefig(out_file_im)
        plt.close(fig_im)
        print(f"    [PLOT] Saved: {out_file_im.name}")
    except Exception as e:
        print(f"    [WARN] Trial Plot failed: {e}")