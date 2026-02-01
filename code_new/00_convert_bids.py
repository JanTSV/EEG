"""
00_convert_bids.py (MEMORY OPTIMIZED)
-------------------------------------
Splits the dual-subject .bdf file into two .fif files.
Change: preload=False to avoid Out-Of-Memory crashes on Mac M1.
"""

import yaml
import mne
import numpy as np
from pathlib import Path
import gc # Garbage Collection

def load_config():
    return yaml.safe_load(open("code_new/config.yaml"))

CONFIG = load_config()

def run_conversion():
    data_root = Path(CONFIG['paths']['data_root'])
    out_dir = Path(CONFIG['paths']['output_raw'])
    out_dir.mkdir(parents=True, exist_ok=True)
    
    p1_prefix = CONFIG['subjects'].get('channel_prefix_p1', '1-')
    p2_prefix = CONFIG['subjects'].get('channel_prefix_p2', '2-')
    
    if CONFIG['subjects']['run_mode'] == 'single':
        pairs = [CONFIG['subjects']['single_pair_id']]
    else:
        all_p = range(CONFIG['subjects']['pair_range'][0], CONFIG['subjects']['pair_range'][1] + 1)
        pairs = [p for p in all_p if p not in CONFIG['subjects']['exclude_pairs']]
        
    print(f"--- STARTING CONVERSION for {len(pairs)} Pair(s) ---")
    
    for pair_id in pairs:
        print(f"\nProcessing Pair {pair_id}...")
        sub_str = f"sub-{pair_id:02d}"
        eeg_dir = data_root / sub_str / 'eeg'
        
        # Find BDF (ignoring ._ files)
        bdf_files = [f for f in eeg_dir.glob("*.bdf") if not f.name.startswith("._")]
        
        if not bdf_files:
            print(f"  [SKIP] No .bdf file found in {eeg_dir}")
            continue
            
        raw_path = bdf_files[0]
        
        try:
            # MEMORY FIX: preload=False loads only metadata, not the heavy data
            raw = mne.io.read_raw_bdf(raw_path, preload=False, verbose='error')
        except Exception as e:
            print(f"  [ERROR] Loading {raw_path.name}: {e}")
            continue
            
        # --- PLAYER 1 ---
        picks_p1 = [ch for ch in raw.ch_names if ch.startswith(p1_prefix)]
        if picks_p1:
            # Note: picking on non-preloaded data works fine
            raw_p1 = raw.copy().pick(picks_p1)
            # Remove prefix
            rename_dict = {ch: ch.replace(p1_prefix, '') for ch in raw_p1.ch_names}
            raw_p1.rename_channels(rename_dict)
            
            out_name = out_dir / f"pair-{pair_id:02d}_player-1_raw.fif"
            # Save handles the streaming from disk to disk
            raw_p1.save(out_name, overwrite=True, verbose=False)
            print(f"  -> Player 1: Saved {len(picks_p1)} chans")
            
            del raw_p1 # Free memory explicitly

        # --- PLAYER 2 ---
        picks_p2 = [ch for ch in raw.ch_names if ch.startswith(p2_prefix)]
        if picks_p2:
            raw_p2 = raw.copy().pick(picks_p2)
            rename_dict = {ch: ch.replace(p2_prefix, '') for ch in raw_p2.ch_names}
            raw_p2.rename_channels(rename_dict)
            
            out_name = out_dir / f"pair-{pair_id:02d}_player-2_raw.fif"
            raw_p2.save(out_name, overwrite=True, verbose=False)
            print(f"  -> Player 2: Saved {len(picks_p2)} chans")
            
            del raw_p2

        # Clean up big raw object
        del raw
        gc.collect() 

if __name__ == "__main__":
    run_conversion()