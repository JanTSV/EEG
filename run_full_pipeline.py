#!/usr/bin/env python3
"""
Full EEG Analysis Pipeline: Preprocess → Decode → Aggregate → Plot

This script orchestrates the complete workflow:
1. Preprocessing: Load raw BDF, apply filters/ICA, save epochs
2. Decoding: Run ML models on all pairs/players
3. Aggregation: Combine results across pairs
4. Plotting: Generate group-level visualizations

Usage:
    python run_full_pipeline.py [--skip-preprocessing] [--skip-decoding] [--skip-aggregation] [--skip-plotting]
"""

import sys
import subprocess
import argparse
import yaml
from pathlib import Path

def load_config(config_name):
    """Load YAML config from code directory."""
    cfg_path = Path(__file__).resolve().parent / "code" / config_name
    with cfg_path.open("r") as f:
        return yaml.safe_load(f)

def run_command(cmd, description):
    """Run a shell command and handle errors."""
    print(f"\n{'='*70}")
    print(f"  {description}")
    print(f"{'='*70}\n")
    
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"\n❌ ERROR: {description} failed with exit code {result.returncode}")
        return False
    print(f"\n✓ {description} completed successfully")
    return True

def step_1_preprocessing(output_dir="derivatives"):
    """Run preprocessing on all pairs."""
    print("\n" + "="*70)
    print("  STEP 1: PREPROCESSING (All Pairs)")
    print("="*70)
    
    base_dir = Path(__file__).resolve().parent
    cmd = f"cd {base_dir} && uv run code/preprocessing.py --out {output_dir}"
    return run_command(cmd, "Preprocessing Pipeline")

def step_2_decoding(output_dir="derivatives"):
    """Run decoding on all pairs."""
    print("\n" + "="*70)
    print("  STEP 2: DECODING (All Pairs)")
    print("="*70)
    
    # Load decoding config to get pair/player info
    cfg = load_config("decoding_config.yaml")
    
    # Extract valid pairs from decoding config
    valid_pairs = []
    for pair_range in cfg["pair_player"]["pair_ranges"]:
        valid_pairs.extend(range(pair_range[0], pair_range[1] + 1))
    
    # Filter by target_pairs if specified
    target_pairs = cfg["pair_player"].get("target_pairs")
    if target_pairs:
        valid_pairs = [p for p in valid_pairs if p in target_pairs]
    
    print(f"\nDecoding for pairs: {valid_pairs}")
    print(f"Players: {cfg['pair_player']['players']}")
    
    # Run decoding once - it will handle all pairs internally
    base_dir = Path(__file__).resolve().parent
    cmd = f"cd {base_dir} && uv run code/decoding.py --out {output_dir}"
    return run_command(cmd, "Decoding Pipeline")

def step_3_aggregation(output_dir="derivatives"):
    """Run aggregation across all pairs."""
    print("\n" + "="*70)
    print("  STEP 3: AGGREGATION (Group-Level)")
    print("="*70)
    
    # Load decoding config to get pair ranges (should match preprocessing)
    cfg = load_config("decoding_config.yaml")
    
    # Extract valid pairs from decoding config
    valid_pairs = []
    for pair_range in cfg["pair_player"]["pair_ranges"]:
        valid_pairs.extend(range(pair_range[0], pair_range[1] + 1))
    
    # Filter by target_pairs if specified
    target_pairs = cfg["pair_player"].get("target_pairs")
    if target_pairs:
        valid_pairs = [p for p in valid_pairs if p in target_pairs]
    
    pairs_str = " ".join(map(str, valid_pairs))
    
    base_dir = Path(__file__).resolve().parent
    cmd = f"cd {base_dir} && uv run code/aggregate_decoding.py --pairs {pairs_str} --targets current_self current_other previous_self previous_other --out {output_dir}"
    return run_command(cmd, "Aggregation Pipeline")

def step_4_plotting():
    """Run plotting."""
    print("\n" + "="*70)
    print("  STEP 4: PLOTTING (Group Results)")
    print("="*70)
    
    print("\n⚠️  plot_results.py is for single-subject plotting.")
    print("    For group-level plots, aggregate_decoding.py generates them automatically.")
    print("    Check: data/derivatives/decoding_results/group_level_plots/")
    
    return True

def main():
    parser = argparse.ArgumentParser(
        description="Run complete EEG analysis pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_full_pipeline.py                 # Run all steps
  python run_full_pipeline.py --skip-preprocessing  # Skip preprocessing
  python run_full_pipeline.py --out results  # Save results to data/results
  python run_full_pipeline.py --skip-decoding --skip-aggregation  # Only preprocess
        """
    )
    
    parser.add_argument("--skip-preprocessing", action="store_true", help="Skip preprocessing step")
    parser.add_argument("--skip-decoding", action="store_true", help="Skip decoding step")
    parser.add_argument("--skip-aggregation", action="store_true", help="Skip aggregation step")
    parser.add_argument("--skip-plotting", action="store_true", help="Skip plotting step")
    parser.add_argument("--out", type=str, default="derivatives", help="Output directory (relative to data/); default: data/derivatives")
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("  EEG ANALYSIS PIPELINE: Preprocess → Decode → Aggregate → Plot")
    print("="*70)
    
    steps_to_run = {
        "preprocessing": not args.skip_preprocessing,
        "decoding": not args.skip_decoding,
        "aggregation": not args.skip_aggregation,
        "plotting": not args.skip_plotting,
    }
    
    output_dir = args.out
    
    print("\nSteps to run:")
    for step, should_run in steps_to_run.items():
        status = "✓" if should_run else "✗"
        print(f"  {status} {step.capitalize()}")
    print(f"\nOutput directory: data/{output_dir}")
    
    # Run pipeline
    if steps_to_run["preprocessing"] and not step_1_preprocessing(output_dir):
        print("\n❌ Pipeline stopped: Preprocessing failed")
        return 1
    
    if steps_to_run["decoding"] and not step_2_decoding(output_dir):
        print("\n⚠️  Pipeline continuing: Some decoding jobs failed, but proceeding to aggregation")
    
    if steps_to_run["aggregation"] and not step_3_aggregation(output_dir):
        print("\n❌ Pipeline stopped: Aggregation failed")
        return 1
    
    if steps_to_run["plotting"] and not step_4_plotting():
        print("\n❌ Pipeline stopped: Plotting failed")
        return 1
    
    print("\n" + "="*70)
    print("  ✓ PIPELINE COMPLETED SUCCESSFULLY")
    print("="*70)
    print("\nResults location:")
    print(f"  - Preprocessed epochs: data/{output_dir}/pair-XX_player-Y_task-RPS_eeg-epo.fif")
    print(f"  - Decoding results: data/{output_dir}/decoding_results/pair-XX_player-Y/target_name/")
    print(f"  - Group plots: data/{output_dir}/decoding_results/group_level_plots/")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
