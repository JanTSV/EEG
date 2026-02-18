import yaml
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import resample_poly

# --- HELPER FUNCTIONS (Validated Logic) ---

def interpolate_fieldtrip_style(epochs, bad_channels, n_neighbors=5):
    """
    Replicates FieldTrip's 'weighted' neighbor interpolation (Inverse Distance Weighting).
    Validated in Debug Step 3.
    """
    if not bad_channels:
        return epochs

    print(f"   -> Custom Interpolation (IDW) for: {bad_channels}")
    data = epochs.get_data().copy()
    info = epochs.info
    ch_names = info['ch_names']
    
    # Get 3D positions
    positions = np.array([ch['loc'][:3] for ch in info['chs']])
    
    # Identify good channels
    good_indices = [i for i, ch in enumerate(ch_names) if ch not in bad_channels]
    
    for bad in bad_channels:
        if bad not in ch_names:
            print(f"WARNUNG: Bad channel {bad} nicht gefunden (Mapping issue?)")
            continue
            
        bad_idx = ch_names.index(bad)
        bad_pos = positions[bad_idx]
        
        # Calculate distances to all good channels
        dists = np.linalg.norm(positions[good_indices] - bad_pos, axis=1)
        
        # Find nearest N neighbors
        sorted_idx = np.argsort(dists)
        closest_indices_local = sorted_idx[:n_neighbors]
        closest_indices_global = [good_indices[i] for i in closest_indices_local]
        closest_dists = dists[closest_indices_local]
        
        # Weights: Inverse Distance
        weights = 1.0 / (closest_dists + 1e-12)
        weights /= np.sum(weights)
        
        # Weighted Average
        neighbor_data = data[:, closest_indices_global, :]
        weights_bc = weights[np.newaxis, :, np.newaxis]
        interpolated_trace = np.sum(neighbor_data * weights_bc, axis=1)
        
        data[:, bad_idx, :] = interpolated_trace
        
    # Write back to Epochs object
    epochs_interp = mne.EpochsArray(data, info, events=epochs.events, 
                                    tmin=epochs.tmin, event_id=epochs.event_id, 
                                    verbose='error')
    epochs_interp.info['bads'] = [] # Clear bads as they are fixed
    return epochs_interp

def resample_polyphase(epochs, target_fs, pad='mean'):
    """
    Replicates MATLAB's 'resample' (Signal Processing Toolbox).
    Uses Polyphase Filter + Padding. Validated in Debug Step 2.
    """
    print(f"   -> Custom Resampling (Polyphase) to {target_fs}Hz")
    current_fs = epochs.info['sfreq']
    down = int(current_fs / target_fs)
    up = 1
    
    data = epochs.get_data()
    # Apply scipy resample_poly
    # axis=2 is time for (epochs, channels, time)
    data_resampled = resample_poly(data, up, down, axis=2, padtype=pad, window=('kaiser', 5.0))
    
    # Create new Info object
    info = epochs.info.copy()
    
    # --- FIX: Unlock Info to change sfreq manually ---
    with info._unlock():
        info['sfreq'] = target_fs
    # -------------------------------------------------
    
    # Re-create Epochs
    # Note: tmin/tmax might need slight adjustment due to rounding, but for MNE array import ok
    return mne.EpochsArray(data_resampled, info, events=epochs.events, 
                           tmin=epochs.tmin, event_id=epochs.event_id, 
                           verbose='error')

# --- MAIN PIPELINE CLASS ---

class EEGPipeline:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.cfg = yaml.safe_load(f)
        
        self.bids_root = Path(self.cfg['paths']['bids_root'])
        self.out_dir = Path(self.cfg['paths']['output_dir'])
        self.report_dir = Path(self.cfg['paths']['report_dir'])
        
        # Ensure dirs exist
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        
        self._check_guardrails()

    def _check_guardrails(self):
        """Ensures we don't accidentally run un-validated features."""
        gr = self.cfg.get('guardrails', {})
        if gr.get('allow_highpass') or gr.get('allow_lowpass'):
            raise NotImplementedError("Filtering is NOT enabled/validated in this pipeline version.")
        if gr.get('allow_rereference'):
            raise NotImplementedError("Re-referencing (CAR) is NOT enabled/validated.")

        self.allow_ica = gr.get('allow_ica', False)

    def get_subjects(self):
        return self.cfg['subjects']['include']

    def get_bad_channels(self, sub_id):
        """Reads participants.tsv to find bad channels."""
        tsv_path = self.bids_root / "participants.tsv"
        df = pd.read_csv(tsv_path, sep='\t')
        
        # Format sub_id string (e.g. 1 -> 'sub-01')
        sub_str = f"sub-{sub_id:02d}"
        
        row = df[df['participant_id'] == sub_str]
        if row.empty:
            return []
        
        # Assuming Player 1 is columns 6/7 (index 5/6) or named explicitly
        try:
            # Fallback: Index (wie MATLAB) -> Column 6 (0-based index 5)
            # Adjust if necessary based on your TSV structure
            bad_str = row.iloc[0, 5] 
        except Exception as e:
            print(f"   WARNING: Could not read bad channels for {sub_str}: {e}")
            return []

        if pd.isna(bad_str) or bad_str == 'n/a':
            return []
            
        return [ch.strip() for ch in str(bad_str).split(',')]

    def run_subject(self, sub_id):
        sub_str = f"sub-{sub_id:02d}"
        print(f"\n==========================================")
        print(f"PROCESSING: {sub_str}")
        print(f"==========================================")
        
        # 1. PATHS
        eeg_dir = self.bids_root / sub_str / "eeg"
        raw_path = eeg_dir / f"{sub_str}_task-{self.cfg['params']['task_name']}_eeg.bdf"
        events_path = eeg_dir / f"{sub_str}_task-{self.cfg['params']['task_name']}_events.tsv"
        
        if not raw_path.exists():
            print(f"   SKIP: File not found {raw_path}")
            return

        # 2. LOAD RAW
        print("   Loading Raw BDF...")
        raw = mne.io.read_raw_bdf(raw_path, preload=True, verbose='error')
        
        # Filter Channels (Player 1 only: 2-A / 2-B)
        # This matches MATLAB logic
        picks = [ch for ch in raw.ch_names if ('2-A' in ch) or ('2-B' in ch)]
        
        # --- FIX: Use .pick() instead of legacy pick_channels() ---
        raw.pick(picks)
        # ----------------------------------------------------------
        
        # 3. MAPPING & MONTAGE (Critical Fix)
        print("   Applying Channel Mapping (BioSemi -> 10-20)...")
        mapping = self.cfg['channel_mapping']
        
        # Check if all channels are in mapping
        # Only rename channels that exist in our pick list
        safe_mapping = {k: v for k, v in mapping.items() if k in raw.ch_names}
        raw.rename_channels(safe_mapping)
        
        # Set Montage
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage, on_missing='warn')

        # 4. ICA
        if self.allow_ica:
            # Average reference (required for ICA)
            # TODO: Maybe not copy again
            raw_copy = raw.copy()
            raw_copy.set_eeg_reference("average", projection=False)

            # Interpolate bad channels
            bad_channels = self.get_bad_channels(sub_id)
            valid_bads = [b for b in bad_channels if b in raw_copy.ch_names]
            if valid_bads:
                print(f"   Interpolating bad channels before ICA: {valid_bads}")
                raw_copy.info["bads"] = valid_bads
                raw_copy.interpolate_bads(reset_bads=True, verbose='error')
            else:
                print("   No bad channels to interpolate.")
            
            # Downsampling for speedup
            ica_sfreq = 256
            print(f"   Downsampling to {ica_sfreq} Hz for ICA...")
            raw_copy.resample(ica_sfreq)

            # High-pass for ICA, see runtime warning:
            # RuntimeWarning: The data has not been high-pass filtered. For good ICA performance, it should be high-pass filtered (e.g., with a 1.0 Hz lower bound) before fitting ICA.
            raw_copy = raw_copy.filter(l_freq=1.0, h_freq=None)

            ica = mne.preprocessing.ICA(
                n_components=20,
                method="fastica",
                random_state=42,
                max_iter="auto"
            )
            ica.fit(raw_copy, picks="eeg")

            # Auto-detect EOG artifacts: uses frontal channels as proxy for eye activity
            # Threshold=3.0: standard; higher = stricter (fewer components excluded)
            try:
                # Use frontal EEG channels as proxies for EOG
                frontal_chs = [ch for ch in ['Fp1', 'Fp2', 'AF7', 'AF8'] if ch in raw_copy.ch_names]

                # Find bad EOG components
                eog_indices, eog_scores = ica.find_bads_eog(raw_copy, ch_name=frontal_chs, threshold=3.0)
                ica.exclude = eog_indices
                if eog_indices:
                    print(f"    ICA detected {len(eog_indices)} EOG component(s): {eog_indices}")
            except RuntimeError:
                print("    No EOG channels found; skipping automatic EOG detection")

            # Apply ICA: removes marked artifact components from raw signal (in-place)
            if ica.exclude:
                ica.apply(raw)  # IMPORTANT: apply to raw now
                print(f"    ICA applied, removed {len(ica.exclude)} component(s)")
            else:
                print("    No artifacts detected by ICA")

        # 5. EPOCHING
        print("   Epoching...")
        events_df = pd.read_csv(events_path, sep='\t')
        
        # Create MNE events array from onset_sample
        onsets = events_df['onset_sample'].values
        events = np.column_stack((onsets, 
                                  np.zeros(len(onsets), dtype=int), 
                                  np.ones(len(onsets), dtype=int))) # Event ID 1
        
        tmin = self.cfg['params']['tmin']
        tmax = self.cfg['params']['tmax']
        
        # Load Raw to uV (MATLAB compatibility)
        epochs = mne.Epochs(raw, events, tmin=tmin, tmax=tmax, 
                            baseline=None, preload=True, verbose='error')
        
        # 6. INTERPOLATION
        bads_1020 = self.get_bad_channels(sub_id)
        # Ensure bad names match current 10-20 names
        valid_bads = [b for b in bads_1020 if b in epochs.ch_names]
        
        if valid_bads:
            print(f"   Interpolating Bad Channels: {valid_bads}")
            epochs.info['bads'] = valid_bads
            
            method = self.cfg['params']['interp_method']
            if method == 'fieldtrip_nearest':
                # Custom Function
                epochs = interpolate_fieldtrip_style(epochs, valid_bads, 
                                                     n_neighbors=self.cfg['params']['neighbor_count'])
            elif method == 'spline':
                # MNE Standard
                epochs.interpolate_bads(reset_bads=True, verbose='error')
            else:
                raise ValueError(f"Unknown interp method: {method}")
        else:
            print("   No bad channels to interpolate.")

        # 7. RESAMPLING
        print("   Resampling...")
        if self.cfg['params']['resample_method'] == 'polyphase':
            epochs = resample_polyphase(epochs, self.cfg['params']['target_fs'], 
                                        pad=self.cfg['params']['resample_pad'])
        else:
            # Fallback / Guardrail check failed
            raise NotImplementedError("Only polyphase resampling is enabled.")

        # 8. QC REPORT
        self._generate_report(epochs, sub_str)

        # 9. SAVE
        out_fname = self.out_dir / f"{sub_str}_task-RPS_desc-preproc_eeg.fif"
        print(f"   Saving to {out_fname}...")
        epochs.save(out_fname, overwrite=True, verbose='error')

    def _generate_report(self, epochs, sub_str):
        """Generates a simple QC plot (Butterfly + Image)."""
        print("   Generating QC Report...")
        fig, ax = plt.subplots(2, 1, figsize=(10, 8))
        
        # Butterfly Plot (Average over trials)
        evoked = epochs.average()
        # Scale to uV for plotting
        times = evoked.times
        data_uv = evoked.get_data() * 1e6
        
        ax[0].plot(times, data_uv.T, color='black', alpha=0.3, linewidth=0.5)
        ax[0].set_title(f"{sub_str}: Butterfly Plot (All Channels)")
        ax[0].set_ylabel("Amplitude (uV)")
        ax[0].grid(True)
        
        # Global Field Power (GFP)
        gfp = np.std(data_uv, axis=0)
        ax[0].plot(times, gfp, color='green', linewidth=1.5, label='GFP')
        ax[0].legend()
        
        # Image Plot (Channels x Time)
        im = ax[1].imshow(data_uv, aspect='auto', origin='lower', 
                          extent=[times[0], times[-1], 0, len(epochs.ch_names)],
                          cmap='RdBu_r', vmin=-30, vmax=30)
        ax[1].set_title("Channel Activity (Heatmap)")
        ax[1].set_xlabel("Time (s)")
        ax[1].set_ylabel("Channel Index")
        plt.colorbar(im, ax=ax[1], label="uV")
        
        plt.tight_layout()
        plot_path = self.report_dir / f"{sub_str}_qc.png"
        plt.savefig(plot_path)
        plt.close(fig)
        print(f"   Report saved: {plot_path}")

    def run(self):
        subs = self.get_subjects()
        for sub in subs:
            try:
                self.run_subject(sub)
            except Exception as e:
                print(f"ERROR processing sub-{sub}: {e}")
                # Continue with next subject
                continue

if __name__ == "__main__":
    # Config Path
    config_file = "config_preprocessing.yaml"
    
    pipeline = EEGPipeline(config_file)
    pipeline.run()