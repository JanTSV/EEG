"""
Run the EEG preprocessing pipeline.

This script converts the raw BioSemi recordings into analysis-ready epoched
data. It follows the preprocessing choices used throughout the project and is
written to be reproducible, configuration-driven, and easy to inspect.

The pipeline performs the following main steps:

- load the raw BDF recording for each subject,
- select the relevant EEG channels,
- map the BioSemi channel names to the standard 10-20 labels,
- apply optional notch and band-pass filtering,
- optionally run ICA-based artifact removal,
- epoch the data around the configured events,
- interpolate bad channels,
- resample the epochs to the analysis sampling rate,
- and save both the cleaned epochs and quality-control figures.

The helper functions in this file intentionally mirror the MATLAB-inspired
processing logic that was validated during debugging.
"""

import yaml
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import resample_poly

# --- HELPER FUNCTIONS ---

def interpolate_fieldtrip_style(epochs, bad_channels, n_neighbors=5):
    """
    Interpolate bad channels using a FieldTrip-style inverse-distance scheme.

    The function mimics the weighted nearest-neighbor interpolation that was
    validated during debugging. For each bad channel, the ``n_neighbors``
    closest good channels are found in sensor space and combined with inverse
    distance weights.

    Parameters
    ----------
    epochs : mne.Epochs
        Epoched EEG data to repair.
    bad_channels : list[str]
        Channel names that should be interpolated.
    n_neighbors : int, default=5
        Number of nearest good channels used in the weighted average.

    Returns
    -------
    mne.Epochs
        A new epochs object with the bad channels replaced by interpolated
        traces.
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
    Resample epochs with a polyphase filter in a MATLAB-compatible manner.

    The implementation mirrors MATLAB's ``resample`` behavior using
    :func:`scipy.signal.resample_poly`. This was validated during debugging to
    keep the final preprocessing output aligned with the reference pipeline.

    Parameters
    ----------
    epochs : mne.Epochs
        Epoched data to resample.
    target_fs : float
        Desired output sampling rate in Hz.
    pad : str, default='mean'
        Padding strategy forwarded to :func:`scipy.signal.resample_poly`.

    Returns
    -------
    mne.Epochs
        Resampled epochs object.
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
    
    with info._unlock():
        info['sfreq'] = target_fs
    
    # Re-create Epochs
    # Note: tmin/tmax might need slight adjustment due to rounding, but for MNE array import ok
    return mne.EpochsArray(data_resampled, info, events=epochs.events, 
                           tmin=epochs.tmin, event_id=epochs.event_id, 
                           verbose='error')

# --- MAIN PIPELINE CLASS ---

class EEGPipeline:
    """End-to-end preprocessing pipeline for one or more EEG subjects."""

    def __init__(self, config_path):
        """
        Load the preprocessing configuration and prepare output directories.

        Parameters
        ----------
        config_path : str or pathlib.Path
            Path to the YAML configuration file.
        """
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
        """
        Read feature-flag style settings that protect unvalidated options.

        At present this is mainly used to gate ICA.
        """
        gr = self.cfg.get('guardrails', {})
        self.allow_ica = gr.get('allow_ica', False)

    def _save_spectrum_report(self, signal_obj, sub_str, stage_label=""):
        """
        Save a frequency-domain quality-control figure.

        The plot summarizes the spectrum after the main cleaning steps and can
        be used to inspect whether the configured filters behaved as expected.

        Parameters
        ----------
        signal_obj : mne.io.BaseRaw | mne.Epochs
            Raw or epoched data object with ``compute_psd`` support.
        sub_str : str
            Subject identifier used in the output filename.
        stage_label : str, default=""
            Optional label appended to the figure title.
        """
        params = self.cfg.get('params', {})
        spec_cfg = params.get('filter_spectrum_report', {})

        # Support both bool and dict style config
        if isinstance(spec_cfg, bool):
            enabled = spec_cfg
            fmin = 0.0
            fmax = min(120.0, signal_obj.info['sfreq'] / 2.0)
        else:
            enabled = spec_cfg.get('enabled', False)
            fmin = float(spec_cfg.get('fmin', 0.0))
            fmax = float(spec_cfg.get('fmax', min(120.0, signal_obj.info['sfreq'] / 2.0)))

        if not enabled:
            return

        if fmax <= fmin:
            raise ValueError(f"Invalid spectrum range: fmin ({fmin}) must be < fmax ({fmax})")

        print(f"   Generating spectrum plot ({fmin}-{fmax} Hz)...")

        psd = signal_obj.compute_psd(method='welch', picks='eeg', fmin=fmin, fmax=fmax, verbose='error')
        psd_data, freqs = psd.get_data(return_freqs=True)

        # Aggregate across all non-frequency dimensions.
        # Raw PSD shape is typically (channels, freqs),
        # Epochs PSD shape is typically (epochs, channels, freqs).
        reduce_axes = tuple(range(psd_data.ndim - 1))
        mean_db = 10 * np.log10(np.mean(psd_data, axis=reduce_axes) + np.finfo(float).eps)
        median_db = 10 * np.log10(np.median(psd_data, axis=reduce_axes) + np.finfo(float).eps)

        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        ax.plot(freqs, mean_db, color='tab:blue', linewidth=1.5, label='Mean PSD (dB)')
        ax.plot(freqs, median_db, color='tab:orange', linewidth=1.2, alpha=0.8, label='Median PSD (dB)')

        # Optional guides for configured filters
        highpass_hz = params.get('highpass_hz', None)
        lowpass_hz = params.get('lowpass_hz', None)
        notch_freqs = params.get('notch_freqs', []) or []

        if highpass_hz is not None:
            ax.axvline(float(highpass_hz), color='green', linestyle='--', linewidth=1, label='High-pass')
        if lowpass_hz is not None:
            ax.axvline(float(lowpass_hz), color='red', linestyle='--', linewidth=1, label='Low-pass')
        for i, nf in enumerate(notch_freqs):
            ax.axvline(float(nf), color='purple', linestyle=':', linewidth=1,
                       label='Notch' if i == 0 else None)

        title = f"{sub_str}: EEG Spectrum"
        if stage_label:
            title += f" ({stage_label})"
        ax.set_title(title)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Power Spectral Density (dB)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        plt.tight_layout()

        spec_path = self.report_dir / f"{sub_str}_spectrum.png"
        plt.savefig(spec_path)
        plt.close(fig)
        print(f"   Spectrum report saved: {spec_path}")

    def _apply_configurable_filters(self, raw):
        """
        Apply the optional filtering stage configured in the YAML file.

        This includes notch filtering for line noise and its harmonics as well
        as optional high-pass and/or low-pass filtering.

        Returns
        -------
        bool
            ``True`` if at least one filter was applied, otherwise ``False``.
        """
        params = self.cfg.get('params', {})

        highpass_hz = params.get('highpass_hz', None)
        lowpass_hz = params.get('lowpass_hz', None)
        notch_freqs = params.get('notch_freqs', [])
        notch_widths = params.get('notch_widths', None)

        # Normalize YAML null values
        if highpass_hz is not None:
            highpass_hz = float(highpass_hz)
        if lowpass_hz is not None:
            lowpass_hz = float(lowpass_hz)
        if notch_freqs is None:
            notch_freqs = []

        if (highpass_hz is not None) and (lowpass_hz is not None) and (highpass_hz >= lowpass_hz):
            raise ValueError(
                f"Invalid filter setup: highpass_hz ({highpass_hz}) must be < lowpass_hz ({lowpass_hz})."
            )

        filters_applied = False

        # 1) Notch first to suppress line-noise peaks and harmonics before broader filtering / ICA.
        if len(notch_freqs) > 0:
            print(f"   Applying notch filter at: {notch_freqs} Hz")
            raw.notch_filter(
                freqs=np.array(notch_freqs, dtype=float),
                picks='eeg',
                notch_widths=notch_widths,
                verbose='error'
            )
            filters_applied = True

        # 2) High/low-pass (band-pass, high-pass only, or low-pass only)
        if (highpass_hz is not None) or (lowpass_hz is not None):
            print(f"   Applying frequency filter: l_freq={highpass_hz}, h_freq={lowpass_hz}")
            raw.filter(
                l_freq=highpass_hz,
                h_freq=lowpass_hz,
                picks='eeg',
                verbose='error'
            )
            filters_applied = True

        return filters_applied

    def get_subjects(self):
        """Return the list of subject IDs that should be processed."""
        return self.cfg['subjects']['include']

    def get_bad_channels(self, sub_id):
        """
        Read the subject-specific bad-channel list from participants.tsv.

        Parameters
        ----------
        sub_id : int
            Numerical subject identifier.

        Returns
        -------
        list[str]
            Bad channel names, or an empty list if no entry is available.
        """
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
            bad_str = row.iloc[0, 5] 
        except Exception as e:
            print(f"   WARNING: Could not read bad channels for {sub_str}: {e}")
            return []

        if pd.isna(bad_str) or bad_str == 'n/a':
            return []
            
        return [ch.strip() for ch in str(bad_str).split(',')]

    def run_subject(self, sub_id):
        """
        Run the full preprocessing workflow for one subject.

        The method handles raw loading, channel mapping, optional filtering,
        ICA, epoching, interpolation, resampling, QC reporting, and final data
        export.
        """
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
        
        raw.pick(picks)
        
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

        # 4. OPTIONAL FILTERING
        # Done on continuous raw before ICA/epoching/resampling to avoid introducing
        # edge artifacts at epoch boundaries and to suppress line-noise harmonics
        # before downsampling.
        self._apply_configurable_filters(raw)

        # 5. ICA
        if self.allow_ica:
            # Average reference (required for ICA)
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

            # Optional extra high-pass only for ICA fitting stability.
            # If main preprocessing already used >= this cutoff, skip duplicate filtering.
            ica_fit_highpass_hz = float(self.cfg['params'].get('ica_fit_highpass_hz', 1.0))
            preproc_highpass_hz = self.cfg['params'].get('highpass_hz', None)
            if (preproc_highpass_hz is None) or (float(preproc_highpass_hz) < ica_fit_highpass_hz):
                print(f"   Additional ICA-fit high-pass at {ica_fit_highpass_hz} Hz")
                raw_copy = raw_copy.filter(l_freq=ica_fit_highpass_hz, h_freq=None)

            ica = mne.preprocessing.ICA(
                n_components=20,
                method="fastica",
                random_state=42,
                max_iter="auto"
            )
            ica.fit(raw_copy, picks="eeg")

            # Plot ICA topos
            print("   Plotting ICA topographies...")
            fig = ica.plot_components(inst=raw_copy, show=False)
            fig.savefig(self.report_dir / f"{sub_str}_ica_components.png")
            plt.close(fig)

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
                # Plot excluded topos
                print("   Plotting excluded ICA topos...")
                figs = ica.plot_properties(raw_copy, picks=ica.exclude, show=False)
                for i, f in enumerate(figs):
                    f.savefig(self.report_dir / f"{sub_str}_ica_bad_comp_{i}.png")
                    plt.close(f)

                ica.apply(raw)  # IMPORTANT: apply to raw now
                print(f"    ICA applied, removed {len(ica.exclude)} component(s)")
            else:
                print("    No artifacts detected by ICA")

        # 6. EPOCHING
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
        
        # 7. INTERPOLATION
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

        # 8. SPECTRUM QC (final cleaned data, before resampling)
        self._save_spectrum_report(epochs, sub_str, stage_label="post-interp")

        # 9. RESAMPLING
        print("   Resampling...")
        if self.cfg['params']['resample_method'] == 'polyphase':
            epochs = resample_polyphase(epochs, self.cfg['params']['target_fs'], 
                                        pad=self.cfg['params']['resample_pad'])
        else:
            # Fallback / Guardrail check failed
            raise NotImplementedError("Only polyphase resampling is enabled.")

        # 10. QC REPORT
        self._generate_report(epochs, sub_str)

        # 11. SAVE
        out_fname = self.out_dir / f"{sub_str}_task-RPS_desc-preproc_eeg.fif"
        print(f"   Saving to {out_fname}...")
        epochs.save(out_fname, overwrite=True, verbose='error')

    def _generate_report(self, epochs, sub_str):
        """
        Generate a compact time-domain quality-control figure.

        The report contains a butterfly plot, a global field power trace, and a
        channel-by-time heatmap to provide a quick visual inspection of the
        cleaned epochs.
        """
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
        """Process all configured subjects and continue past recoverable errors."""
        subs = self.get_subjects()
        for sub in subs:
            try:
                self.run_subject(sub)
            except Exception as e:
                print(f"ERROR processing sub-{sub}: {e}")
                # Continue with next subject
                continue

if __name__ == "__main__":
    # Use a script-relative config path so the pipeline stays portable.
    config_file = Path(__file__).resolve().parent / "config_preprocessing.yaml"
    
    pipeline = EEGPipeline(config_file)
    pipeline.run()
