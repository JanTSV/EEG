## Research Questions
Can multivariate pattern analysis/decoding on hyperscanning EEG data reveal:
1. Information about the player's own decision?
2. Information about the opponent's decision?
3. Information about previous outcomes (history bias)? & crucially: Do winners and losers encode this information differently? (Finding: Losers encode previous trials more, which is a suboptimal strategy)

## Methodology
- Multivariate Decoding (LDA)
- No Filtering (Pre-processing) 

---

## Step-by-step analysis matlab
### step1_preprocessing.m
- Mismatch paper & code:
  - paper: "First, we re-referenced to common average... then interpolated."
  - code: Loads -> Interpolates -> Resamples. It does NOT re-reference in this script.
  - Hypothesis: They likely do the re-referencing in step2 or on the fly during the decoding epochs. If we re-reference in Python now, we will get different values than their intermediate files. We will make this a configurable toggle.
1. Subject Mapping (The "Swap"):
   - Logic: The EEG hardware labeled players as 1 and 2, but the behavioral data/demographics labeled them oppositely.
   - Code: chan_idx selects channels 2-A... for Player 1 and 1-A... for Player 2.
   - Python Action: We need a strict channel mapping dictionary in our config.
2. Epoching (Rough):
    - Logic: Cut -0.2s to +5.0s around the 'Decision' trigger.
    - Code: TRL matrix calculation based on events.tsv.
3. Channel Naming:
   - Logic: Renames BioSemi channels (e.g., 1-A1) to standard 10-20 system (e.g., Fp1).
   - Code: Uses biosemi64.lay.
   - Python Action: We need a standard montage file or dictionary mapping BioSemi -> 10-20.
4. Bad Channel Interpolation:
   - Logic: Reads bad channels from participants.tsv.
   - Method: ft_channelrepair using 'distance' method (0.5 cm).
   - Python Action: MNE's interpolate_bads usually defaults to spherical splines. To replicate exactly, we might need to check if MNE supports simple distance-weighted averaging or accept that spherical spline is scientifically superior (and document why we changed it).
5. Downsampling:
   - Logic: 2048 Hz -> 256 Hz.
   - Code: ft_resampledata.
   - Python Action: raw.resample(256). Note: MATLAB's resample often includes an anti-aliasing filter by default. MNE does too. We must check the filter length to ensure we don't violate the "no filtering" rule (though anti-aliasing is usually acceptable/necessary).
6. Saving:
   - Saves as .mat files.


---

## PREPROCESSING 

| Category     | Step.                              | Paper ?   | Matlab ?| Python ?| notes                                      |
| ---          | ----                               | --        | --     | --     | -----------                                  |
| 0. Setup     | Channel Mapping/ Renaming          | implicit  | yes    | yes    | format handling for BioSemi                  |
| 0. Setup     | Player Swapping.                   | no.       | yes.   | yes    |                                              |
| 1. Signal    | Re-referencing (CAR)               | yes       | no.    | yes    | **MISMATCH**                                 |
| 2. Epoching  | Trial segmentation.                | yes       | yes    | yes    |                                              |
| 3. Cleaning  | Identify bad samples               | yes       | yes    | yes    |                                              |
| 3. Cleaning  | Interpolate bad channels           | yes       | yes    | yes    |                                              |
| 4. Signal    | Filtering.                         | no        | no     | no     |                                              |
| 5. Resample  | Downsampling                       | yes       | yes    | yes    |                                              |
| 6. Finish    | Baseline Correction                | yes       | yes    | yes    |                                              |
| 7. Saving.   | Export.                            | -         | yes    | yes    |                                              |

---

## DECODING 

| Category     | Step.                              | Paper ?   | Matlab ?| Python ?| notes                                      |
| ---          | ----                               | --        | --     | --     | -----------                                  |
| 0. Setup     | Define Target.                     | yes.      | yes    | yes    | format handling for BioSemi                  |
| 0. Setup     | clean trials.                      | yes.      | yes    | yes    | format handling for BioSemi                  |
| 1. Features  | Time Binning                       | yes       | yes    | yes    | **MISMATCH**: *paper: 250ms bins, matlab: 20ms* |
| 2. Strategy  | cross-validation                   | yes       | yes    | yes    |                                              |
| 2. Strategy  | balancing.                         | implicit  | yes    | yes    |                                              |
| 3. Features  | pseudo-trials                      | yes.      | yes    | yes    |                                              |
| 4. Model.    | classifier.                        | yes.      | yes    | yes    |                                              |
| 5. Loop      | iterations                         | yes.      | yes    | yes    |                                              |
| 6. Output    | metric.                            | yes.      | yes    | yes    |                                              |

---

## MARKOV CHAIN

| Category  | Step.        | Paper ?            | Matlab ?       | Python ?       | notes        |
| --------- | ------------ | ------------------ | -------------- | -------------- | ------------ |
| 0. Setup  | cleaning.    | not noted          | yes            | yes            | class 0/ nan |
| 1. model  | markov order | markov chain model | order 1        | order 1        |              |
| 2. params | window size  | 5:100              | 0:100          | 5:100          |              |
| 3. Metric | accuracy     | pred vs actual     | pred vs actual | pred vs actual |              |
