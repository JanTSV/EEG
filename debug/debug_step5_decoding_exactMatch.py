"""
debug_step5c_exact_classifier.py
----------------------------------
Classifier identity check: feeds MATLAB's pre-built super-trials directly
into the Python LDA to isolate whether any remaining accuracy gap is caused
by the classifier itself or by upstream data differences.

By bypassing the Python averaging step and using MATLAB's super-trials as
input, this script answers one specific question:
  "Given identical input data, do the MATLAB and Python classifiers agree?"

Pipeline
--------
1. Load MATLAB super-trials (X, y, chunks) — produced by cosmo_average_samples.
2. Reconstruct the per-bin feature layout.
   Assumed CoSMoMVPA flattening order: [channels_bin0 | channels_bin1 | …]
   i.e. features 0..63 = bin 0, features 64..127 = bin 1, etc.
3. Run Leave-One-Chunk-Out (LOCO) cross-validation — matching CoSMoMVPA's
   default partitioning scheme.
4. Compare per-bin decoding accuracies against the MATLAB ground truth.
"""

import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score

# ── CONFIG ────────────────────────────────────────────────────────────────────
mat_supertrials_path = Path("originalCode/debug_step2c_supertrials.mat")
mat_results_path     = Path("originalCode/debug_step2b_results.mat")

# Expected number of EEG channels — used to reconstruct the bin layout
N_CHANS = 64
# ─────────────────────────────────────────────────────────────────────────────


def run_exact_match():
    print("─── START: debug_step5c (EXACT CLASSIFIER MATCH) ───\n")

    # ── 1. Load MATLAB super-trials ───────────────────────────────────────────
    # CoSMoMVPA exports a flat 2-D feature matrix after averaging.
    # Assumed layout: (n_samples, n_channels × n_bins), C-order block structure.
    try:
        mat      = scipy.io.loadmat(mat_supertrials_path, squeeze_me=True)
        X        = mat['super_trials']    # (n_samples, n_channels × n_bins)
        y        = mat['super_targets']   # (n_samples,)
        chunks   = mat['super_chunks']    # (n_samples,) — CoSMoMVPA chunk labels
        print(f"  Super-trial matrix shape : {X.shape}")
    except Exception as e:
        print(f"ERROR – could not load super-trials: {e}")
        return

    # ── 2. Load MATLAB ground-truth decoding accuracies ───────────────────────
    try:
        mat_res    = scipy.io.loadmat(mat_results_path, squeeze_me=True)
        dec_struct = mat_res['decoding_accuracy']

        # Handle variable struct shapes returned by scipy.io
        if dec_struct.shape in [(1, 4), (4,)]:
            res = dec_struct[0]
        else:
            res = dec_struct

        gt_acc = np.squeeze(res['samples'].item())
        print(f"  Ground-truth accuracy shape : {gt_acc.shape}")
    except Exception as e:
        print(f"ERROR – could not load MATLAB results: {e}")
        return

    # ── 3. Reconstruct per-bin feature layout ─────────────────────────────────
    # CoSMoMVPA's cosmo_interval_neighborhood with radius=0 (searchlight over
    # time) produces one feature block per time bin, each of size n_channels.
    # Assumed order: [channels_bin0 | channels_bin1 | … | channels_binN]
    n_samples, n_features_total = X.shape
    n_bins = n_features_total // N_CHANS

    if n_features_total % N_CHANS != 0:
        print(f"  WARNING: total feature count ({n_features_total}) is not "
              f"evenly divisible by N_CHANS ({N_CHANS}). "
              f"Check CoSMoMVPA's flattening order.")

    print(f"  Reshaping to ({n_samples}, {N_CHANS}, {n_bins})")

    # ── 4. Classifier ─────────────────────────────────────────────────────────
    # LDA with LSQR solver and automatic Ledoit-Wolf shrinkage — matches the
    # classifier used by CoSMoMVPA's cosmo_classify_lda.
    clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')

    # ── 5. Per-bin Leave-One-Chunk-Out decoding ───────────────────────────────
    # CoSMoMVPA partitions data by 'chunks'; each unique chunk value serves as
    # the test fold once while all remaining chunks form the training set.
    unique_chunks = np.unique(chunks)
    py_accuracies = []

    for b in range(n_bins):
        # Extract the channel features that belong to this time bin
        idx_start = b * N_CHANS
        idx_end   = (b + 1) * N_CHANS
        X_bin     = X[:, idx_start:idx_end]

        scores = []
        for test_chunk in unique_chunks:
            test_mask  = (chunks == test_chunk)
            train_mask = ~test_mask

            X_train, X_test = X_bin[train_mask], X_bin[test_mask]
            y_train, y_test = y[train_mask],     y[test_mask]

            clf.fit(X_train, y_train)
            pred = clf.predict(X_test)
            scores.append(accuracy_score(y_test, pred))

        py_accuracies.append(np.mean(scores))

    py_accuracies = np.array(py_accuracies)

    # ── 6. Summary metrics ────────────────────────────────────────────────────
    corr = np.corrcoef(gt_acc, py_accuracies)[0, 1]
    rmse = np.sqrt(np.mean((gt_acc - py_accuracies) ** 2))

    print(f"\n─── RESULTS ────────────────────────────────────")
    print(f"  Correlation (Python vs. MATLAB) : {corr:.5f}")
    print(f"  RMSE                            : {rmse:.5f}")

    if corr > 0.999:
        print("  >>> SUCCESS: classifier behaviour is effectively identical.")
    else:
        print("  >>> CHECK: residual gap suggests a classifier or CV difference.")

    # ── 7. Diagnostic plot ────────────────────────────────────────────────────
    plt.figure(figsize=(10, 5))

    plt.axhline(1 / 3, color='gray', linestyle=':', label='Chance level (1/3)')
    plt.plot(gt_acc,        'r-o',  label='MATLAB (CoSMoMVPA)', linewidth=2, alpha=0.7)
    plt.plot(py_accuracies, 'b--x', label=f'Python (exact input,  r = {corr:.4f})',
             linewidth=2, alpha=0.7)

    plt.title(f"Classifier identity check  "
              f"(corr = {corr:.4f},  RMSE = {rmse:.4f})")
    plt.xlabel("Time bin")
    plt.ylabel("Decoding accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_file = "debug_step5_exact_match.png"
    plt.savefig(out_file)
    print(f"\nPlot saved: {out_file}")
    plt.show()


if __name__ == "__main__":
    run_exact_match()
