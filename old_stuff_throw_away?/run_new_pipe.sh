#!/bin/bash
# Master Pipeline Script for RPS EEG Replication
# Usage: ./run_pipeline.sh

echo "=========================================================="
echo "STARTING FULL PIPELINE PROCESSING (Moerel et al., 2025)"
echo "=========================================================="

# 1. Activate Environment (if not using 'uv run' explicitly inside scripts)
# We use 'uv run' for every command to ensure reproducibility.

# # 2. Step 0: Conversion (Raw -> FIF)
# echo "----------------------------------------------------------"
# echo "STEP 0: Data Conversion & Mapping"
# uv run code_new/00_convert_bids.py

# # 3. Step 1: Preprocessing (Filter, Epoch, Interp)
# echo "----------------------------------------------------------"
# echo "STEP 1: Signal Preprocessing"
# uv run code_new/01_preprocess.py

# # 4. Step 2: Diagnosis (Optional - skipped for speed in batch, run manually if needed)
# echo "STEP 2: Diagnosis Plots"
# uv run code_new/03_diagnosis.py

# # 5. Step 3: Feature Extraction (Binning)
# echo "----------------------------------------------------------"
# echo "STEP 3: Feature Extraction"
# uv run code_new/04_features.py

# # 6. Step 4: Decoding Analysis (LDA)
# echo "----------------------------------------------------------"
# echo "STEP 4: Decoding (LDA)"
# uv run code_new/05_decoding.py

# # 7. Step 5: Behavioral Analysis (Markov)
# echo "----------------------------------------------------------"
# echo "STEP 5: Behavioral Analysis"
# uv run code_new/06_markov.py

# 8. Step 6: Paper Replication Plots (Aggregated)
echo "----------------------------------------------------------"
echo "STEP 6: Generating Final Paper Figures"
uv run code_new/07_plot_paper_fig1.py
uv run code_new/08_plot_paper_fig2_3.py

echo "=========================================================="
echo "PIPELINE COMPLETED SUCCESSFULLY"
echo "Check results in 'figures/'"
echo "=========================================================="

# 9. Step 7: Compare models
echo "----------------------------------------------------------"
echo "STEP 7: Compare models"
uv run code_new/09_model_comparison.py
echo "=========================================================="

# 10. Step 8: Continuous analysis
echo "----------------------------------------------------------"
echo "STEP 8: Continuous analysis"
uv run code_new/10_continuous_analysis.py
echo "=========================================================="

# 11. Step 9: Time Generalization
echo "----------------------------------------------------------"
echo "STEP 9: Time Generalization"
uv run code_new/11_time_gen.py
echo "=========================================================="

# 12. Step 10: plot Extension
echo "----------------------------------------------------------"
echo "STEP 10: Plot Extension results"
uv run code_new/12_extension_plots.py
echo "=========================================================="


echo "=========================================================="
echo "PIPELINE COMPLETED SUCCESSFULLY"
echo "Check results in 'figures/'"
echo "=========================================================="