# MRS-SHAP: Multi-Objective Explainable EEG Feature Selection Framework for ADHD Detection

This repository implements the complete processing, feature extraction, feature selection, and classification pipeline described in the paper:

> **MRS-SHAP: Multi-objective explainable EEG feature selection framework for ADHD detection**  
> Mohammed Atheef G A, Shyam Pranav G, and Omkar S Powar  
> *Discover Applied Sciences* (2026)  
> DOI: [10.1007/s42452-026-08788-7](https://doi.org/10.1007/s42452-026-08788-7)

The framework combines SHAP-based feature importance, Dynamic Time Warping (DTW)-based temporal stability, and inner-loop cross-validation performance weighting into a unified ranking score (MRS-SHAP) to select robust, explainable, and neurophysiologically relevant EEG features for ADHD classification.

## Repository Structure

- `config.py`: Configuration file for local dataset paths, signal acquisition parameters (128 Hz sampling frequency, 19-channel standard configuration), and window sliding values.
- `requirements.txt`: Python package dependencies.
- `preprocessing.py`: Multi-level signal preprocessing steps (50Hz/60Hz Notch filters, ±500 µV spike clipping, 1-45 Hz Butterworth bandpass filter, and multi-metric weighted ICA artifact removal).
- `feature_extraction.py`: Extraction of all 998 features across Spectral, Nonlinear, and Global/Connectivity domains (including band powers, Engagement Index, TBR, CWT wavelets, PAC, microstates, channel correlation, and band coherence).
- `feature_selection.py`: Pipeline for SMOTE-ENN inside outer CV folds, variance thresholding (0.01), correlation filtering (Pearson > 0.85), and the proposed MRS-SHAP ranking.
- `modeling.py`: Definition of individual models (SVM, LR, GB, CatBoost) and the soft Voting Ensemble Classifier with optimized weights `[0.3, 0.3, 0.25, 0.15]`.
- `run_pipeline.py`: Main execution script that runs the entire pipeline, computes leakage-free subject-level CV, evaluates on the held-out set, and saves results/plots.

## Installation

Install all required dependencies using `pip`:

```bash
pip install -r requirements.txt
```

## How to Run

### 1. Dry Run / Simulation Mode (No Dataset Needed)
To verify that the complete pipeline compiles, runs successfully, and generates correct outputs, run the pipeline in simulation mode:

```bash
python3 run_pipeline.py --simulation
```

This will automatically generate synthetic EEG segments for 10 subjects, run them through preprocessing, feature extraction (998 features), cross-validation, ensemble training, held-out test evaluation, and generate the corresponding plots.

### 2. Run on Actual Dataset
First, modify the directory paths in `config.py` to point to your local dataset folders:

```python
ADHD_PART1 = "/path/to/your/ADHD_part1"
ADHD_PART2 = "/path/to/your/ADHD_part2"
CONTROL_PART1 = "/path/to/your/Control_part1"
CONTROL_PART2 = "/path/to/your/Control_part2"
```

Once the paths are set, execute the main pipeline:

```bash
python3 run_pipeline.py
```

## Generated Outputs

Upon execution, the script will output the average subject-level cross-validation metrics, followed by held-out test set performance. It also generates the following files:

- `selected_features.txt`: List of the top 60% features selected by MRS-SHAP.
- `final_test_metrics.csv`: Final classification metrics (Sensitivity, Specificity, F1, Accuracy, AUC) on the held-out test set.
- `confusion_matrix_test.png`: Confusion matrix plot for the Voting Ensemble model on the test set.
- `roc_curve_test.png`: ROC-AUC curve plot for the ensemble model on the test set.
