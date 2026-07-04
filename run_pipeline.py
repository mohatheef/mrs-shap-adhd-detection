import os
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import loadmat
import shap

from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from imblearn.combine import SMOTEENN

# Import config parameters
import config

# Import pipeline steps
from preprocessing import (
    notch_filter, clip_spikes, butterworth_filter,
    find_optimal_weights, apply_ica, compare_ica_methods
)
from feature_extraction import extract_all_features
from feature_selection import (
    select_top_features, remove_highly_correlated_features,
    shap_selection, dtw_selection, mrs_selection
)
from modeling import get_base_models, build_voting_ensemble

# ==========================================
# DATA LOADING UTILITIES
# ==========================================

def load_and_segment_file(file_path, window_size=512, stride=256, expected_channels=19):
    """Load a single MAT file and segment into overlapping windows."""
    try:
        mat_data = loadmat(file_path)
        # Exclude metadata variables
        var_names = [k for k in mat_data.keys() if not k.startswith('__')]
        if not var_names:
            return None
        data = mat_data[var_names[0]]

        if data.ndim == 2:
            data = data[np.newaxis, :, :] # Add segment dimension if 2D

        segments = []
        for seg in data:
            # Segment along the time dimension (axis 0 of the segment)
            for start in range(0, seg.shape[0] - window_size, stride):
                window = seg[start:start + window_size, :]
                if window.shape[1] == expected_channels:
                    segments.append(window)

        return np.array(segments) if segments else None
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def load_from_folder_with_ids(folder_path, label, window_size=512, stride=256, expected_channels=19):
    """Load and segment all MAT files in a folder and associate patient IDs."""
    if not os.path.exists(folder_path):
        print(f"Directory not found: {folder_path}")
        return None, None, None

    files = sorted([f for f in os.listdir(folder_path) if f.endswith('.mat')])
    if not files:
        return None, None, None

    all_segments = []
    patient_ids = []
    labels = []

    for f in files:
        file_path = os.path.join(folder_path, f)
        segments = load_and_segment_file(file_path, window_size, stride, expected_channels)
        if segments is not None and len(segments) > 0:
            all_segments.append(segments)
            pid = f.split(".")[0]
            patient_ids.extend([pid] * len(segments))
            labels.extend([label] * len(segments))

    if not all_segments:
        return None, None, None

    return np.concatenate(all_segments), patient_ids, labels

def load_dataset(window_size=512, stride=256, expected_channels=19):
    """Load ADHD and Control data from all configured paths."""
    print("Loading resting-state EEG dataset...")
    
    # Load ADHD parts
    X_a1, pids_a1, y_a1 = load_from_folder_with_ids(config.ADHD_PART1, 1, window_size, stride, expected_channels)
    X_a2, pids_a2, y_a2 = load_from_folder_with_ids(config.ADHD_PART2, 1, window_size, stride, expected_channels)
    
    # Load Control parts
    X_c1, pids_c1, y_c1 = load_from_folder_with_ids(config.CONTROL_PART1, 0, window_size, stride, expected_channels)
    X_c2, pids_c2, y_c2 = load_from_folder_with_ids(config.CONTROL_PART2, 0, window_size, stride, expected_channels)

    # Combine parts
    X_list, pids_list, y_list = [], [], []
    for dataset in [(X_a1, pids_a1, y_a1), (X_a2, pids_a2, y_a2), (X_c1, pids_c1, y_c1), (X_c2, pids_c2, y_c2)]:
        if dataset[0] is not None:
            X_list.append(dataset[0])
            pids_list.extend(dataset[1])
            y_list.extend(dataset[2])

    if not X_list:
        raise FileNotFoundError("No EEG data found. Check your config.py dataset directories or run with --simulation.")

    X = np.concatenate(X_list)
    y = np.array(y_list)
    patient_ids = np.array(pids_list)

    print(f"Total segments loaded : {X.shape[0]}")
    print(f"Unique subjects        : {len(np.unique(patient_ids))}")
    print(f"ADHD segments (Class 1): {np.sum(y == 1)}")
    print(f"Control segments (Class 0): {np.sum(y == 0)}")
    return X, y, patient_ids

# ==========================================
# SIMULATION DATA GENERATION (DRY RUN)
# ==========================================

def generate_simulation_data(n_subjects=30, segments_per_subject=15):
    """Generate synthetic EEG data for dry run testing of the pipeline."""
    print("Generating simulation EEG data...")
    n_samples = config.WINDOW_SIZE
    n_channels = config.EXPECTED_CHANNELS
    
    X_list, pids_list, y_list = [], [], []
    
    for s_idx in range(n_subjects):
        subject_id = f"Sub{s_idx:03d}"
        # Split subjects equally between ADHD (1) and Control (0)
        label = 1 if s_idx < n_subjects // 2 else 0
        
        # Generate random segments with differing frequency attributes
        for _ in range(segments_per_subject):
            if label == 1:
                # ADHD has slightly elevated slow waves (4-8Hz theta)
                t = np.linspace(0, n_samples/config.FS, n_samples)
                sig = np.sin(2 * np.pi * 6 * t)[:, np.newaxis] * 1.5 + np.random.randn(n_samples, n_channels)
            else:
                # Controls have normal alpha/beta
                t = np.linspace(0, n_samples/config.FS, n_samples)
                sig = np.sin(2 * np.pi * 10 * t)[:, np.newaxis] * 1.2 + np.random.randn(n_samples, n_channels)
                
            X_list.append(sig)
            pids_list.append(subject_id)
            y_list.append(label)
            
    X = np.array(X_list)
    y = np.array(y_list)
    patient_ids = np.array(pids_list)
    
    print(f"Simulated segments : {X.shape[0]}")
    print(f"Simulated subjects  : {len(np.unique(patient_ids))}")
    return X, y, patient_ids

# ==========================================
# MAIN EXECUTION PIPELINE
# ==========================================

def run_experiment(X, y, patient_ids, run_full_features=False):
    """Run preprocessing, feature extraction, selection, CV, and test set evaluation."""
    # Step 1: Preprocessing
    print("\n--- Preprocessing ---")
    print("Applying notch and Butterworth bandpass filters...")
    X_pre = notch_filter(X, config.FS, freqs=[50, 60], Q=30)
    X_pre = clip_spikes(X_pre, threshold=500)
    X_pre = butterworth_filter(X_pre, config.FS, lowcut=1.0, highcut=45.0, order=4)

    print("Searching for optimal Weighted ICA weights...")
    # Find optimal weights on a subset of data for execution speed
    subset_idx = np.random.choice(len(X_pre), min(10, len(X_pre)), replace=False)
    best_weights, _, _ = find_optimal_weights(X_pre[subset_idx], X[subset_idx], fs=config.FS, n_components=config.EXPECTED_CHANNELS)
    
    print("Applying weighted ICA artifact removal...")
    X_cleaned, _ = apply_ica(X_pre, fs=config.FS, method='weighted', weights=best_weights)

    # Step 2: Feature Extraction
    print("\n--- Feature Extraction ---")
    X_features = extract_all_features(X_cleaned)
    print(f"Extracted feature matrix shape: {X_features.shape}")

    # Build feature name list matching the 998 feature set
    n_channels = config.EXPECTED_CHANNELS
    feature_names = []
    for ch in range(n_channels):
        feature_names.extend([
            f'engagement_ch{ch}', f'theta_ch{ch}', f'alpha_ch{ch}', f'beta_ch{ch}',
            f'pac_theta_alpha_ch{ch}', f'pac_theta_beta_ch{ch}', f'pac_alpha_beta_ch{ch}',
            f'wavelet_mean1_ch{ch}', f'wavelet_max1_ch{ch}',
            f'wavelet_mean2_ch{ch}', f'wavelet_max2_ch{ch}',
            f'wavelet_mean3_ch{ch}', f'wavelet_max3_ch{ch}',
            f'theta_beta_ratio_ch{ch}'
        ])
    microstate_names = [f'microstate_duration_{i}' for i in range(4)] + [f'microstate_coverage_{i}' for i in range(4)] + ['gfp_mean', 'gfp_std']
    corr_names = [f'corr_ch{i}_ch{j}' for i in range(n_channels) for j in range(i+1, n_channels)]
    coh_names = [f'coh_{band}_ch{i}_ch{j}' for band in ['theta', 'alpha', 'beta'] for i in range(n_channels) for j in range(i+1, n_channels)]
    feature_names.extend(microstate_names + corr_names + coh_names)

    # Step 3: Train/Test Split (Subject-Level)
    print("\n--- Train/Test Split (Subject-Level) ---")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X_features, y, groups=patient_ids))

    X_train_all, X_test_held = X_features[train_idx], X_features[test_idx]
    y_train_all, y_test_held = y[train_idx], y[test_idx]
    sids_train = patient_ids[train_idx]
    sids_test = patient_ids[test_idx]

    print(f"Training subjects      : {len(np.unique(sids_train))} ({len(X_train_all)} segments)")
    print(f"Held-out test subjects : {len(np.unique(sids_test))} ({len(X_test_held)} segments)")

    # Step 4: Subject-Level 5-Fold CV (Outer Loop)
    print("\n--- Running Subject-Level 5-Fold CV (Leakage-Free) ---")
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = []
    
    # Store selected features for test set evaluation
    fold_selected_features = []

    for fold, (tr_idx, te_idx) in enumerate(sgkf.split(X_train_all, y_train_all, groups=sids_train), 1):
        X_tr_raw, X_te = X_train_all[tr_idx], X_train_all[te_idx]
        y_tr_raw, y_te = y_train_all[tr_idx], y_train_all[te_idx]
        sids_tr = sids_train[tr_idx]

        # Inner-loop 5-fold CV on training fold to compute metrics for Equation 21 (BW)
        print(f"Fold {fold}: Computing inner-loop metrics for Balanced Weight (BW)...")
        inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        inner_metrics = []
        
        for val_tr_idx, val_te_idx in inner_cv.split(X_tr_raw, y_tr_raw):
            X_val_tr_raw, X_val_te = X_tr_raw[val_tr_idx], X_tr_raw[val_te_idx]
            y_val_tr_raw, y_val_te = y_tr_raw[val_tr_idx], y_tr_raw[val_te_idx]
            
            smote_enn = SMOTEENN(random_state=42)
            X_val_tr, y_val_tr = smote_enn.fit_resample(X_val_tr_raw, y_val_tr_raw)
            
            # Simple modeling in inner loop for speed
            models = get_base_models()
            inner_clf = models['cat']
            inner_clf.fit(X_val_tr[:, :50], y_val_tr) # Evaluate on a subset of features for speed
            preds = inner_clf.predict(X_val_te[:, :50])
            
            cm_val = confusion_matrix(y_val_te, preds)
            tn, fp, fn, tp = cm_val.ravel()
            sens = tp / (tp + fn + 1e-8)
            spec = tn / (tn + fp + 1e-8)
            acc = accuracy_score(y_val_te, preds)
            f1 = f1_score(y_val_te, preds)
            inner_metrics.append([sens, spec, f1, acc])
            
        mean_metrics = np.mean(inner_metrics, axis=0)
        
        # Calculate BW using Equations 18-20 (WITHOUT division by sum)
        metrics_norm = (mean_metrics - mean_metrics.min()) / (mean_metrics.max() - mean_metrics.min() + 1e-10)
        metric_weights = 1.0 - metrics_norm
        BW = np.mean(metric_weights)
        print(f"Fold {fold} Balanced Weight (BW): {BW:.4f}")

        # Standardize training data per fold
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr_raw)
        X_tr_scaled = np.nan_to_num(X_tr_scaled, nan=0.0, posinf=0.0, neginf=0.0)

        # Apply SMOTE-ENN boundary refinement
        smote_enn = SMOTEENN(random_state=42)
        X_tr_bal, y_tr_bal = smote_enn.fit_resample(X_tr_scaled, y_tr_raw)

        # Variance thresholding
        var_sel = VarianceThreshold(threshold=0.01)
        X_tr_red = var_sel.fit_transform(X_tr_bal)
        support_mask = var_sel.get_support()
        feat_names_red = np.array(feature_names)[support_mask].tolist()

        # Correlation filtering
        X_tr_red, feat_names_red = remove_highly_correlated_features(X_tr_red, feat_names_red, threshold=0.85)

        # Feature selection using MRS-SHAP (incorporating BW)
        shap_scores = shap_selection(X_tr_red, y_tr_bal)
        
        X_tr_raw_red = X_tr_scaled[:, [feature_names.index(f) for f in feat_names_red]]
        dtw_scores = dtw_selection(X_tr_raw_red, sids_tr)
        
        mrs_scores = mrs_selection(shap_scores, dtw_scores, w_shap=0.6, w_dtw=0.4, BW=BW)
        selected_idx = select_top_features(mrs_scores, percentage=0.6)
        
        selected_feats = [feat_names_red[idx] for idx in selected_idx]
        fold_selected_features.append(selected_feats)
        
        # Train and evaluate CatBoost classifier on selected features
        selected_indices_orig = [feature_names.index(f) for f in selected_feats]
        
        # Scale test fold
        X_te_scaled = scaler.transform(X_te)
        X_te_scaled = np.nan_to_num(X_te_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        
        model = get_base_models()['cat']
        model.fit(X_tr_bal[:, [feat_names_red.index(f) for f in selected_feats]], y_tr_bal)
        
        y_pred = model.predict(X_te_scaled[:, selected_indices_orig])
        y_prob = model.predict_proba(X_te_scaled[:, selected_indices_orig])[:, 1]

        cm = confusion_matrix(y_te, y_pred)
        tn, fp, fn, tp = cm.ravel()
        fold_sens = tp / (tp + fn + 1e-8)
        fold_spec = tn / (tn + fp + 1e-8)
        fold_acc = accuracy_score(y_te, y_pred)
        fold_f1 = f1_score(y_te, y_pred)
        fold_auc = roc_auc_score(y_te, y_prob)

        print(f"Fold {fold} - Accuracy: {fold_acc:.4f}, F1: {fold_f1:.4f}, AUC: {fold_auc:.4f}")
        cv_results.append([fold, fold_sens, fold_spec, fold_f1, fold_acc, fold_auc])

    df_cv = pd.DataFrame(cv_results, columns=["Fold", "Sensitivity", "Specificity", "F1", "Accuracy", "AUC"])
    print("\n--- Average Subject-Level CV Performance ---")
    print(df_cv.mean().round(4))

    # Step 5: Final Evaluation on Held-out Test Set using Voting Ensemble
    print("\n--- Final Evaluation on Held-out Test Set ---")
    # Fit MRS-SHAP on full training set
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_all)
    X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    
    # SMOTE-ENN
    smote_enn = SMOTEENN(random_state=42)
    X_train_bal, y_train_bal = smote_enn.fit_resample(X_train_scaled, y_train_all)
    
    # Variance and correlation
    var_sel = VarianceThreshold(threshold=0.01)
    X_train_red = var_sel.fit_transform(X_train_bal)
    support_mask = var_sel.get_support()
    feat_names_red = np.array(feature_names)[support_mask].tolist()
    X_train_red, feat_names_red = remove_highly_correlated_features(X_train_red, feat_names_red, threshold=0.85)

    # Compute final select indices
    shap_scores = shap_selection(X_train_red, y_train_bal)
    X_train_raw_red = X_train_scaled[:, [feature_names.index(f) for f in feat_names_red]]
    dtw_scores = dtw_selection(X_train_raw_red, sids_train)
    mrs_scores = mrs_selection(shap_scores, dtw_scores, w_shap=0.6, w_dtw=0.4, BW=0.25)
    selected_idx = select_top_features(mrs_scores, percentage=0.6)
    selected_feats = [feat_names_red[idx] for idx in selected_idx]
    
    # Scale test set
    X_test_scaled = scaler.transform(X_test_held)
    X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    selected_indices_orig = [feature_names.index(f) for f in selected_feats]
    X_tr_ensemble = X_train_bal[:, [feat_names_red.index(f) for f in selected_feats]]
    X_te_ensemble = X_test_scaled[:, selected_indices_orig]

    # Train ensemble model
    ensemble = build_voting_ensemble()
    ensemble.fit(X_tr_ensemble, y_train_bal)
    
    y_pred_held = ensemble.predict(X_te_ensemble)
    y_prob_held = ensemble.predict_proba(X_te_ensemble)[:, 1]

    cm_held = confusion_matrix(y_test_held, y_pred_held)
    tn, fp, fn, tp = cm_held.ravel()
    sens_held = tp / (tp + fn + 1e-8)
    spec_held = tn / (tn + fp + 1e-8)
    acc_held = accuracy_score(y_test_held, y_pred_held)
    f1_held = f1_score(y_test_held, y_pred_held)
    auc_held = roc_auc_score(y_test_held, y_prob_held)

    print("\nHeld-out Test Set Performance:")
    print(f"Confusion Matrix:\n{cm_held}")
    print(f"Sensitivity: {sens_held:.4f}")
    print(f"Specificity: {spec_held:.4f}")
    print(f"F1-Score: {f1_held:.4f}")
    print(f"Accuracy: {acc_held:.4f}")
    print(f"AUC: {auc_held:.4f}")

    # Generate Figures
    print("\nSaving figures...")
    # 1. Confusion Matrix
    plt.figure(figsize=(6, 5), dpi=300)
    sns.heatmap(cm_held, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix (Test Set)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig('confusion_matrix_test.png')
    plt.close()

    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_test_held, y_prob_held)
    plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(fpr, tpr, color='orange', label=f'ROC Curve (AUC = {auc_held:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('ROC Curve (Test Set)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('roc_curve_test.png')
    plt.close()

    # Save outputs
    print("\nSaving selected features list to 'selected_features.txt'...")
    with open('selected_features.txt', 'w') as f:
        f.write('\n'.join(selected_feats))

    # Save final test metrics
    final_metrics_df = pd.DataFrame({
        'Metric': ['Sensitivity', 'Specificity', 'F1-Score', 'Accuracy', 'AUC'],
        'Value': [sens_held, spec_held, f1_held, acc_held, auc_held]
    })
    final_metrics_df.to_csv('final_test_metrics.csv', index=False)
    print("Final test metrics saved to 'final_test_metrics.csv'")
    print("✔ Pipeline execution completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the MRS-SHAP ADHD EEG pipeline.")
    parser.add_argument('--simulation', action='store_true', help='Run pipeline with simulated data for test/dry-run.')
    args = parser.parse_args()

    if args.simulation:
        X, y, patient_ids = generate_simulation_data(n_subjects=10, segments_per_subject=5)
    else:
        X, y, patient_ids = load_dataset(config.WINDOW_SIZE, config.STRIDE, config.EXPECTED_CHANNELS)
        
    run_experiment(X, y, patient_ids)
