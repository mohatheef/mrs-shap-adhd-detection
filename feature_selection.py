import time
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from dtaidistance import dtw
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from imblearn.combine import SMOTEENN
import psutil

def select_top_features(scores, percentage=0.6):
    """Retain the top percentage of features based on scores."""
    k = max(1, int(len(scores) * percentage))
    return np.argsort(scores)[::-1][:k]

def remove_highly_correlated_features(X, feature_names, threshold=0.85):
    """Remove highly correlated features (Pearson correlation > threshold)."""
    corr_matrix = np.abs(np.corrcoef(X.T))
    to_remove = set()
    for i in range(len(corr_matrix)):
        for j in range(i + 1, len(corr_matrix)):
            if corr_matrix[i, j] > threshold:
                to_remove.add(j)
    keep_indices = [i for i in range(X.shape[1]) if i not in to_remove]
    return X[:, keep_indices], np.array(feature_names)[keep_indices].tolist()

def shap_selection(X, y):
    """Estimate feature importance using mean absolute SHAP values from XGBoost."""
    xgb_model = xgb.XGBClassifier(random_state=42, n_estimators=100, eval_metric='logloss', reg_lambda=0.1)
    xgb_model.fit(X, y)
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        # In binary classification, SHAP values are a list of length 2
        shap_values = shap_values[1]
    shap_importance = np.abs(shap_values).mean(axis=0)
    return shap_importance

def dtw_selection(X, subject_ids, min_window_size=50):
    """Calculate temporal stability scores using Dynamic Time Warping (DTW)."""
    dtw_scores = np.zeros(X.shape[1])
    for feature_idx in range(X.shape[1]):
        feature_series = X[:, feature_idx]
        if np.any(np.isnan(feature_series)) or np.std(feature_series) == 0:
            dtw_scores[feature_idx] = 0
            continue
        
        # Segment the feature series into sub-windows
        sub_windows = [feature_series[i:i+min_window_size] for i in range(0, len(feature_series)-min_window_size, min_window_size)]
        if len(sub_windows) < 2:
            dtw_scores[feature_idx] = 0
            continue
            
        try:
            dtw_sum = 0
            count = 0
            # Pairwise DTW alignment
            for i in range(len(sub_windows)):
                for j in range(i+1, len(sub_windows)):
                    dist = dtw.distance(sub_windows[i], sub_windows[j])
                    if np.isnan(dist):
                        continue
                    dtw_sum += dist
                    count += 1
            dtw_scores[feature_idx] = dtw_sum / count if count > 0 else 0
        except Exception:
            dtw_scores[feature_idx] = 0
            
    # Stability: invert the DTW alignment cost (smaller cost means higher stability)
    # Normalize stable features to have high values
    dtw_inverted = np.max(dtw_scores) - dtw_scores
    return dtw_inverted

def mrs_selection(shap_scores, dtw_scores, w_shap=0.6, w_dtw=0.4, BW=0.0):
    """Calculate the Multi-Objective Ranking Score (MRS) for feature selection."""
    sn = (shap_scores - shap_scores.min()) / (shap_scores.max() - shap_scores.min() + 1e-10)
    dn = (dtw_scores - dtw_scores.min()) / (dtw_scores.max() - dtw_scores.min() + 1e-10)
    
    # Equation 21: suboptimal model performance amplifies SHAP weighting
    mo_score = w_shap * sn * (1.0 + BW) + w_dtw * dn
    return mo_score

def select_features_mrs_shap_full(X, y, feature_names, subject_ids, n_features_to_select=0.6, min_window_size=50, BW=0.0):
    """Wrapper function to perform the full MRS-SHAP feature selection pipeline."""
    start_time = time.time()
    process = psutil.Process()
    mem_before = process.memory_info().rss / 1024**2

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    # SMOTE-ENN boundary cleaning
    smote_enn = SMOTEENN(random_state=42, sampling_strategy='auto')
    X_balanced, y_balanced = smote_enn.fit_resample(X_scaled, y)

    # Variance Thresholding
    selector = VarianceThreshold(threshold=0.01)
    X_reduced = selector.fit_transform(X_balanced)
    mask = selector.get_support()
    feature_names_reduced = np.array(feature_names)[mask].tolist()

    # Correlation Filtering
    X_reduced, feature_names_reduced = remove_highly_correlated_features(X_reduced, feature_names_reduced, threshold=0.85)

    # Compute SHAP and DTW stability
    shap_scores = shap_selection(X_reduced, y_balanced)
    
    # DTW uses raw/un-oversampled data for accurate subject-level temporal stability
    X_reduced_raw = X_scaled[:, [feature_names.index(f) for f in feature_names_reduced]]
    dtw_scores = dtw_selection(X_reduced_raw, subject_ids, min_window_size)
    
    # Compute MRS
    mrs_scores = mrs_selection(shap_scores, dtw_scores, w_shap=0.6, w_dtw=0.4, BW=BW)
    selected_indices = select_top_features(mrs_scores, percentage=n_features_to_select)
    
    selected_features = [feature_names_reduced[idx] for idx in selected_indices]
    selected_indices_orig = [feature_names.index(f) for f in selected_features]
    
    mem_after = process.memory_info().rss / 1024**2
    print(f"MRS-SHAP complete: Features selected={len(selected_features)}, Time={time.time()-start_time:.2f}s, Memory={mem_after-mem_before:.2f}MB")
    
    return selected_features, X[:, selected_indices_orig]
