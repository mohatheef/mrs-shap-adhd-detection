import time
import numpy as np
from scipy import signal
from scipy.signal import hilbert, find_peaks
from sklearn.cluster import KMeans
import pywt
from tqdm import tqdm

# Import preprocessing filters
from preprocessing import butterworth_filter

def calculate_psd(data, fs=128, freq_range=None):
    """Calculate power spectral density using Welch's method."""
    nperseg = min(len(data), 512)
    f, psd = signal.welch(data, fs=fs, nperseg=nperseg, axis=0)
    if freq_range:
        freq_mask = (f >= freq_range[0]) & (f <= freq_range[1])
        return np.mean(psd[freq_mask], axis=0) if freq_mask.any() else 0
    return psd.mean(axis=0)

def calculate_engagement_index(data, fs=128, prev_value=None):
    """Calculate engagement index: (theta + alpha) / beta."""
    theta = calculate_psd(data, fs, freq_range=(4, 8))
    alpha = calculate_psd(data, fs, freq_range=(8, 13))
    beta = calculate_psd(data, fs, freq_range=(13, 30))
    if np.any(np.isnan([theta, alpha, beta])) or beta == 0:
        return prev_value if prev_value is not None else 0
    return (theta + alpha) / (beta + 1e-10)

def phase_amplitude_coupling(data, fs=128, prev_pac=None):
    """Calculate phase-amplitude coupling for theta-alpha, theta-beta, alpha-beta."""
    # Ensure correct shape for the filter (1 x samples x 1)
    data_reshaped = data[np.newaxis, :, np.newaxis]
    theta = butterworth_filter(data_reshaped, fs, 4.0, 8.0)[0, :, 0]
    alpha = butterworth_filter(data_reshaped, fs, 8.0, 13.0)[0, :, 0]
    beta = butterworth_filter(data_reshaped, fs, 13.0, 30.0)[0, :, 0]
    
    if np.any(np.isnan([theta, alpha, beta])) or np.std(theta) == 0 or np.std(alpha) == 0 or np.std(beta) == 0:
        return prev_pac if prev_pac is not None else np.zeros(3)
        
    theta_phase = np.angle(hilbert(theta))
    alpha_phase = np.angle(hilbert(alpha))
    beta_phase = np.angle(hilbert(beta))
    
    alpha_amp = np.abs(hilbert(alpha))
    beta_amp = np.abs(hilbert(beta))
    
    pac_theta_alpha = np.abs(np.mean(alpha_amp * np.exp(1j * theta_phase)))
    pac_theta_beta = np.abs(np.mean(beta_amp * np.exp(1j * theta_phase)))
    pac_alpha_beta = np.abs(np.mean(beta_amp * np.exp(1j * alpha_phase)))
    
    return np.array([pac_theta_alpha, pac_theta_beta, pac_alpha_beta])

def wavelet_features(data, prev_wavelet=None):
    """Extract wavelet features using Mexican Hat wavelet (CWT)."""
    scales = np.arange(1, 4)
    coeffs, _ = pywt.cwt(data, scales, 'mexh', axis=0)
    if np.any(np.isnan(coeffs)):
        return prev_wavelet if prev_wavelet is not None else np.zeros(6)
    features = []
    for c in coeffs:
        features.extend([np.mean(np.abs(c)), np.max(np.abs(c))])
    return np.array(features)

def theta_beta_ratio(data, fs=128, prev_value=None):
    """Calculate theta/beta ratio."""
    theta = calculate_psd(data, fs, freq_range=(4, 8))
    beta = calculate_psd(data, fs, freq_range=(13, 30))
    if np.any(np.isnan([theta, beta])) or beta == 0:
        return prev_value if prev_value is not None else 0
    return theta / (beta + 1e-10)

def channel_coherence(data, fs=128, prev_coherence=None):
    """Calculate coherence between channel pairs in theta, alpha, and beta bands."""
    n_channels = data.shape[1]
    coherence_features = []
    expected_length = 3 * (n_channels * (n_channels - 1) // 2)
    nperseg = min(len(data), 512)
    
    for band, freq_range in [('theta', (4, 8)), ('alpha', (8, 13)), ('beta', (13, 30))]:
        for i in range(n_channels):
            for j in range(i + 1, n_channels):
                f, coh = signal.coherence(data[:, i], data[:, j], fs=fs, nperseg=nperseg)
                freq_mask = (f >= freq_range[0]) & (f <= freq_range[1])
                coh_value = np.mean(coh[freq_mask]) if freq_mask.any() else 0
                if np.isnan(coh_value):
                    return prev_coherence if prev_coherence is not None else np.zeros(expected_length)
                coherence_features.append(coh_value)
    return np.array(coherence_features)

def channel_correlation(data, prev_correlation=None):
    """Calculate Pearson correlation coefficient between channel pairs."""
    n_channels = data.shape[1]
    corr_features = []
    expected_length = n_channels * (n_channels - 1) // 2
    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            corr = np.corrcoef(data[:, i], data[:, j])[0, 1]
            if np.isnan(corr):
                return prev_correlation if prev_correlation is not None else np.zeros(expected_length)
            corr_features.append(corr)
    return np.array(corr_features)

def microstate_analysis(data, fs=128, n_states=4, prev_microstates=None):
    """Perform EEG microstate analysis using K-means clustering."""
    expected_length = n_states + n_states + 2 # Durations + Coverages + GFP (mean, std)
    gfp = np.std(data, axis=1)
    if np.any(np.isnan(gfp)):
        return prev_microstates if prev_microstates is not None else np.zeros(expected_length)
        
    peaks, _ = find_peaks(gfp, distance=fs//10)
    if len(peaks) < n_states:
        peak_data = data
    else:
        peak_data = data[peaks]
        
    try:
        kmeans = KMeans(n_clusters=n_states, random_state=42, n_init=10)
        kmeans.fit(peak_data)
        templates = kmeans.cluster_centers_
        
        distances = np.zeros((data.shape[0], n_states))
        for i, template in enumerate(templates):
            template_norm = np.linalg.norm(template)
            data_norm = np.linalg.norm(data, axis=1)
            distances[:, i] = np.abs(np.sum(data * template, axis=1) / (data_norm * template_norm + 1e-10))
            
        labels = np.argmax(distances, axis=1)
        durations = np.zeros(n_states)
        gfp_peaks = gfp[peaks] if len(peaks) > 0 else gfp
        gfp_mean = np.mean(gfp_peaks)
        gfp_std = np.std(gfp_peaks)
        
        for state in range(n_states):
            state_indices = np.where(labels == state)[0]
            if len(state_indices) > 0:
                diff = np.diff(state_indices)
                breaks = np.where(diff > 1)[0]
                segment_lengths = np.diff(np.concatenate([[0], breaks + 1, [len(state_indices)]]))
                durations[state] = np.mean(segment_lengths) / fs * 1000 if len(segment_lengths) > 0 else 0
                
        coverage = np.bincount(labels, minlength=n_states).astype(float) / len(labels)
        microstate_features = np.hstack([durations, coverage, [gfp_mean, gfp_std]])
        return microstate_features
    except Exception:
        return prev_microstates if prev_microstates is not None else np.zeros(expected_length)

def extract_features_single(sample, idx, prev_features=None):
    """Extract all features for a single multi-channel EEG segment."""
    n_channels = sample.shape[1]
    
    # Standardize data per segment
    normalized = np.zeros_like(sample)
    for ch in range(n_channels):
        channel_data = sample[:, ch]
        std = np.std(channel_data)
        if std > 0:
            normalized[:, ch] = (channel_data - np.mean(channel_data)) / std
        else:
            normalized[:, ch] = channel_data
            
    channel_features = []
    features_per_channel = 14
    total_per_channel = n_channels * features_per_channel
    microstate_len = 10
    corr_len = (n_channels * (n_channels - 1) // 2)
    coh_len = 3 * corr_len
    total_prev_len = total_per_channel + microstate_len + corr_len + coh_len
    
    if prev_features is not None and len(prev_features) != total_prev_len:
        raise ValueError(f"Expected prev_features length {total_prev_len}, got {len(prev_features)}")
        
    for ch in range(n_channels):
        channel_data = normalized[:, ch]
        offset = ch * features_per_channel
        prev_values = prev_features[offset:offset + features_per_channel] if prev_features is not None else None
        
        engagement = calculate_engagement_index(channel_data, prev_value=prev_values[0] if prev_values is not None else None)
        theta = calculate_psd(channel_data, freq_range=(4, 8))
        alpha = calculate_psd(channel_data, freq_range=(8, 13))
        beta = calculate_psd(channel_data, freq_range=(13, 30))
        pac = phase_amplitude_coupling(channel_data, prev_pac=prev_values[4:7] if prev_values is not None else None)
        wavelet = wavelet_features(channel_data, prev_wavelet=prev_values[7:13] if prev_values is not None else None)
        tbr = theta_beta_ratio(channel_data, prev_value=prev_values[13] if prev_values is not None else None)
        
        ch_features = np.concatenate([
            [engagement], [theta], [alpha], [beta], pac, wavelet, [tbr]
        ])
        channel_features.append(ch_features)
        
    per_channel_features = np.array(channel_features).flatten()
    
    # Extract global and connectivity features
    microstates = microstate_analysis(normalized, prev_microstates=prev_features[total_per_channel:total_per_channel + microstate_len] if prev_features is not None else None)
    correlations = channel_correlation(normalized, prev_correlation=prev_features[total_per_channel + microstate_len:total_per_channel + microstate_len + corr_len] if prev_features is not None else None)
    coherences = channel_coherence(normalized, prev_coherence=prev_features[total_per_channel + microstate_len + corr_len:] if prev_features is not None else None)
    
    sample_features = np.concatenate([per_channel_features, microstates, correlations, coherences])
    return sample_features

def extract_all_features(data, chunk_size=100):
    """Extract features for all EEG samples."""
    start_time = time.time()
    n_samples, n_samples_per_segment, n_channels = data.shape
    feature_matrix = []
    prev_features = None
    
    for i in tqdm(range(n_samples), desc="Extracting features"):
        sample_features = extract_features_single(data[i], i, prev_features)
        feature_matrix.append(sample_features)
        prev_features = sample_features
        
    feature_matrix = np.array(feature_matrix)
    print(f"Feature matrix shape: {feature_matrix.shape}, Time={time.time()-start_time:.2f}s")
    return feature_matrix
