import time
import numpy as np
from scipy import signal
from scipy.stats import kurtosis
import mne
from tqdm import tqdm

def notch_filter(data, fs=128, freqs=[50], Q=30):
    """Apply IIR notch filter to remove power-line interference."""
    data_filtered = data.copy()
    for freq in freqs:
        b, a = signal.iirnotch(freq, Q, fs)
        # Apply along the time axis (axis 1 for shape: segments x samples x channels)
        data_filtered = signal.filtfilt(b, a, data_filtered, axis=1)
    return data_filtered

def clip_spikes(data, threshold=500):
    """Clip transient spikes exceeding threshold."""
    return np.clip(data, -threshold, threshold)

def butterworth_filter(data, fs=128, lowcut=1.0, highcut=45.0, order=4):
    """Apply Butterworth bandpass filter to EEG data."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.filtfilt(b, a, data, axis=1)

def apply_ica(data, fs=128, n_components=None, method='weighted', weights=None):
    """Apply ICA with weighted (kurtosis, variance, peak, Fp2, frequency) or standard method."""
    start_time = time.time()
    print(f"Applying ICA (method={method}) with input shape={data.shape}")
    
    if data.ndim == 2:
        data = data[np.newaxis, :, :]
    elif data.ndim != 3:
        raise ValueError(f"Expected 3D data, got {data.ndim}D")
        
    n_segments, n_samples, n_channels = data.shape
    if n_components is None or n_components > n_channels:
        n_components = min(19, n_channels)
        
    print(f"Using n_components={n_components}")
    if n_components < 1:
        print("Warning: n_components < 1, skipping ICA exclusion")
        return data, data
        
    total_samples = n_segments * n_samples
    concatenated_data = data.reshape(total_samples, n_channels)
    
    ch_names = [f"EEG {i}" for i in range(n_channels)]
    montage_labels = ['EEG1', 'EEG2', 'EEG3', 'EEG4', 'EEG5', 'EEG6', 'EEG7', 'EEG8', 'EEG9', 'EEG10', 'EEG11', 'EEG12', 'EEG13', 'EEG14', 'EEG15', 'EEG16', 'EEG17', 'EEG18', 'EEG19']
    
    info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types="eeg")
    raw = mne.io.RawArray(concatenated_data.T, info)
    
    channel_mapping = {f"EEG {i}": montage_labels[i] for i in range(n_channels)}
    raw.rename_channels(channel_mapping)
    
    try:
        theta_values = [-18, 18, -54, -39, 0, 39, 54, -90, -90, 90, 90, 90, -126, -141, 180, 141, 126, -162, 162]
        radius_values_cm = [51.3, 51.3, 51.3, 33.3, 25.6, 33.3, 51.3, 51.3, 25.6, 0, 25.6, 51.3, 51.3, 33.3, 25.6, 33.3, 51.3, 51.3, 51.3]
        radius_values_m = [r / 100 for r in radius_values_cm]
        ch_pos = {label: [radius * np.cos(np.deg2rad(theta)), radius * np.sin(np.deg2rad(theta)), 0.0]
                  for label, theta, radius in zip(montage_labels, theta_values, radius_values_m)}
        montage = mne.channels.make_dig_montage(ch_pos, coord_frame='head')
        raw.set_montage(montage)
    except Exception:
        print("Warning: Failed to set custom montage, using standard_1020")
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage)
        
    # Highpass at 0.5Hz for ICA stability
    raw.filter(l_freq=0.5, h_freq=45.0, method='iir', iir_params=dict(order=4, ftype='butter'), verbose=False)
    
    ica = mne.preprocessing.ICA(n_components=n_components, random_state=42, max_iter=500)
    ica.fit(raw, verbose=False)
    
    ica_sources = ica.get_sources(raw).get_data().T
    kurtosis_values = np.array([kurtosis(ica_sources[:, i]) for i in range(n_components)])
    variance_values = np.var(ica_sources, axis=0)
    peak_amplitude = np.max(np.abs(ica_sources), axis=0)
    fp2_weight = np.mean(np.abs(ica_sources[:, 1])) / np.mean(np.abs(ica_sources)) if n_channels > 1 else 0
    
    nperseg = min(total_samples, 512)
    f, psd = signal.welch(ica_sources.T, fs=fs, nperseg=nperseg, axis=1)
    artifact_band = (f >= 45) & (f <= 60)
    artifact_power = np.mean(psd[:, artifact_band], axis=1) + 1e-10
    
    if method == 'weighted':
        if weights is None:
            weights = {'kurtosis': 0.45, 'variance': 0.15, 'peak': 0.15, 'fp2': 0.10, 'artifact': 0.15}
            
        kurtosis_norm = (kurtosis_values - kurtosis_values.min()) / (kurtosis_values.max() - kurtosis_values.min() + 1e-10)
        variance_norm = (variance_values - variance_values.min()) / (variance_values.max() - variance_values.min() + 1e-10)
        peak_norm = (peak_amplitude - peak_amplitude.min()) / (peak_amplitude.max() - peak_amplitude.min() + 1e-10)
        artifact_norm = (artifact_power - artifact_power.min()) / (artifact_power.max() - artifact_power.min() + 1e-10)
        
        combined_score = (weights['kurtosis'] * kurtosis_norm +
                         weights['variance'] * variance_norm +
                         weights['peak'] * peak_norm +
                         weights['fp2'] * fp2_weight +
                         weights['artifact'] * artifact_norm)
                         
        score_threshold = np.percentile(combined_score, 90)
        top_artifact_indices = np.where(combined_score > score_threshold)[0].tolist()
        if not top_artifact_indices or len(top_artifact_indices) < 2:
            top_artifact_indices = np.argsort(combined_score)[-min(2, n_components):].tolist()
        ica.exclude = top_artifact_indices
    else:
        score_threshold = np.percentile(kurtosis_values, 90)
        top_artifact_indices = np.where(kurtosis_values > score_threshold)[0].tolist()
        if not top_artifact_indices:
            top_artifact_indices = np.argsort(kurtosis_values)[-min(2, n_components):].tolist()
        ica.exclude = top_artifact_indices
        
    raw_clean = ica.apply(raw.copy(), verbose=False)
    cleaned_data_raw = raw_clean.get_data()
    cleaned_data = cleaned_data_raw.T.reshape(n_segments, n_samples, n_channels)
    
    print(f"ICA ({method}): Time={time.time()-start_time:.2f}s, Output shape={cleaned_data.shape}")
    return cleaned_data, data

def compare_ica_methods(X_raw, X_cleaned_ica, X_cleaned_weighted, fs=128):
    """Compare standard ICA and weighted ICA using SNR in multiple EEG bands."""
    nperseg = min(X_raw.shape[1], 512)
    
    f, psd_raw = signal.welch(X_raw, fs=fs, nperseg=nperseg, axis=1, scaling='density')
    _, psd_ica = signal.welch(X_cleaned_ica, fs=fs, nperseg=nperseg, axis=1, scaling='density')
    _, psd_weighted = signal.welch(X_cleaned_weighted, fs=fs, nperseg=nperseg, axis=1, scaling='density')
    
    signal_bands = [(1, 4), (4, 8), (8, 13), (13, 30)]
    noise_band = (45, 60)
    
    signal_power_ica = 0
    signal_power_weighted = 0
    band_weights = [0.2, 0.3, 0.3, 0.2]
    
    for (low, high), w in zip(signal_bands, band_weights):
        band_mask = (f >= low) & (f <= high)
        signal_power_ica += np.mean(psd_ica[:, band_mask, :], axis=1) * w
        signal_power_weighted += np.mean(psd_weighted[:, band_mask, :], axis=1) * w
        
    signal_power_ica = signal_power_ica + 1e-10
    signal_power_weighted = signal_power_weighted + 1e-10
    
    noise_mask = (f >= noise_band[0]) & (f <= noise_band[1])
    noise_power_ica = np.mean(psd_ica[:, noise_mask, :], axis=1) + 1e-9
    noise_power_weighted = np.mean(psd_weighted[:, noise_mask, :], axis=1) + 1e-9
    
    snr_ica = 10 * np.log10(np.clip(signal_power_ica / noise_power_ica, 1e-10, 1e10))
    snr_weighted = 10 * np.log10(np.clip(signal_power_weighted / noise_power_weighted, 1e-10, 1e10))
    
    snr_ica_mean = np.mean(snr_ica)
    snr_weighted_mean = np.mean(snr_weighted)
    
    print(f"Standard ICA SNR (mean): {snr_ica_mean:.4f} dB")
    print(f"Weighted ICA SNR (mean): {snr_weighted_mean:.4f} dB")
    return snr_ica_mean, snr_weighted_mean

def find_optimal_weights(X, X_raw, fs=128, n_components=19):
    """Perform random search to find optimal weights for Weighted ICA, maximizing SNR."""
    print("Starting random search for optimal weights...")
    start_time = time.time()
    
    n_iterations = 15  # Scaled down slightly for faster execution, can be configured
    best_snr_weighted = -np.inf
    best_snr_diff = -np.inf
    best_weights = None
    
    for i in range(n_iterations):
        weights = {
            'kurtosis': np.random.uniform(0.4, 0.5),
            'variance': np.random.uniform(0.1, 0.2),
            'peak': np.random.uniform(0.15, 0.25),
            'fp2': np.random.uniform(0.05, 0.15),
            'artifact': np.random.uniform(0.15, 0.25)
        }
        total = sum(weights.values())
        weights = {k: v / total for k, v in weights.items()}
        try:
            X_cleaned_weighted, _ = apply_ica(X, fs=fs, n_components=n_components, method='weighted', weights=weights)
            X_cleaned_ica, _ = apply_ica(X, fs=fs, n_components=n_components, method='standard')
            snr_ica, snr_weighted = compare_ica_methods(X_raw, X_cleaned_ica, X_cleaned_weighted, fs=fs)
            snr_diff = snr_weighted - snr_ica
            if snr_weighted > best_snr_weighted:
                best_snr_weighted = snr_weighted
                best_snr_diff = snr_diff
                best_weights = weights
                print(f"Iter {i+1} best composite SNR: {snr_weighted:.4f} dB (Diff over standard: {snr_diff:.4f} dB)")
        except Exception as e:
            print(f"Iteration {i+1} failed: {str(e)}")
            continue
            
    print(f"Random search completed in {time.time()-start_time:.2f}s")
    return best_weights, best_snr_weighted, best_snr_diff
