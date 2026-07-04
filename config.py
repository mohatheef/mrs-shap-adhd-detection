import os

# ==========================================
# DATASET PATHS
# ==========================================
# Default directories on local system. Please modify these if your dataset is located elsewhere.
ADHD_PART1 = os.environ.get("ADHD_PART1", "dataset/ADHD_part1")
ADHD_PART2 = os.environ.get("ADHD_PART2", "dataset/ADHD_part2")
CONTROL_PART1 = os.environ.get("CONTROL_PART1", "dataset/Control_part1")
CONTROL_PART2 = os.environ.get("CONTROL_PART2", "dataset/Control_part2")

# ==========================================
# SIGNAL ACQUISITION PARAMETERS
# ==========================================
FS = 128                 # Sampling frequency (Hz)
WINDOW_SIZE = 512        # Sliding window size (epochs)
STRIDE = 256            # Stride size (50% overlap)
EXPECTED_CHANNELS = 19   # Standard 10-20 EEG configuration

# Channel names matching the 10-20 standard placement system
CHANNEL_NAMES = [
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", 
    "O1", "O2", "F7", "F8", "T3", "T4", "T5", "T6", 
    "Fz", "Cz", "Pz"
]
