"""
Configuration settings for EMG Gesture Recognition Application.

This module centralizes all configurable parameters to make the system
easy to tune and deploy in different environments.
"""
import os

# =============================================================================
# PATHS CONFIGURATION
# =============================================================================
# Base directory of the application
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to training data - uses the existing dataset from the adjacent project
DATA_DIR = os.path.join(BASE_DIR, 'data')
TRAINING_DATA_PATH = os.path.join(DATA_DIR, 'emg_data.csv')

# Directory for persisted models and preprocessing artifacts
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Model file paths
RANDOM_FOREST_MODEL_PATH = os.path.join(MODELS_DIR, 'random_forest.joblib')
HISTGB_MODEL_PATH = os.path.join(MODELS_DIR, 'hist_gradient_boosting.joblib')
LOGISTIC_REGRESSION_MODEL_PATH = os.path.join(MODELS_DIR, 'logistic_regression.joblib')

# Preprocessing artifacts
NORMALIZATION_STATS_PATH = os.path.join(MODELS_DIR, 'normalization_stats.joblib')
LABEL_ENCODER_PATH = os.path.join(MODELS_DIR, 'label_encoder.joblib')

# =============================================================================
# EMG SIGNAL CONFIGURATION
# =============================================================================
# Number of EMG channels expected from the sensor/input
NUM_EMG_CHANNELS = 8

# Gesture classes supported by the system
GESTURE_CLASSES = ['fist', 'open', 'pinch', 'rest']

# =============================================================================
# SIGNAL PROCESSING CONFIGURATION
# =============================================================================
# Z-score threshold for artifact detection and removal
# Values beyond this threshold are considered artifacts and capped
ARTIFACT_ZSCORE_THRESHOLD = 3.0

# Minimum and maximum values for Min-Max normalization bounds
# These are used if training data doesn't provide reasonable bounds
NORMALIZATION_MIN_DEFAULT = 0.0
NORMALIZATION_MAX_DEFAULT = 1.0

# =============================================================================
# FEATURE EXTRACTION CONFIGURATION
# =============================================================================
# Number of features per channel (MAV + RMS = 2)
FEATURES_PER_CHANNEL = 2

# Total features = channels * features_per_channel
TOTAL_FEATURES = NUM_EMG_CHANNELS * FEATURES_PER_CHANNEL  # 16

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================
# Train/test split ratio (80% training, 20% testing)
TRAIN_TEST_SPLIT_RATIO = 0.2

# Random seed for reproducibility
RANDOM_SEED = 42

# Random Forest hyperparameters
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = 15
RF_MIN_SAMPLES_SPLIT = 5

# Histogram Gradient Boosting hyperparameters
HISTGB_MAX_ITER = 100
HISTGB_MAX_DEPTH = 10
HISTGB_LEARNING_RATE = 0.1

# Logistic Regression hyperparameters
LR_MAX_ITER = 1000
LR_C = 1.0

# =============================================================================
# INFERENCE CONFIGURATION
# =============================================================================
# Confidence variance threshold for uncertainty detection
# If variance across model predictions exceeds this, return "uncertain"
UNCERTAINTY_VARIANCE_THRESHOLD = 0.25

# Minimum confidence required for a valid prediction
MIN_CONFIDENCE_THRESHOLD = 0.4

# Target inference latency in milliseconds
TARGET_LATENCY_MS = 100

# =============================================================================
# SIMULATED EMG STREAM CONFIGURATION
# =============================================================================
# Interval between simulated samples in milliseconds
STREAM_INTERVAL_MS = 50

# Base amplitude range for simulated signals
SIMULATED_BASE_AMPLITUDE = (0.02, 0.1)

# Noise standard deviation for simulated signals
SIMULATED_NOISE_STD = 0.01

# Amplitude drift rate per sample
SIMULATED_DRIFT_RATE = 0.001

# =============================================================================
# CALIBRATION CONFIGURATION
# =============================================================================
# Number of samples to collect for rest baseline calibration
CALIBRATION_WINDOW_SIZE = 50

# =============================================================================
# GESTURE TO ACTION MAPPING
# =============================================================================
# Maps recognized gestures to virtual actions
# This abstraction allows the same gesture recognition to control
# different systems without changing the core logic
GESTURE_ACTION_MAP = {
    'fist': 'ON',
    'open': 'OFF',
    'pinch': 'TOGGLE',
    'rest': 'IDLE',
    'uncertain': 'HOLD'  # When uncertain, maintain current state
}

# =============================================================================
# FLASK SERVER CONFIGURATION
# =============================================================================
FLASK_HOST = '0.0.0.0'
FLASK_PORT = 5000
FLASK_DEBUG = False

# Maximum file upload size (16 MB)
MAX_UPLOAD_SIZE_MB = 16
