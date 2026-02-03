"""
EMG Feature Extraction

This module extracts time-domain features from preprocessed EMG signals.
Features are designed to capture muscle activation patterns that distinguish
between different gestures.

Features extracted per channel:
- Mean Absolute Value (MAV): Average amplitude indicating overall activation
- Root Mean Square (RMS): Power-related measure of muscle activity

Total features: 16 (2 features × 8 channels)
"""
import numpy as np
from typing import Optional, Tuple

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import NUM_EMG_CHANNELS, FEATURES_PER_CHANNEL, TOTAL_FEATURES


class FeatureExtractor:
    """
    Extracts time-domain features from EMG signals.
    
    This class computes two features per channel:
    1. Mean Absolute Value (MAV): Simple and robust amplitude measure
    2. Root Mean Square (RMS): Captures signal energy/power
    
    Together, these 16 features provide sufficient information to
    distinguish between gestures while remaining computationally efficient
    for real-time inference.
    
    Why these features?
    ------------------
    - MAV and RMS are proven effective for EMG gesture recognition
    - They're computationally cheap (important for <100ms latency target)
    - They're relatively robust to noise after preprocessing
    - More complex features (frequency domain) showed minimal improvement
      in preliminary testing for this gesture set
    
    Attributes:
        num_channels: Number of EMG channels
        features_per_channel: Number of features computed per channel
    """
    
    def __init__(self, num_channels: int = NUM_EMG_CHANNELS):
        """
        Initialize the feature extractor.
        
        Args:
            num_channels: Number of EMG channels to process
        """
        self.num_channels = num_channels
        self.features_per_channel = FEATURES_PER_CHANNEL
        self.total_features = self.num_channels * self.features_per_channel
    
    def extract(self, data: np.ndarray) -> np.ndarray:
        """
        Extract features from EMG data.
        
        Args:
            data: EMG data of shape (n_samples, num_channels) for windowed data,
                  or (num_channels,) for a single sample
                  
        Returns:
            Feature vector of shape (total_features,) for single sample/window,
            or (n_windows, total_features) for multiple windows
            
        For a single sample, MAV and RMS are computed as the absolute value
        and the value itself (since N=1).
        
        For windowed data, features are computed over the window.
        """
        # Handle single sample case - treat as a window of 1
        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        features = []
        
        for channel in range(self.num_channels):
            channel_data = data[:, channel]
            
            # Mean Absolute Value (MAV)
            # MAV = (1/N) * sum(|x_i|)
            # Represents average muscle activation level
            mav = self._compute_mav(channel_data)
            
            # Root Mean Square (RMS)
            # RMS = sqrt((1/N) * sum(x_i^2))
            # Related to signal power, captures activation intensity
            rms = self._compute_rms(channel_data)
            
            features.extend([mav, rms])
        
        return np.array(features)
    
    def extract_batch(self, data: np.ndarray, window_size: int = 1) -> np.ndarray:
        """
        Extract features from multiple windows of EMG data.
        
        Args:
            data: EMG data of shape (n_samples, num_channels)
            window_size: Number of samples per window (1 for sample-by-sample)
            
        Returns:
            Feature matrix of shape (n_windows, total_features)
            
        This is useful for batch processing of CSV files or training data.
        """
        if window_size == 1:
            # For window_size=1, each sample becomes its own feature vector
            features_list = []
            for i in range(len(data)):
                features = self.extract(data[i])
                features_list.append(features)
            return np.array(features_list)
        
        # For larger windows, compute features over sliding windows
        n_samples = len(data)
        n_windows = n_samples // window_size
        
        features_list = []
        for i in range(n_windows):
            start_idx = i * window_size
            end_idx = start_idx + window_size
            window_data = data[start_idx:end_idx]
            features = self.extract(window_data)
            features_list.append(features)
        
        return np.array(features_list)
    
    def _compute_mav(self, signal: np.ndarray) -> float:
        """
        Compute Mean Absolute Value of a signal.
        
        Args:
            signal: 1D array of signal values
            
        Returns:
            MAV value
            
        MAV is widely used in EMG analysis because:
        - It's simple and fast to compute
        - It's relatively resistant to noise
        - It correlates well with muscle force
        """
        return np.mean(np.abs(signal))
    
    def _compute_rms(self, signal: np.ndarray) -> float:
        """
        Compute Root Mean Square of a signal.
        
        Args:
            signal: 1D array of signal values
            
        Returns:
            RMS value
            
        RMS is preferred for power-related analysis because:
        - It represents signal energy
        - It's more sensitive to peaks than MAV
        - It's the standard measure for AC signal magnitude
        """
        return np.sqrt(np.mean(signal ** 2))
    
    def get_feature_names(self) -> list:
        """
        Get descriptive names for all features.
        
        Returns:
            List of feature names like ['ch1_mav', 'ch1_rms', 'ch2_mav', ...]
            
        Useful for debugging and feature importance analysis.
        """
        names = []
        for ch in range(1, self.num_channels + 1):
            names.append(f'ch{ch}_mav')
            names.append(f'ch{ch}_rms')
        return names
    
    def get_channel_feature_indices(self, channel: int) -> Tuple[int, int]:
        """
        Get the feature indices for a specific channel.
        
        Args:
            channel: Channel number (0-indexed)
            
        Returns:
            Tuple of (mav_index, rms_index) in the feature vector
            
        Useful for analyzing channel-specific contributions.
        """
        base_idx = channel * self.features_per_channel
        return (base_idx, base_idx + 1)


def extract_features_from_raw(raw_data: np.ndarray, preprocessor=None) -> np.ndarray:
    """
    Convenience function to preprocess and extract features in one call.
    
    Args:
        raw_data: Raw EMG data of shape (n_samples, num_channels)
        preprocessor: Optional fitted EMGPreprocessor instance
        
    Returns:
        Feature matrix of shape (n_samples, total_features)
        
    This is the typical pipeline for inference:
    raw data -> preprocess -> extract features -> model prediction
    """
    if preprocessor is not None:
        data = preprocessor.transform(raw_data)
    else:
        data = raw_data
    
    extractor = FeatureExtractor()
    return extractor.extract_batch(data)
