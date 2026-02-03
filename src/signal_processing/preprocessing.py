"""
EMG Signal Preprocessing

This module handles signal conditioning before feature extraction:
- Missing value imputation
- Artifact detection and removal
- Normalization

All preprocessing maintains statistics from training data to ensure
consistent transformations during inference.
"""
import numpy as np
from typing import Optional, Tuple, Dict
import joblib

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import (
    ARTIFACT_ZSCORE_THRESHOLD,
    NORMALIZATION_STATS_PATH,
    NUM_EMG_CHANNELS
)


class EMGPreprocessor:
    """
    Preprocessor for EMG signals that handles imputation, artifact removal,
    and normalization.
    
    The preprocessor can operate in two modes:
    1. Training mode: Computes and stores statistics from training data
    2. Inference mode: Uses stored statistics for consistent transformations
    
    Attributes:
        channel_means: Mean values per channel (for imputation)
        channel_stds: Standard deviations per channel (for artifact detection)
        min_vals: Minimum values per channel (for normalization)
        max_vals: Maximum values per channel (for normalization)
        is_fitted: Whether the preprocessor has been fitted on training data
    """
    
    def __init__(self, num_channels: int = NUM_EMG_CHANNELS):
        """
        Initialize the preprocessor.
        
        Args:
            num_channels: Number of EMG channels to process
        """
        self.num_channels = num_channels
        
        # Statistics computed during fitting
        self.channel_means: Optional[np.ndarray] = None
        self.channel_stds: Optional[np.ndarray] = None
        self.min_vals: Optional[np.ndarray] = None
        self.max_vals: Optional[np.ndarray] = None
        
        self.is_fitted = False
    
    def fit(self, data: np.ndarray) -> 'EMGPreprocessor':
        """
        Compute preprocessing statistics from training data.
        
        Args:
            data: Training data of shape (n_samples, num_channels)
            
        Returns:
            self (for method chaining)
            
        This method must be called before transform() during training.
        Statistics computed here are reused during inference.
        """
        # Handle any existing NaN values for statistics computation
        # using numpy's nanmean/nanstd which ignore NaN values
        self.channel_means = np.nanmean(data, axis=0)
        self.channel_stds = np.nanstd(data, axis=0)
        
        # Prevent division by zero in case of constant channels
        self.channel_stds = np.where(self.channel_stds == 0, 1.0, self.channel_stds)
        
        # Compute min/max for normalization after handling imputation
        # First impute missing values, then find min/max
        imputed_data = self._impute_missing(data.copy())
        artifact_removed = self._remove_artifacts(imputed_data)
        
        self.min_vals = np.min(artifact_removed, axis=0)
        self.max_vals = np.max(artifact_removed, axis=0)
        
        # Ensure min != max to prevent division by zero
        range_vals = self.max_vals - self.min_vals
        self.max_vals = np.where(range_vals == 0, self.min_vals + 1.0, self.max_vals)
        
        self.is_fitted = True
        return self
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Apply preprocessing pipeline to data.
        
        Args:
            data: EMG data of shape (n_samples, num_channels) or (num_channels,)
            
        Returns:
            Preprocessed data with same shape
            
        Pipeline:
        1. Mean imputation for missing/NaN values
        2. Z-score artifact detection with capping
        3. Min-Max normalization to [0, 1]
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transform. Call fit() first.")
        
        # Handle single sample case
        single_sample = data.ndim == 1
        if single_sample:
            data = data.reshape(1, -1)
        
        # Step 1: Impute missing values using channel means from training
        processed = self._impute_missing(data.copy())
        
        # Step 2: Detect and cap artifacts using Z-score thresholding
        processed = self._remove_artifacts(processed)
        
        # Step 3: Apply Min-Max normalization using training statistics
        processed = self._normalize(processed)
        
        # Restore original shape if single sample
        if single_sample:
            processed = processed.flatten()
        
        return processed
    
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Convenience method to fit and transform in one call.
        
        Args:
            data: Training data of shape (n_samples, num_channels)
            
        Returns:
            Preprocessed training data
        """
        return self.fit(data).transform(data)
    
    def _impute_missing(self, data: np.ndarray) -> np.ndarray:
        """
        Replace missing values (NaN, None, 0) with channel means.
        
        Args:
            data: Data with potential missing values
            
        Returns:
            Data with missing values imputed
            
        Mean imputation is used because:
        - It's computationally efficient
        - EMG signals are approximately normally distributed
        - It preserves channel-level characteristics
        
        For EMG, we also treat exact zeros as potentially missing
        since true EMG signals rarely have exactly zero amplitude.
        """
        # Replace NaN and None with mean
        nan_mask = np.isnan(data) | (data == 0)
        
        for channel in range(self.num_channels):
            channel_nan_mask = nan_mask[:, channel]
            if np.any(channel_nan_mask):
                data[channel_nan_mask, channel] = self.channel_means[channel]
        
        return data
    
    def _remove_artifacts(self, data: np.ndarray) -> np.ndarray:
        """
        Detect and cap artifacts using Z-score thresholding.
        
        Args:
            data: EMG data after imputation
            
        Returns:
            Data with artifacts capped
            
        Artifacts in EMG signals can come from:
        - Motion artifacts (electrode movement)
        - Power line interference (50/60 Hz)
        - Muscle crosstalk
        - Equipment noise
        
        Z-score thresholding identifies values that deviate significantly
        from the expected distribution and caps them to prevent
        downstream processing issues.
        """
        # Compute Z-scores using training statistics
        z_scores = (data - self.channel_means) / self.channel_stds
        
        # Find values exceeding threshold
        artifact_mask = np.abs(z_scores) > ARTIFACT_ZSCORE_THRESHOLD
        
        # Cap positive artifacts to mean + threshold * std
        positive_mask = artifact_mask & (z_scores > 0)
        data[positive_mask] = (self.channel_means + 
                               ARTIFACT_ZSCORE_THRESHOLD * self.channel_stds)[np.where(positive_mask)[1]]
        
        # Cap negative artifacts to mean - threshold * std
        negative_mask = artifact_mask & (z_scores < 0)
        data[negative_mask] = (self.channel_means - 
                               ARTIFACT_ZSCORE_THRESHOLD * self.channel_stds)[np.where(negative_mask)[1]]
        
        return data
    
    def _normalize(self, data: np.ndarray) -> np.ndarray:
        """
        Apply Min-Max normalization to scale data to [0, 1].
        
        Args:
            data: Data after imputation and artifact removal
            
        Returns:
            Normalized data in range [0, 1]
            
        Min-Max normalization is used because:
        - EMG amplitudes are inherently non-negative
        - It preserves relative patterns within each channel
        - It's interpretable (0 = min activity, 1 = max activity)
        
        IMPORTANT: Uses min/max from training data, not the current batch.
        This ensures consistent normalization between training and inference.
        """
        # Apply Min-Max: (x - min) / (max - min)
        normalized = (data - self.min_vals) / (self.max_vals - self.min_vals)
        
        # Clip to [0, 1] in case inference data exceeds training bounds
        normalized = np.clip(normalized, 0.0, 1.0)
        
        return normalized
    
    def save(self, filepath: str = NORMALIZATION_STATS_PATH) -> None:
        """
        Save preprocessing statistics to disk.
        
        Args:
            filepath: Path to save the statistics
            
        These statistics must be loaded for inference to ensure
        consistent transformations.
        """
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted preprocessor")
        
        stats = {
            'channel_means': self.channel_means,
            'channel_stds': self.channel_stds,
            'min_vals': self.min_vals,
            'max_vals': self.max_vals,
            'num_channels': self.num_channels
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(stats, filepath)
    
    def load(self, filepath: str = NORMALIZATION_STATS_PATH) -> 'EMGPreprocessor':
        """
        Load preprocessing statistics from disk.
        
        Args:
            filepath: Path to load the statistics from
            
        Returns:
            self (for method chaining)
        """
        stats = joblib.load(filepath)
        
        self.channel_means = stats['channel_means']
        self.channel_stds = stats['channel_stds']
        self.min_vals = stats['min_vals']
        self.max_vals = stats['max_vals']
        self.num_channels = stats['num_channels']
        self.is_fitted = True
        
        return self
    
    def get_stats(self) -> Dict[str, np.ndarray]:
        """
        Get the computed preprocessing statistics.
        
        Returns:
            Dictionary containing all statistics
        """
        return {
            'channel_means': self.channel_means,
            'channel_stds': self.channel_stds,
            'min_vals': self.min_vals,
            'max_vals': self.max_vals
        }
