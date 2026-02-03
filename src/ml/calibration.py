"""
Session-Based EMG Calibration

This module handles per-session calibration to correct for:
- Electrode placement differences between sessions
- Skin conductivity variations
- Ambient noise levels
- Amplitude drift over time

The calibrator captures a "rest" baseline at session start and uses it
to normalize subsequent readings.
"""
import numpy as np
from typing import Optional, Dict
import time

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import CALIBRATION_WINDOW_SIZE, NUM_EMG_CHANNELS


class SessionCalibrator:
    """
    Per-session calibration for EMG amplitude drift correction.
    
    The calibration process:
    1. User maintains a relaxed, rest position
    2. Collect CALIBRATION_WINDOW_SIZE samples
    3. Compute per-channel baseline statistics
    4. Apply correction to subsequent readings
    
    This compensates for day-to-day and session-to-session variations
    that would otherwise reduce prediction accuracy.
    
    Attributes:
        baseline_mean: Mean amplitude per channel during rest
        baseline_std: Standard deviation per channel during rest
        is_calibrated: Whether calibration has been performed
        calibration_timestamp: When calibration was performed
    """
    
    def __init__(self, num_channels: int = NUM_EMG_CHANNELS):
        """
        Initialize the calibrator.
        
        Args:
            num_channels: Number of EMG channels
        """
        self.num_channels = num_channels
        self.baseline_mean: Optional[np.ndarray] = None
        self.baseline_std: Optional[np.ndarray] = None
        self.is_calibrated = False
        self.calibration_timestamp: Optional[float] = None
        
        # Buffer for collecting calibration samples
        self._calibration_buffer: list = []
        self._target_samples = CALIBRATION_WINDOW_SIZE
    
    def add_calibration_sample(self, sample: np.ndarray) -> bool:
        """
        Add a sample to the calibration buffer.
        
        Args:
            sample: Single EMG sample of shape (num_channels,)
            
        Returns:
            True if calibration is complete, False if more samples needed
            
        Call this repeatedly with rest samples until it returns True.
        """
        self._calibration_buffer.append(sample.copy())
        
        if len(self._calibration_buffer) >= self._target_samples:
            return self._complete_calibration()
        
        return False
    
    def calibrate_from_batch(self, rest_samples: np.ndarray) -> bool:
        """
        Perform calibration from a batch of rest samples.
        
        Args:
            rest_samples: Array of shape (n_samples, num_channels)
            
        Returns:
            True if calibration successful, False otherwise
            
        Use this for immediate calibration with pre-collected data.
        """
        if rest_samples.shape[1] != self.num_channels:
            print(f"Invalid sample shape: expected {self.num_channels} channels")
            return False
        
        self._calibration_buffer = list(rest_samples)
        return self._complete_calibration()
    
    def _complete_calibration(self) -> bool:
        """
        Complete the calibration by computing baseline statistics.
        
        Returns:
            True if successful
        """
        if len(self._calibration_buffer) < 10:
            # Need minimum samples for reliable statistics
            return False
        
        # Convert buffer to array
        buffer_array = np.array(self._calibration_buffer)
        
        # Compute per-channel statistics
        self.baseline_mean = np.mean(buffer_array, axis=0)
        self.baseline_std = np.std(buffer_array, axis=0)
        
        # Prevent division by zero
        self.baseline_std = np.where(self.baseline_std == 0, 1.0, self.baseline_std)
        
        self.is_calibrated = True
        self.calibration_timestamp = time.time()
        
        # Clear buffer
        self._calibration_buffer = []
        
        print(f"Calibration complete with {len(buffer_array)} samples")
        print(f"Baseline means: {self.baseline_mean}")
        
        return True
    
    def apply_correction(self, data: np.ndarray) -> np.ndarray:
        """
        Apply calibration correction to EMG data.
        
        Args:
            data: EMG data of shape (num_channels,) or (n_samples, num_channels)
            
        Returns:
            Calibrated data with same shape
            
        The correction removes the rest baseline, effectively converting
        the signal to "activity relative to rest". This helps normalize
        across different electrode placements and skin conditions.
        """
        if not self.is_calibrated:
            # No calibration available, return unchanged
            return data
        
        # Handle single sample
        single_sample = data.ndim == 1
        if single_sample:
            data = data.reshape(1, -1)
        
        # Apply baseline subtraction
        # This centers the signal around the rest baseline
        corrected = data - self.baseline_mean
        
        # Optional: Scale by baseline std to normalize variance
        # This makes the system more robust to different signal amplitudes
        # corrected = corrected / self.baseline_std
        
        # Re-center to positive range (EMG shouldn't be negative)
        # We add a small offset to avoid exactly zero values
        corrected = corrected + 0.5  # Center around 0.5
        
        # Clip to valid range
        corrected = np.clip(corrected, 0.0, 1.0)
        
        if single_sample:
            corrected = corrected.flatten()
        
        return corrected
    
    def reset(self) -> None:
        """
        Reset calibration state for a new session.
        
        Call this when starting a new usage session.
        """
        self.baseline_mean = None
        self.baseline_std = None
        self.is_calibrated = False
        self.calibration_timestamp = None
        self._calibration_buffer = []
    
    def get_calibration_progress(self) -> Dict[str, any]:
        """
        Get the current calibration progress.
        
        Returns:
            Dictionary with progress information
        """
        return {
            'is_calibrated': self.is_calibrated,
            'samples_collected': len(self._calibration_buffer),
            'samples_needed': self._target_samples,
            'progress_percent': min(100, int(len(self._calibration_buffer) / self._target_samples * 100)),
            'calibration_timestamp': self.calibration_timestamp
        }
    
    def get_baseline_info(self) -> Optional[Dict[str, np.ndarray]]:
        """
        Get the computed baseline statistics.
        
        Returns:
            Dictionary with baseline mean and std, or None if not calibrated
        """
        if not self.is_calibrated:
            return None
        
        return {
            'baseline_mean': self.baseline_mean.tolist(),
            'baseline_std': self.baseline_std.tolist(),
            'calibrated_at': self.calibration_timestamp
        }
