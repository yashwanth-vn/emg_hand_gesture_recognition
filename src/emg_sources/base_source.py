"""
Abstract Base Class for EMG Data Sources

This module defines the contract that all EMG input sources must implement.
By abstracting the data source, we ensure that preprocessing, feature extraction,
and inference work identically regardless of whether data comes from:
- Uploaded CSV files
- Simulated EMG streams
- Real hardware sensors

HARDWARE INTEGRATION GUIDE:
---------------------------
When connecting real EMG sensors, implement a new class that inherits from
EMGSource. The key methods to implement are:
1. get_sample() - Return a single reading from all 8 channels
2. get_batch() - Return multiple readings (for batch processing)
3. is_streaming() - Indicate if this is a live stream
4. start_stream() / stop_stream() - Control live acquisition

See hardware_source.py for a template with detailed integration points.
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Generator
import numpy as np


class EMGSource(ABC):
    """
    Abstract base class defining the interface for all EMG data sources.
    
    All EMG sources must provide 8-channel float values between 0 and 1
    (or values that can be normalized to this range during preprocessing).
    
    Attributes:
        num_channels: Number of EMG channels (always 8 for this system)
        is_active: Whether the source is currently providing data
    """
    
    def __init__(self, num_channels: int = 8):
        """
        Initialize the EMG source.
        
        Args:
            num_channels: Number of EMG channels to expect. Default is 8
                         to match typical surface EMG sensor configurations.
        """
        # The system is designed for 8-channel EMG which provides good
        # spatial resolution while remaining practical for wearable devices
        self.num_channels = num_channels
        self.is_active = False
    
    @abstractmethod
    def get_sample(self) -> Optional[np.ndarray]:
        """
        Get a single EMG sample from all channels.
        
        Returns:
            numpy array of shape (num_channels,) containing float values,
            or None if no data is available.
            
        Each value represents the EMG amplitude for that channel at this
        moment in time. Values should ideally be in [0, 1] range but
        preprocessing will normalize if needed.
        """
        pass
    
    @abstractmethod
    def get_batch(self, batch_size: int) -> Optional[np.ndarray]:
        """
        Get multiple EMG samples at once (for batch processing).
        
        Args:
            batch_size: Number of samples to retrieve
            
        Returns:
            numpy array of shape (batch_size, num_channels), or
            numpy array of shape (actual_samples, num_channels) if fewer
            samples are available, or None if no data.
            
        This method is used for:
        - Processing uploaded CSV files
        - Batch inference for efficiency
        - Training data loading
        """
        pass
    
    @abstractmethod
    def is_streaming(self) -> bool:
        """
        Check if this source provides real-time streaming data.
        
        Returns:
            True if this is a live stream (simulated or hardware),
            False if this is a batch source (CSV file).
            
        The streaming flag affects how the frontend handles updates and
        whether SSE (Server-Sent Events) should be used.
        """
        pass
    
    def start_stream(self) -> bool:
        """
        Start the data stream (for live sources).
        
        Returns:
            True if stream started successfully, False otherwise.
            
        Override this method for sources that need initialization,
        such as opening serial ports or starting sensor sampling.
        """
        self.is_active = True
        return True
    
    def stop_stream(self) -> None:
        """
        Stop the data stream and clean up resources.
        
        Override this method to properly close connections,
        stop sampling threads, or release hardware resources.
        """
        self.is_active = False
    
    def stream_samples(self) -> Generator[np.ndarray, None, None]:
        """
        Generator that yields samples continuously (for live sources).
        
        Yields:
            numpy array of shape (num_channels,) for each sample
            
        This generator pattern allows the Flask backend to efficiently
        stream data using Server-Sent Events (SSE).
        
        Default implementation calls get_sample() repeatedly.
        Override for more efficient implementations if needed.
        """
        while self.is_active:
            sample = self.get_sample()
            if sample is not None:
                yield sample
    
    def validate_sample(self, sample: np.ndarray) -> bool:
        """
        Validate that a sample has the expected shape and reasonable values.
        
        Args:
            sample: The EMG sample to validate
            
        Returns:
            True if sample is valid, False otherwise
            
        This helps catch data corruption or sensor malfunction early.
        """
        if sample is None:
            return False
        if sample.shape != (self.num_channels,):
            return False
        # Check for NaN or infinite values that would break processing
        if not np.isfinite(sample).all():
            return False
        return True
    
    def get_source_info(self) -> dict:
        """
        Get metadata about this EMG source.
        
        Returns:
            Dictionary containing source type, channels, and status.
            
        Useful for debugging and UI display.
        """
        return {
            'source_type': self.__class__.__name__,
            'num_channels': self.num_channels,
            'is_streaming': self.is_streaming(),
            'is_active': self.is_active
        }
