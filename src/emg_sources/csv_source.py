"""
CSV File EMG Source

This module handles EMG data from uploaded CSV files, providing batch
processing capabilities for offline analysis and testing.

Expected CSV format:
- 8 columns (one per EMG channel)
- No headers or labels (raw data only)
- No timestamps (each row is one sample)
- Float values representing EMG amplitude
"""
import numpy as np
import pandas as pd
from typing import Optional, List
from io import StringIO, BytesIO

from .base_source import EMGSource


class CSVSource(EMGSource):
    """
    EMG source that reads data from CSV files.
    
    This is the primary interface for batch processing uploaded files.
    Users can upload their EMG recordings for gesture classification
    without needing live hardware.
    
    Attributes:
        data: numpy array containing all loaded EMG samples
        current_index: Position in the data for sequential reading
    """
    
    def __init__(self, num_channels: int = 8):
        """
        Initialize CSV source.
        
        Args:
            num_channels: Expected number of EMG channels (default 8)
        """
        super().__init__(num_channels)
        self.data: Optional[np.ndarray] = None
        self.current_index = 0
    
    def load_from_file(self, file_content: bytes) -> bool:
        """
        Load EMG data from uploaded file content.
        
        Args:
            file_content: Raw bytes from the uploaded CSV file
            
        Returns:
            True if data loaded successfully, False otherwise
            
        This method parses the CSV, validates the structure, and stores
        the data for subsequent access via get_sample() and get_batch().
        """
        try:
            # Decode bytes to string for pandas parsing
            # Handle both UTF-8 and Latin-1 encodings commonly used in data files
            try:
                content_str = file_content.decode('utf-8')
            except UnicodeDecodeError:
                content_str = file_content.decode('latin-1')
            
            # Parse CSV - we expect no header since format specifies raw data
            # However, some files may have headers so we detect and handle both
            df = pd.read_csv(StringIO(content_str), header=None)
            
            # Check if first row looks like a header (contains non-numeric data)
            first_row = df.iloc[0]
            if first_row.dtype == object or not pd.to_numeric(first_row, errors='coerce').notna().all():
                # First row is likely a header, skip it
                df = df.iloc[1:]
            
            # Convert to float, handling any string values
            df = df.apply(pd.to_numeric, errors='coerce')
            
            # Validate column count matches expected channels
            if df.shape[1] != self.num_channels:
                raise ValueError(
                    f"CSV has {df.shape[1]} columns but expected {self.num_channels} channels. "
                    "Please ensure your CSV has exactly 8 EMG channel columns."
                )
            
            # Convert to numpy array and store
            self.data = df.values.astype(np.float64)
            self.current_index = 0
            self.is_active = True
            
            return True
            
        except Exception as e:
            # Log the error for debugging but don't crash
            print(f"Error loading CSV: {e}")
            self.data = None
            return False
    
    def load_from_array(self, data: np.ndarray) -> bool:
        """
        Load EMG data from a numpy array directly.
        
        Args:
            data: numpy array of shape (n_samples, num_channels)
            
        Returns:
            True if data loaded successfully, False otherwise
            
        Useful for programmatic data loading and testing.
        """
        if data.ndim != 2 or data.shape[1] != self.num_channels:
            print(f"Invalid data shape: {data.shape}. Expected (n, {self.num_channels})")
            return False
        
        self.data = data.astype(np.float64)
        self.current_index = 0
        self.is_active = True
        return True
    
    def get_sample(self) -> Optional[np.ndarray]:
        """
        Get the next EMG sample from the loaded CSV data.
        
        Returns:
            numpy array of shape (num_channels,) or None if no more data
            
        Samples are returned sequentially. Call reset() to start over.
        """
        if self.data is None or self.current_index >= len(self.data):
            return None
        
        sample = self.data[self.current_index]
        self.current_index += 1
        return sample
    
    def get_batch(self, batch_size: int) -> Optional[np.ndarray]:
        """
        Get a batch of EMG samples from the loaded CSV data.
        
        Args:
            batch_size: Maximum number of samples to return
            
        Returns:
            numpy array of shape (actual_size, num_channels) where
            actual_size <= batch_size depending on remaining data
        """
        if self.data is None or self.current_index >= len(self.data):
            return None
        
        # Calculate how many samples we can actually return
        remaining = len(self.data) - self.current_index
        actual_size = min(batch_size, remaining)
        
        batch = self.data[self.current_index:self.current_index + actual_size]
        self.current_index += actual_size
        return batch
    
    def get_all_data(self) -> Optional[np.ndarray]:
        """
        Get all loaded EMG data at once.
        
        Returns:
            numpy array of shape (n_samples, num_channels) or None
            
        This returns the full dataset without affecting the current_index.
        Useful for batch inference on entire files.
        """
        return self.data.copy() if self.data is not None else None
    
    def is_streaming(self) -> bool:
        """
        CSV is a batch source, not a streaming source.
        
        Returns:
            Always False for CSV sources
        """
        return False
    
    def reset(self) -> None:
        """
        Reset the read position to the beginning of the data.
        
        Call this to re-process the same CSV file from the start.
        """
        self.current_index = 0
    
    def get_sample_count(self) -> int:
        """
        Get the total number of samples in the loaded CSV.
        
        Returns:
            Number of samples, or 0 if no data loaded
        """
        return len(self.data) if self.data is not None else 0
    
    def get_remaining_samples(self) -> int:
        """
        Get the number of samples not yet read.
        
        Returns:
            Number of remaining samples
        """
        if self.data is None:
            return 0
        return max(0, len(self.data) - self.current_index)
