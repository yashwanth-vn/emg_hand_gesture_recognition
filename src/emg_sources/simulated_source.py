"""
Simulated EMG Source

This module generates realistic EMG-like signals for testing and demonstration.
It mimics the behavior of real EMG sensors without requiring hardware, enabling:
- Development and debugging without physical sensors
- Consistent testing with reproducible patterns
- Demonstration of the system to stakeholders

The simulation includes realistic artifacts:
- Gaussian noise (electronic interference)
- Amplitude drift (electrode impedance changes)
- Non-stationarity (muscle fatigue effects)
"""
import numpy as np
import time
from typing import Optional, Generator
from threading import Thread, Event

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import (
    STREAM_INTERVAL_MS,
    SIMULATED_BASE_AMPLITUDE,
    SIMULATED_NOISE_STD,
    SIMULATED_DRIFT_RATE,
    GESTURE_CLASSES
)

from .base_source import EMGSource


class SimulatedSource(EMGSource):
    """
    EMG source that generates realistic simulated signals.
    
    The simulation models typical EMG characteristics:
    - Different activation patterns for each gesture
    - Realistic noise levels
    - Time-varying amplitude (drift)
    - Smooth transitions between gestures
    
    Attributes:
        current_gesture: The gesture currently being simulated
        drift_offset: Cumulative amplitude drift
        sample_count: Number of samples generated
    """
    
    def __init__(self, num_channels: int = 8):
        """
        Initialize the simulated EMG source.
        
        Args:
            num_channels: Number of EMG channels to simulate
        """
        super().__init__(num_channels)
        
        # Current simulated gesture - default to rest
        self.current_gesture = 'rest'
        
        # Drift state - accumulates over time to simulate electrode changes
        self.drift_offset = np.zeros(num_channels)
        
        # Sample count for time-based effects
        self.sample_count = 0
        
        # Stream interval in seconds
        self.stream_interval = STREAM_INTERVAL_MS / 1000.0
        
        # Threading control for async streaming
        self._stop_event = Event()
        self._stream_thread: Optional[Thread] = None
        
        # Define characteristic activation patterns for each gesture
        # These patterns represent which channels are typically active
        # for each gesture based on forearm muscle anatomy
        self._gesture_patterns = self._create_gesture_patterns()
    
    def _create_gesture_patterns(self) -> dict:
        """
        Create characteristic EMG patterns for each gesture.
        
        Returns:
            Dictionary mapping gesture names to channel activation patterns
            
        These patterns are based on typical forearm muscle activation:
        - Fist: Strong activation in flexor muscles (channels 1-4)
        - Open: Strong activation in extensor muscles (channels 5-8)
        - Pinch: Mixed activation, particularly in thumb muscles
        - Rest: Minimal activation across all channels
        """
        patterns = {
            'rest': {
                # Rest shows minimal, baseline activity
                'base_amplitude': np.array([0.03, 0.03, 0.02, 0.03, 0.02, 0.03, 0.02, 0.03]),
                'variability': 0.02
            },
            'fist': {
                # Fist engages wrist flexors strongly
                'base_amplitude': np.array([0.7, 0.6, 0.5, 0.65, 0.3, 0.25, 0.35, 0.4]),
                'variability': 0.1
            },
            'open': {
                # Open hand engages wrist extensors
                'base_amplitude': np.array([0.25, 0.3, 0.35, 0.25, 0.6, 0.7, 0.65, 0.55]),
                'variability': 0.1
            },
            'pinch': {
                # Pinch uses a combination for precision grip
                'base_amplitude': np.array([0.5, 0.4, 0.55, 0.6, 0.45, 0.5, 0.4, 0.35]),
                'variability': 0.08
            }
        }
        return patterns
    
    def set_gesture(self, gesture: str) -> bool:
        """
        Set the gesture to simulate.
        
        Args:
            gesture: One of 'fist', 'open', 'pinch', 'rest'
            
        Returns:
            True if gesture is valid, False otherwise
        """
        if gesture.lower() in GESTURE_CLASSES:
            self.current_gesture = gesture.lower()
            return True
        return False
    
    def get_sample(self) -> Optional[np.ndarray]:
        """
        Generate a single simulated EMG sample.
        
        Returns:
            numpy array of shape (num_channels,) with EMG values
            
        The sample includes:
        - Base pattern for current gesture
        - Gaussian noise (simulates electronic noise)
        - Amplitude drift (simulates electrode impedance changes)
        - Non-stationarity (simulates muscle fatigue)
        """
        if not self.is_active:
            return None
        
        pattern = self._gesture_patterns.get(self.current_gesture, self._gesture_patterns['rest'])
        
        # Start with base amplitude for this gesture
        sample = pattern['base_amplitude'].copy()
        
        # Add gesture-specific variability (muscle tremor, natural variation)
        variability = np.random.normal(0, pattern['variability'], self.num_channels)
        sample += variability
        
        # Add Gaussian noise (electronic interference, amplifier noise)
        noise = np.random.normal(0, SIMULATED_NOISE_STD, self.num_channels)
        sample += noise
        
        # Apply amplitude drift (electrode impedance changes over time)
        # Drift is a random walk that accumulates slowly
        drift_change = np.random.normal(0, SIMULATED_DRIFT_RATE, self.num_channels)
        self.drift_offset += drift_change
        
        # Limit drift to reasonable bounds
        self.drift_offset = np.clip(self.drift_offset, -0.1, 0.1)
        sample += self.drift_offset
        
        # Add non-stationarity (muscle fatigue effect)
        # Amplitude gradually increases as muscles tire
        fatigue_factor = 1.0 + 0.0001 * (self.sample_count % 1000)
        sample *= fatigue_factor
        
        # Ensure values are in valid range [0, 1]
        sample = np.clip(sample, 0.0, 1.0)
        
        self.sample_count += 1
        return sample
    
    def get_batch(self, batch_size: int) -> Optional[np.ndarray]:
        """
        Generate a batch of simulated EMG samples.
        
        Args:
            batch_size: Number of samples to generate
            
        Returns:
            numpy array of shape (batch_size, num_channels)
        """
        if not self.is_active:
            self.start_stream()
        
        samples = []
        for _ in range(batch_size):
            sample = self.get_sample()
            if sample is not None:
                samples.append(sample)
        
        return np.array(samples) if samples else None
    
    def is_streaming(self) -> bool:
        """
        Simulated source is always a streaming source.
        
        Returns:
            Always True for simulated sources
        """
        return True
    
    def start_stream(self) -> bool:
        """
        Start generating simulated EMG data.
        
        Returns:
            True if stream started successfully
        """
        self.is_active = True
        self._stop_event.clear()
        self.sample_count = 0
        self.drift_offset = np.zeros(self.num_channels)
        return True
    
    def stop_stream(self) -> None:
        """
        Stop the simulated EMG stream.
        """
        self.is_active = False
        self._stop_event.set()
    
    def stream_samples(self) -> Generator[np.ndarray, None, None]:
        """
        Generator that yields samples at realistic intervals.
        
        Yields:
            numpy array of shape (num_channels,) at regular intervals
            
        This mimics real-time EMG acquisition with proper timing.
        """
        self.start_stream()
        
        while self.is_active and not self._stop_event.is_set():
            sample = self.get_sample()
            if sample is not None:
                yield sample
            
            # Wait for next sample interval to simulate real-time acquisition
            time.sleep(self.stream_interval)
    
    def randomize_gesture(self) -> str:
        """
        Randomly change to a new gesture (for demo/testing).
        
        Returns:
            The new gesture name
        """
        new_gesture = np.random.choice(GESTURE_CLASSES)
        self.current_gesture = new_gesture
        return new_gesture
    
    def reset_drift(self) -> None:
        """
        Reset the amplitude drift to zero.
        
        Call this to simulate applying fresh electrode gel or
        repositioning electrodes.
        """
        self.drift_offset = np.zeros(self.num_channels)
        self.sample_count = 0
