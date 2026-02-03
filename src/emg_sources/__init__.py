"""
EMG Data Sources - Hardware Abstraction Layer

This module provides a unified interface for EMG data acquisition,
supporting multiple input modes while keeping downstream processing identical.

Supported sources:
- CSVSource: Batch processing from uploaded CSV files
- SimulatedSource: Realistic EMG simulation for testing
- HardwareSource: Placeholder for real sensor integration
"""
from .base_source import EMGSource
from .csv_source import CSVSource
from .simulated_source import SimulatedSource
from .hardware_source import HardwareSource

__all__ = ['EMGSource', 'CSVSource', 'SimulatedSource', 'HardwareSource']
