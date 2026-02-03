"""
Hardware EMG Source - Integration Placeholder

This module provides a template for integrating real EMG sensors.
It defines all the connection points and data flow patterns needed
for hardware integration without requiring actual sensor libraries.

============================================================================
HARDWARE INTEGRATION GUIDE FOR ENGINEERS
============================================================================

When connecting a real EMG sensor (e.g., MyoWare, OpenBCI, Myo armband),
follow these steps:

1. DEPENDENCIES:
   Add your sensor's Python library to requirements.txt
   Example: pyserial, pylsl, bluetooth-library, etc.

2. INITIALIZATION (__init__):
   - Initialize serial ports, Bluetooth connections, or USB devices
   - Configure sensor sampling rate and gain settings
   - Set up any required calibration

3. CONNECTION (connect_sensor):
   - Open communication channel to sensor
   - Verify sensor is responding
   - Start data acquisition if supported

4. DATA READING (get_sample):
   - Read raw bytes from sensor
   - Parse according to sensor's data protocol
   - Convert to 8-channel float array
   - Apply any sensor-specific calibration

5. CLEANUP (stop_stream):
   - Close connections properly
   - Release any hardware resources
   - Save calibration if needed

COMMON SENSOR TYPES AND THEIR TYPICAL INTERFACES:
-------------------------------------------------
| Sensor Type     | Interface    | Library           |
|-----------------|--------------|-------------------|
| MyoWare/DIY     | Serial/ADC   | pyserial, pyfirmata|
| OpenBCI         | Serial/WiFi  | pyOpenBCI, brainflow|
| Myo Armband     | Bluetooth    | myo-python        |
| BITalino        | Bluetooth    | bitalino          |
| Lab Streaming   | Network      | pylsl             |

============================================================================
"""
import numpy as np
from typing import Optional, Dict, Any
import time

from .base_source import EMGSource


class HardwareSource(EMGSource):
    """
    Template class for real EMG hardware integration.
    
    This class provides the structure and documentation for connecting
    actual EMG sensors. Override the marked methods to implement your
    specific sensor's communication protocol.
    
    Attributes:
        connection: Handle to the hardware connection (serial port, socket, etc.)
        sensor_config: Configuration parameters for the sensor
        is_connected: Whether the sensor is currently connected
    """
    
    def __init__(self, num_channels: int = 8, sensor_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the hardware EMG source.
        
        Args:
            num_channels: Number of EMG channels on your sensor
            sensor_config: Dictionary of sensor-specific settings
            
        Example sensor_config for serial sensor:
        {
            'port': 'COM3',  # or '/dev/ttyUSB0' on Linux
            'baud_rate': 115200,
            'timeout': 1.0
        }
        
        Example sensor_config for Bluetooth sensor:
        {
            'device_name': 'MyoSensor',
            'mac_address': 'XX:XX:XX:XX:XX:XX'
        }
        """
        super().__init__(num_channels)
        
        self.sensor_config = sensor_config or {}
        self.connection = None  # Placeholder for serial.Serial, socket, etc.
        self.is_connected = False
        
        # Buffer for storing raw sensor data
        self._data_buffer = []
        
        # Calibration values (sensor-specific)
        self._calibration_offset = np.zeros(num_channels)
        self._calibration_scale = np.ones(num_channels)
    
    def connect_sensor(self) -> bool:
        """
        Establish connection to the EMG sensor.
        
        Returns:
            True if connection successful, False otherwise
            
        =====================================================================
        INTEGRATION POINT: Implement your sensor connection logic here
        =====================================================================
        
        Example for serial connection:
        ```
        import serial
        
        try:
            self.connection = serial.Serial(
                port=self.sensor_config.get('port', 'COM3'),
                baudrate=self.sensor_config.get('baud_rate', 115200),
                timeout=self.sensor_config.get('timeout', 1.0)
            )
            self.is_connected = True
            return True
        except serial.SerialException as e:
            print(f"Failed to connect: {e}")
            return False
        ```
        """
        # Placeholder implementation - simulates successful connection
        print("[HARDWARE PLACEHOLDER] connect_sensor() called")
        print("  -> In a real implementation, this would open a serial port,")
        print("     Bluetooth connection, or network socket to the sensor.")
        
        self.is_connected = False  # Would be True on successful connection
        return False  # Return False since no real hardware is connected
    
    def disconnect_sensor(self) -> None:
        """
        Close the connection to the EMG sensor.
        
        =====================================================================
        INTEGRATION POINT: Close your sensor connection properly
        =====================================================================
        
        Example:
        ```
        if self.connection is not None:
            self.connection.close()
            self.connection = None
        self.is_connected = False
        ```
        """
        print("[HARDWARE PLACEHOLDER] disconnect_sensor() called")
        self.is_connected = False
    
    def _read_raw_data(self) -> Optional[bytes]:
        """
        Read raw bytes from the sensor.
        
        Returns:
            Raw bytes from sensor, or None if read failed
            
        =====================================================================
        INTEGRATION POINT: Implement your sensor's read protocol
        =====================================================================
        
        Example for serial:
        ```
        if self.connection and self.connection.in_waiting > 0:
            return self.connection.read(self.num_channels * 2)  # 2 bytes per channel
        return None
        ```
        """
        return None  # Placeholder
    
    def _parse_raw_data(self, raw_data: bytes) -> Optional[np.ndarray]:
        """
        Parse raw sensor bytes into channel values.
        
        Args:
            raw_data: Raw bytes from the sensor
            
        Returns:
            numpy array of shape (num_channels,) with parsed values
            
        =====================================================================
        INTEGRATION POINT: Implement your sensor's data format parsing
        =====================================================================
        
        Example for 16-bit ADC values (common format):
        ```
        import struct
        values = struct.unpack(f'<{self.num_channels}H', raw_data)
        return np.array(values) / 65535.0  # Normalize to [0, 1]
        ```
        """
        return None  # Placeholder
    
    def _apply_calibration(self, sample: np.ndarray) -> np.ndarray:
        """
        Apply calibration to raw sensor values.
        
        Args:
            sample: Raw normalized values from sensor
            
        Returns:
            Calibrated values
            
        This adjusts for electrode placement variation and sensor drift.
        """
        calibrated = (sample - self._calibration_offset) * self._calibration_scale
        return np.clip(calibrated, 0.0, 1.0)
    
    def calibrate(self, num_samples: int = 100) -> bool:
        """
        Perform sensor calibration by collecting baseline readings.
        
        Args:
            num_samples: Number of samples to collect for calibration
            
        Returns:
            True if calibration successful, False otherwise
            
        This should be called with the user in a relaxed, rest position.
        The calibration values are used to normalize subsequent readings.
        """
        print("[HARDWARE PLACEHOLDER] calibrate() called")
        print(f"  -> Would collect {num_samples} samples in rest position")
        print("  -> Calculate mean and standard deviation per channel")
        print("  -> Store as calibration offset and scale")
        return False
    
    def get_sample(self) -> Optional[np.ndarray]:
        """
        Get a single EMG sample from the sensor.
        
        Returns:
            numpy array of shape (num_channels,) or None if no data
            
        This is the main data acquisition method. It:
        1. Reads raw bytes from the sensor
        2. Parses them according to the sensor's protocol
        3. Applies calibration
        4. Returns normalized channel values
        """
        if not self.is_connected:
            return None
        
        raw_data = self._read_raw_data()
        if raw_data is None:
            return None
        
        sample = self._parse_raw_data(raw_data)
        if sample is None:
            return None
        
        return self._apply_calibration(sample)
    
    def get_batch(self, batch_size: int) -> Optional[np.ndarray]:
        """
        Get multiple samples from the sensor.
        
        Args:
            batch_size: Number of samples to collect
            
        Returns:
            numpy array of shape (batch_size, num_channels) or None
        """
        if not self.is_connected:
            return None
        
        samples = []
        for _ in range(batch_size):
            sample = self.get_sample()
            if sample is not None:
                samples.append(sample)
            else:
                time.sleep(0.001)  # Brief wait if no data available
        
        return np.array(samples) if samples else None
    
    def is_streaming(self) -> bool:
        """
        Hardware is a streaming source when connected.
        
        Returns:
            True if connected, False otherwise
        """
        return self.is_connected
    
    def start_stream(self) -> bool:
        """
        Start the hardware data stream.
        
        Returns:
            True if stream started successfully
        """
        if not self.is_connected:
            success = self.connect_sensor()
            if not success:
                return False
        
        self.is_active = True
        return True
    
    def stop_stream(self) -> None:
        """
        Stop the hardware data stream and clean up.
        """
        self.is_active = False
        self.disconnect_sensor()
    
    def get_sensor_status(self) -> Dict[str, Any]:
        """
        Get detailed status information about the sensor.
        
        Returns:
            Dictionary with connection status, config, and health metrics
        """
        return {
            'is_connected': self.is_connected,
            'is_active': self.is_active,
            'config': self.sensor_config,
            'calibrated': not np.allclose(self._calibration_scale, 1.0),
            'message': 'Hardware placeholder - no real sensor connected'
        }
