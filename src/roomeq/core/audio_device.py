"""Audio device detection and management.

Handles enumeration of audio devices for room measurement.
Works with any CoreAudio-compatible interface.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import sounddevice as sd
from numpy.typing import NDArray

# Known manufacturer patterns (for informational purposes only, not filtering)
KNOWN_MANUFACTURERS = {
    "rme": [
        r"RME", r"Fireface", r"UCX", r"UFX", r"Babyface", r"MADIface", r"ADI-2", r"802", r"Digiface"
    ],
    "motu": [r"MOTU", r"M2", r"M4", r"UltraLite", r"828"],
    "focusrite": [r"Focusrite", r"Scarlett", r"Clarett"],
    "universal_audio": [r"Universal Audio", r"Apollo", r"Volt"],
    "presonus": [r"PreSonus", r"AudioBox", r"Studio"],
    "behringer": [r"Behringer", r"UMC", r"U-PHORIA"],
    "steinberg": [r"Steinberg", r"UR"],
    "audient": [r"Audient", r"iD"],
    "native_instruments": [r"Native Instruments", r"Komplete Audio"],
    "apogee": [r"Apogee", r"Duet", r"Quartet"],
}

# Legacy: RME patterns for backward compatibility
RME_PATTERNS = KNOWN_MANUFACTURERS["rme"]


@dataclass
class AudioDevice:
    """Represents an audio device."""

    id: int
    name: str
    max_input_channels: int
    max_output_channels: int
    default_sample_rate: float
    hostapi: str
    manufacturer: str = ""  # Detected manufacturer (if known)

    # Backward compatibility: is_rme is now computed from manufacturer
    _is_rme_override: bool | None = field(default=None, repr=False)

    @property
    def is_rme(self) -> bool:
        """Check if this is an RME device (for backward compatibility)."""
        if self._is_rme_override is not None:
            return self._is_rme_override
        return self.manufacturer == "rme"

    @property
    def has_inputs(self) -> bool:
        """Check if device has input channels."""
        return self.max_input_channels > 0

    @property
    def has_outputs(self) -> bool:
        """Check if device has output channels."""
        return self.max_output_channels > 0


class AudioDeviceManager:
    """Manages audio device detection and selection."""

    def __init__(self):
        """Initialize the device manager."""
        self._devices: list[AudioDevice] | None = None

    def refresh(self) -> None:
        """Refresh the device list."""
        self._devices = None

    def get_devices(self) -> list[AudioDevice]:
        """
        Return list of available audio devices.

        Returns:
            List of AudioDevice objects
        """
        if self._devices is None:
            self._devices = self._enumerate_devices()
        return self._devices

    def get_rme_devices(self) -> list[AudioDevice]:
        """
        Return RME devices only (for backward compatibility).

        Returns:
            List of AudioDevice objects that are RME interfaces
        """
        return [d for d in self.get_devices() if d.is_rme]

    def get_devices_by_manufacturer(self, manufacturer: str) -> list[AudioDevice]:
        """
        Return devices from a specific manufacturer.

        Args:
            manufacturer: Manufacturer ID (e.g., "rme", "motu", "focusrite")

        Returns:
            List of AudioDevice objects from that manufacturer
        """
        return [d for d in self.get_devices() if d.manufacturer == manufacturer.lower()]

    def get_input_devices(self) -> list[AudioDevice]:
        """
        Return devices with input channels.

        Returns:
            List of AudioDevice objects with inputs
        """
        return [d for d in self.get_devices() if d.has_inputs]

    def get_output_devices(self) -> list[AudioDevice]:
        """
        Return devices with output channels.

        Returns:
            List of AudioDevice objects with outputs
        """
        return [d for d in self.get_devices() if d.has_outputs]

    def get_default_device(self) -> AudioDevice | None:
        """
        Return the default/preferred device.

        Prefers devices with both inputs and outputs (suitable for measurement).
        Falls back to system default if available.

        Returns:
            Preferred AudioDevice or None if no devices available
        """
        devices = self.get_devices()
        if not devices:
            return None

        # First, prefer devices with both inputs and outputs
        full_devices = [d for d in devices if d.has_inputs and d.has_outputs]
        if full_devices:
            # Sort by name for consistency
            full_devices.sort(key=lambda d: d.name)
            return full_devices[0]

        # Fall back to system default
        try:
            default_input = sd.default.device[0]
            default_output = sd.default.device[1]

            # Try to find a device that matches the defaults
            for device in devices:
                if device.id == default_input or device.id == default_output:
                    return device
        except Exception:
            pass

        # Just return first device
        return devices[0] if devices else None

    def get_device_by_id(self, device_id: int) -> AudioDevice | None:
        """
        Get a device by its ID.

        Args:
            device_id: Device ID

        Returns:
            AudioDevice or None if not found
        """
        for device in self.get_devices():
            if device.id == device_id:
                return device
        return None

    def get_device_by_name(self, name: str) -> AudioDevice | None:
        """
        Get a device by name (partial match).

        Args:
            name: Device name to search for

        Returns:
            AudioDevice or None if not found
        """
        name_lower = name.lower()
        for device in self.get_devices():
            if name_lower in device.name.lower():
                return device
        return None

    def test_device(
        self,
        device: AudioDevice,
        duration: float = 0.1,
    ) -> bool:
        """
        Test if device is accessible by attempting a short recording.

        Args:
            device: Device to test
            duration: Test duration in seconds

        Returns:
            True if device is accessible
        """
        if not device.has_inputs:
            return False

        try:
            # Attempt a short recording
            sd.rec(
                int(duration * device.default_sample_rate),
                samplerate=int(device.default_sample_rate),
                channels=1,
                device=device.id,
                blocking=True,
            )
            return True
        except Exception:
            return False

    def get_input_level(
        self,
        device: AudioDevice,
        channel: int = 0,
        duration: float = 0.05,
    ) -> float:
        """
        Get current input level from device.

        Args:
            device: Input device
            channel: Input channel (0-indexed)
            duration: Measurement duration in seconds

        Returns:
            RMS level in dB (relative to full scale)
        """
        if not device.has_inputs:
            return -100.0

        if channel >= device.max_input_channels:
            return -100.0

        try:
            # Record from the device - we need to record enough channels to include
            # the one we want, then extract it
            num_channels = channel + 1
            recording = sd.rec(
                int(duration * device.default_sample_rate),
                samplerate=int(device.default_sample_rate),
                channels=num_channels,
                device=device.id,
                blocking=True,
            )

            # Extract the specific channel we want
            if recording.ndim > 1:
                channel_data = recording[:, channel]
            else:
                channel_data = recording.flatten()

            # Calculate RMS level in dB
            rms = np.sqrt(np.mean(channel_data**2))
            if rms > 0:
                return float(20 * np.log10(rms))
            return -100.0
        except Exception:
            return -100.0

    def _enumerate_devices(self) -> list[AudioDevice]:
        """Enumerate all audio devices."""
        devices = []
        try:
            device_list = sd.query_devices()
            hostapis = sd.query_hostapis()

            for idx, dev in enumerate(device_list):
                # Get host API name
                hostapi_idx = dev.get("hostapi", 0)
                hostapi_name = hostapis[hostapi_idx]["name"] if hostapi_idx < len(hostapis) else ""

                # Detect manufacturer
                manufacturer = self._detect_manufacturer(dev["name"])

                devices.append(
                    AudioDevice(
                        id=idx,
                        name=dev["name"],
                        max_input_channels=dev["max_input_channels"],
                        max_output_channels=dev["max_output_channels"],
                        default_sample_rate=dev["default_samplerate"],
                        hostapi=hostapi_name,
                        manufacturer=manufacturer,
                    )
                )
        except Exception:
            pass

        return devices

    @staticmethod
    def _detect_manufacturer(name: str) -> str:
        """
        Detect device manufacturer from name.

        Args:
            name: Device name

        Returns:
            Manufacturer ID or empty string if unknown
        """
        for manufacturer, patterns in KNOWN_MANUFACTURERS.items():
            for pattern in patterns:
                if re.search(pattern, name, re.IGNORECASE):
                    return manufacturer
        return ""

    @staticmethod
    def _is_rme_device(name: str) -> bool:
        """
        Check if device name matches RME patterns.

        Kept for backward compatibility.
        """
        for pattern in RME_PATTERNS:
            if re.search(pattern, name, re.IGNORECASE):
                return True
        return False


class LevelMonitor:
    """Monitors input level in real-time using a callback."""

    def __init__(
        self,
        device: AudioDevice,
        channel: int = 0,
        callback: Callable[[float], None] | None = None,
    ):
        """
        Initialize level monitor.

        Args:
            device: Audio device to monitor
            channel: Input channel (0-indexed)
            callback: Function called with dB level on each update
        """
        self.device = device
        self.channel = channel
        self.callback = callback
        self._stream: sd.InputStream | None = None
        self._running = False
        self._current_level = -100.0

    @property
    def current_level(self) -> float:
        """Get the current level in dB."""
        return self._current_level

    @property
    def is_running(self) -> bool:
        """Check if monitor is running."""
        return self._running

    def start(self) -> None:
        """Start monitoring."""
        if self._running:
            return

        if not self.device.has_inputs:
            raise ValueError("Device has no input channels")

        def audio_callback(indata: NDArray, frames: int, time_info, status):
            if status:
                return

            # Get the channel we're monitoring
            if indata.ndim > 1 and self.channel < indata.shape[1]:
                channel_data = indata[:, self.channel]
            else:
                channel_data = indata.flatten()

            # Calculate RMS level
            rms = np.sqrt(np.mean(channel_data**2))
            if rms > 0:
                self._current_level = 20 * np.log10(rms)
            else:
                self._current_level = -100.0

            if self.callback:
                self.callback(self._current_level)

        self._stream = sd.InputStream(
            device=self.device.id,
            channels=max(self.channel + 1, 1),
            samplerate=int(self.device.default_sample_rate),
            callback=audio_callback,
            blocksize=1024,
        )
        self._stream.start()
        self._running = True

    def stop(self) -> None:
        """Stop monitoring."""
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._running = False
        self._current_level = -100.0
