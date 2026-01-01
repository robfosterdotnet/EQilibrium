"""Audio device detection and management.

Handles enumeration of audio devices with special detection for
RME interfaces (UCX II, 802 FS, UFX+, etc.).
"""

import re
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import sounddevice as sd
from numpy.typing import NDArray

# RME device name patterns
RME_PATTERNS = [
    r"RME",
    r"Fireface",
    r"UCX",
    r"UFX",
    r"Babyface",
    r"MADIface",
    r"ADI-2",
    r"802",
    r"Digiface",
]


@dataclass
class AudioDevice:
    """Represents an audio device."""

    id: int
    name: str
    max_input_channels: int
    max_output_channels: int
    default_sample_rate: float
    is_rme: bool
    hostapi: str

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
        Return RME devices only.

        Returns:
            List of AudioDevice objects that are RME interfaces
        """
        return [d for d in self.get_devices() if d.is_rme]

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
        Return preferred device (RME if available, otherwise system default).

        Returns:
            Preferred AudioDevice or None if no devices available
        """
        rme_devices = self.get_rme_devices()
        if rme_devices:
            # Prefer RME device with both inputs and outputs
            for device in rme_devices:
                if device.has_inputs and device.has_outputs:
                    return device
            return rme_devices[0]

        # Fall back to system default
        devices = self.get_devices()
        if devices:
            try:
                default_input = sd.default.device[0]
                default_output = sd.default.device[1]

                # Try to find a device that matches both defaults
                for device in devices:
                    if device.id == default_input or device.id == default_output:
                        return device

                return devices[0]
            except Exception:
                return devices[0] if devices else None

        return None

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

                # Check if it's an RME device
                is_rme = self._is_rme_device(dev["name"])

                devices.append(
                    AudioDevice(
                        id=idx,
                        name=dev["name"],
                        max_input_channels=dev["max_input_channels"],
                        max_output_channels=dev["max_output_channels"],
                        default_sample_rate=dev["default_samplerate"],
                        is_rme=is_rme,
                        hostapi=hostapi_name,
                    )
                )
        except Exception:
            pass

        return devices

    @staticmethod
    def _is_rme_device(name: str) -> bool:
        """Check if device name matches RME patterns."""
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
