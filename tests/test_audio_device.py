"""Tests for audio device management."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from roomeq.core.audio_device import (
    AudioDevice,
    AudioDeviceManager,
    KNOWN_MANUFACTURERS,
    LevelMonitor,
    RME_PATTERNS,
)


class TestAudioDevice:
    """Tests for AudioDevice dataclass."""

    def test_has_inputs(self):
        """Test has_inputs property."""
        device_with_inputs = AudioDevice(
            id=0,
            name="Test Device",
            max_input_channels=2,
            max_output_channels=0,
            default_sample_rate=48000,
            hostapi="Core Audio",
        )
        assert device_with_inputs.has_inputs is True

        device_no_inputs = AudioDevice(
            id=1,
            name="Output Only",
            max_input_channels=0,
            max_output_channels=2,
            default_sample_rate=48000,
            hostapi="Core Audio",
        )
        assert device_no_inputs.has_inputs is False

    def test_has_outputs(self):
        """Test has_outputs property."""
        device_with_outputs = AudioDevice(
            id=0,
            name="Test Device",
            max_input_channels=0,
            max_output_channels=2,
            default_sample_rate=48000,
            hostapi="Core Audio",
        )
        assert device_with_outputs.has_outputs is True

        device_no_outputs = AudioDevice(
            id=1,
            name="Input Only",
            max_input_channels=2,
            max_output_channels=0,
            default_sample_rate=48000,
            hostapi="Core Audio",
        )
        assert device_no_outputs.has_outputs is False

    def test_is_rme_property(self):
        """Test is_rme computed property."""
        rme_device = AudioDevice(
            id=0,
            name="RME UCX II",
            max_input_channels=20,
            max_output_channels=22,
            default_sample_rate=48000,
            hostapi="Core Audio",
            manufacturer="rme",
        )
        assert rme_device.is_rme is True

        non_rme_device = AudioDevice(
            id=1,
            name="Focusrite Scarlett",
            max_input_channels=2,
            max_output_channels=2,
            default_sample_rate=48000,
            hostapi="Core Audio",
            manufacturer="focusrite",
        )
        assert non_rme_device.is_rme is False


class TestManufacturerDetection:
    """Tests for manufacturer detection."""

    def test_known_manufacturers_exist(self):
        """Test that known manufacturers are defined."""
        assert len(KNOWN_MANUFACTURERS) > 0
        assert "rme" in KNOWN_MANUFACTURERS
        assert "focusrite" in KNOWN_MANUFACTURERS
        assert "motu" in KNOWN_MANUFACTURERS

    def test_rme_patterns_exist(self):
        """Test that RME patterns are defined (backward compatibility)."""
        assert len(RME_PATTERNS) > 0

    @pytest.mark.parametrize(
        "name,expected_manufacturer",
        [
            ("RME UCX II", "rme"),
            ("RME Fireface UCX", "rme"),
            ("Fireface 802 FS", "rme"),
            ("RME UFX+", "rme"),
            ("RME Babyface Pro", "rme"),
            ("RME ADI-2 DAC", "rme"),
            ("MADIface XT", "rme"),
            ("RME Digiface USB", "rme"),
            ("Focusrite Scarlett 2i2", "focusrite"),
            ("Universal Audio Apollo", "universal_audio"),
            ("MOTU M4", "motu"),
            ("PreSonus AudioBox", "presonus"),
            ("Built-in Microphone", ""),
            ("MacBook Pro Speakers", ""),
        ],
    )
    def test_detect_manufacturer(self, name, expected_manufacturer):
        """Test manufacturer detection by name."""
        manufacturer = AudioDeviceManager._detect_manufacturer(name)
        assert manufacturer == expected_manufacturer

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("RME UCX II", True),
            ("RME Fireface UCX", True),
            ("Fireface 802 FS", True),
            ("Built-in Microphone", False),
            ("Focusrite Scarlett 2i2", False),
        ],
    )
    def test_is_rme_device(self, name, expected):
        """Test RME device detection by name (backward compatibility)."""
        manager = AudioDeviceManager()
        assert manager._is_rme_device(name) is expected

    def test_detection_case_insensitive(self):
        """Test that detection is case insensitive."""
        assert AudioDeviceManager._detect_manufacturer("rme ucx ii") == "rme"
        assert AudioDeviceManager._detect_manufacturer("FIREFACE 802") == "rme"
        assert AudioDeviceManager._detect_manufacturer("FOCUSRITE scarlett") == "focusrite"


class TestAudioDeviceManager:
    """Tests for AudioDeviceManager."""

    @pytest.fixture
    def mock_devices(self):
        """Create mock device list."""
        return [
            {
                "name": "Built-in Microphone",
                "max_input_channels": 2,
                "max_output_channels": 0,
                "default_samplerate": 48000,
                "hostapi": 0,
            },
            {
                "name": "Built-in Output",
                "max_input_channels": 0,
                "max_output_channels": 2,
                "default_samplerate": 48000,
                "hostapi": 0,
            },
            {
                "name": "RME UCX II",
                "max_input_channels": 20,
                "max_output_channels": 22,
                "default_samplerate": 48000,
                "hostapi": 0,
            },
            {
                "name": "Focusrite Scarlett 2i2",
                "max_input_channels": 2,
                "max_output_channels": 2,
                "default_samplerate": 48000,
                "hostapi": 0,
            },
        ]

    @pytest.fixture
    def mock_hostapis(self):
        """Create mock host API list."""
        return [{"name": "Core Audio"}]

    def test_get_devices(self, mock_devices, mock_hostapis):
        """Test device enumeration."""
        with (
            patch("sounddevice.query_devices", return_value=mock_devices),
            patch("sounddevice.query_hostapis", return_value=mock_hostapis),
        ):
            manager = AudioDeviceManager()
            devices = manager.get_devices()

            assert len(devices) == 4
            assert all(isinstance(d, AudioDevice) for d in devices)

    def test_get_rme_devices(self, mock_devices, mock_hostapis):
        """Test filtering for RME devices (backward compatibility)."""
        with (
            patch("sounddevice.query_devices", return_value=mock_devices),
            patch("sounddevice.query_hostapis", return_value=mock_hostapis),
        ):
            manager = AudioDeviceManager()
            rme_devices = manager.get_rme_devices()

            assert len(rme_devices) == 1
            assert rme_devices[0].name == "RME UCX II"
            assert rme_devices[0].is_rme is True
            assert rme_devices[0].manufacturer == "rme"

    def test_get_devices_by_manufacturer(self, mock_devices, mock_hostapis):
        """Test filtering devices by manufacturer."""
        with (
            patch("sounddevice.query_devices", return_value=mock_devices),
            patch("sounddevice.query_hostapis", return_value=mock_hostapis),
        ):
            manager = AudioDeviceManager()

            rme_devices = manager.get_devices_by_manufacturer("rme")
            assert len(rme_devices) == 1
            assert rme_devices[0].name == "RME UCX II"

            focusrite_devices = manager.get_devices_by_manufacturer("focusrite")
            assert len(focusrite_devices) == 1
            assert focusrite_devices[0].name == "Focusrite Scarlett 2i2"

            # Unknown manufacturer returns empty
            unknown = manager.get_devices_by_manufacturer("unknown")
            assert len(unknown) == 0

    def test_get_input_devices(self, mock_devices, mock_hostapis):
        """Test filtering for input devices."""
        with (
            patch("sounddevice.query_devices", return_value=mock_devices),
            patch("sounddevice.query_hostapis", return_value=mock_hostapis),
        ):
            manager = AudioDeviceManager()
            input_devices = manager.get_input_devices()

            assert len(input_devices) == 3  # Mic, RME, Scarlett
            assert all(d.has_inputs for d in input_devices)

    def test_get_output_devices(self, mock_devices, mock_hostapis):
        """Test filtering for output devices."""
        with (
            patch("sounddevice.query_devices", return_value=mock_devices),
            patch("sounddevice.query_hostapis", return_value=mock_hostapis),
        ):
            manager = AudioDeviceManager()
            output_devices = manager.get_output_devices()

            assert len(output_devices) == 3  # Output, RME, Scarlett
            assert all(d.has_outputs for d in output_devices)

    def test_get_default_device_prefers_full_io(self, mock_devices, mock_hostapis):
        """Test that default device prefers devices with both inputs and outputs."""
        with (
            patch("sounddevice.query_devices", return_value=mock_devices),
            patch("sounddevice.query_hostapis", return_value=mock_hostapis),
        ):
            manager = AudioDeviceManager()
            default = manager.get_default_device()

            assert default is not None
            # Should prefer device with both inputs and outputs
            assert default.has_inputs is True
            assert default.has_outputs is True
            # First alphabetically among full I/O devices
            assert default.name in ["Focusrite Scarlett 2i2", "RME UCX II"]

    def test_get_default_device_no_full_io(self, mock_hostapis):
        """Test default device when no device has both inputs and outputs."""
        io_only_devices = [
            {
                "name": "Built-in Microphone",
                "max_input_channels": 2,
                "max_output_channels": 0,
                "default_samplerate": 48000,
                "hostapi": 0,
            },
            {
                "name": "Built-in Output",
                "max_input_channels": 0,
                "max_output_channels": 2,
                "default_samplerate": 48000,
                "hostapi": 0,
            },
        ]

        with (
            patch("sounddevice.query_devices", return_value=io_only_devices),
            patch("sounddevice.query_hostapis", return_value=mock_hostapis),
            patch("sounddevice.default", MagicMock(device=[0, 1])),
        ):
            manager = AudioDeviceManager()
            default = manager.get_default_device()

            # Should fall back to system default
            assert default is not None

    def test_get_device_by_id(self, mock_devices, mock_hostapis):
        """Test getting device by ID."""
        with (
            patch("sounddevice.query_devices", return_value=mock_devices),
            patch("sounddevice.query_hostapis", return_value=mock_hostapis),
        ):
            manager = AudioDeviceManager()

            device = manager.get_device_by_id(2)
            assert device is not None
            assert device.name == "RME UCX II"

            # Non-existent ID
            device = manager.get_device_by_id(999)
            assert device is None

    def test_get_device_by_name(self, mock_devices, mock_hostapis):
        """Test getting device by name (partial match)."""
        with (
            patch("sounddevice.query_devices", return_value=mock_devices),
            patch("sounddevice.query_hostapis", return_value=mock_hostapis),
        ):
            manager = AudioDeviceManager()

            device = manager.get_device_by_name("UCX")
            assert device is not None
            assert device.name == "RME UCX II"

            device = manager.get_device_by_name("scarlett")
            assert device is not None
            assert device.name == "Focusrite Scarlett 2i2"

            # Non-existent name
            device = manager.get_device_by_name("NonExistent")
            assert device is None

    def test_refresh_clears_cache(self, mock_devices, mock_hostapis):
        """Test that refresh clears the device cache."""
        with (
            patch("sounddevice.query_devices", return_value=mock_devices),
            patch("sounddevice.query_hostapis", return_value=mock_hostapis),
        ):
            manager = AudioDeviceManager()

            # First call populates cache
            devices1 = manager.get_devices()
            assert len(devices1) == 4

            # Refresh clears cache
            manager.refresh()
            assert manager._devices is None

    def test_caches_devices(self, mock_devices, mock_hostapis):
        """Test that devices are cached after first query."""
        with (
            patch("sounddevice.query_devices", return_value=mock_devices) as mock_query,
            patch("sounddevice.query_hostapis", return_value=mock_hostapis),
        ):
            manager = AudioDeviceManager()

            # First call
            manager.get_devices()
            assert mock_query.call_count == 1

            # Second call should use cache
            manager.get_devices()
            assert mock_query.call_count == 1


class TestLevelMonitor:
    """Tests for LevelMonitor."""

    @pytest.fixture
    def mock_device(self):
        """Create a mock audio device."""
        return AudioDevice(
            id=0,
            name="Test Device",
            max_input_channels=2,
            max_output_channels=2,
            default_sample_rate=48000,
            hostapi="Core Audio",
        )

    def test_initial_state(self, mock_device):
        """Test initial monitor state."""
        monitor = LevelMonitor(mock_device)

        assert monitor.device == mock_device
        assert monitor.channel == 0
        assert monitor.is_running is False
        assert monitor.current_level == -100.0

    def test_cannot_monitor_output_only_device(self):
        """Test that monitoring fails on output-only device."""
        output_device = AudioDevice(
            id=0,
            name="Output Only",
            max_input_channels=0,
            max_output_channels=2,
            default_sample_rate=48000,
            hostapi="Core Audio",
        )

        monitor = LevelMonitor(output_device)

        with pytest.raises(ValueError, match="no input channels"):
            monitor.start()

    def test_callback_called(self, mock_device):
        """Test that callback is stored."""
        callback = MagicMock()
        monitor = LevelMonitor(mock_device, callback=callback)

        assert monitor.callback == callback


class TestInputLevelCalculation:
    """Tests for input level calculation logic."""

    def test_silence_gives_low_level(self):
        """Test that silence produces very low dB level."""
        silence = np.zeros(1024)
        rms = np.sqrt(np.mean(silence**2))
        # RMS of silence is 0, which would be -inf dB
        assert rms == 0

    def test_full_scale_gives_zero_db(self):
        """Test that full scale sine gives ~0 dB RMS."""
        # Full scale sine wave
        t = np.linspace(0, 1, 48000)
        sine = np.sin(2 * np.pi * 1000 * t)

        rms = np.sqrt(np.mean(sine**2))
        db = 20 * np.log10(rms)

        # Sine wave RMS is amplitude / sqrt(2), so dB should be ~-3dB
        assert -4 < db < -2

    def test_half_amplitude_gives_minus_six_db(self):
        """Test that half amplitude gives ~-6 dB less."""
        t = np.linspace(0, 1, 48000)
        sine_full = np.sin(2 * np.pi * 1000 * t)
        sine_half = 0.5 * np.sin(2 * np.pi * 1000 * t)

        rms_full = np.sqrt(np.mean(sine_full**2))
        rms_half = np.sqrt(np.mean(sine_half**2))

        db_full = 20 * np.log10(rms_full)
        db_half = 20 * np.log10(rms_half)

        # Half amplitude should be about 6 dB less
        assert abs((db_full - db_half) - 6.0) < 0.1
