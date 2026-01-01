"""Measurement capture and orchestration.

Handles synchronized playback and recording for room measurements
using sweep signals.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path

import numpy as np
import sounddevice as sd
from numpy.typing import NDArray

from roomeq.core.sweep import (
    Deconvolver,
    SweepGenerator,
    SweepParameters,
    calculate_frequency_response,
)


class MeasurementChannel(Enum):
    """Speaker channel to measure."""

    LEFT = "left"
    RIGHT = "right"
    BOTH = "both"  # Both speakers simultaneously (quick test mode)


@dataclass
class MeasurementConfig:
    """Configuration for a measurement session."""

    device_id: int
    input_channel: int  # 0-indexed
    output_channel_left: int  # 0-indexed
    output_channel_right: int  # 0-indexed
    sample_rate: int = 48000
    sweep_duration: float = 5.0
    pre_delay: float = 0.5  # Silence before sweep
    post_delay: float = 1.0  # Silence after sweep (capture reverb tail)
    ir_length: float = 0.5  # Desired impulse response length

    def __post_init__(self):
        """Validate configuration."""
        if self.sample_rate <= 0:
            raise ValueError("Sample rate must be positive")
        if self.sweep_duration <= 0:
            raise ValueError("Sweep duration must be positive")
        if self.pre_delay < 0 or self.post_delay < 0:
            raise ValueError("Delays must be non-negative")


@dataclass
class MeasurementResult:
    """Result of a single measurement."""

    position_id: int
    position_name: str
    channel: MeasurementChannel
    recording: NDArray[np.float64]
    impulse_response: NDArray[np.float64]
    frequencies: NDArray[np.float64]
    magnitude_db: NDArray[np.float64]
    sample_rate: int
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def peak_level_db(self) -> float:
        """Get peak recording level in dB."""
        peak = np.max(np.abs(self.recording))
        if peak > 0:
            return float(20 * np.log10(peak))
        return -100.0


@dataclass
class MeasurementPosition:
    """Describes a measurement position."""

    id: int
    name: str
    description: str
    x_offset: float  # cm from center
    y_offset: float  # cm from center (forward positive)
    z_offset: float  # cm from center (up positive)


# Standard 7-position measurement layout
STANDARD_POSITIONS = [
    MeasurementPosition(0, "Center", "Center at ear height", 0, 0, 0),
    MeasurementPosition(1, "Left", "30cm left of center", -30, 0, 0),
    MeasurementPosition(2, "Right", "30cm right of center", 30, 0, 0),
    MeasurementPosition(3, "Front Left", "30cm forward, 15cm left", -15, 30, 0),
    MeasurementPosition(4, "Front Right", "30cm forward, 15cm right", 15, 30, 0),
    MeasurementPosition(5, "Back", "30cm back of center", 0, -30, 0),
    MeasurementPosition(6, "Up", "30cm above center", 0, 0, 30),
]

# Quick 5-position layout
QUICK_POSITIONS = STANDARD_POSITIONS[:5]

# Thorough 9-position layout
THOROUGH_POSITIONS = STANDARD_POSITIONS + [
    MeasurementPosition(7, "Back Left", "30cm back, 15cm left", -15, -30, 0),
    MeasurementPosition(8, "Back Right", "30cm back, 15cm right", 15, -30, 0),
]


def get_positions(num_positions: int = 7) -> list[MeasurementPosition]:
    """Get measurement positions for a given count."""
    if num_positions <= 5:
        return QUICK_POSITIONS
    elif num_positions <= 7:
        return STANDARD_POSITIONS
    else:
        return THOROUGH_POSITIONS


class MeasurementEngine:
    """Orchestrates measurement capture process."""

    def __init__(self, config: MeasurementConfig):
        """
        Initialize measurement engine.

        Args:
            config: Measurement configuration
        """
        self.config = config
        self.sweep_gen = SweepGenerator()
        self.deconvolver = Deconvolver()

        # Pre-generate sweep and inverse filter
        self._sweep_params = SweepParameters(
            duration=config.sweep_duration,
            sample_rate=config.sample_rate,
            start_freq=20.0,
            end_freq=20000.0,
            amplitude=0.8,
        )
        self._sweep = self.sweep_gen.generate(self._sweep_params)
        self._inverse_filter = self.sweep_gen.generate_inverse_filter(
            self._sweep, self._sweep_params
        )

    @property
    def sweep_signal(self) -> NDArray[np.float64]:
        """Get the sweep signal used for measurements."""
        return self._sweep.copy()

    @property
    def total_duration(self) -> float:
        """Get total duration of one measurement in seconds."""
        return self.config.pre_delay + self.config.sweep_duration + self.config.post_delay

    def capture(
        self,
        position: MeasurementPosition,
        channel: MeasurementChannel,
        progress_callback: Callable[[float], None] | None = None,
    ) -> MeasurementResult:
        """
        Capture a single measurement.

        Args:
            position: Measurement position info
            channel: Which speaker channel to measure
            progress_callback: Optional callback with progress (0.0 to 1.0)

        Returns:
            MeasurementResult with recording and analysis
        """
        # Build output signal with pre/post silence
        pre_samples = int(self.config.pre_delay * self.config.sample_rate)
        post_samples = int(self.config.post_delay * self.config.sample_rate)

        signal = np.concatenate(
            [
                np.zeros(pre_samples),
                self._sweep,
                np.zeros(post_samples),
            ]
        )

        # Determine which output channel(s) to use (0-indexed internally)
        if channel == MeasurementChannel.LEFT:
            output_channels = [self.config.output_channel_left]
        elif channel == MeasurementChannel.RIGHT:
            output_channels = [self.config.output_channel_right]
        else:  # BOTH
            output_channels = [
                self.config.output_channel_left,
                self.config.output_channel_right,
            ]

        # Report progress at start
        if progress_callback:
            progress_callback(0.0)

        # Create output signal - mono for single channel, stereo for both
        if len(output_channels) == 1:
            output = signal.reshape(-1, 1)
            output_mapping = [output_channels[0] + 1]  # 1-indexed
        else:
            # Stereo output - same signal to both channels
            output = np.column_stack([signal, signal])
            output_mapping = [ch + 1 for ch in output_channels]  # 1-indexed

        # Synchronized playback and recording
        # output_mapping uses 1-indexed channel numbers for sounddevice
        recording = sd.playrec(
            output,
            samplerate=self.config.sample_rate,
            device=self.config.device_id,
            input_mapping=[self.config.input_channel + 1],  # 1-indexed
            output_mapping=output_mapping,
            blocking=True,
        )

        # Extract mono recording
        if recording.ndim > 1:
            recording = recording[:, 0]
        else:
            recording = recording.flatten()

        if progress_callback:
            progress_callback(0.5)

        # Deconvolve to get impulse response
        full_ir = self.deconvolver.deconvolve(recording, self._inverse_filter)
        linear_ir = self.deconvolver.extract_linear_ir(
            full_ir, self._sweep_params, self.config.ir_length
        )

        if progress_callback:
            progress_callback(0.75)

        # Calculate frequency response
        frequencies, magnitude_db = calculate_frequency_response(
            linear_ir, self.config.sample_rate
        )

        if progress_callback:
            progress_callback(1.0)

        return MeasurementResult(
            position_id=position.id,
            position_name=position.name,
            channel=channel,
            recording=recording,
            impulse_response=linear_ir,
            frequencies=frequencies,
            magnitude_db=magnitude_db,
            sample_rate=self.config.sample_rate,
        )

    def capture_test_signal(self, duration: float = 0.1) -> NDArray[np.float64]:
        """
        Capture a short test recording (no playback).

        Useful for testing input levels before measurement.

        Args:
            duration: Recording duration in seconds

        Returns:
            Recorded audio data
        """
        # Record from the configured input channel
        # sd.rec doesn't support input_mapping, so record enough channels
        # and extract the one we need
        num_channels = self.config.input_channel + 1
        recording = sd.rec(
            int(duration * self.config.sample_rate),
            samplerate=self.config.sample_rate,
            channels=num_channels,
            device=self.config.device_id,
            blocking=True,
        )
        # Extract the configured input channel
        if recording.ndim > 1:
            channel_data = recording[:, self.config.input_channel]
        else:
            channel_data = recording.flatten()
        return channel_data.astype(np.float64)  # type: ignore[no-any-return]


class MeasurementSession:
    """Manages a complete measurement session with multiple positions."""

    def __init__(self, config: MeasurementConfig, num_positions: int = 7):
        """
        Initialize measurement session.

        Args:
            config: Measurement configuration
            num_positions: Number of measurement positions (5, 7, or 9)
        """
        self.config = config
        self.positions = get_positions(num_positions)
        self.engine = MeasurementEngine(config)

        # Store measurements by channel and position
        self.measurements: dict[MeasurementChannel, dict[int, MeasurementResult]] = {
            MeasurementChannel.LEFT: {},
            MeasurementChannel.RIGHT: {},
        }

    @property
    def total_measurements(self) -> int:
        """Total number of measurements required (positions x 2 channels)."""
        return len(self.positions) * 2

    @property
    def completed_measurements(self) -> int:
        """Number of measurements completed."""
        left_count = len(self.measurements[MeasurementChannel.LEFT])
        right_count = len(self.measurements[MeasurementChannel.RIGHT])
        return left_count + right_count

    @property
    def progress(self) -> float:
        """Overall progress (0.0 to 1.0)."""
        if self.total_measurements == 0:
            return 0.0
        return self.completed_measurements / self.total_measurements

    def get_measurement(
        self, channel: MeasurementChannel, position_id: int
    ) -> MeasurementResult | None:
        """Get a specific measurement result."""
        return self.measurements[channel].get(position_id)

    def add_measurement(self, result: MeasurementResult) -> None:
        """Add a measurement result to the session."""
        self.measurements[result.channel][result.position_id] = result

    def get_channel_measurements(
        self, channel: MeasurementChannel
    ) -> list[MeasurementResult]:
        """Get all measurements for a channel."""
        return list(self.measurements[channel].values())

    def is_channel_complete(self, channel: MeasurementChannel) -> bool:
        """Check if all positions are measured for a channel."""
        return len(self.measurements[channel]) >= len(self.positions)

    def is_complete(self) -> bool:
        """Check if all measurements are complete."""
        return self.is_channel_complete(MeasurementChannel.LEFT) and self.is_channel_complete(
            MeasurementChannel.RIGHT
        )

    def clear_channel(self, channel: MeasurementChannel) -> None:
        """Clear all measurements for a channel."""
        self.measurements[channel] = {}

    def clear_all(self) -> None:
        """Clear all measurements."""
        self.measurements = {
            MeasurementChannel.LEFT: {},
            MeasurementChannel.RIGHT: {},
        }

    def save(self, filepath: Path) -> None:
        """Save session data to a JSON file.

        Args:
            filepath: Path to save the session data
        """
        data = {
            "config": {
                "device_id": self.config.device_id,
                "input_channel": self.config.input_channel,
                "output_channel_left": self.config.output_channel_left,
                "output_channel_right": self.config.output_channel_right,
                "sample_rate": self.config.sample_rate,
                "sweep_duration": self.config.sweep_duration,
            },
            "num_positions": len(self.positions),
            "measurements": {},
        }

        # Serialize measurements
        for channel in [MeasurementChannel.LEFT, MeasurementChannel.RIGHT]:
            channel_key = channel.value
            data["measurements"][channel_key] = {}
            for pos_id, result in self.measurements[channel].items():
                data["measurements"][channel_key][str(pos_id)] = {
                    "position_id": result.position_id,
                    "position_name": result.position_name,
                    "channel": result.channel.value,
                    "frequencies": result.frequencies.tolist(),
                    "magnitude_db": result.magnitude_db.tolist(),
                    "sample_rate": result.sample_rate,
                    "timestamp": result.timestamp.isoformat(),
                    # Skip recording and impulse_response to save space
                }

        filepath.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, filepath: Path) -> "MeasurementSession":
        """Load session data from a JSON file.

        Args:
            filepath: Path to load the session data from

        Returns:
            Loaded MeasurementSession
        """
        data = json.loads(filepath.read_text())

        # Recreate config
        config = MeasurementConfig(
            device_id=data["config"]["device_id"],
            input_channel=data["config"]["input_channel"],
            output_channel_left=data["config"]["output_channel_left"],
            output_channel_right=data["config"]["output_channel_right"],
            sample_rate=data["config"]["sample_rate"],
            sweep_duration=data["config"].get("sweep_duration", 5.0),
        )

        # Create session
        session = cls(config, data["num_positions"])

        # Load measurements
        for channel_str, positions in data["measurements"].items():
            channel = MeasurementChannel(channel_str)
            for _pos_id_str, result_data in positions.items():
                result = MeasurementResult(
                    position_id=result_data["position_id"],
                    position_name=result_data["position_name"],
                    channel=MeasurementChannel(result_data["channel"]),
                    recording=np.array([]),  # Not saved
                    impulse_response=np.array([]),  # Not saved
                    frequencies=np.array(result_data["frequencies"]),
                    magnitude_db=np.array(result_data["magnitude_db"]),
                    sample_rate=result_data["sample_rate"],
                    timestamp=datetime.fromisoformat(result_data["timestamp"]),
                )
                session.measurements[channel][result.position_id] = result

        return session
