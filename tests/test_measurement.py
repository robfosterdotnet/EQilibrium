"""Tests for measurement capture and orchestration."""

from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from roomeq.core.measurement import (
    QUICK_POSITIONS,
    STANDARD_POSITIONS,
    THOROUGH_POSITIONS,
    MeasurementChannel,
    MeasurementConfig,
    MeasurementEngine,
    MeasurementPosition,
    MeasurementResult,
    MeasurementSession,
    get_positions,
)


class TestMeasurementConfig:
    """Tests for MeasurementConfig."""

    def test_default_values(self):
        """Test that config has sensible defaults."""
        config = MeasurementConfig(
            device_id=0,
            input_channel=0,
            output_channel_left=0,
            output_channel_right=1,
        )
        assert config.sample_rate == 48000
        assert config.sweep_duration == 5.0
        assert config.pre_delay == 0.5
        assert config.post_delay == 1.0

    def test_invalid_sample_rate(self):
        """Test that invalid sample rate raises error."""
        with pytest.raises(ValueError, match="Sample rate must be positive"):
            MeasurementConfig(
                device_id=0,
                input_channel=0,
                output_channel_left=0,
                output_channel_right=1,
                sample_rate=0,
            )

    def test_invalid_sweep_duration(self):
        """Test that invalid sweep duration raises error."""
        with pytest.raises(ValueError, match="Sweep duration must be positive"):
            MeasurementConfig(
                device_id=0,
                input_channel=0,
                output_channel_left=0,
                output_channel_right=1,
                sweep_duration=-1.0,
            )

    def test_invalid_delays(self):
        """Test that negative delays raise error."""
        with pytest.raises(ValueError, match="Delays must be non-negative"):
            MeasurementConfig(
                device_id=0,
                input_channel=0,
                output_channel_left=0,
                output_channel_right=1,
                pre_delay=-0.5,
            )


class TestMeasurementResult:
    """Tests for MeasurementResult."""

    @pytest.fixture
    def sample_result(self, sample_rate):
        """Create a sample measurement result."""
        n_samples = sample_rate
        # Use a deterministic signal with known amplitude
        t = np.arange(n_samples) / sample_rate
        recording = 0.5 * np.sin(2 * np.pi * 1000 * t)  # 0.5 amplitude sine
        ir = np.zeros(int(0.5 * sample_rate))
        ir[0] = 1.0

        return MeasurementResult(
            position_id=0,
            position_name="Center",
            channel=MeasurementChannel.LEFT,
            recording=recording,
            impulse_response=ir,
            frequencies=np.linspace(0, sample_rate / 2, 1000),
            magnitude_db=np.zeros(1000),
            sample_rate=sample_rate,
        )

    def test_peak_level_db(self, sample_result):
        """Test peak level calculation."""
        level = sample_result.peak_level_db
        # 0.5 amplitude sine has peak of 0.5, which is -6dB
        assert -7 < level < -5

    def test_peak_level_silence(self, sample_rate):
        """Test peak level for silence."""
        result = MeasurementResult(
            position_id=0,
            position_name="Center",
            channel=MeasurementChannel.LEFT,
            recording=np.zeros(1000),
            impulse_response=np.zeros(100),
            frequencies=np.zeros(100),
            magnitude_db=np.zeros(100),
            sample_rate=sample_rate,
        )
        assert result.peak_level_db == -100.0

    def test_timestamp_auto_generated(self, sample_result):
        """Test that timestamp is automatically set."""
        assert isinstance(sample_result.timestamp, datetime)


class TestMeasurementPositions:
    """Tests for measurement position definitions."""

    def test_standard_positions_count(self):
        """Test standard position count."""
        assert len(STANDARD_POSITIONS) == 7

    def test_quick_positions_count(self):
        """Test quick position count."""
        assert len(QUICK_POSITIONS) == 5

    def test_thorough_positions_count(self):
        """Test thorough position count."""
        assert len(THOROUGH_POSITIONS) == 9

    def test_positions_have_unique_ids(self):
        """Test that all positions have unique IDs."""
        ids = [p.id for p in THOROUGH_POSITIONS]
        assert len(ids) == len(set(ids))

    def test_center_position_is_origin(self):
        """Test that center position is at origin."""
        center = STANDARD_POSITIONS[0]
        assert center.x_offset == 0
        assert center.y_offset == 0
        assert center.z_offset == 0

    def test_get_positions_5(self):
        """Test getting 5 positions."""
        positions = get_positions(5)
        assert len(positions) == 5

    def test_get_positions_7(self):
        """Test getting 7 positions."""
        positions = get_positions(7)
        assert len(positions) == 7

    def test_get_positions_9(self):
        """Test getting 9 positions."""
        positions = get_positions(9)
        assert len(positions) == 9

    def test_get_positions_default(self):
        """Test default position count."""
        positions = get_positions()
        assert len(positions) == 7


class TestMeasurementEngine:
    """Tests for MeasurementEngine."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return MeasurementConfig(
            device_id=0,
            input_channel=0,
            output_channel_left=0,
            output_channel_right=1,
            sample_rate=48000,
            sweep_duration=0.5,  # Short for testing
            pre_delay=0.1,
            post_delay=0.2,
        )

    @pytest.fixture
    def engine(self, config):
        """Create measurement engine."""
        return MeasurementEngine(config)

    def test_sweep_signal_generated(self, engine):
        """Test that sweep signal is generated."""
        sweep = engine.sweep_signal
        assert len(sweep) > 0
        assert sweep.dtype == np.float64

    def test_total_duration(self, engine, config):
        """Test total duration calculation."""
        expected = config.pre_delay + config.sweep_duration + config.post_delay
        assert engine.total_duration == expected

    def test_capture_returns_result(self, engine):
        """Test that capture returns a MeasurementResult."""
        position = MeasurementPosition(0, "Test", "Test position", 0, 0, 0)

        # Mock the sounddevice playrec function
        def mock_playrec(data, **kwargs):
            # Return a simulated recording (sweep + noise)
            n_samples = len(data)
            # Simulate delayed sweep with some noise
            recording = np.roll(data[:, 0], 100) + np.random.randn(n_samples) * 0.01
            return recording.reshape(-1, 1)

        with patch("sounddevice.playrec", side_effect=mock_playrec):
            result = engine.capture(position, MeasurementChannel.LEFT)

        assert isinstance(result, MeasurementResult)
        assert result.position_id == 0
        assert result.position_name == "Test"
        assert result.channel == MeasurementChannel.LEFT
        assert len(result.recording) > 0
        assert len(result.impulse_response) > 0
        assert len(result.frequencies) > 0
        assert len(result.magnitude_db) > 0

    def test_capture_calls_progress_callback(self, engine):
        """Test that progress callback is called."""
        position = MeasurementPosition(0, "Test", "Test position", 0, 0, 0)
        progress_values = []

        def progress_callback(progress):
            progress_values.append(progress)

        def mock_playrec(data, **kwargs):
            return np.zeros((len(data), 1))

        with patch("sounddevice.playrec", side_effect=mock_playrec):
            engine.capture(position, MeasurementChannel.LEFT, progress_callback)

        # Should have called with 0, 0.5, 0.75, 1.0
        assert len(progress_values) == 4
        assert progress_values[0] == 0.0
        assert progress_values[-1] == 1.0

    def test_capture_left_channel_uses_left_output(self, engine):
        """Test that left channel measurement uses left output."""
        position = MeasurementPosition(0, "Test", "Test position", 0, 0, 0)
        captured_kwargs = {}

        def mock_playrec(data, **kwargs):
            captured_kwargs.update(kwargs)
            return np.zeros((len(data), 1))

        with patch("sounddevice.playrec", side_effect=mock_playrec):
            engine.capture(position, MeasurementChannel.LEFT)

        # output_mapping should route to left channel (0-indexed + 1 = 1)
        assert captured_kwargs["output_mapping"] == [engine.config.output_channel_left + 1]

    def test_capture_right_channel_uses_right_output(self, engine):
        """Test that right channel measurement uses right output."""
        position = MeasurementPosition(0, "Test", "Test position", 0, 0, 0)
        captured_kwargs = {}

        def mock_playrec(data, **kwargs):
            captured_kwargs.update(kwargs)
            return np.zeros((len(data), 1))

        with patch("sounddevice.playrec", side_effect=mock_playrec):
            engine.capture(position, MeasurementChannel.RIGHT)

        # output_mapping should route to right channel (0-indexed + 1 = 2)
        assert captured_kwargs["output_mapping"] == [engine.config.output_channel_right + 1]

    def test_capture_test_signal(self, engine):
        """Test capturing test signal."""
        def mock_rec(frames, **kwargs):
            return np.random.randn(frames, 1) * 0.1

        with patch("sounddevice.rec", side_effect=mock_rec):
            recording = engine.capture_test_signal(0.1)

        expected_length = int(0.1 * engine.config.sample_rate)
        assert len(recording) == expected_length


class TestMeasurementSession:
    """Tests for MeasurementSession."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return MeasurementConfig(
            device_id=0,
            input_channel=0,
            output_channel_left=0,
            output_channel_right=1,
            sample_rate=48000,
            sweep_duration=0.5,
        )

    @pytest.fixture
    def session(self, config):
        """Create measurement session."""
        return MeasurementSession(config, num_positions=7)

    def test_total_measurements(self, session):
        """Test total measurement count."""
        # 7 positions x 2 channels = 14
        assert session.total_measurements == 14

    def test_initial_progress_zero(self, session):
        """Test that initial progress is zero."""
        assert session.progress == 0.0
        assert session.completed_measurements == 0

    def test_add_measurement_updates_progress(self, session, sample_rate):
        """Test that adding measurements updates progress."""
        result = MeasurementResult(
            position_id=0,
            position_name="Center",
            channel=MeasurementChannel.LEFT,
            recording=np.zeros(1000),
            impulse_response=np.zeros(100),
            frequencies=np.zeros(100),
            magnitude_db=np.zeros(100),
            sample_rate=sample_rate,
        )

        session.add_measurement(result)

        assert session.completed_measurements == 1
        assert session.progress == 1 / 14

    def test_get_measurement(self, session, sample_rate):
        """Test getting a specific measurement."""
        result = MeasurementResult(
            position_id=2,
            position_name="Right",
            channel=MeasurementChannel.LEFT,
            recording=np.zeros(1000),
            impulse_response=np.zeros(100),
            frequencies=np.zeros(100),
            magnitude_db=np.zeros(100),
            sample_rate=sample_rate,
        )

        session.add_measurement(result)

        retrieved = session.get_measurement(MeasurementChannel.LEFT, 2)
        assert retrieved is not None
        assert retrieved.position_name == "Right"

        # Non-existent measurement
        assert session.get_measurement(MeasurementChannel.LEFT, 99) is None

    def test_is_channel_complete(self, session, sample_rate):
        """Test channel completion check."""
        assert session.is_channel_complete(MeasurementChannel.LEFT) is False

        # Add all left channel measurements
        for pos in session.positions:
            result = MeasurementResult(
                position_id=pos.id,
                position_name=pos.name,
                channel=MeasurementChannel.LEFT,
                recording=np.zeros(1000),
                impulse_response=np.zeros(100),
                frequencies=np.zeros(100),
                magnitude_db=np.zeros(100),
                sample_rate=sample_rate,
            )
            session.add_measurement(result)

        assert session.is_channel_complete(MeasurementChannel.LEFT) is True
        assert session.is_channel_complete(MeasurementChannel.RIGHT) is False

    def test_is_complete(self, session, sample_rate):
        """Test overall completion check."""
        assert session.is_complete() is False

        # Add all measurements for both channels
        for channel in [MeasurementChannel.LEFT, MeasurementChannel.RIGHT]:
            for pos in session.positions:
                result = MeasurementResult(
                    position_id=pos.id,
                    position_name=pos.name,
                    channel=channel,
                    recording=np.zeros(1000),
                    impulse_response=np.zeros(100),
                    frequencies=np.zeros(100),
                    magnitude_db=np.zeros(100),
                    sample_rate=sample_rate,
                )
                session.add_measurement(result)

        assert session.is_complete() is True

    def test_clear_channel(self, session, sample_rate):
        """Test clearing a channel's measurements."""
        result = MeasurementResult(
            position_id=0,
            position_name="Center",
            channel=MeasurementChannel.LEFT,
            recording=np.zeros(1000),
            impulse_response=np.zeros(100),
            frequencies=np.zeros(100),
            magnitude_db=np.zeros(100),
            sample_rate=sample_rate,
        )
        session.add_measurement(result)
        assert session.completed_measurements == 1

        session.clear_channel(MeasurementChannel.LEFT)
        assert session.completed_measurements == 0

    def test_clear_all(self, session, sample_rate):
        """Test clearing all measurements."""
        for channel in [MeasurementChannel.LEFT, MeasurementChannel.RIGHT]:
            result = MeasurementResult(
                position_id=0,
                position_name="Center",
                channel=channel,
                recording=np.zeros(1000),
                impulse_response=np.zeros(100),
                frequencies=np.zeros(100),
                magnitude_db=np.zeros(100),
                sample_rate=sample_rate,
            )
            session.add_measurement(result)

        assert session.completed_measurements == 2

        session.clear_all()
        assert session.completed_measurements == 0

    def test_get_channel_measurements(self, session, sample_rate):
        """Test getting all measurements for a channel."""
        # Add a couple measurements
        for pos_id in [0, 2]:
            result = MeasurementResult(
                position_id=pos_id,
                position_name=f"Position {pos_id}",
                channel=MeasurementChannel.LEFT,
                recording=np.zeros(1000),
                impulse_response=np.zeros(100),
                frequencies=np.zeros(100),
                magnitude_db=np.zeros(100),
                sample_rate=sample_rate,
            )
            session.add_measurement(result)

        measurements = session.get_channel_measurements(MeasurementChannel.LEFT)
        assert len(measurements) == 2

    def test_different_position_counts(self, config):
        """Test sessions with different position counts."""
        session_5 = MeasurementSession(config, num_positions=5)
        session_7 = MeasurementSession(config, num_positions=7)
        session_9 = MeasurementSession(config, num_positions=9)

        assert session_5.total_measurements == 10
        assert session_7.total_measurements == 14
        assert session_9.total_measurements == 18
