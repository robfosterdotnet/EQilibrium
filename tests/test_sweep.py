"""Tests for sweep generation and deconvolution."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from roomeq.core.sweep import (
    Deconvolver,
    SweepGenerator,
    SweepParameters,
    calculate_frequency_response,
)


class TestSweepParameters:
    """Tests for SweepParameters validation."""

    def test_default_parameters(self):
        """Test default parameters are valid."""
        params = SweepParameters()
        assert params.duration == 5.0
        assert params.sample_rate == 48000
        assert params.start_freq == 20.0
        assert params.end_freq == 20000.0
        assert params.amplitude == 0.8

    def test_invalid_duration(self):
        """Test that negative duration raises error."""
        with pytest.raises(ValueError, match="Duration must be positive"):
            SweepParameters(duration=-1.0)

    def test_invalid_sample_rate(self):
        """Test that zero sample rate raises error."""
        with pytest.raises(ValueError, match="Sample rate must be positive"):
            SweepParameters(sample_rate=0)

    def test_invalid_frequency_order(self):
        """Test that start_freq >= end_freq raises error."""
        with pytest.raises(ValueError, match="Start frequency must be less"):
            SweepParameters(start_freq=20000.0, end_freq=20.0)

    def test_invalid_amplitude(self):
        """Test that amplitude > 1 raises error."""
        with pytest.raises(ValueError, match="Amplitude must be between"):
            SweepParameters(amplitude=1.5)


class TestSweepGenerator:
    """Tests for SweepGenerator."""

    @pytest.fixture
    def generator(self):
        """Create a sweep generator."""
        return SweepGenerator()

    @pytest.fixture
    def short_params(self, sample_rate):
        """Create short sweep parameters for fast tests."""
        return SweepParameters(
            duration=0.5,
            sample_rate=sample_rate,
            start_freq=100.0,
            end_freq=10000.0,
            amplitude=0.8,
        )

    def test_sweep_length(self, generator, short_params):
        """Test that sweep has correct length."""
        sweep = generator.generate(short_params)
        expected_length = int(short_params.duration * short_params.sample_rate)
        assert len(sweep) == expected_length

    def test_sweep_amplitude(self, generator, short_params):
        """Test that sweep amplitude is within bounds."""
        sweep = generator.generate(short_params)
        assert np.max(np.abs(sweep)) <= short_params.amplitude
        # Most of the signal should be close to amplitude (except fades)
        assert np.max(np.abs(sweep)) > short_params.amplitude * 0.9

    def test_sweep_starts_at_low_frequency(self, generator, sample_rate):
        """Test that sweep starts at low frequency."""
        params = SweepParameters(
            duration=1.0,
            sample_rate=sample_rate,
            start_freq=100.0,
            end_freq=10000.0,
        )
        sweep = generator.generate(params)

        # Measure frequency at start (zero crossings in first 0.1s)
        start_segment = sweep[: int(0.1 * sample_rate)]
        zero_crossings = np.where(np.diff(np.signbit(start_segment)))[0]

        # Estimated frequency from zero crossings
        if len(zero_crossings) > 1:
            avg_period = np.mean(np.diff(zero_crossings)) * 2 / sample_rate
            estimated_freq = 1 / avg_period
            # Should be close to start frequency
            assert 50 < estimated_freq < 500  # Reasonable range around 100Hz

    def test_sweep_ends_at_high_frequency(self, generator, sample_rate):
        """Test that sweep ends at high frequency."""
        params = SweepParameters(
            duration=1.0,
            sample_rate=sample_rate,
            start_freq=100.0,
            end_freq=10000.0,
        )
        sweep = generator.generate(params)

        # Measure frequency at end (zero crossings in last 0.01s)
        end_segment = sweep[-int(0.01 * sample_rate) :]
        zero_crossings = np.where(np.diff(np.signbit(end_segment)))[0]

        # Estimated frequency from zero crossings
        if len(zero_crossings) > 1:
            avg_period = np.mean(np.diff(zero_crossings)) * 2 / sample_rate
            estimated_freq = 1 / avg_period
            # Should be close to end frequency
            assert 5000 < estimated_freq < 15000

    def test_fade_in_applied(self, generator, sample_rate):
        """Test that fade in is applied at start."""
        params = SweepParameters(
            duration=1.0,
            sample_rate=sample_rate,
            fade_in=0.05,  # 50ms fade
        )
        sweep = generator.generate(params)

        # First sample should be near zero
        assert abs(sweep[0]) < 0.01

        # Sample at middle of fade should be growing
        mid_fade = int(0.025 * sample_rate)
        assert abs(sweep[mid_fade]) > 0.01

    def test_fade_out_applied(self, generator, sample_rate):
        """Test that fade out is applied at end."""
        params = SweepParameters(
            duration=1.0,
            sample_rate=sample_rate,
            fade_out=0.05,
        )
        sweep = generator.generate(params)

        # Last sample should be near zero
        assert abs(sweep[-1]) < 0.01

    def test_inverse_filter_length(self, generator, short_params):
        """Test that inverse filter has same length as sweep."""
        sweep = generator.generate(short_params)
        inverse = generator.generate_inverse_filter(sweep, short_params)
        assert len(inverse) == len(sweep)

    def test_inverse_filter_normalized(self, generator, short_params):
        """Test that inverse filter is normalized to max 1."""
        sweep = generator.generate(short_params)
        inverse = generator.generate_inverse_filter(sweep, short_params)
        assert_allclose(np.max(np.abs(inverse)), 1.0, rtol=1e-6)


class TestDeconvolver:
    """Tests for Deconvolver."""

    @pytest.fixture
    def deconvolver(self):
        """Create a deconvolver."""
        return Deconvolver()

    @pytest.fixture
    def generator(self):
        """Create a sweep generator."""
        return SweepGenerator()

    def test_next_power_of_2(self, deconvolver):
        """Test next power of 2 calculation."""
        assert deconvolver._next_power_of_2(1) == 1
        assert deconvolver._next_power_of_2(2) == 2
        assert deconvolver._next_power_of_2(3) == 4
        assert deconvolver._next_power_of_2(5) == 8
        assert deconvolver._next_power_of_2(1000) == 1024

    def test_perfect_deconvolution(self, generator, deconvolver, sample_rate):
        """Test that deconvolving sweep with itself gives impulse."""
        params = SweepParameters(
            duration=0.5,
            sample_rate=sample_rate,
            start_freq=100.0,
            end_freq=10000.0,
        )
        sweep = generator.generate(params)
        inverse = generator.generate_inverse_filter(sweep, params)

        # Deconvolve sweep with its inverse
        ir = deconvolver.deconvolve(sweep, inverse)

        # Should produce an impulse-like response
        # The peak should be significantly larger than the noise
        peak_idx = np.argmax(np.abs(ir))
        peak_value = np.abs(ir[peak_idx])
        rms = np.sqrt(np.mean(ir**2))

        # Peak should be much larger than RMS (impulse characteristic)
        assert peak_value > 10 * rms

    def test_deconvolution_with_delay(self, generator, deconvolver, sample_rate):
        """Test deconvolution detects delay in recording."""
        params = SweepParameters(
            duration=0.5,
            sample_rate=sample_rate,
            start_freq=100.0,
            end_freq=10000.0,
        )
        sweep = generator.generate(params)
        inverse = generator.generate_inverse_filter(sweep, params)

        # Simulate delayed recording (like room delay)
        delay_samples = int(0.01 * sample_rate)  # 10ms delay
        delayed = np.concatenate([np.zeros(delay_samples), sweep])

        # Deconvolve
        ir = deconvolver.deconvolve(delayed, inverse)

        # The peak should be shifted by approximately the delay
        peak_idx = np.argmax(np.abs(ir))
        sweep_peak_idx = np.argmax(np.abs(deconvolver.deconvolve(sweep, inverse)))

        # Peak difference should be approximately the delay
        peak_diff = peak_idx - sweep_peak_idx
        assert abs(peak_diff - delay_samples) < 10  # Allow small error

    def test_extract_linear_ir(self, generator, deconvolver, sample_rate):
        """Test extraction of linear impulse response."""
        params = SweepParameters(
            duration=0.5,
            sample_rate=sample_rate,
            start_freq=100.0,
            end_freq=10000.0,
        )
        sweep = generator.generate(params)
        inverse = generator.generate_inverse_filter(sweep, params)

        full_ir = deconvolver.deconvolve(sweep, inverse)
        linear_ir = deconvolver.extract_linear_ir(full_ir, params, ir_length=0.1)

        # Linear IR should be shorter than full IR
        expected_length = int(0.1 * sample_rate)
        assert len(linear_ir) == expected_length

        # Should be normalized to 1.0
        assert_allclose(np.max(np.abs(linear_ir)), 1.0, rtol=1e-6)


class TestFrequencyResponse:
    """Tests for frequency response calculation."""

    def test_impulse_has_flat_response(self, sample_rate):
        """Test that a perfect impulse has flat frequency response."""
        # Create perfect impulse
        ir = np.zeros(1024)
        ir[0] = 1.0

        freqs, mag_db = calculate_frequency_response(ir, sample_rate)

        # Should be flat (all values equal)
        # Allow for small variations due to windowing
        variation = np.max(mag_db) - np.min(mag_db)
        assert variation < 1.0  # Less than 1dB variation

    def test_frequency_range(self, sample_rate):
        """Test that frequency array covers expected range."""
        ir = np.zeros(4096)
        ir[0] = 1.0

        freqs, mag_db = calculate_frequency_response(ir, sample_rate, n_points=4096)

        # Should start at 0 Hz
        assert freqs[0] == 0.0

        # Should end near Nyquist
        assert freqs[-1] <= sample_rate / 2

    def test_low_pass_filter_response(self, sample_rate):
        """Test frequency response of a simple low-pass IR."""
        # Create simple exponential decay (acts like low-pass)
        n_samples = 4096
        t = np.arange(n_samples) / sample_rate
        ir = np.exp(-t * 1000)  # Fast decay = more low-pass

        freqs, mag_db = calculate_frequency_response(ir, sample_rate)

        # Low frequencies should be louder than high frequencies
        low_freq_idx = np.where((freqs > 100) & (freqs < 500))[0]
        high_freq_idx = np.where((freqs > 5000) & (freqs < 10000))[0]

        if len(low_freq_idx) > 0 and len(high_freq_idx) > 0:
            low_avg = np.mean(mag_db[low_freq_idx])
            high_avg = np.mean(mag_db[high_freq_idx])
            assert low_avg > high_avg  # Low frequencies louder

    def test_response_normalized_to_zero_db(self, sample_rate):
        """Test that response is normalized so max is 0dB."""
        ir = np.zeros(1024)
        ir[0] = 0.5  # Quieter impulse

        freqs, mag_db = calculate_frequency_response(ir, sample_rate)

        # Max should be 0dB
        assert_allclose(np.max(mag_db), 0.0, atol=0.1)
