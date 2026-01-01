"""Tests for measurement averaging and frequency response processing."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from roomeq.core.averaging import (
    AveragingMethod,
    average_frequency_responses,
    calculate_deviation_from_target,
    calculate_rms_deviation,
    calculate_weighted_deviation,
    fractional_octave_smoothing,
    interpolate_to_common_frequencies,
)


class TestAverageFrequencyResponses:
    """Tests for frequency response averaging."""

    @pytest.fixture
    def sample_frequencies(self):
        """Create sample frequency array."""
        return np.logspace(np.log10(20), np.log10(20000), 500)

    def test_single_measurement(self, sample_frequencies):
        """Test averaging with single measurement."""
        mag_db = np.zeros(500)  # Flat response

        freqs, avg_db = average_frequency_responses(
            [sample_frequencies], [mag_db], AveragingMethod.POWER
        )

        assert len(freqs) == 500
        assert_allclose(avg_db, mag_db, atol=0.1)

    def test_identical_measurements_unchanged(self, sample_frequencies):
        """Test that averaging identical measurements gives same result."""
        mag_db = np.zeros(500)
        mag_db[100:200] = 6.0  # Add a bump

        freqs, avg_db = average_frequency_responses(
            [sample_frequencies, sample_frequencies],
            [mag_db, mag_db],
            AveragingMethod.POWER,
        )

        assert_allclose(avg_db, mag_db, atol=0.1)

    def test_averaging_reduces_variance(self, sample_frequencies):
        """Test that averaging reduces noise."""
        # Create two measurements with opposite noise
        base = np.zeros(500)
        noise1 = np.random.randn(500) * 3
        noise2 = -noise1  # Opposite noise

        mag1 = base + noise1
        mag2 = base + noise2

        freqs, avg_db = average_frequency_responses(
            [sample_frequencies, sample_frequencies],
            [mag1, mag2],
            AveragingMethod.POWER,
        )

        # Averaged result should have less variance
        assert np.std(avg_db) < np.std(mag1)

    def test_power_vs_magnitude_averaging(self, sample_frequencies):
        """Test difference between power and magnitude averaging."""
        mag1 = np.full(500, 0.0)
        mag2 = np.full(500, 6.0)

        _, avg_power = average_frequency_responses(
            [sample_frequencies, sample_frequencies],
            [mag1, mag2],
            AveragingMethod.POWER,
        )

        _, avg_magnitude = average_frequency_responses(
            [sample_frequencies, sample_frequencies],
            [mag1, mag2],
            AveragingMethod.MAGNITUDE,
        )

        # Power averaging gives higher result due to logarithmic nature
        assert np.mean(avg_power) > np.mean(avg_magnitude)

    def test_empty_list_raises_error(self):
        """Test that empty lists raise error."""
        with pytest.raises(ValueError, match="at least one measurement"):
            average_frequency_responses([], [], AveragingMethod.POWER)

    def test_mismatched_lists_raise_error(self, sample_frequencies):
        """Test that mismatched list lengths raise error."""
        mag1 = np.zeros(500)
        mag2 = np.zeros(500)

        with pytest.raises(ValueError, match="same length"):
            average_frequency_responses(
                [sample_frequencies],
                [mag1, mag2],
                AveragingMethod.POWER,
            )


class TestFractionalOctaveSmoothing:
    """Tests for fractional-octave smoothing."""

    @pytest.fixture
    def noisy_response(self):
        """Create a noisy frequency response."""
        frequencies = np.logspace(np.log10(20), np.log10(20000), 500)
        # Base response with some noise
        response = np.sin(np.linspace(0, 10, 500)) * 3  # Smooth variation
        response += np.random.randn(500) * 2  # Add noise
        return frequencies, response

    def test_smoothing_reduces_noise(self, noisy_response):
        """Test that smoothing reduces high-frequency noise."""
        frequencies, response = noisy_response

        smoothed = fractional_octave_smoothing(frequencies, response, 1 / 6)

        # Smoothed should have less variance
        assert np.std(smoothed) < np.std(response)

    def test_smoothing_preserves_mean(self, noisy_response):
        """Test that smoothing approximately preserves mean level."""
        frequencies, response = noisy_response

        smoothed = fractional_octave_smoothing(frequencies, response, 1 / 6)

        # Mean should be similar (within a few dB)
        assert abs(np.mean(smoothed) - np.mean(response)) < 1.0

    def test_wider_smoothing_smoother_result(self, noisy_response):
        """Test that wider smoothing gives smoother result."""
        frequencies, response = noisy_response

        smoothed_narrow = fractional_octave_smoothing(frequencies, response, 1 / 24)
        smoothed_wide = fractional_octave_smoothing(frequencies, response, 1 / 3)

        # Wider smoothing should have less variance
        assert np.std(smoothed_wide) < np.std(smoothed_narrow)

    def test_invalid_octave_fraction_raises_error(self, noisy_response):
        """Test that invalid octave fraction raises error."""
        frequencies, response = noisy_response

        with pytest.raises(ValueError, match="positive"):
            fractional_octave_smoothing(frequencies, response, 0)

        with pytest.raises(ValueError, match="positive"):
            fractional_octave_smoothing(frequencies, response, -0.1)

    def test_mismatched_arrays_raise_error(self):
        """Test that mismatched arrays raise error."""
        frequencies = np.logspace(1, 4, 100)
        response = np.zeros(50)  # Different length

        with pytest.raises(ValueError, match="same length"):
            fractional_octave_smoothing(frequencies, response, 1 / 6)


class TestDeviationFromTarget:
    """Tests for target deviation calculation."""

    @pytest.fixture
    def flat_response(self):
        """Create a perfectly flat response."""
        frequencies = np.logspace(np.log10(20), np.log10(20000), 500)
        response = np.zeros(500)
        return frequencies, response

    def test_flat_response_zero_deviation(self, flat_response):
        """Test that flat response has zero deviation from flat target."""
        frequencies, response = flat_response

        deviation = calculate_deviation_from_target(frequencies, response, "flat")

        assert_allclose(deviation, 0.0, atol=1e-10)

    def test_positive_response_positive_deviation(self, flat_response):
        """Test that response above target gives positive deviation."""
        frequencies, _ = flat_response
        response = np.full(500, 6.0)  # 6dB above target

        deviation = calculate_deviation_from_target(frequencies, response, "flat")

        assert_allclose(deviation, 6.0, atol=1e-10)

    def test_negative_response_negative_deviation(self, flat_response):
        """Test that response below target gives negative deviation."""
        frequencies, _ = flat_response
        response = np.full(500, -3.0)  # 3dB below target

        deviation = calculate_deviation_from_target(frequencies, response, "flat")

        assert_allclose(deviation, -3.0, atol=1e-10)

    def test_house_curve_target(self, flat_response):
        """Test house curve target calculation."""
        frequencies, response = flat_response

        deviation = calculate_deviation_from_target(frequencies, response, "house_curve")

        # Bass should show negative deviation (flat is below house curve)
        bass_idx = frequencies < 100
        if np.any(bass_idx):
            assert np.mean(deviation[bass_idx]) < 0

    def test_unknown_target_raises_error(self, flat_response):
        """Test that unknown target raises error."""
        frequencies, response = flat_response

        with pytest.raises(ValueError, match="Unknown target"):
            calculate_deviation_from_target(frequencies, response, "unknown")


class TestRMSDeviation:
    """Tests for RMS deviation calculation."""

    def test_zero_deviation(self):
        """Test RMS of zero deviation."""
        deviation = np.zeros(100)
        rms = calculate_rms_deviation(deviation)
        assert rms == 0.0

    def test_constant_deviation(self):
        """Test RMS of constant deviation."""
        deviation = np.full(100, 6.0)
        rms = calculate_rms_deviation(deviation)
        assert_allclose(rms, 6.0)

    def test_symmetric_deviation(self):
        """Test RMS of symmetric positive/negative deviation."""
        deviation = np.array([6.0, -6.0, 6.0, -6.0])
        rms = calculate_rms_deviation(deviation)
        assert_allclose(rms, 6.0)


class TestWeightedDeviation:
    """Tests for weighted deviation calculation."""

    @pytest.fixture
    def sample_data(self):
        """Create sample frequencies and deviation."""
        frequencies = np.logspace(np.log10(20), np.log10(20000), 500)
        deviation = np.full(500, 6.0)
        return frequencies, deviation

    def test_flat_weighting(self, sample_data):
        """Test flat weighting equals RMS."""
        frequencies, deviation = sample_data

        weighted = calculate_weighted_deviation(frequencies, deviation, "flat")
        rms = calculate_rms_deviation(deviation)

        assert_allclose(weighted, rms)

    def test_audible_weighting_reduces_extremes(self, sample_data):
        """Test that audible weighting de-emphasizes extreme frequencies."""
        frequencies, _ = sample_data

        # Create deviation only at extreme frequencies
        deviation = np.zeros(500)
        deviation[frequencies < 30] = 10.0
        deviation[frequencies > 16000] = 10.0

        weighted = calculate_weighted_deviation(frequencies, deviation, "audible")
        unweighted = calculate_rms_deviation(deviation)

        # Weighted should be less due to de-emphasis of extremes
        assert weighted < unweighted

    def test_bass_weighting_emphasizes_bass(self, sample_data):
        """Test that bass weighting emphasizes low frequencies."""
        frequencies, _ = sample_data

        # Create deviation only at low frequencies
        deviation = np.zeros(500)
        bass_mask = frequencies < 200
        deviation[bass_mask] = 6.0

        bass_weighted = calculate_weighted_deviation(frequencies, deviation, "bass")
        flat_weighted = calculate_weighted_deviation(frequencies, deviation, "flat")

        # Bass weighting should give higher result for bass deviation
        assert bass_weighted > flat_weighted

    def test_unknown_emphasis_raises_error(self, sample_data):
        """Test that unknown emphasis raises error."""
        frequencies, deviation = sample_data

        with pytest.raises(ValueError, match="Unknown emphasis"):
            calculate_weighted_deviation(frequencies, deviation, "unknown")


class TestInterpolateToCommonFrequencies:
    """Tests for frequency interpolation."""

    def test_same_frequencies_unchanged(self):
        """Test that identical frequencies don't change values."""
        frequencies = np.logspace(1, 4, 100)
        magnitude = np.random.randn(100)

        common_freq, interpolated = interpolate_to_common_frequencies(
            [frequencies], [magnitude]
        )

        assert_allclose(common_freq, frequencies)
        assert_allclose(interpolated[0], magnitude)

    def test_different_frequencies_interpolated(self):
        """Test interpolation between different frequency sets."""
        freq1 = np.logspace(1, 4, 100)
        freq2 = np.logspace(1, 4, 200)  # Different resolution
        mag1 = np.zeros(100)
        mag2 = np.full(200, 6.0)

        common_freq, interpolated = interpolate_to_common_frequencies(
            [freq1, freq2], [mag1, mag2]
        )

        # Should use first frequency set
        assert len(common_freq) == 100
        assert len(interpolated) == 2
        assert_allclose(interpolated[0], mag1)  # First unchanged

    def test_custom_target_frequencies(self):
        """Test using custom target frequencies."""
        freq1 = np.logspace(1, 4, 100)
        mag1 = np.zeros(100)
        target = np.logspace(1, 4, 50)  # Different size

        common_freq, interpolated = interpolate_to_common_frequencies(
            [freq1], [mag1], target_frequencies=target
        )

        assert len(common_freq) == 50
        assert len(interpolated[0]) == 50
