"""Tests for biquad filter coefficient calculation."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from roomeq.core.biquad import (
    BiquadCoefficients,
    EQBand,
    FilterType,
    calculate_coefficients,
    calculate_combined_response,
    calculate_frequency_response,
    calculate_high_shelf,
    calculate_low_shelf,
    calculate_peaking_eq,
    create_correction_band,
)


class TestEQBand:
    """Tests for EQBand dataclass."""

    def test_valid_band(self):
        """Test creating a valid EQ band."""
        band = EQBand(
            filter_type=FilterType.PEAKING,
            frequency=1000.0,
            gain=6.0,
            q=2.0,
        )
        assert band.frequency == 1000.0
        assert band.gain == 6.0
        assert band.q == 2.0
        assert band.enabled is True

    def test_invalid_frequency(self):
        """Test that invalid frequency raises error."""
        with pytest.raises(ValueError, match="Frequency must be positive"):
            EQBand(FilterType.PEAKING, frequency=0, gain=0, q=1)

        with pytest.raises(ValueError, match="Frequency must be positive"):
            EQBand(FilterType.PEAKING, frequency=-100, gain=0, q=1)

    def test_invalid_gain(self):
        """Test that excessive gain raises error."""
        with pytest.raises(ValueError, match="Gain must be between"):
            EQBand(FilterType.PEAKING, frequency=1000, gain=50, q=1)

    def test_invalid_q(self):
        """Test that invalid Q raises error."""
        with pytest.raises(ValueError, match="Q must be positive"):
            EQBand(FilterType.PEAKING, frequency=1000, gain=0, q=0)


class TestPeakingEQ:
    """Tests for peaking EQ filter."""

    @pytest.fixture
    def sample_rate(self):
        return 48000

    @pytest.fixture
    def test_frequencies(self):
        return np.logspace(np.log10(20), np.log10(20000), 500)

    def test_zero_gain_is_unity(self, sample_rate, test_frequencies):
        """Test that zero gain gives unity response."""
        coeffs = calculate_peaking_eq(1000, 0.0, 1.0, sample_rate)
        response = calculate_frequency_response(coeffs, test_frequencies, sample_rate)

        # Should be approximately 0 dB everywhere
        assert_allclose(response, 0.0, atol=0.01)

    def test_positive_gain_boosts(self, sample_rate, test_frequencies):
        """Test that positive gain boosts at center frequency."""
        coeffs = calculate_peaking_eq(1000, 6.0, 2.0, sample_rate)
        response = calculate_frequency_response(coeffs, test_frequencies, sample_rate)

        # Find response at 1000 Hz
        idx_1k = np.argmin(np.abs(test_frequencies - 1000))
        assert response[idx_1k] > 5.0  # Should be close to +6 dB

    def test_negative_gain_cuts(self, sample_rate, test_frequencies):
        """Test that negative gain cuts at center frequency."""
        coeffs = calculate_peaking_eq(1000, -6.0, 2.0, sample_rate)
        response = calculate_frequency_response(coeffs, test_frequencies, sample_rate)

        idx_1k = np.argmin(np.abs(test_frequencies - 1000))
        assert response[idx_1k] < -5.0  # Should be close to -6 dB

    def test_higher_q_narrower_peak(self, sample_rate, test_frequencies):
        """Test that higher Q gives narrower peak."""
        coeffs_low_q = calculate_peaking_eq(1000, 6.0, 1.0, sample_rate)
        coeffs_high_q = calculate_peaking_eq(1000, 6.0, 8.0, sample_rate)

        response_low_q = calculate_frequency_response(coeffs_low_q, test_frequencies, sample_rate)
        response_high_q = calculate_frequency_response(
            coeffs_high_q, test_frequencies, sample_rate
        )

        # High Q should have less effect at frequencies away from center
        idx_500 = np.argmin(np.abs(test_frequencies - 500))
        assert abs(response_high_q[idx_500]) < abs(response_low_q[idx_500])

    def test_coefficients_normalized(self, sample_rate):
        """Test that coefficients are properly normalized."""
        coeffs = calculate_peaking_eq(1000, 6.0, 2.0, sample_rate)

        # a0 should be 1.0 after normalization
        assert coeffs.a0 == 1.0


class TestLowShelf:
    """Tests for low shelf filter."""

    @pytest.fixture
    def sample_rate(self):
        return 48000

    @pytest.fixture
    def test_frequencies(self):
        return np.logspace(np.log10(20), np.log10(20000), 500)

    def test_zero_gain_is_unity(self, sample_rate, test_frequencies):
        """Test that zero gain gives unity response."""
        coeffs = calculate_low_shelf(100, 0.0, 0.707, sample_rate)
        response = calculate_frequency_response(coeffs, test_frequencies, sample_rate)

        assert_allclose(response, 0.0, atol=0.01)

    def test_boosts_below_frequency(self, sample_rate, test_frequencies):
        """Test that low shelf boosts below shelf frequency."""
        coeffs = calculate_low_shelf(200, 6.0, 0.707, sample_rate)
        response = calculate_frequency_response(coeffs, test_frequencies, sample_rate)

        # Low frequencies should be boosted
        idx_50 = np.argmin(np.abs(test_frequencies - 50))
        assert response[idx_50] > 4.0

        # High frequencies should be unaffected
        idx_2k = np.argmin(np.abs(test_frequencies - 2000))
        assert abs(response[idx_2k]) < 1.0

    def test_cuts_below_frequency(self, sample_rate, test_frequencies):
        """Test that negative gain cuts below shelf frequency."""
        coeffs = calculate_low_shelf(200, -6.0, 0.707, sample_rate)
        response = calculate_frequency_response(coeffs, test_frequencies, sample_rate)

        idx_50 = np.argmin(np.abs(test_frequencies - 50))
        assert response[idx_50] < -4.0


class TestHighShelf:
    """Tests for high shelf filter."""

    @pytest.fixture
    def sample_rate(self):
        return 48000

    @pytest.fixture
    def test_frequencies(self):
        return np.logspace(np.log10(20), np.log10(20000), 500)

    def test_zero_gain_is_unity(self, sample_rate, test_frequencies):
        """Test that zero gain gives unity response."""
        coeffs = calculate_high_shelf(5000, 0.0, 0.707, sample_rate)
        response = calculate_frequency_response(coeffs, test_frequencies, sample_rate)

        assert_allclose(response, 0.0, atol=0.01)

    def test_boosts_above_frequency(self, sample_rate, test_frequencies):
        """Test that high shelf boosts above shelf frequency."""
        coeffs = calculate_high_shelf(5000, 6.0, 0.707, sample_rate)
        response = calculate_frequency_response(coeffs, test_frequencies, sample_rate)

        # High frequencies should be boosted
        idx_10k = np.argmin(np.abs(test_frequencies - 10000))
        assert response[idx_10k] > 4.0

        # Low frequencies should be unaffected
        idx_100 = np.argmin(np.abs(test_frequencies - 100))
        assert abs(response[idx_100]) < 1.0


class TestCalculateCoefficients:
    """Tests for the calculate_coefficients function."""

    @pytest.fixture
    def sample_rate(self):
        return 48000

    def test_peaking_type(self, sample_rate):
        """Test that peaking type uses peaking EQ."""
        band = EQBand(FilterType.PEAKING, 1000, 6.0, 2.0)
        coeffs = calculate_coefficients(band, sample_rate)

        # Should match direct calculation
        expected = calculate_peaking_eq(1000, 6.0, 2.0, sample_rate)
        assert coeffs == expected

    def test_low_shelf_type(self, sample_rate):
        """Test that low shelf type uses low shelf."""
        band = EQBand(FilterType.LOW_SHELF, 100, 6.0, 0.707)
        coeffs = calculate_coefficients(band, sample_rate)

        expected = calculate_low_shelf(100, 6.0, 0.707, sample_rate)
        assert coeffs == expected

    def test_high_shelf_type(self, sample_rate):
        """Test that high shelf type uses high shelf."""
        band = EQBand(FilterType.HIGH_SHELF, 5000, 6.0, 0.707)
        coeffs = calculate_coefficients(band, sample_rate)

        expected = calculate_high_shelf(5000, 6.0, 0.707, sample_rate)
        assert coeffs == expected


class TestCombinedResponse:
    """Tests for combined response calculation."""

    @pytest.fixture
    def sample_rate(self):
        return 48000

    @pytest.fixture
    def test_frequencies(self):
        return np.logspace(np.log10(20), np.log10(20000), 500)

    def test_empty_bands(self, test_frequencies, sample_rate):
        """Test that empty bands gives unity response."""
        response = calculate_combined_response([], test_frequencies, sample_rate)

        assert_allclose(response, 0.0)

    def test_single_band_matches_individual(self, test_frequencies, sample_rate):
        """Test that single band matches individual calculation."""
        band = EQBand(FilterType.PEAKING, 1000, 6.0, 2.0)
        combined = calculate_combined_response([band], test_frequencies, sample_rate)

        coeffs = calculate_coefficients(band, sample_rate)
        individual = calculate_frequency_response(coeffs, test_frequencies, sample_rate)

        assert_allclose(combined, individual)

    def test_multiple_bands_sum(self, test_frequencies, sample_rate):
        """Test that multiple bands sum in dB."""
        band1 = EQBand(FilterType.PEAKING, 100, 3.0, 2.0)
        band2 = EQBand(FilterType.PEAKING, 1000, 3.0, 2.0)

        combined = calculate_combined_response([band1, band2], test_frequencies, sample_rate)

        # At a frequency affected by both, response should be summed
        # This is approximate since bands overlap differently
        assert np.max(combined) > 2.5

    def test_disabled_bands_excluded(self, test_frequencies, sample_rate):
        """Test that disabled bands are excluded."""
        band1 = EQBand(FilterType.PEAKING, 100, 6.0, 2.0, enabled=True)
        band2 = EQBand(FilterType.PEAKING, 1000, 6.0, 2.0, enabled=False)

        combined = calculate_combined_response([band1, band2], test_frequencies, sample_rate)

        # Only band1 should contribute
        idx_100 = np.argmin(np.abs(test_frequencies - 100))
        idx_1k = np.argmin(np.abs(test_frequencies - 1000))

        assert combined[idx_100] > 4.0  # Band1 active
        assert abs(combined[idx_1k]) < 1.0  # Band2 disabled


class TestCreateCorrectionBand:
    """Tests for correction band creation."""

    def test_positive_deviation_negative_gain(self):
        """Test that positive deviation creates negative gain."""
        band = create_correction_band(100, 6.0, 4.0)

        assert band.gain == -6.0
        assert band.frequency == 100.0
        assert band.q == 4.0

    def test_negative_deviation_positive_gain(self):
        """Test that negative deviation creates positive gain."""
        band = create_correction_band(100, -6.0, 4.0)

        assert band.gain == 6.0

    def test_gain_clamped(self):
        """Test that gain is clamped to reasonable range."""
        band_high = create_correction_band(100, 30.0, 4.0)
        band_low = create_correction_band(100, -30.0, 4.0)

        assert band_high.gain >= -20.0
        assert band_low.gain <= 20.0

    def test_filter_type_default(self):
        """Test default filter type is peaking."""
        band = create_correction_band(100, 6.0, 4.0)
        assert band.filter_type == FilterType.PEAKING

    def test_custom_filter_type(self):
        """Test custom filter type."""
        band = create_correction_band(100, 6.0, 4.0, FilterType.LOW_SHELF)
        assert band.filter_type == FilterType.LOW_SHELF


class TestFrequencyResponseAccuracy:
    """Tests for frequency response calculation accuracy."""

    @pytest.fixture
    def sample_rate(self):
        return 48000

    def test_gain_at_center_frequency(self, sample_rate):
        """Test that gain at center frequency matches specified gain."""
        for gain in [-12, -6, 0, 6, 12]:
            coeffs = calculate_peaking_eq(1000, gain, 2.0, sample_rate)
            response = calculate_frequency_response(
                coeffs, np.array([1000.0]), sample_rate
            )

            assert_allclose(response[0], gain, atol=0.1)

    def test_symmetric_boost_cut(self, sample_rate):
        """Test that boost and cut are symmetric."""
        frequencies = np.logspace(np.log10(20), np.log10(20000), 500)

        coeffs_boost = calculate_peaking_eq(1000, 6.0, 2.0, sample_rate)
        coeffs_cut = calculate_peaking_eq(1000, -6.0, 2.0, sample_rate)

        response_boost = calculate_frequency_response(coeffs_boost, frequencies, sample_rate)
        response_cut = calculate_frequency_response(coeffs_cut, frequencies, sample_rate)

        # Boost and cut should be approximately symmetric
        assert_allclose(response_boost, -response_cut, atol=0.5)
