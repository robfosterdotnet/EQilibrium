"""Tests for EQ optimization algorithm."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from roomeq.core.analysis import ProblemType, RoomProblem, Severity
from roomeq.core.biquad import EQBand, FilterType
from roomeq.core.eq_optimizer import (
    RME_MAX_BANDS,
    RME_MAX_FREQ,
    RME_MAX_GAIN,
    RME_MAX_Q,
    RME_MIN_FREQ,
    RME_MIN_GAIN,
    RME_MIN_Q,
    EQSettings,
    OptimizationResult,
    bands_to_params,
    get_parameter_bounds,
    initialize_bands_from_problems,
    optimize_eq,
    params_to_bands,
    quick_optimize,
    round_to_rme_precision,
    validate_for_rme,
)


class TestRoundToRMEPrecision:
    """Tests for RME precision rounding."""

    def test_frequency_rounded(self):
        """Test that frequency is rounded to 1 Hz."""
        band = EQBand(FilterType.PEAKING, 100.4, 6.0, 2.0)
        rounded = round_to_rme_precision(band)
        assert rounded.frequency == 100.0

        band = EQBand(FilterType.PEAKING, 100.6, 6.0, 2.0)
        rounded = round_to_rme_precision(band)
        assert rounded.frequency == 101.0

    def test_gain_rounded(self):
        """Test that gain is rounded to 0.1 dB."""
        band = EQBand(FilterType.PEAKING, 1000, 6.04, 2.0)
        rounded = round_to_rme_precision(band)
        assert rounded.gain == pytest.approx(6.0, abs=0.001)

        band = EQBand(FilterType.PEAKING, 1000, 6.06, 2.0)
        rounded = round_to_rme_precision(band)
        assert rounded.gain == pytest.approx(6.1, abs=0.001)

    def test_q_rounded(self):
        """Test that Q is rounded to 0.1."""
        band = EQBand(FilterType.PEAKING, 1000, 6.0, 2.04)
        rounded = round_to_rme_precision(band)
        assert rounded.q == 2.0

        band = EQBand(FilterType.PEAKING, 1000, 6.0, 2.06)
        rounded = round_to_rme_precision(band)
        assert rounded.q == 2.1

    def test_frequency_clamped(self):
        """Test that frequency is clamped to RME range."""
        band = EQBand(FilterType.PEAKING, 10, 6.0, 2.0)
        rounded = round_to_rme_precision(band)
        assert rounded.frequency == RME_MIN_FREQ

        band = EQBand(FilterType.PEAKING, 25000, 6.0, 2.0)
        rounded = round_to_rme_precision(band)
        assert rounded.frequency == RME_MAX_FREQ

    def test_gain_clamped(self):
        """Test that gain is clamped to RME range."""
        band = EQBand(FilterType.PEAKING, 1000, -25.0, 2.0)
        rounded = round_to_rme_precision(band)
        assert rounded.gain == RME_MIN_GAIN

        band = EQBand(FilterType.PEAKING, 1000, 25.0, 2.0)
        rounded = round_to_rme_precision(band)
        assert rounded.gain == RME_MAX_GAIN

    def test_q_clamped(self):
        """Test that Q is clamped to RME range."""
        band = EQBand(FilterType.PEAKING, 1000, 6.0, 0.2)
        rounded = round_to_rme_precision(band)
        assert rounded.q == RME_MIN_Q

        band = EQBand(FilterType.PEAKING, 1000, 6.0, 15.0)
        rounded = round_to_rme_precision(band)
        assert rounded.q == RME_MAX_Q


class TestValidateForRME:
    """Tests for RME validation."""

    def test_valid_settings(self):
        """Test that valid settings pass validation."""
        bands = [
            EQBand(FilterType.PEAKING, 100, 6.0, 2.0),
            EQBand(FilterType.PEAKING, 1000, -6.0, 4.0),
        ]
        errors = validate_for_rme(bands)
        assert errors == []

    def test_too_many_bands(self):
        """Test that too many bands fails validation."""
        bands = [EQBand(FilterType.PEAKING, 100 * (i + 1), 0.0, 1.0) for i in range(10)]
        errors = validate_for_rme(bands)
        assert len(errors) == 1
        assert "Too many bands" in errors[0]

    def test_frequency_out_of_range(self):
        """Test that out-of-range frequency fails validation."""
        bands = [EQBand(FilterType.PEAKING, 10, 6.0, 2.0)]
        errors = validate_for_rme(bands)
        assert any("Frequency" in e for e in errors)

    def test_gain_out_of_range(self):
        """Test that out-of-range gain fails validation."""
        bands = [EQBand(FilterType.PEAKING, 1000, -25.0, 2.0)]
        errors = validate_for_rme(bands)
        assert any("Gain" in e for e in errors)

    def test_q_out_of_range(self):
        """Test that out-of-range Q fails validation."""
        bands = [EQBand(FilterType.PEAKING, 1000, 6.0, 0.2)]
        errors = validate_for_rme(bands)
        assert any("Q" in e for e in errors)


class TestInitializeBandsFromProblems:
    """Tests for band initialization from detected problems."""

    @pytest.fixture
    def sample_problems(self):
        """Create sample room problems."""
        return [
            RoomProblem(ProblemType.PEAK, 63.0, 8.0, 4.0, Severity.SEVERE, "Peak at 63Hz"),
            RoomProblem(ProblemType.DIP, 125.0, -5.0, 3.0, Severity.MODERATE, "Dip at 125Hz"),
            RoomProblem(ProblemType.PEAK, 250.0, 4.0, 5.0, Severity.MINOR, "Peak at 250Hz"),
        ]

    def test_creates_bands_from_problems(self, sample_problems):
        """Test that bands are created from problems."""
        bands = initialize_bands_from_problems(sample_problems)

        assert len(bands) > 0
        assert len(bands) <= RME_MAX_BANDS

    def test_bands_are_rounded(self, sample_problems):
        """Test that created bands are rounded to RME precision."""
        bands = initialize_bands_from_problems(sample_problems)

        for band in bands:
            # Check that values are at RME step resolution
            assert band.frequency == round(band.frequency)
            assert band.gain == round(band.gain * 10) / 10
            assert band.q == round(band.q * 10) / 10

    def test_correction_inverts_deviation(self, sample_problems):
        """Test that correction gain is opposite of deviation."""
        bands = initialize_bands_from_problems(sample_problems)

        # Find band for 63Hz peak (8dB deviation)
        band_63 = None
        for band in bands:
            if 60 < band.frequency < 70:
                band_63 = band
                break

        if band_63:
            # Correction should be negative (to counter positive deviation)
            assert band_63.gain < 0

    def test_respects_max_bands(self, sample_problems):
        """Test that max_bands from profile is respected."""
        # RME profile has max_bands=9, all 3 problems should fit
        bands = initialize_bands_from_problems(sample_problems)
        assert len(bands) <= RME_MAX_BANDS
        # With 3 problems, should have at most 3 bands
        assert len(bands) <= len(sample_problems)

    def test_empty_problems_empty_bands(self):
        """Test that empty problems gives empty bands."""
        bands = initialize_bands_from_problems([])
        assert len(bands) == 0


class TestParamsConversion:
    """Tests for parameter array conversion."""

    def test_bands_to_params_round_trip(self):
        """Test that bands can be converted to params and back."""
        original_bands = [
            EQBand(FilterType.PEAKING, 100, 6.0, 2.0),
            EQBand(FilterType.PEAKING, 1000, -3.0, 4.0),
        ]

        params = bands_to_params(original_bands)
        recovered_bands = params_to_bands(params)

        assert len(recovered_bands) == len(original_bands)

        for orig, recov in zip(original_bands, recovered_bands):
            assert_allclose(recov.frequency, orig.frequency, rtol=1e-6)
            assert_allclose(recov.gain, orig.gain, rtol=1e-6)
            assert_allclose(recov.q, orig.q, rtol=1e-6)

    def test_params_length(self):
        """Test that params array has correct length."""
        bands = [EQBand(FilterType.PEAKING, 100, 6.0, 2.0) for _ in range(5)]
        params = bands_to_params(bands)

        # 3 params per band
        assert len(params) == 15

    def test_get_parameter_bounds_length(self):
        """Test that bounds have correct length."""
        bounds = get_parameter_bounds(5)
        assert len(bounds) == 15  # 3 per band


class TestEQSettings:
    """Tests for EQSettings dataclass."""

    def test_num_active_bands(self):
        """Test counting of active bands."""
        bands = [
            EQBand(FilterType.PEAKING, 100, 6.0, 2.0, enabled=True),
            EQBand(FilterType.PEAKING, 200, 6.0, 2.0, enabled=False),
            EQBand(FilterType.PEAKING, 300, 6.0, 2.0, enabled=True),
        ]
        settings = EQSettings(bands=bands, channel="left")

        assert settings.num_active_bands == 2


class TestOptimizeEQ:
    """Tests for full EQ optimization."""

    @pytest.fixture
    def sample_response(self):
        """Create a sample response with problems."""
        frequencies = np.logspace(np.log10(20), np.log10(20000), 500)
        response_db = np.zeros(500)

        # Add a peak at 63Hz
        for i, f in enumerate(frequencies):
            if f > 0:
                response_db[i] += 8.0 * np.exp(-((np.log2(f / 63)) ** 2) / (0.1**2))

        # Add a dip at 125Hz
        for i, f in enumerate(frequencies):
            if f > 0:
                response_db[i] -= 5.0 * np.exp(-((np.log2(f / 125)) ** 2) / (0.1**2))

        return frequencies, response_db

    @pytest.fixture
    def sample_problems(self):
        """Create sample problems matching the response."""
        return [
            RoomProblem(ProblemType.PEAK, 63.0, 8.0, 4.0, Severity.SEVERE, "Peak"),
            RoomProblem(ProblemType.DIP, 125.0, -5.0, 3.0, Severity.MODERATE, "Dip"),
        ]

    def test_returns_optimization_result(self, sample_response, sample_problems):
        """Test that optimization returns OptimizationResult."""
        frequencies, response_db = sample_response
        target_db = np.zeros_like(response_db)

        result = optimize_eq(
            frequencies,
            response_db,
            target_db,
            sample_problems,
            max_iterations=50,
        )

        assert isinstance(result, OptimizationResult)
        assert isinstance(result.settings, EQSettings)

    def test_optimization_improves_response(self, sample_response, sample_problems):
        """Test that optimization reduces deviation."""
        frequencies, response_db = sample_response
        target_db = np.zeros_like(response_db)

        result = optimize_eq(
            frequencies,
            response_db,
            target_db,
            sample_problems,
            max_iterations=100,
        )

        # Corrected RMS should be less than original
        assert result.corrected_deviation_rms < result.original_deviation_rms
        assert result.improvement_db > 0

    def test_empty_problems_no_bands(self):
        """Test that empty problems gives no bands."""
        frequencies = np.logspace(np.log10(20), np.log10(20000), 500)
        response_db = np.zeros(500)
        target_db = np.zeros(500)

        result = optimize_eq(frequencies, response_db, target_db, [])

        assert len(result.settings.bands) == 0

    def test_bands_within_rme_limits(self, sample_response, sample_problems):
        """Test that optimized bands are within RME limits."""
        frequencies, response_db = sample_response
        target_db = np.zeros_like(response_db)

        result = optimize_eq(
            frequencies,
            response_db,
            target_db,
            sample_problems,
            max_iterations=50,
        )

        errors = validate_for_rme(result.settings.bands)
        assert errors == []


class TestQuickOptimize:
    """Tests for quick optimization."""

    @pytest.fixture
    def sample_problems(self):
        """Create sample problems."""
        return [
            RoomProblem(ProblemType.PEAK, 63.0, 8.0, 4.0, Severity.SEVERE, "Peak"),
            RoomProblem(ProblemType.DIP, 125.0, -5.0, 3.0, Severity.MODERATE, "Dip"),
        ]

    def test_returns_eq_settings(self, sample_problems):
        """Test that quick_optimize returns EQSettings."""
        frequencies = np.logspace(np.log10(20), np.log10(20000), 500)
        response_db = np.zeros(500)
        target_db = np.zeros(500)

        result = quick_optimize(frequencies, response_db, target_db, sample_problems)

        assert isinstance(result, EQSettings)
        assert len(result.bands) > 0

    def test_creates_bands_from_problems(self, sample_problems):
        """Test that bands are created from problems."""
        frequencies = np.logspace(np.log10(20), np.log10(20000), 500)
        response_db = np.zeros(500)
        target_db = np.zeros(500)

        result = quick_optimize(frequencies, response_db, target_db, sample_problems)

        # Should have bands for each problem
        assert len(result.bands) == len(sample_problems)
