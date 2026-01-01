"""Tests for frequency response analysis and problem detection."""

import numpy as np
import pytest

from roomeq.core.analysis import (
    AnalysisResult,
    ProblemType,
    RoomProblem,
    Severity,
    analyze_response,
    classify_severity,
    detect_problems,
    estimate_q_from_width,
    prioritize_problems,
)


class TestClassifySeverity:
    """Tests for severity classification."""

    def test_minor_severity(self):
        """Test minor severity classification."""
        assert classify_severity(3.0) == Severity.MINOR
        assert classify_severity(-4.0) == Severity.MINOR
        assert classify_severity(4.9) == Severity.MINOR

    def test_moderate_severity(self):
        """Test moderate severity classification."""
        assert classify_severity(5.0) == Severity.MODERATE
        assert classify_severity(-6.0) == Severity.MODERATE
        assert classify_severity(7.9) == Severity.MODERATE

    def test_severe_severity(self):
        """Test severe severity classification."""
        assert classify_severity(8.0) == Severity.SEVERE
        assert classify_severity(-10.0) == Severity.SEVERE
        assert classify_severity(15.0) == Severity.SEVERE


class TestEstimateQ:
    """Tests for Q factor estimation."""

    @pytest.fixture
    def sample_peak(self):
        """Create a sample response with a known peak."""
        frequencies = np.logspace(np.log10(20), np.log10(20000), 1000)
        response = np.zeros(1000)

        # Add a peak at 100Hz with known Q
        target_q = 4.0
        center_freq = 100.0
        bandwidth = center_freq / target_q

        for i, f in enumerate(frequencies):
            # Simplified peak shape
            if f > 0:
                distance = abs(np.log2(f / center_freq))
                response[i] = 6.0 * np.exp(-(distance**2) / (0.1**2))

        return frequencies, response, center_freq

    def test_q_estimation_reasonable_range(self, sample_peak):
        """Test that Q estimation gives reasonable value."""
        frequencies, response, center_freq = sample_peak

        # Find peak index
        peak_idx = np.argmax(response)

        q = estimate_q_from_width(center_freq, frequencies, response, peak_idx)

        # Should be in reasonable range
        assert 0.5 <= q <= 15.0

    def test_q_estimation_narrow_peak_higher_q(self):
        """Test that narrower peaks give higher Q."""
        frequencies = np.logspace(np.log10(20), np.log10(20000), 1000)

        # Narrow peak
        narrow_response = np.zeros(1000)
        for i, f in enumerate(frequencies):
            if f > 0:
                narrow_response[i] = 6.0 * np.exp(-((np.log2(f / 100)) ** 2) / (0.05**2))

        # Wide peak
        wide_response = np.zeros(1000)
        for i, f in enumerate(frequencies):
            if f > 0:
                wide_response[i] = 6.0 * np.exp(-((np.log2(f / 100)) ** 2) / (0.2**2))

        narrow_peak_idx = np.argmax(narrow_response)
        wide_peak_idx = np.argmax(wide_response)

        q_narrow = estimate_q_from_width(100.0, frequencies, narrow_response, narrow_peak_idx)
        q_wide = estimate_q_from_width(100.0, frequencies, wide_response, wide_peak_idx)

        assert q_narrow > q_wide


class TestDetectProblems:
    """Tests for problem detection."""

    @pytest.fixture
    def flat_response(self):
        """Create a flat frequency response."""
        frequencies = np.logspace(np.log10(20), np.log10(20000), 1000)
        deviation = np.zeros(1000)
        return frequencies, deviation

    @pytest.fixture
    def response_with_peak(self):
        """Create a response with a peak at 63Hz."""
        frequencies = np.logspace(np.log10(20), np.log10(20000), 1000)
        deviation = np.zeros(1000)

        # Add peak at 63Hz
        for i, f in enumerate(frequencies):
            if f > 0:
                deviation[i] = 8.0 * np.exp(-((np.log2(f / 63)) ** 2) / (0.1**2))

        return frequencies, deviation

    @pytest.fixture
    def response_with_dip(self):
        """Create a response with a dip at 125Hz."""
        frequencies = np.logspace(np.log10(20), np.log10(20000), 1000)
        deviation = np.zeros(1000)

        # Add dip at 125Hz
        for i, f in enumerate(frequencies):
            if f > 0:
                deviation[i] = -6.0 * np.exp(-((np.log2(f / 125)) ** 2) / (0.1**2))

        return frequencies, deviation

    def test_flat_response_no_problems(self, flat_response):
        """Test that flat response has no problems."""
        frequencies, deviation = flat_response

        problems = detect_problems(frequencies, deviation, threshold_db=3.0)

        assert len(problems) == 0

    def test_detect_peak(self, response_with_peak):
        """Test detection of peak."""
        frequencies, deviation = response_with_peak

        problems = detect_problems(frequencies, deviation, threshold_db=3.0)

        # Should detect at least one peak
        peaks = [p for p in problems if p.problem_type == ProblemType.PEAK]
        assert len(peaks) >= 1

        # Peak should be near 63Hz
        peak = peaks[0]
        assert 50 < peak.frequency < 80
        assert peak.magnitude > 0

    def test_detect_dip(self, response_with_dip):
        """Test detection of dip."""
        frequencies, deviation = response_with_dip

        problems = detect_problems(frequencies, deviation, threshold_db=3.0)

        # Should detect at least one dip
        dips = [p for p in problems if p.problem_type in (ProblemType.DIP, ProblemType.NULL)]
        assert len(dips) >= 1

        # Dip should be near 125Hz
        dip = dips[0]
        assert 100 < dip.frequency < 150
        assert dip.magnitude < 0

    def test_threshold_filters_small_problems(self, response_with_peak):
        """Test that threshold filters small problems."""
        frequencies, _ = response_with_peak

        # Create small deviation
        small_deviation = np.zeros(1000)
        for i, f in enumerate(frequencies):
            if f > 0:
                small_deviation[i] = 2.0 * np.exp(-((np.log2(f / 63)) ** 2) / (0.1**2))

        problems = detect_problems(frequencies, small_deviation, threshold_db=3.0)

        # Should not detect the small peak
        assert len(problems) == 0

    def test_frequency_range_filtering(self, response_with_peak):
        """Test frequency range filtering."""
        frequencies, deviation = response_with_peak

        # Detect with range that excludes the peak
        problems = detect_problems(
            frequencies, deviation, threshold_db=3.0, min_frequency=100, max_frequency=20000
        )

        # Peak at 63Hz should be excluded
        peaks_near_63 = [p for p in problems if 50 < p.frequency < 80]
        assert len(peaks_near_63) == 0


class TestRoomProblem:
    """Tests for RoomProblem dataclass."""

    def test_is_peak(self):
        """Test is_peak property."""
        peak = RoomProblem(
            problem_type=ProblemType.PEAK,
            frequency=100.0,
            magnitude=6.0,
            q_factor=4.0,
            severity=Severity.MODERATE,
            description="Test peak",
        )
        assert peak.is_peak is True
        assert peak.is_dip is False

    def test_is_dip(self):
        """Test is_dip property."""
        dip = RoomProblem(
            problem_type=ProblemType.DIP,
            frequency=100.0,
            magnitude=-6.0,
            q_factor=4.0,
            severity=Severity.MODERATE,
            description="Test dip",
        )
        assert dip.is_peak is False
        assert dip.is_dip is True

    def test_null_is_dip(self):
        """Test that NULL type is classified as dip."""
        null = RoomProblem(
            problem_type=ProblemType.NULL,
            frequency=100.0,
            magnitude=-15.0,
            q_factor=4.0,
            severity=Severity.SEVERE,
            description="Test null",
        )
        assert null.is_dip is True


class TestPrioritizeProblems:
    """Tests for problem prioritization."""

    @pytest.fixture
    def sample_problems(self):
        """Create sample problems for testing."""
        return [
            RoomProblem(ProblemType.PEAK, 63.0, 8.0, 4.0, Severity.SEVERE, "Severe peak"),
            RoomProblem(ProblemType.DIP, 125.0, -4.0, 3.0, Severity.MINOR, "Minor dip"),
            RoomProblem(ProblemType.PEAK, 250.0, 5.0, 5.0, Severity.MODERATE, "Moderate peak"),
            RoomProblem(ProblemType.NULL, 100.0, -12.0, 2.0, Severity.SEVERE, "Severe null"),
        ]

    def test_prioritize_returns_max_count(self, sample_problems):
        """Test that prioritize returns at most max_count problems."""
        prioritized = prioritize_problems(sample_problems, max_count=2)
        assert len(prioritized) == 2

    def test_prioritize_returns_all_if_fewer(self, sample_problems):
        """Test that prioritize returns all if fewer than max_count."""
        prioritized = prioritize_problems(sample_problems, max_count=10)
        assert len(prioritized) == 4

    def test_prioritize_severe_first(self, sample_problems):
        """Test that severe problems come first."""
        prioritized = prioritize_problems(sample_problems, max_count=4)

        # First problem should be severe
        assert prioritized[0].severity == Severity.SEVERE

    def test_prioritize_prefers_peaks(self, sample_problems):
        """Test that peaks are preferred when option is set."""
        prioritized = prioritize_problems(sample_problems, max_count=4, prefer_peaks=True)

        # Check that peaks are generally earlier
        peak_indices = [i for i, p in enumerate(prioritized) if p.is_peak]
        dip_indices = [i for i, p in enumerate(prioritized) if p.is_dip]

        if peak_indices and dip_indices:
            # Average peak index should be lower (earlier)
            assert np.mean(peak_indices) <= np.mean(dip_indices)

    def test_prioritize_empty_list(self):
        """Test prioritizing empty list."""
        prioritized = prioritize_problems([], max_count=9)
        assert prioritized == []


class TestAnalyzeResponse:
    """Tests for complete response analysis."""

    @pytest.fixture
    def sample_response(self, sample_frequency_response):
        """Get sample response from conftest fixture."""
        return sample_frequency_response

    def test_analyze_returns_result(self, sample_response):
        """Test that analyze_response returns AnalysisResult."""
        frequencies, magnitude_db = sample_response

        result = analyze_response(frequencies, magnitude_db)

        assert isinstance(result, AnalysisResult)
        # Note: result arrays may be smaller than input due to frequency filtering
        # The function filters to the audible range (20-20000 Hz by default)
        assert len(result.frequencies) > 0
        assert len(result.frequencies) <= len(frequencies)
        assert len(result.response_db) == len(result.frequencies)
        assert len(result.smoothed_response_db) == len(result.frequencies)
        assert len(result.deviation_db) == len(result.frequencies)

    def test_analyze_detects_problems(self, sample_response):
        """Test that analysis detects problems in sample response."""
        frequencies, magnitude_db = sample_response

        result = analyze_response(frequencies, magnitude_db, threshold_db=2.0)

        # Sample response has known peaks/dips
        assert len(result.problems) > 0

    def test_analyze_calculates_rms(self, sample_response):
        """Test that analysis calculates RMS deviation."""
        frequencies, magnitude_db = sample_response

        result = analyze_response(frequencies, magnitude_db)

        assert result.rms_deviation >= 0

    def test_analyze_flat_response(self):
        """Test analysis of flat response."""
        frequencies = np.logspace(np.log10(20), np.log10(20000), 500)
        magnitude_db = np.zeros(500)

        result = analyze_response(frequencies, magnitude_db)

        assert len(result.problems) == 0
        assert result.rms_deviation < 1.0  # Should be very small
