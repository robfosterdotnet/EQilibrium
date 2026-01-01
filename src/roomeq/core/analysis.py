"""Frequency response analysis and problem detection.

Analyzes room frequency response to identify peaks, dips, and
other acoustic problems that can be corrected with EQ.
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


class ProblemType(Enum):
    """Type of room acoustic problem."""

    PEAK = "peak"  # Room mode causing boost
    DIP = "dip"  # Cancellation causing cut
    NULL = "null"  # Severe cancellation (>10dB)


class Severity(Enum):
    """Severity of room problem."""

    MINOR = "minor"  # 3-5 dB deviation
    MODERATE = "moderate"  # 5-8 dB deviation
    SEVERE = "severe"  # >8 dB deviation


@dataclass
class RoomProblem:
    """Represents a detected room acoustic problem."""

    problem_type: ProblemType
    frequency: float  # Hz
    magnitude: float  # dB deviation from target
    q_factor: float  # Estimated Q of the problem
    severity: Severity
    description: str

    @property
    def is_peak(self) -> bool:
        """Check if this is a peak (positive deviation)."""
        return self.problem_type == ProblemType.PEAK

    @property
    def is_dip(self) -> bool:
        """Check if this is a dip (negative deviation)."""
        return self.problem_type in (ProblemType.DIP, ProblemType.NULL)


@dataclass
class AnalysisWarning:
    """Warning about a problem that cannot be fully corrected with EQ."""

    message: str
    frequency: float | None = None
    severity: str = "info"  # "info", "warning", "critical"


@dataclass
class AnalysisResult:
    """Complete analysis of room frequency response."""

    frequencies: NDArray[np.float64]
    response_db: NDArray[np.float64]
    smoothed_response_db: NDArray[np.float64]
    target_db: NDArray[np.float64]
    deviation_db: NDArray[np.float64]
    problems: list[RoomProblem]
    rms_deviation: float
    warnings: list[AnalysisWarning] | None = None


def classify_severity(magnitude_db: float) -> Severity:
    """Classify severity based on dB deviation."""
    abs_mag = abs(magnitude_db)
    if abs_mag >= 8.0:
        return Severity.SEVERE
    elif abs_mag >= 5.0:
        return Severity.MODERATE
    else:
        return Severity.MINOR


def estimate_q_from_width(
    center_freq: float,
    frequencies: NDArray[np.float64],
    response_db: NDArray[np.float64],
    peak_idx: int,
) -> float:
    """
    Estimate Q factor from the width of a peak/dip at -3dB point.

    Args:
        center_freq: Center frequency of the peak/dip
        frequencies: Frequency array
        response_db: Response array in dB
        peak_idx: Index of the peak/dip

    Returns:
        Estimated Q factor
    """
    peak_value = response_db[peak_idx]
    target_level = peak_value - 3.0 if peak_value > 0 else peak_value + 3.0

    # Find -3dB points on each side
    left_idx = peak_idx
    right_idx = peak_idx

    # Search left
    for i in range(peak_idx, 0, -1):
        if (peak_value > 0 and response_db[i] <= target_level) or (
            peak_value < 0 and response_db[i] >= target_level
        ):
            left_idx = i
            break

    # Search right
    for i in range(peak_idx, len(response_db)):
        if (peak_value > 0 and response_db[i] <= target_level) or (
            peak_value < 0 and response_db[i] >= target_level
        ):
            right_idx = i
            break

    # Calculate bandwidth
    f_low = frequencies[left_idx]
    f_high = frequencies[right_idx]
    bandwidth = f_high - f_low

    if bandwidth > 0:
        q = center_freq / bandwidth
        # Clamp to reasonable range
        return float(np.clip(q, 0.5, 15.0))

    return 4.0  # Default Q if calculation fails


def detect_problems(
    frequencies: NDArray[np.float64],
    deviation_db: NDArray[np.float64],
    threshold_db: float = 3.0,
    min_frequency: float = 20.0,
    max_frequency: float = 20000.0,
) -> list[RoomProblem]:
    """
    Detect peaks and dips in the frequency response.

    Args:
        frequencies: Frequency array in Hz
        deviation_db: Deviation from target in dB
        threshold_db: Minimum deviation to consider a problem
        min_frequency: Minimum frequency to analyze
        max_frequency: Maximum frequency to analyze

    Returns:
        List of detected RoomProblem instances
    """
    problems: list[RoomProblem] = []

    # Smooth slightly to reduce noise sensitivity
    smoothed = gaussian_filter1d(deviation_db, sigma=2)

    # Create frequency mask
    freq_mask = (frequencies >= min_frequency) & (frequencies <= max_frequency)
    valid_indices = np.where(freq_mask)[0]

    if len(valid_indices) == 0:
        return problems

    # Find peaks (positive deviations)
    peak_indices, _ = find_peaks(
        smoothed[freq_mask],
        height=threshold_db,
        prominence=2.0,
        distance=10,  # Minimum distance between peaks
    )

    for rel_idx in peak_indices:
        idx = valid_indices[rel_idx]
        freq = frequencies[idx]
        mag = deviation_db[idx]

        q = estimate_q_from_width(freq, frequencies, deviation_db, idx)
        severity = classify_severity(mag)

        problems.append(
            RoomProblem(
                problem_type=ProblemType.PEAK,
                frequency=float(freq),
                magnitude=float(mag),
                q_factor=q,
                severity=severity,
                description=f"Room mode at {freq:.0f}Hz causing {mag:.1f}dB boost",
            )
        )

    # Find dips (negative deviations)
    dip_indices, _ = find_peaks(
        -smoothed[freq_mask],  # Invert to find dips
        height=threshold_db,
        prominence=2.0,
        distance=10,
    )

    for rel_idx in dip_indices:
        idx = valid_indices[rel_idx]
        freq = frequencies[idx]
        mag = deviation_db[idx]  # Will be negative

        q = estimate_q_from_width(freq, frequencies, deviation_db, idx)
        severity = classify_severity(mag)

        # Classify as null if very severe
        if abs(mag) > 10:
            problem_type = ProblemType.NULL
            desc = f"Severe cancellation at {freq:.0f}Hz ({mag:.1f}dB)"
        else:
            problem_type = ProblemType.DIP
            desc = f"Cancellation at {freq:.0f}Hz causing {abs(mag):.1f}dB cut"

        problems.append(
            RoomProblem(
                problem_type=problem_type,
                frequency=float(freq),
                magnitude=float(mag),
                q_factor=q,
                severity=severity,
                description=desc,
            )
        )

    # Sort by frequency
    problems.sort(key=lambda p: p.frequency)

    return problems


def analyze_response(
    frequencies: NDArray[np.float64],
    magnitude_db: NDArray[np.float64],
    target: str = "flat",
    smoothing_octave: float = 1 / 24,
    threshold_db: float = 3.0,
    min_freq: float = 20.0,
    max_freq: float = 20000.0,
) -> AnalysisResult:
    """
    Perform complete analysis of frequency response.

    Args:
        frequencies: Frequency array in Hz
        magnitude_db: Magnitude array in dB
        target: Target response type ("flat" or "house_curve")
        smoothing_octave: Smoothing amount in octaves
        threshold_db: Problem detection threshold
        min_freq: Minimum frequency to analyze (default 20 Hz)
        max_freq: Maximum frequency to analyze (default 20 kHz)

    Returns:
        AnalysisResult with full analysis
    """
    from roomeq.core.averaging import (
        calculate_deviation_from_target,
        calculate_rms_deviation,
        fractional_octave_smoothing,
    )

    # Filter to audible frequency range (avoids log(0) issues and focuses analysis)
    freq_mask = (frequencies >= min_freq) & (frequencies <= max_freq)
    frequencies = frequencies[freq_mask]
    magnitude_db = magnitude_db[freq_mask]

    if len(frequencies) == 0:
        # Return empty result if no data in range
        empty = np.array([])
        return AnalysisResult(
            frequencies=empty,
            response_db=empty,
            smoothed_response_db=empty,
            target_db=empty,
            deviation_db=empty,
            problems=[],
            rms_deviation=0.0,
        )

    # Normalize: set average level in 200-2000 Hz range to 0 dB
    # This is standard practice for room measurements
    ref_mask = (frequencies >= 200) & (frequencies <= 2000)
    if np.any(ref_mask):
        ref_level = np.mean(magnitude_db[ref_mask])
        magnitude_db = magnitude_db - ref_level

    # Apply smoothing
    smoothed_db = fractional_octave_smoothing(frequencies, magnitude_db, smoothing_octave)

    # Calculate target response
    if target == "flat":
        target_db = np.zeros_like(frequencies)
    elif target == "house_curve":
        # Common house curve: +3dB at 20Hz, flat from 200Hz, -3dB at 20kHz
        target_db = np.zeros_like(frequencies)
        for i, f in enumerate(frequencies):
            if f < 200:
                # Bass boost: +3dB at 20Hz, rolling off to flat at 200Hz
                if f > 0:
                    target_db[i] = 3 * (1 - np.log10(f / 20) / np.log10(200 / 20))
                else:
                    target_db[i] = 3
            elif f > 2000:
                # Treble rolloff: flat at 2kHz, -3dB at 20kHz
                target_db[i] = -3 * np.log10(f / 2000) / np.log10(20000 / 2000)
    else:
        target_db = np.zeros_like(frequencies)

    # Calculate deviation
    deviation_db = calculate_deviation_from_target(frequencies, smoothed_db, target)

    # Calculate RMS deviation
    rms_dev = calculate_rms_deviation(deviation_db)

    # Detect problems
    problems = detect_problems(frequencies, deviation_db, threshold_db)

    # Generate warnings about uncorrectable issues
    warnings = generate_analysis_warnings(problems)

    return AnalysisResult(
        frequencies=frequencies,
        response_db=magnitude_db,
        smoothed_response_db=smoothed_db,
        target_db=target_db,
        deviation_db=deviation_db,
        problems=problems,
        rms_deviation=rms_dev,
        warnings=warnings,
    )


def generate_analysis_warnings(problems: list[RoomProblem]) -> list[AnalysisWarning]:
    """
    Generate warnings about problems that cannot be fully corrected with EQ.

    Args:
        problems: List of detected room problems

    Returns:
        List of AnalysisWarning objects
    """
    warnings: list[AnalysisWarning] = []

    # Check for nulls (deep dips that can't be corrected)
    nulls = [p for p in problems if p.problem_type == ProblemType.NULL]
    if nulls:
        for null in nulls:
            warnings.append(AnalysisWarning(
                message=(
                    f"Deep null at {null.frequency:.0f} Hz ({null.magnitude:.1f} dB) "
                    "cannot be corrected with EQ. This is likely caused by phase "
                    "cancellation. Consider acoustic treatment or speaker/listener repositioning."
                ),
                frequency=null.frequency,
                severity="critical",
            ))

    # Check for moderate dips
    dips = [p for p in problems if p.problem_type == ProblemType.DIP and abs(p.magnitude) > 6]
    if dips:
        for dip in dips:
            warnings.append(AnalysisWarning(
                message=(
                    f"Significant dip at {dip.frequency:.0f} Hz ({dip.magnitude:.1f} dB). "
                    "EQ can only partially compensate. Boosting may cause distortion."
                ),
                frequency=dip.frequency,
                severity="warning",
            ))

    # Check for very low frequency problems
    very_low = [p for p in problems if p.frequency < 30 and p.is_peak]
    if very_low:
        warnings.append(AnalysisWarning(
            message=(
                "Room modes detected below 30 Hz. These are difficult to correct with EQ "
                "and may require bass traps or subwoofer repositioning."
            ),
            frequency=None,
            severity="info",
        ))

    # General advice if we have a lot of problems
    if len(problems) > 9:
        warnings.append(AnalysisWarning(
            message=(
                f"Detected {len(problems)} acoustic issues. Only 9 EQ bands available. "
                "Consider acoustic treatment for comprehensive correction."
            ),
            frequency=None,
            severity="info",
        ))

    return warnings


def prioritize_problems(
    problems: list[RoomProblem],
    max_count: int = 9,
    prefer_peaks: bool = True,
) -> list[RoomProblem]:
    """
    Prioritize problems for EQ correction.

    IMPORTANT: Based on room acoustics best practices:
    - Peaks (room modes causing boost) are highly correctable with EQ cuts
    - Dips are much harder to correct - often caused by phase cancellation
    - Nulls (severe dips) CANNOT be fixed with EQ - don't waste bands on them

    References:
    - PS Audio: https://www.psaudio.com/blogs/copper/using-eq-with-speakers-some-limitations
    - HouseCurve: https://housecurve.com/docs/tuning/equalization

    Args:
        problems: List of detected problems
        max_count: Maximum number of problems to return
        prefer_peaks: Whether to prefer correcting peaks over dips

    Returns:
        Prioritized list of problems (most important first)
    """
    if not problems:
        return []

    def priority_score(p: RoomProblem) -> float:
        """Calculate priority score (higher = more important)."""
        score = abs(p.magnitude)

        # Severe problems get priority
        if p.severity == Severity.SEVERE:
            score *= 1.5
        elif p.severity == Severity.MODERATE:
            score *= 1.2

        # CRITICAL: Peaks are much more effectively corrected than dips
        # Peaks (positive deviation) can be cut with EQ - works great
        # Dips (negative deviation) boosting is often ineffective
        if p.is_peak:
            score *= 2.0  # Strongly prefer peaks
        else:
            # Dips get very low priority - EQ boosts often don't help
            score *= 0.3

        # Bass region (30-200 Hz) is where room modes are most problematic
        # and where EQ correction is most effective
        if 30 <= p.frequency <= 200:
            score *= 1.3
        elif p.frequency < 30:
            # Very low bass is hard to correct
            score *= 0.5

        # Nulls (severe dips >10dB) CANNOT be fixed with EQ
        # They're caused by destructive interference - boosting just wastes power
        if p.problem_type == ProblemType.NULL:
            score *= 0.1  # Almost exclude them

        return score

    # Sort by priority score (descending)
    sorted_problems = sorted(problems, key=priority_score, reverse=True)

    return sorted_problems[:max_count]
