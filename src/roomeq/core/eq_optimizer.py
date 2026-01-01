"""EQ optimization algorithm.

Optimizes parametric EQ settings to correct room frequency response,
respecting RME TotalMix constraints (9 bands max, Q 0.4-9.9, etc.).
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from roomeq.core.analysis import RoomProblem, prioritize_problems
from roomeq.core.biquad import (
    EQBand,
    FilterType,
    calculate_combined_response,
)

# RME TotalMix Room EQ constraints
RME_MIN_FREQ = 20.0
RME_MAX_FREQ = 20000.0
RME_FREQ_STEP = 1.0
RME_MIN_GAIN = -20.0
RME_MAX_GAIN = 20.0
RME_GAIN_STEP = 0.1
RME_MIN_Q = 0.4
RME_MAX_Q = 9.9
RME_Q_STEP = 0.1
RME_MAX_BANDS = 9

# Room correction best practices constraints
# Boosting dips is generally ineffective and can cause distortion
# See: https://housecurve.com/docs/tuning/equalization
# See: https://www.psaudio.com/blogs/copper/using-eq-with-speakers-some-limitations
CORRECTION_MAX_BOOST = 3.0  # Maximum boost in dB (prefer cuts over boosts)
CORRECTION_MAX_CUT = -15.0  # Maximum cut in dB
CORRECTION_MIN_FREQ = 30.0  # Below this, room modes dominate
CORRECTION_MAX_FREQ = 10000.0  # Above this, corrections often unnecessary
DIP_BOOST_THRESHOLD = -6.0  # Don't try to correct dips deeper than this


@dataclass
class EQSettings:
    """Complete EQ configuration for one channel."""

    bands: list[EQBand]
    channel: str  # "left" or "right"

    @property
    def num_active_bands(self) -> int:
        """Number of enabled bands."""
        return sum(1 for b in self.bands if b.enabled)


@dataclass
class OptimizationResult:
    """Result of EQ optimization."""

    settings: EQSettings
    original_deviation_rms: float
    corrected_deviation_rms: float
    improvement_db: float


def round_to_rme_precision(band: EQBand) -> EQBand:
    """
    Round EQ band parameters to RME precision.

    Args:
        band: EQ band to round

    Returns:
        New EQBand with rounded values
    """
    freq = round(band.frequency / RME_FREQ_STEP) * RME_FREQ_STEP
    freq = np.clip(freq, RME_MIN_FREQ, RME_MAX_FREQ)

    gain = round(band.gain / RME_GAIN_STEP) * RME_GAIN_STEP
    gain = np.clip(gain, RME_MIN_GAIN, RME_MAX_GAIN)

    q = round(band.q / RME_Q_STEP) * RME_Q_STEP
    q = np.clip(q, RME_MIN_Q, RME_MAX_Q)

    return EQBand(
        filter_type=band.filter_type,
        frequency=float(freq),
        gain=float(gain),
        q=float(q),
        enabled=band.enabled,
    )


def validate_for_rme(bands: list[EQBand]) -> list[str]:
    """
    Validate EQ bands against RME constraints.

    Args:
        bands: List of EQ bands to validate

    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []

    if len(bands) > RME_MAX_BANDS:
        errors.append(f"Too many bands: {len(bands)} (max {RME_MAX_BANDS})")

    for i, band in enumerate(bands):
        if not RME_MIN_FREQ <= band.frequency <= RME_MAX_FREQ:
            errors.append(f"Band {i + 1}: Frequency {band.frequency} Hz out of range")

        if not RME_MIN_GAIN <= band.gain <= RME_MAX_GAIN:
            errors.append(f"Band {i + 1}: Gain {band.gain} dB out of range")

        if not RME_MIN_Q <= band.q <= RME_MAX_Q:
            errors.append(f"Band {i + 1}: Q {band.q} out of range")

    return errors


def initialize_bands_from_problems(
    problems: list[RoomProblem],
    max_bands: int = RME_MAX_BANDS,
) -> list[EQBand]:
    """
    Initialize EQ bands from detected room problems.

    IMPORTANT: Room correction best practices (per REW, HouseCurve, and acoustic
    engineering literature):
    - Peaks (positive deviations) can be effectively corrected with cuts
    - Dips/nulls (negative deviations) are often caused by phase cancellation
      and CANNOT be effectively corrected with EQ boosts
    - Boosting dips wastes amplifier power and can cause distortion
    - We limit boosts to small corrections for minor dips only

    Args:
        problems: List of detected room problems
        max_bands: Maximum number of bands to create

    Returns:
        List of initial EQ bands
    """
    # Prioritize problems
    prioritized = prioritize_problems(problems, max_count=max_bands)

    bands = []
    for problem in prioritized:
        # Skip problems outside effective correction range
        if problem.frequency < CORRECTION_MIN_FREQ or problem.frequency > CORRECTION_MAX_FREQ:
            continue

        # For dips (negative magnitude), check if it's too deep to correct
        if problem.magnitude < 0:
            # Deep dips (nulls) cannot be fixed with EQ - skip them
            if problem.magnitude < DIP_BOOST_THRESHOLD:
                continue
            # For shallow dips, limit the boost amount
            correction_gain = min(-problem.magnitude, CORRECTION_MAX_BOOST)
        else:
            # For peaks, apply full correction (cut)
            correction_gain = max(-problem.magnitude, CORRECTION_MAX_CUT)

        band = EQBand(
            filter_type=FilterType.PEAKING,
            frequency=problem.frequency,
            gain=float(correction_gain),
            q=problem.q_factor,
            enabled=True,
        )
        band = round_to_rme_precision(band)

        # Skip bands with negligible correction
        if abs(band.gain) < 0.5:
            continue

        bands.append(band)

    return bands


def bands_to_params(bands: list[EQBand]) -> NDArray[np.float64]:
    """
    Convert EQ bands to optimization parameter array.

    Each band has 3 parameters: frequency (log), gain, Q (log)
    Using log scale for frequency and Q improves optimization.
    """
    params = []
    for band in bands:
        params.extend(
            [
                np.log10(band.frequency),  # Log frequency
                band.gain,
                np.log10(band.q),  # Log Q
            ]
        )
    return np.array(params)


def params_to_bands(
    params: NDArray[np.float64],
    filter_types: list[FilterType] | None = None,
) -> list[EQBand]:
    """
    Convert optimization parameter array back to EQ bands.
    """
    n_bands = len(params) // 3

    if filter_types is None:
        filter_types = [FilterType.PEAKING] * n_bands

    bands = []
    for i in range(n_bands):
        idx = i * 3
        freq = 10 ** params[idx]
        gain = params[idx + 1]
        q = 10 ** params[idx + 2]

        bands.append(
            EQBand(
                filter_type=filter_types[i],
                frequency=float(freq),
                gain=float(gain),
                q=float(q),
                enabled=True,
            )
        )

    return bands


def get_parameter_bounds(
    n_bands: int,
    initial_bands: list[EQBand] | None = None,
) -> list[tuple[float, float]]:
    """
    Get parameter bounds for optimization.

    Uses tighter bounds based on room correction best practices:
    - Frequency limited to effective correction range
    - Gain limited based on whether correcting a peak or dip
    """
    bounds = []
    for i in range(n_bands):
        # Frequency bounds - focus on effective correction range
        bounds.append((np.log10(CORRECTION_MIN_FREQ), np.log10(CORRECTION_MAX_FREQ)))

        # Gain bounds - asymmetric: allow full cuts but limit boosts
        # If we know this band started as a cut (correcting a peak), allow full cut range
        # If it started as a boost (correcting a dip), limit boost amount
        if initial_bands and i < len(initial_bands):
            if initial_bands[i].gain >= 0:
                # This is a boost (correcting a dip) - limit it
                bounds.append((0, CORRECTION_MAX_BOOST))
            else:
                # This is a cut (correcting a peak) - allow full range
                bounds.append((CORRECTION_MAX_CUT, CORRECTION_MAX_BOOST))
        else:
            # Default: allow cuts, limit boosts
            bounds.append((CORRECTION_MAX_CUT, CORRECTION_MAX_BOOST))

        # Q bounds
        bounds.append((np.log10(RME_MIN_Q), np.log10(RME_MAX_Q)))

    return bounds


def calculate_objective(
    params: NDArray[np.float64],
    frequencies: NDArray[np.float64],
    current_response_db: NDArray[np.float64],
    target_db: NDArray[np.float64],
    sample_rate: int,
    filter_types: list[FilterType],
) -> float:
    """
    Calculate optimization objective function.

    Minimizes weighted RMS error from target.
    """
    bands = params_to_bands(params, filter_types)

    # Calculate EQ response
    eq_response = calculate_combined_response(bands, frequencies, sample_rate)

    # Corrected response
    corrected = current_response_db + eq_response

    # Error from target
    error = corrected - target_db

    # Perceptual weighting
    weights = np.ones_like(frequencies)
    # De-emphasize extreme frequencies
    weights = np.where(frequencies < 30, 0.5, weights)
    weights = np.where(frequencies > 16000, 0.5, weights)
    # Emphasize midrange
    weights = np.where((frequencies >= 200) & (frequencies <= 4000), 1.2, weights)

    weighted_error = error * weights

    # Base cost: RMS error
    cost = np.sqrt(np.mean(weighted_error**2))

    # Penalty for extreme Q values with high gain (can cause ringing)
    for band in bands:
        if band.q > 6.0 and abs(band.gain) > 6.0:
            cost += 0.3 * (band.q - 6.0) * abs(band.gain) / 6.0

    return float(cost)


def optimize_eq(
    frequencies: NDArray[np.float64],
    response_db: NDArray[np.float64],
    target_db: NDArray[np.float64],
    problems: list[RoomProblem],
    sample_rate: int = 48000,
    max_bands: int = RME_MAX_BANDS,
    max_iterations: int = 500,
) -> OptimizationResult:
    """
    Optimize EQ settings to minimize deviation from target.

    Args:
        frequencies: Frequency array in Hz
        response_db: Current frequency response in dB
        target_db: Target frequency response in dB
        problems: Detected room problems (for initialization)
        sample_rate: Sample rate in Hz
        max_bands: Maximum number of EQ bands
        max_iterations: Maximum optimization iterations

    Returns:
        OptimizationResult with optimized settings
    """
    # Calculate original deviation
    original_deviation = response_db - target_db
    original_rms = float(np.sqrt(np.mean(original_deviation**2)))

    # Initialize bands from detected problems
    initial_bands = initialize_bands_from_problems(problems, max_bands)

    if not initial_bands:
        # No problems detected, return empty settings
        return OptimizationResult(
            settings=EQSettings(bands=[], channel=""),
            original_deviation_rms=original_rms,
            corrected_deviation_rms=original_rms,
            improvement_db=0.0,
        )

    # Get filter types for reconstruction
    filter_types = [b.filter_type for b in initial_bands]

    # Convert to parameter array
    initial_params = bands_to_params(initial_bands)

    # Get bounds (with knowledge of initial bands for asymmetric gain limits)
    bounds = get_parameter_bounds(len(initial_bands), initial_bands)

    # Optimize
    result = minimize(
        calculate_objective,
        initial_params,
        args=(frequencies, response_db, target_db, sample_rate, filter_types),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": max_iterations, "ftol": 1e-6},
    )

    # Convert back to bands
    optimized_bands = params_to_bands(result.x, filter_types)

    # Round to RME precision
    final_bands = [round_to_rme_precision(b) for b in optimized_bands]

    # Remove bands with very small gain (< 0.5 dB)
    final_bands = [b for b in final_bands if abs(b.gain) >= 0.5]

    # Calculate corrected deviation
    eq_response = calculate_combined_response(final_bands, frequencies, sample_rate)
    corrected = response_db + eq_response
    corrected_deviation = corrected - target_db
    corrected_rms = float(np.sqrt(np.mean(corrected_deviation**2)))

    improvement = original_rms - corrected_rms

    return OptimizationResult(
        settings=EQSettings(bands=final_bands, channel=""),
        original_deviation_rms=original_rms,
        corrected_deviation_rms=corrected_rms,
        improvement_db=improvement,
    )


def quick_optimize(
    frequencies: NDArray[np.float64],
    response_db: NDArray[np.float64],
    target_db: NDArray[np.float64],
    problems: list[RoomProblem],
    sample_rate: int = 48000,
) -> EQSettings:
    """
    Quick optimization using only initial problem-based bands.

    No iterative optimization - just uses inverse of detected problems.
    Faster but less optimal than full optimize_eq.
    """
    bands = initialize_bands_from_problems(problems, RME_MAX_BANDS)

    return EQSettings(bands=bands, channel="")
