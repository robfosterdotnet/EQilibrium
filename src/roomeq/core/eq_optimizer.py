"""EQ optimization algorithm.

Optimizes parametric EQ settings to correct room frequency response,
respecting interface-specific constraints (band count, Q range, etc.).
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
from roomeq.core.interface_profiles import (
    DEFAULT_PROFILE,
    InterfaceProfile,
)

# Room correction best practices constraints (universal, not interface-specific)
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


def round_to_profile_precision(
    band: EQBand,
    profile: InterfaceProfile | None = None,
) -> EQBand:
    """
    Round EQ band parameters to interface precision.

    Args:
        band: EQ band to round
        profile: Interface profile with constraints (defaults to RME)

    Returns:
        New EQBand with rounded values
    """
    if profile is None:
        profile = DEFAULT_PROFILE

    c = profile.constraints

    freq = round(band.frequency / c.freq_step) * c.freq_step
    freq = np.clip(freq, c.min_freq, c.max_freq)

    gain = round(band.gain / c.gain_step) * c.gain_step
    gain = np.clip(gain, c.min_gain, c.max_gain)

    q = round(band.q / c.q_step) * c.q_step
    q = np.clip(q, c.min_q, c.max_q)

    return EQBand(
        filter_type=band.filter_type,
        frequency=float(freq),
        gain=float(gain),
        q=float(q),
        enabled=band.enabled,
    )


def validate_for_profile(
    bands: list[EQBand],
    profile: InterfaceProfile | None = None,
) -> list[str]:
    """
    Validate EQ bands against interface constraints.

    Args:
        bands: List of EQ bands to validate
        profile: Interface profile with constraints (defaults to RME)

    Returns:
        List of validation error messages (empty if valid)
    """
    if profile is None:
        profile = DEFAULT_PROFILE

    c = profile.constraints
    errors = []

    if len(bands) > c.max_bands:
        errors.append(f"Too many bands: {len(bands)} (max {c.max_bands})")

    for i, band in enumerate(bands):
        if not c.min_freq <= band.frequency <= c.max_freq:
            errors.append(f"Band {i + 1}: Frequency {band.frequency} Hz out of range")

        if not c.min_gain <= band.gain <= c.max_gain:
            errors.append(f"Band {i + 1}: Gain {band.gain} dB out of range")

        if not c.min_q <= band.q <= c.max_q:
            errors.append(f"Band {i + 1}: Q {band.q} out of range")

    return errors


def initialize_bands_from_problems(
    problems: list[RoomProblem],
    profile: InterfaceProfile | None = None,
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
        profile: Interface profile with constraints (defaults to RME)

    Returns:
        List of initial EQ bands
    """
    if profile is None:
        profile = DEFAULT_PROFILE

    max_bands = profile.constraints.max_bands

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
        band = round_to_profile_precision(band, profile)

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
    profile: InterfaceProfile | None = None,
) -> list[tuple[float, float]]:
    """
    Get parameter bounds for optimization.

    Uses tighter bounds based on room correction best practices:
    - Frequency limited to effective correction range
    - Gain limited based on whether correcting a peak or dip
    - Q bounds from interface profile

    Args:
        n_bands: Number of bands
        initial_bands: Initial band settings (for asymmetric gain bounds)
        profile: Interface profile with constraints (defaults to RME)
    """
    if profile is None:
        profile = DEFAULT_PROFILE

    c = profile.constraints
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

        # Q bounds from profile
        bounds.append((np.log10(c.min_q), np.log10(c.max_q)))

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
    profile: InterfaceProfile | None = None,
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
        profile: Interface profile with constraints (defaults to RME)
        max_iterations: Maximum optimization iterations

    Returns:
        OptimizationResult with optimized settings
    """
    if profile is None:
        profile = DEFAULT_PROFILE

    # Calculate original deviation
    original_deviation = response_db - target_db
    original_rms = float(np.sqrt(np.mean(original_deviation**2)))

    # Initialize bands from detected problems
    initial_bands = initialize_bands_from_problems(problems, profile)

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
    bounds = get_parameter_bounds(len(initial_bands), initial_bands, profile)

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

    # Round to interface precision
    final_bands = [round_to_profile_precision(b, profile) for b in optimized_bands]

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
    profile: InterfaceProfile | None = None,
) -> EQSettings:
    """
    Quick optimization using only initial problem-based bands.

    No iterative optimization - just uses inverse of detected problems.
    Faster but less optimal than full optimize_eq.
    """
    if profile is None:
        profile = DEFAULT_PROFILE

    bands = initialize_bands_from_problems(problems, profile)

    return EQSettings(bands=bands, channel="")


# =============================================================================
# Backward Compatibility Aliases
# =============================================================================
# These maintain compatibility with existing code that uses RME-specific names


def round_to_rme_precision(band: EQBand) -> EQBand:
    """Backward compatibility alias for round_to_profile_precision with RME profile."""
    return round_to_profile_precision(band, DEFAULT_PROFILE)


def validate_for_rme(bands: list[EQBand]) -> list[str]:
    """Backward compatibility alias for validate_for_profile with RME profile."""
    return validate_for_profile(bands, DEFAULT_PROFILE)


# Re-export RME constants for backward compatibility
RME_MIN_FREQ = DEFAULT_PROFILE.constraints.min_freq
RME_MAX_FREQ = DEFAULT_PROFILE.constraints.max_freq
RME_FREQ_STEP = DEFAULT_PROFILE.constraints.freq_step
RME_MIN_GAIN = DEFAULT_PROFILE.constraints.min_gain
RME_MAX_GAIN = DEFAULT_PROFILE.constraints.max_gain
RME_GAIN_STEP = DEFAULT_PROFILE.constraints.gain_step
RME_MIN_Q = DEFAULT_PROFILE.constraints.min_q
RME_MAX_Q = DEFAULT_PROFILE.constraints.max_q
RME_Q_STEP = DEFAULT_PROFILE.constraints.q_step
RME_MAX_BANDS = DEFAULT_PROFILE.constraints.max_bands
