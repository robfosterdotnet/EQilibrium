"""Multi-position measurement averaging and frequency response analysis.

Provides functions to average multiple frequency response measurements
and apply fractional-octave smoothing for cleaner analysis.
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import uniform_filter1d


class AveragingMethod(Enum):
    """Method for averaging frequency responses."""

    COMPLEX = "complex"  # Average complex values (preserves phase)
    MAGNITUDE = "magnitude"  # Average magnitudes
    POWER = "power"  # Average power (magnitude squared)


@dataclass
class AveragingConfig:
    """Configuration for measurement averaging."""

    method: AveragingMethod = AveragingMethod.COMPLEX
    smoothing_octave: float = 1 / 24  # 1/24 octave smoothing


def average_frequency_responses(
    frequencies_list: list[NDArray[np.float64]],
    magnitude_db_list: list[NDArray[np.float64]],
    method: AveragingMethod = AveragingMethod.POWER,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Average multiple frequency response measurements.

    Args:
        frequencies_list: List of frequency arrays (should be identical)
        magnitude_db_list: List of magnitude arrays in dB
        method: Averaging method to use

    Returns:
        Tuple of (frequencies, averaged_magnitude_db)
    """
    if not frequencies_list or not magnitude_db_list:
        raise ValueError("Need at least one measurement to average")

    if len(frequencies_list) != len(magnitude_db_list):
        raise ValueError("Frequency and magnitude lists must have same length")

    # Use first frequency array as reference
    frequencies = frequencies_list[0]

    if method == AveragingMethod.MAGNITUDE:
        # Convert dB to linear, average, convert back
        linear_sum = np.zeros_like(frequencies)
        for mag_db in magnitude_db_list:
            linear_sum += 10 ** (mag_db / 20)
        averaged_linear = linear_sum / len(magnitude_db_list)
        averaged_db = 20 * np.log10(averaged_linear + 1e-10)

    elif method == AveragingMethod.POWER:
        # Convert dB to power, average, convert back
        power_sum = np.zeros_like(frequencies)
        for mag_db in magnitude_db_list:
            power_sum += 10 ** (mag_db / 10)
        averaged_power = power_sum / len(magnitude_db_list)
        averaged_db = 10 * np.log10(averaged_power + 1e-10)

    else:  # COMPLEX - treat as magnitude for now (would need phase data)
        # For simplicity, use power averaging as default "complex" method
        # True complex averaging would require the original complex spectra
        power_sum = np.zeros_like(frequencies)
        for mag_db in magnitude_db_list:
            power_sum += 10 ** (mag_db / 10)
        averaged_power = power_sum / len(magnitude_db_list)
        averaged_db = 10 * np.log10(averaged_power + 1e-10)

    return frequencies, averaged_db


def fractional_octave_smoothing(
    frequencies: NDArray[np.float64],
    magnitude_db: NDArray[np.float64],
    octave_fraction: float = 1 / 24,
) -> NDArray[np.float64]:
    """
    Apply fractional-octave smoothing to frequency response.

    For each frequency, averages values within the specified fraction
    of an octave. This provides perceptually-appropriate smoothing.

    Args:
        frequencies: Frequency array in Hz
        magnitude_db: Magnitude array in dB
        octave_fraction: Fraction of an octave to smooth over (e.g., 1/3, 1/6, 1/24)

    Returns:
        Smoothed magnitude array in dB
    """
    if len(frequencies) != len(magnitude_db):
        raise ValueError("Frequency and magnitude arrays must have same length")

    if octave_fraction <= 0:
        raise ValueError("Octave fraction must be positive")

    # Convert to log frequency for uniform octave spacing
    # Avoid log(0) by using small minimum frequency
    min_freq = max(frequencies[frequencies > 0].min() if np.any(frequencies > 0) else 1, 1)
    log_frequencies = np.log2(np.maximum(frequencies, min_freq))

    # Calculate smoothing width in log-frequency space
    # One octave = 1 in log2 space
    smoothing_width = octave_fraction

    # Convert magnitude to linear for averaging
    linear_mag = 10 ** (magnitude_db / 20)

    # Smooth using a moving average in log-frequency space
    # This requires resampling to uniform log-frequency spacing
    n_points = len(frequencies)

    # Create uniform log-frequency grid
    log_freq_min = log_frequencies[log_frequencies > -np.inf].min()
    log_freq_max = log_frequencies.max()
    uniform_log_freq = np.linspace(log_freq_min, log_freq_max, n_points)

    # Interpolate to uniform grid
    valid_mask = np.isfinite(log_frequencies)
    if np.sum(valid_mask) < 2:
        return magnitude_db.copy()

    uniform_linear_mag = np.interp(
        uniform_log_freq, log_frequencies[valid_mask], linear_mag[valid_mask]
    )

    # Calculate filter width in samples
    log_freq_step = (log_freq_max - log_freq_min) / (n_points - 1) if n_points > 1 else 1
    filter_width_samples = max(1, int(smoothing_width / log_freq_step))

    # Apply uniform filter (moving average)
    smoothed_uniform = uniform_filter1d(uniform_linear_mag, filter_width_samples, mode="nearest")

    # Interpolate back to original frequencies
    smoothed_linear = np.interp(log_frequencies, uniform_log_freq, smoothed_uniform)

    # Convert back to dB
    smoothed_db = 20 * np.log10(smoothed_linear + 1e-10)

    return smoothed_db.astype(np.float64)  # type: ignore[no-any-return]


def calculate_deviation_from_target(
    frequencies: NDArray[np.float64],
    magnitude_db: NDArray[np.float64],
    target: str = "flat",
    target_db: float = 0.0,
) -> NDArray[np.float64]:
    """
    Calculate deviation from target response.

    Args:
        frequencies: Frequency array in Hz
        magnitude_db: Magnitude array in dB
        target: Target type ("flat", "house_curve")
        target_db: Target level in dB (for flat response)

    Returns:
        Deviation array in dB (positive = above target, negative = below)
    """
    if target == "flat":
        # Flat response at target_db
        target_response = np.full_like(magnitude_db, target_db)
    elif target == "house_curve":
        # Common house curve: +3dB at 20Hz, flat from 200Hz, -3dB at 20kHz
        # This is a gentle slope that many people find pleasing
        target_response = np.zeros_like(magnitude_db)

        for i, f in enumerate(frequencies):
            if f < 200:
                # Bass boost: +3dB at 20Hz, rolling off to flat at 200Hz
                if f > 0:
                    target_response[i] = 3 * (1 - np.log10(f / 20) / np.log10(200 / 20))
                else:
                    target_response[i] = 3
            elif f > 2000:
                # Treble rolloff: flat at 2kHz, -3dB at 20kHz
                target_response[i] = -3 * np.log10(f / 2000) / np.log10(20000 / 2000)
    else:
        raise ValueError(f"Unknown target type: {target}")

    return magnitude_db - target_response


def calculate_rms_deviation(deviation_db: NDArray[np.float64]) -> float:
    """
    Calculate RMS deviation from target.

    Args:
        deviation_db: Deviation array in dB

    Returns:
        RMS deviation in dB
    """
    return float(np.sqrt(np.mean(deviation_db**2)))


def calculate_weighted_deviation(
    frequencies: NDArray[np.float64],
    deviation_db: NDArray[np.float64],
    emphasis: str = "audible",
) -> float:
    """
    Calculate weighted deviation emphasizing audible frequencies.

    Args:
        frequencies: Frequency array in Hz
        deviation_db: Deviation array in dB
        emphasis: Weighting type ("audible", "bass", "flat")

    Returns:
        Weighted RMS deviation in dB
    """
    if emphasis == "flat":
        weights = np.ones_like(frequencies)
    elif emphasis == "bass":
        # Emphasize bass frequencies
        weights = np.where(frequencies < 200, 2.0, 1.0)
    elif emphasis == "audible":
        # De-emphasize extreme frequencies
        weights = np.ones_like(frequencies)
        weights = np.where(frequencies < 30, 0.5, weights)
        weights = np.where(frequencies > 16000, 0.5, weights)
    else:
        raise ValueError(f"Unknown emphasis type: {emphasis}")

    weighted_squared = deviation_db**2 * weights
    return float(np.sqrt(np.sum(weighted_squared) / np.sum(weights)))


def interpolate_to_common_frequencies(
    frequencies_list: list[NDArray[np.float64]],
    magnitude_db_list: list[NDArray[np.float64]],
    target_frequencies: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], list[NDArray[np.float64]]]:
    """
    Interpolate multiple measurements to common frequency points.

    Args:
        frequencies_list: List of frequency arrays
        magnitude_db_list: List of magnitude arrays in dB
        target_frequencies: Target frequency array (if None, uses first measurement)

    Returns:
        Tuple of (common_frequencies, interpolated_magnitude_list)
    """
    if target_frequencies is None:
        target_frequencies = frequencies_list[0]

    interpolated_list = []
    for freq, mag in zip(frequencies_list, magnitude_db_list, strict=False):
        interpolated = np.interp(target_frequencies, freq, mag)
        interpolated_list.append(interpolated)

    return target_frequencies, interpolated_list
