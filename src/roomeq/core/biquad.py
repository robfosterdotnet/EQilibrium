"""Biquad filter coefficient calculation.

Implements the Audio EQ Cookbook formulas by Robert Bristow-Johnson
for calculating biquad filter coefficients.

Reference:
    https://webaudio.github.io/Audio-EQ-Cookbook/audio-eq-cookbook.html
"""

from dataclasses import dataclass
from enum import Enum
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray


class FilterType(Enum):
    """Type of biquad filter."""

    PEAKING = "peaking"  # Parametric EQ band
    LOW_SHELF = "low_shelf"  # Low shelf filter
    HIGH_SHELF = "high_shelf"  # High shelf filter
    LOW_PASS = "low_pass"  # Low pass filter
    HIGH_PASS = "high_pass"  # High pass filter


class BiquadCoefficients(NamedTuple):
    """Biquad filter coefficients (normalized)."""

    b0: float
    b1: float
    b2: float
    a0: float  # Always 1.0 after normalization
    a1: float
    a2: float


@dataclass
class EQBand:
    """Single parametric EQ band."""

    filter_type: FilterType
    frequency: float  # Hz (20-20000)
    gain: float  # dB (-20 to +20)
    q: float  # Q factor (0.4 to 9.9 for RME)
    enabled: bool = True

    def __post_init__(self):
        """Validate parameters."""
        if self.frequency <= 0:
            raise ValueError("Frequency must be positive")
        if not -30 <= self.gain <= 30:
            raise ValueError("Gain must be between -30 and +30 dB")
        if self.q <= 0:
            raise ValueError("Q must be positive")


def calculate_peaking_eq(
    frequency: float,
    gain_db: float,
    q: float,
    sample_rate: int,
) -> BiquadCoefficients:
    """
    Calculate biquad coefficients for peaking EQ filter.

    This is the standard parametric EQ band that boosts or cuts
    around a center frequency.

    Args:
        frequency: Center frequency in Hz
        gain_db: Gain in dB (positive = boost, negative = cut)
        q: Q factor (bandwidth)
        sample_rate: Sample rate in Hz

    Returns:
        BiquadCoefficients for the filter
    """
    A = 10 ** (gain_db / 40)  # sqrt(10^(dB/20))  # noqa: N806
    w0 = 2 * np.pi * frequency / sample_rate
    sin_w0 = np.sin(w0)
    cos_w0 = np.cos(w0)
    alpha = sin_w0 / (2 * q)

    b0 = 1 + alpha * A
    b1 = -2 * cos_w0
    b2 = 1 - alpha * A
    a0 = 1 + alpha / A
    a1 = -2 * cos_w0
    a2 = 1 - alpha / A

    # Normalize by a0
    return BiquadCoefficients(
        b0=b0 / a0,
        b1=b1 / a0,
        b2=b2 / a0,
        a0=1.0,
        a1=a1 / a0,
        a2=a2 / a0,
    )


def calculate_low_shelf(
    frequency: float,
    gain_db: float,
    q: float,
    sample_rate: int,
) -> BiquadCoefficients:
    """
    Calculate biquad coefficients for low shelf filter.

    Boosts or cuts all frequencies below the shelf frequency.

    Args:
        frequency: Shelf frequency in Hz
        gain_db: Gain in dB
        q: Q factor (controls shelf slope)
        sample_rate: Sample rate in Hz

    Returns:
        BiquadCoefficients for the filter
    """
    A = 10 ** (gain_db / 40)  # noqa: N806
    w0 = 2 * np.pi * frequency / sample_rate
    sin_w0 = np.sin(w0)
    cos_w0 = np.cos(w0)
    alpha = sin_w0 / (2 * q)
    sqrt_A = np.sqrt(A)  # noqa: N806
    two_sqrt_A_alpha = 2 * sqrt_A * alpha  # noqa: N806

    b0 = A * ((A + 1) - (A - 1) * cos_w0 + two_sqrt_A_alpha)
    b1 = 2 * A * ((A - 1) - (A + 1) * cos_w0)
    b2 = A * ((A + 1) - (A - 1) * cos_w0 - two_sqrt_A_alpha)
    a0 = (A + 1) + (A - 1) * cos_w0 + two_sqrt_A_alpha
    a1 = -2 * ((A - 1) + (A + 1) * cos_w0)
    a2 = (A + 1) + (A - 1) * cos_w0 - two_sqrt_A_alpha

    return BiquadCoefficients(
        b0=b0 / a0,
        b1=b1 / a0,
        b2=b2 / a0,
        a0=1.0,
        a1=a1 / a0,
        a2=a2 / a0,
    )


def calculate_high_shelf(
    frequency: float,
    gain_db: float,
    q: float,
    sample_rate: int,
) -> BiquadCoefficients:
    """
    Calculate biquad coefficients for high shelf filter.

    Boosts or cuts all frequencies above the shelf frequency.

    Args:
        frequency: Shelf frequency in Hz
        gain_db: Gain in dB
        q: Q factor (controls shelf slope)
        sample_rate: Sample rate in Hz

    Returns:
        BiquadCoefficients for the filter
    """
    A = 10 ** (gain_db / 40)  # noqa: N806
    w0 = 2 * np.pi * frequency / sample_rate
    sin_w0 = np.sin(w0)
    cos_w0 = np.cos(w0)
    alpha = sin_w0 / (2 * q)
    sqrt_A = np.sqrt(A)  # noqa: N806
    two_sqrt_A_alpha = 2 * sqrt_A * alpha  # noqa: N806

    b0 = A * ((A + 1) + (A - 1) * cos_w0 + two_sqrt_A_alpha)
    b1 = -2 * A * ((A - 1) + (A + 1) * cos_w0)
    b2 = A * ((A + 1) + (A - 1) * cos_w0 - two_sqrt_A_alpha)
    a0 = (A + 1) - (A - 1) * cos_w0 + two_sqrt_A_alpha
    a1 = 2 * ((A - 1) - (A + 1) * cos_w0)
    a2 = (A + 1) - (A - 1) * cos_w0 - two_sqrt_A_alpha

    return BiquadCoefficients(
        b0=b0 / a0,
        b1=b1 / a0,
        b2=b2 / a0,
        a0=1.0,
        a1=a1 / a0,
        a2=a2 / a0,
    )


def calculate_coefficients(
    band: EQBand,
    sample_rate: int,
) -> BiquadCoefficients:
    """
    Calculate biquad coefficients for an EQ band.

    Args:
        band: EQ band parameters
        sample_rate: Sample rate in Hz

    Returns:
        BiquadCoefficients for the filter
    """
    if band.filter_type == FilterType.PEAKING:
        return calculate_peaking_eq(band.frequency, band.gain, band.q, sample_rate)
    elif band.filter_type == FilterType.LOW_SHELF:
        return calculate_low_shelf(band.frequency, band.gain, band.q, sample_rate)
    elif band.filter_type == FilterType.HIGH_SHELF:
        return calculate_high_shelf(band.frequency, band.gain, band.q, sample_rate)
    else:
        raise ValueError(f"Unsupported filter type: {band.filter_type}")


def calculate_frequency_response(
    coefficients: BiquadCoefficients,
    frequencies: NDArray[np.float64],
    sample_rate: int,
) -> NDArray[np.float64]:
    """
    Calculate frequency response of a biquad filter.

    Args:
        coefficients: Biquad filter coefficients
        frequencies: Array of frequencies in Hz
        sample_rate: Sample rate in Hz

    Returns:
        Magnitude response in dB
    """
    # Normalized frequency
    w = 2 * np.pi * frequencies / sample_rate

    # Complex frequency response: H(e^jw) = (b0 + b1*e^-jw + b2*e^-2jw) / (1 + a1*e^-jw + a2*e^-2jw)
    ejw = np.exp(-1j * w)
    ejw2 = np.exp(-2j * w)

    numerator = coefficients.b0 + coefficients.b1 * ejw + coefficients.b2 * ejw2
    denominator = 1.0 + coefficients.a1 * ejw + coefficients.a2 * ejw2

    H = numerator / denominator  # noqa: N806

    # Magnitude in dB
    magnitude_db = 20 * np.log10(np.abs(H) + 1e-10)

    return magnitude_db.astype(np.float64)  # type: ignore[no-any-return]


def calculate_combined_response(
    bands: list[EQBand],
    frequencies: NDArray[np.float64],
    sample_rate: int,
) -> NDArray[np.float64]:
    """
    Calculate combined frequency response of multiple EQ bands.

    Args:
        bands: List of EQ bands
        frequencies: Array of frequencies in Hz
        sample_rate: Sample rate in Hz

    Returns:
        Combined magnitude response in dB
    """
    combined_db = np.zeros_like(frequencies)

    for band in bands:
        if not band.enabled:
            continue

        coefficients = calculate_coefficients(band, sample_rate)
        band_response = calculate_frequency_response(coefficients, frequencies, sample_rate)
        combined_db += band_response

    return combined_db


def create_correction_band(
    frequency: float,
    deviation_db: float,
    q: float,
    filter_type: FilterType = FilterType.PEAKING,
) -> EQBand:
    """
    Create an EQ band to correct a deviation.

    The gain is inverted from the deviation (if response is +6dB, we need -6dB).

    Args:
        frequency: Center frequency in Hz
        deviation_db: Deviation from target in dB
        q: Q factor for the correction
        filter_type: Type of filter to use

    Returns:
        EQBand configured for correction
    """
    # Correction gain is opposite of deviation
    correction_gain = -deviation_db

    # Clamp to reasonable range
    correction_gain = np.clip(correction_gain, -20.0, 20.0)

    return EQBand(
        filter_type=filter_type,
        frequency=float(frequency),
        gain=float(correction_gain),
        q=float(q),
        enabled=True,
    )
