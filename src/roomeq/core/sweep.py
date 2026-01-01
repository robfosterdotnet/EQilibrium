"""Logarithmic sine sweep generation and deconvolution.

Implements the Farina exponential sine sweep method for measuring
impulse responses. This technique is robust against noise and
non-linearities.

References:
    - Farina, A. "Simultaneous Measurement of Impulse Response and
      Distortion with a Swept-Sine Technique" (AES 2000)
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class SweepParameters:
    """Parameters for sweep generation."""

    duration: float = 5.0  # seconds
    sample_rate: int = 48000  # Hz
    start_freq: float = 20.0  # Hz
    end_freq: float = 20000.0  # Hz
    amplitude: float = 0.8  # 0.0 to 1.0
    fade_in: float = 0.01  # seconds
    fade_out: float = 0.01  # seconds

    def __post_init__(self):
        """Validate parameters."""
        if self.duration <= 0:
            raise ValueError("Duration must be positive")
        if self.sample_rate <= 0:
            raise ValueError("Sample rate must be positive")
        if self.start_freq <= 0 or self.end_freq <= 0:
            raise ValueError("Frequencies must be positive")
        if self.start_freq >= self.end_freq:
            raise ValueError("Start frequency must be less than end frequency")
        if not 0 < self.amplitude <= 1.0:
            raise ValueError("Amplitude must be between 0 and 1")
        if self.fade_in < 0 or self.fade_out < 0:
            raise ValueError("Fade times must be non-negative")


class SweepGenerator:
    """Generates exponential sine sweep signals using Farina method."""

    def generate(self, params: SweepParameters) -> NDArray[np.float64]:
        """
        Generate exponential sine sweep.

        The exponential (logarithmic) sweep has the property that each
        octave takes the same amount of time, resulting in pink spectrum.

        Args:
            params: Sweep generation parameters

        Returns:
            NumPy array containing the sweep signal
        """
        n_samples = int(params.duration * params.sample_rate)
        t = np.arange(n_samples) / params.sample_rate

        # Sweep rate (log ratio)
        R = np.log(params.end_freq / params.start_freq)  # noqa: N806

        # Exponential sine sweep (Farina formula)
        # x(t) = sin(2*pi*f1*T/R * (exp(t*R/T) - 1))
        phase = (
            2
            * np.pi
            * params.start_freq
            * params.duration
            / R
            * (np.exp(t * R / params.duration) - 1)
        )
        sweep = params.amplitude * np.sin(phase)

        # Apply fade in/out to prevent clicks
        sweep = self._apply_fades(sweep, params)

        return sweep

    def generate_inverse_filter(
        self, sweep: NDArray[np.float64], params: SweepParameters
    ) -> NDArray[np.float64]:
        """
        Generate inverse filter for deconvolution.

        The inverse filter is the time-reversed sweep with amplitude
        modulation (+6dB/octave) to compensate for the pink spectrum
        of the log sweep.

        Args:
            sweep: The sweep signal
            params: Parameters used to generate the sweep

        Returns:
            NumPy array containing the inverse filter
        """
        # Time-reverse the sweep
        inverse = sweep[::-1].copy()

        # Apply amplitude modulation envelope (+6dB/octave)
        # This compensates for the pink spectrum
        n_samples = len(inverse)
        t = np.arange(n_samples) / params.sample_rate
        R = np.log(params.end_freq / params.start_freq)  # noqa: N806

        # Exponential amplitude growth
        modulation = np.exp(t * R / params.duration)

        inverse = inverse * modulation

        # Normalize
        inverse = inverse / np.max(np.abs(inverse))

        return inverse.astype(np.float64)  # type: ignore[no-any-return]

    def _apply_fades(
        self, signal: NDArray[np.float64], params: SweepParameters
    ) -> NDArray[np.float64]:
        """Apply raised cosine fade in/out to signal."""
        signal = signal.copy()

        fade_in_samples = int(params.fade_in * params.sample_rate)
        fade_out_samples = int(params.fade_out * params.sample_rate)

        if fade_in_samples > 0:
            # Raised cosine fade in
            fade_in = 0.5 * (1 - np.cos(np.pi * np.arange(fade_in_samples) / fade_in_samples))
            signal[:fade_in_samples] *= fade_in

        if fade_out_samples > 0:
            # Raised cosine fade out
            fade_out = 0.5 * (1 + np.cos(np.pi * np.arange(fade_out_samples) / fade_out_samples))
            signal[-fade_out_samples:] *= fade_out

        return signal


class Deconvolver:
    """Extracts impulse response from recorded sweep via deconvolution."""

    def deconvolve(
        self,
        recording: NDArray[np.float64],
        inverse_filter: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Deconvolve recording with inverse filter to extract impulse response.

        Uses FFT-based convolution for efficiency.

        Args:
            recording: Recorded sweep response
            inverse_filter: Inverse filter from SweepGenerator

        Returns:
            Full impulse response (includes harmonic distortion artifacts)
        """
        # Calculate output length
        n = len(recording) + len(inverse_filter) - 1

        # Use power of 2 for FFT efficiency
        n_fft = self._next_power_of_2(n)

        # FFT of both signals
        rec_fft = np.fft.rfft(recording, n_fft)
        inv_fft = np.fft.rfft(inverse_filter, n_fft)

        # Multiply in frequency domain (convolution)
        ir_fft = rec_fft * inv_fft

        # Inverse FFT
        ir = np.fft.irfft(ir_fft, n_fft)

        # Trim to relevant portion
        return ir[:n]

    def extract_linear_ir(
        self,
        full_ir: NDArray[np.float64],
        params: SweepParameters,
        ir_length: float = 0.5,
    ) -> NDArray[np.float64]:
        """
        Extract linear impulse response from full deconvolution result.

        The log sweep deconvolution produces harmonic distortion products
        that appear at negative time (before the linear IR). This function
        extracts just the linear impulse response.

        Args:
            full_ir: Full impulse response from deconvolve()
            params: Sweep parameters
            ir_length: Desired IR length in seconds

        Returns:
            Linear impulse response (without harmonic artifacts)
        """
        # The linear IR peak is approximately at the sweep length position
        # Find the main peak
        peak_idx = int(np.argmax(np.abs(full_ir)))

        # Calculate desired number of samples
        ir_samples = int(ir_length * params.sample_rate)

        # Extract IR starting slightly before peak
        pre_samples = int(0.001 * params.sample_rate)  # 1ms before peak
        start_idx = max(0, peak_idx - pre_samples)
        end_idx = min(len(full_ir), start_idx + ir_samples)

        linear_ir = full_ir[start_idx:end_idx]

        # Normalize so peak is at 1.0
        max_val = np.max(np.abs(linear_ir))
        if max_val > 0:
            linear_ir = linear_ir / max_val

        return linear_ir

    @staticmethod
    def _next_power_of_2(n: int) -> int:
        """Return the next power of 2 >= n."""
        return 1 << (n - 1).bit_length()


def calculate_frequency_response(
    impulse_response: NDArray[np.float64],
    sample_rate: int,
    n_points: int = 4096,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Calculate frequency response from impulse response.

    Args:
        impulse_response: Time-domain impulse response
        sample_rate: Sample rate in Hz
        n_points: Number of FFT points (controls frequency resolution)

    Returns:
        Tuple of (frequencies, magnitude_db)
    """
    # Use at least the IR length for FFT
    n_fft = max(n_points, len(impulse_response))

    # Apply window to reduce spectral leakage
    window = np.hanning(len(impulse_response))
    windowed_ir = impulse_response * window

    # FFT
    spectrum = np.fft.rfft(windowed_ir, n_fft)
    frequencies = np.fft.rfftfreq(n_fft, 1 / sample_rate)

    # Magnitude in dB (add small value to avoid log(0))
    magnitude_db = 20 * np.log10(np.abs(spectrum) + 1e-10)

    # Normalize to 0dB at peak
    magnitude_db = magnitude_db - np.max(magnitude_db)

    return frequencies.astype(np.float64), magnitude_db.astype(np.float64)
