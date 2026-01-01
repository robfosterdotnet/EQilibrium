"""Pytest configuration and fixtures."""

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def set_random_seed():
    """Set random seed for reproducible tests."""
    np.random.seed(42)
    yield


@pytest.fixture
def sample_rate():
    """Standard sample rate for tests."""
    return 48000


@pytest.fixture
def sweep_duration():
    """Standard sweep duration for tests."""
    return 1.0  # 1 second for faster tests


@pytest.fixture
def frequency_range():
    """Standard frequency range for tests."""
    return (20.0, 20000.0)


@pytest.fixture
def sample_impulse_response(sample_rate):
    """Generate a simple test impulse response with known characteristics."""
    # Create a simple IR: direct sound + one reflection
    duration = 0.5  # 500ms
    n_samples = int(duration * sample_rate)
    ir = np.zeros(n_samples)

    # Direct sound at sample 0
    ir[0] = 1.0

    # Reflection at 10ms with 0.5 amplitude
    reflection_sample = int(0.01 * sample_rate)
    ir[reflection_sample] = 0.5

    # Add some decay (simple exponential)
    decay = np.exp(-np.arange(n_samples) / (0.1 * sample_rate))
    noise = np.random.randn(n_samples) * 0.01
    ir += noise * decay

    return ir


@pytest.fixture
def sample_frequency_response():
    """Generate a sample frequency response with known peaks/dips."""
    frequencies = np.logspace(np.log10(20), np.log10(20000), 1000)

    # Start with flat response
    response_db = np.zeros_like(frequencies)

    # Add a peak at 63 Hz (+6 dB, Q=4)
    peak1 = 6.0 * np.exp(-((np.log10(frequencies) - np.log10(63)) ** 2) / (2 * 0.05**2))
    response_db += peak1

    # Add a dip at 125 Hz (-4 dB, Q=3)
    dip1 = -4.0 * np.exp(-((np.log10(frequencies) - np.log10(125)) ** 2) / (2 * 0.06**2))
    response_db += dip1

    # Add a peak at 250 Hz (+3 dB, Q=5)
    peak2 = 3.0 * np.exp(-((np.log10(frequencies) - np.log10(250)) ** 2) / (2 * 0.04**2))
    response_db += peak2

    return frequencies, response_db
