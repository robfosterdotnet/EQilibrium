"""Interface profiles for different audio hardware.

Defines EQ constraints and export options for various audio interfaces.
Each profile specifies the hardware limitations (band count, Q range, etc.)
and available export formats.
"""

from dataclasses import dataclass
from enum import Enum


class ExportFormatType(Enum):
    """Available export format types."""

    RME_TMREQ = "tmreq"  # RME TotalMix .tmreq XML format
    REW_TEXT = "rew_text"  # REW-compatible text format
    MANUAL = "manual"  # Human-readable display for manual entry


@dataclass(frozen=True)
class EQConstraints:
    """EQ parameter constraints for a specific interface.

    Attributes:
        max_bands: Maximum number of EQ bands available
        min_freq: Minimum frequency in Hz
        max_freq: Maximum frequency in Hz
        freq_step: Frequency adjustment step size in Hz
        min_gain: Minimum gain in dB
        max_gain: Maximum gain in dB
        gain_step: Gain adjustment step size in dB
        min_q: Minimum Q factor
        max_q: Maximum Q factor
        q_step: Q factor adjustment step size
    """

    max_bands: int
    min_freq: float
    max_freq: float
    freq_step: float
    min_gain: float
    max_gain: float
    gain_step: float
    min_q: float
    max_q: float
    q_step: float


@dataclass(frozen=True)
class InterfaceProfile:
    """Profile defining capabilities of an audio interface.

    Attributes:
        id: Unique identifier (e.g., "rme", "generic")
        name: Human-readable name (e.g., "RME TotalMix")
        constraints: EQ parameter constraints
        export_formats: List of supported export format types
        import_instructions: Instructions for importing EQ settings
    """

    id: str
    name: str
    constraints: EQConstraints
    export_formats: list[ExportFormatType]
    import_instructions: str


# =============================================================================
# Built-in Interface Profiles
# =============================================================================

# RME TotalMix Room EQ constraints
RME_CONSTRAINTS = EQConstraints(
    max_bands=9,
    min_freq=20.0,
    max_freq=20000.0,
    freq_step=1.0,
    min_gain=-20.0,
    max_gain=20.0,
    gain_step=0.1,
    min_q=0.4,
    max_q=9.9,
    q_step=0.1,
)

RME_PROFILE = InterfaceProfile(
    id="rme",
    name="RME TotalMix",
    constraints=RME_CONSTRAINTS,
    export_formats=[ExportFormatType.RME_TMREQ, ExportFormatType.REW_TEXT],
    import_instructions="""To import into RME TotalMix FX:

1. Open TotalMix FX
2. Click the Room EQ button (or press F8)
3. Click Options (gear icon)
4. Select "Load Preset..."
5. Navigate to the exported .tmreq file
6. Click Open

The EQ settings will be loaded. Enable Room EQ to hear the correction.""",
)

# Generic profile for interfaces without specific support
GENERIC_CONSTRAINTS = EQConstraints(
    max_bands=31,  # Common 31-band EQ
    min_freq=20.0,
    max_freq=20000.0,
    freq_step=0.1,
    min_gain=-20.0,
    max_gain=20.0,
    gain_step=0.1,
    min_q=0.1,
    max_q=30.0,
    q_step=0.1,
)

GENERIC_PROFILE = InterfaceProfile(
    id="generic",
    name="Generic (REW Format)",
    constraints=GENERIC_CONSTRAINTS,
    export_formats=[ExportFormatType.REW_TEXT, ExportFormatType.MANUAL],
    import_instructions="""The exported REW text format is compatible with many applications:

- Room EQ Wizard (REW)
- Equalizer APO (Windows)
- Many audio plugins and DSP software

Consult your software's documentation for import instructions.""",
)

# Manual profile for any interface (just displays values)
MANUAL_PROFILE = InterfaceProfile(
    id="manual",
    name="Manual Entry",
    constraints=GENERIC_CONSTRAINTS,
    export_formats=[ExportFormatType.MANUAL],
    import_instructions="""Use the displayed EQ values to manually configure your audio interface:

1. Open your interface's EQ/DSP settings
2. Add parametric EQ bands for each filter shown
3. Set the frequency, gain, and Q values as displayed

Tip: Start with the largest corrections first.""",
)

# Registry of all available profiles
PROFILES: dict[str, InterfaceProfile] = {
    "rme": RME_PROFILE,
    "generic": GENERIC_PROFILE,
    "manual": MANUAL_PROFILE,
}

# Default profile
DEFAULT_PROFILE = RME_PROFILE


def get_profile(profile_id: str) -> InterfaceProfile:
    """Get an interface profile by ID.

    Args:
        profile_id: Profile identifier (e.g., "rme", "generic")

    Returns:
        The requested InterfaceProfile

    Raises:
        KeyError: If profile_id is not found
    """
    return PROFILES[profile_id]


def get_all_profiles() -> list[InterfaceProfile]:
    """Get all available interface profiles.

    Returns:
        List of all InterfaceProfile instances
    """
    return list(PROFILES.values())


def get_profile_choices() -> list[tuple[str, str]]:
    """Get profile choices for UI dropdowns.

    Returns:
        List of (id, name) tuples for each profile
    """
    return [(p.id, p.name) for p in PROFILES.values()]
