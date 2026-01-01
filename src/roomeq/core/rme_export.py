"""RME TotalMix export format.

This module provides backward compatibility with existing code.
New code should use export_formats.py directly.
"""

from pathlib import Path

from roomeq.core.eq_optimizer import EQSettings
from roomeq.core.export_formats import (
    RMETotalMixExport,
    get_export_format,
)
from roomeq.core.interface_profiles import ExportFormatType

# Re-export for backward compatibility
__all__ = [
    "generate_tmreq_format",
    "export_to_file",
    "get_totalmix_import_instructions",
    "generate_rew_format",
]


def generate_tmreq_format(
    left_settings: EQSettings | None = None,
    right_settings: EQSettings | None = None,
    delay: float = 0.0,
    channel_gain: float = 0.0,
) -> str:
    """
    Generate TotalMix Room EQ preset XML content.

    Args:
        left_settings: Left channel EQ settings
        right_settings: Right channel EQ settings
        delay: Room EQ delay in ms (0-30)
        channel_gain: Overall channel gain in dB

    Returns:
        XML content string
    """
    exporter = RMETotalMixExport(delay=delay, channel_gain=channel_gain)
    return exporter.export(left_settings, right_settings)


def export_to_file(
    filepath: Path | str,
    left_settings: EQSettings | None = None,
    right_settings: EQSettings | None = None,
    delay: float = 0.0,
    channel_gain: float = 0.0,
) -> None:
    """
    Export EQ settings to a .tmreq file for TotalMix import.

    Args:
        filepath: Output file path (should use .tmreq extension)
        left_settings: Left channel EQ settings
        right_settings: Right channel EQ settings
        delay: Room EQ delay in ms
        channel_gain: Overall channel gain in dB
    """
    exporter = RMETotalMixExport(delay=delay, channel_gain=channel_gain)
    exporter.export_to_file(filepath, left_settings, right_settings)


def get_totalmix_import_instructions() -> str:
    """
    Get user instructions for importing into TotalMix.

    Returns:
        Instruction text
    """
    exporter = get_export_format(ExportFormatType.RME_TMREQ)
    return exporter.get_import_instructions()


# Legacy function for backward compatibility
def generate_rew_format(settings: EQSettings, config: object = None) -> str:
    """
    Legacy function - generates tmreq format instead.

    Kept for backward compatibility with existing code.
    """
    return generate_tmreq_format(left_settings=settings)
