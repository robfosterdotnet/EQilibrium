"""RME TotalMix export format.

Generates .tmreq XML files that can be directly imported into
RME TotalMix FX Room EQ via the Preset menu.
"""

from pathlib import Path
from xml.etree import ElementTree as ET

from roomeq.core.biquad import EQBand, FilterType
from roomeq.core.eq_optimizer import (
    RME_MAX_BANDS,
    EQSettings,
    round_to_rme_precision,
    validate_for_rme,
)

# Default frequencies for unused bands (matches TotalMix defaults)
DEFAULT_FREQUENCIES = [50, 100, 150, 200, 250, 300, 400, 600, 800]

# Filter type values in TotalMix format
# 0 = Peaking, 1 = Low Shelf, 2 = High Shelf (for bands 1, 8, 9)
FILTER_TYPE_VALUES = {
    FilterType.PEAKING: 0,
    FilterType.LOW_SHELF: 1,
    FilterType.HIGH_SHELF: 2,
}


def _add_val(parent: ET.Element, name: str, value: float) -> None:
    """Add a val element with the TotalMix format."""
    val = ET.SubElement(parent, "val")
    val.set("e", name)
    val.set("v", f"{value:.2f},")


def _generate_channel_params(
    params: ET.Element,
    settings: EQSettings | None,
    delay: float = 0.0,
    channel_gain: float = 0.0,
) -> None:
    """Generate parameter elements for a single channel."""
    # Delay parameter
    _add_val(params, "REQ Delay", delay)

    # Get bands list, pad to 9 with None
    bands: list[EQBand | None] = []
    if settings and settings.bands:
        bands = list(settings.bands)[:RME_MAX_BANDS]
    while len(bands) < RME_MAX_BANDS:
        bands.append(None)

    # Add band parameters
    for i, band in enumerate(bands, 1):
        if band is not None and band.enabled:
            rounded = round_to_rme_precision(band)
            freq = rounded.frequency
            q = rounded.q
            gain = rounded.gain
        else:
            # Use default values for unused bands
            freq = float(DEFAULT_FREQUENCIES[i - 1])
            q = 5.0
            gain = 0.0

        _add_val(params, f"REQ Band{i} Freq", freq)
        _add_val(params, f"REQ Band{i} Q", q)
        _add_val(params, f"REQ Band{i} Gain", gain)

    # Band types (only bands 1, 8, 9 can be shelf filters)
    for band_num in [1, 8, 9]:
        band = bands[band_num - 1]
        if band is not None and band.enabled:
            type_val = FILTER_TYPE_VALUES.get(band.filter_type, 0)
        else:
            type_val = 0
        # Band 1 has no space before "Type", bands 8 and 9 do
        type_name = f"REQ Band{band_num}Type" if band_num == 1 else f"REQ Band{band_num} Type"
        _add_val(params, type_name, float(type_val))

    # Channel gain
    _add_val(params, "Chan Gain", channel_gain)


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
    # Create root element
    preset = ET.Element("Preset")

    # Left channel
    room_eq_l = ET.SubElement(preset, "Room EQ L")
    params_l = ET.SubElement(room_eq_l, "Params")
    _generate_channel_params(params_l, left_settings, delay, channel_gain)

    # Right channel
    room_eq_r = ET.SubElement(preset, "Room EQ R")
    params_r = ET.SubElement(room_eq_r, "Params")
    _generate_channel_params(params_r, right_settings, delay, channel_gain)

    # Convert to string with proper formatting
    ET.indent(preset, space="\t")
    xml_str = ET.tostring(preset, encoding="unicode")

    # Remove space before /> to match TotalMix format exactly
    xml_str = xml_str.replace(" />", "/>")

    return xml_str + "\n"


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
    filepath = Path(filepath)

    # Validate before export
    if left_settings:
        errors = validate_for_rme(left_settings.bands)
        if errors:
            raise ValueError(f"Invalid left channel EQ settings: {'; '.join(errors)}")

    if right_settings:
        errors = validate_for_rme(right_settings.bands)
        if errors:
            raise ValueError(f"Invalid right channel EQ settings: {'; '.join(errors)}")

    content = generate_tmreq_format(left_settings, right_settings, delay, channel_gain)
    filepath.write_text(content)


# Legacy function for backward compatibility
def generate_rew_format(settings: EQSettings, config: object = None) -> str:
    """
    Legacy function - generates tmreq format instead.

    Kept for backward compatibility with existing code.
    """
    return generate_tmreq_format(left_settings=settings)


def get_totalmix_import_instructions() -> str:
    """
    Get user instructions for importing into TotalMix.

    Returns:
        Instruction text
    """
    return """
To import the EQ settings into RME TotalMix FX:

1. Open TotalMix FX
2. Select the output channel you want to correct
3. Click the "Room EQ" button to open the Room EQ panel
4. Click "Preset" in the Room EQ panel
5. Select "Load Preset..." from the menu
6. Navigate to and select the exported .tmreq file
7. The EQ bands will be loaded for both L and R channels

Note: The preset contains settings for both left and right channels.
You may need to enable Room EQ after import if it's not already enabled.
""".strip()
