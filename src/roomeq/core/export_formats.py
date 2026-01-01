"""Export formats for EQ settings.

Provides multiple export formats for different audio interfaces and applications:
- RME TotalMix (.tmreq) - XML format for RME interfaces
- REW Text (.txt) - Universal format compatible with many applications
- Manual Display - Human-readable format for manual entry
"""

from abc import ABC, abstractmethod
from pathlib import Path
from xml.etree import ElementTree as ET

from roomeq.core.biquad import EQBand, FilterType
from roomeq.core.eq_optimizer import EQSettings
from roomeq.core.interface_profiles import (
    DEFAULT_PROFILE,
    ExportFormatType,
    InterfaceProfile,
)


class ExportFormat(ABC):
    """Abstract base class for export formats."""

    @abstractmethod
    def get_format_type(self) -> ExportFormatType:
        """Return the format type identifier."""
        ...

    @abstractmethod
    def get_file_extension(self) -> str:
        """Return the file extension (including dot)."""
        ...

    @abstractmethod
    def get_display_name(self) -> str:
        """Return human-readable format name."""
        ...

    @abstractmethod
    def export(
        self,
        left_settings: EQSettings | None = None,
        right_settings: EQSettings | None = None,
    ) -> str:
        """Generate export content string."""
        ...

    @abstractmethod
    def get_import_instructions(self) -> str:
        """Return instructions for importing the exported file."""
        ...

    def export_to_file(
        self,
        filepath: Path | str,
        left_settings: EQSettings | None = None,
        right_settings: EQSettings | None = None,
    ) -> None:
        """Export settings to a file."""
        filepath = Path(filepath)
        content = self.export(left_settings, right_settings)
        filepath.write_text(content)


# =============================================================================
# RME TotalMix Export Format
# =============================================================================


class RMETotalMixExport(ExportFormat):
    """Export format for RME TotalMix Room EQ (.tmreq XML files)."""

    # Default frequencies for unused bands (matches TotalMix defaults)
    DEFAULT_FREQUENCIES = [50, 100, 150, 200, 250, 300, 400, 600, 800]

    # Filter type values in TotalMix format
    FILTER_TYPE_VALUES = {
        FilterType.PEAKING: 0,
        FilterType.LOW_SHELF: 1,
        FilterType.HIGH_SHELF: 2,
    }

    def __init__(self, delay: float = 0.0, channel_gain: float = 0.0):
        """Initialize with optional delay and gain settings."""
        self.delay = delay
        self.channel_gain = channel_gain

    def get_format_type(self) -> ExportFormatType:
        return ExportFormatType.RME_TMREQ

    def get_file_extension(self) -> str:
        return ".tmreq"

    def get_display_name(self) -> str:
        return "RME TotalMix Preset"

    def _add_val(self, parent: ET.Element, name: str, value: float) -> None:
        """Add a val element with the TotalMix format."""
        val = ET.SubElement(parent, "val")
        val.set("e", name)
        val.set("v", f"{value:.2f},")

    def _round_band(self, band: EQBand) -> EQBand:
        """Round band to RME precision."""
        from roomeq.core.eq_optimizer import round_to_rme_precision
        return round_to_rme_precision(band)

    def _generate_channel_params(
        self,
        params: ET.Element,
        settings: EQSettings | None,
    ) -> None:
        """Generate parameter elements for a single channel."""
        max_bands = 9

        # Delay parameter
        self._add_val(params, "REQ Delay", self.delay)

        # Get bands list, pad to 9 with None
        bands: list[EQBand | None] = []
        if settings and settings.bands:
            bands = list(settings.bands)[:max_bands]
        while len(bands) < max_bands:
            bands.append(None)

        # Add band parameters
        for i, band in enumerate(bands, 1):
            if band is not None and band.enabled:
                rounded = self._round_band(band)
                freq = rounded.frequency
                q = rounded.q
                gain = rounded.gain
            else:
                # Use default values for unused bands
                freq = float(self.DEFAULT_FREQUENCIES[i - 1])
                q = 5.0
                gain = 0.0

            self._add_val(params, f"REQ Band{i} Freq", freq)
            self._add_val(params, f"REQ Band{i} Q", q)
            self._add_val(params, f"REQ Band{i} Gain", gain)

        # Band types (only bands 1, 8, 9 can be shelf filters)
        for band_num in [1, 8, 9]:
            band = bands[band_num - 1]
            if band is not None and band.enabled:
                type_val = self.FILTER_TYPE_VALUES.get(band.filter_type, 0)
            else:
                type_val = 0
            # Band 1 has no space before "Type", bands 8 and 9 do
            type_name = f"REQ Band{band_num}Type" if band_num == 1 else f"REQ Band{band_num} Type"
            self._add_val(params, type_name, float(type_val))

        # Channel gain
        self._add_val(params, "Chan Gain", self.channel_gain)

    def export(
        self,
        left_settings: EQSettings | None = None,
        right_settings: EQSettings | None = None,
    ) -> str:
        """Generate TotalMix Room EQ preset XML content."""
        # Validate before export
        from roomeq.core.eq_optimizer import validate_for_rme

        if left_settings:
            errors = validate_for_rme(left_settings.bands)
            if errors:
                raise ValueError(f"Invalid left channel settings: {'; '.join(errors)}")

        if right_settings:
            errors = validate_for_rme(right_settings.bands)
            if errors:
                raise ValueError(f"Invalid right channel settings: {'; '.join(errors)}")

        # Create root element
        preset = ET.Element("Preset")

        # Left channel
        room_eq_l = ET.SubElement(preset, "Room EQ L")
        params_l = ET.SubElement(room_eq_l, "Params")
        self._generate_channel_params(params_l, left_settings)

        # Right channel
        room_eq_r = ET.SubElement(preset, "Room EQ R")
        params_r = ET.SubElement(room_eq_r, "Params")
        self._generate_channel_params(params_r, right_settings)

        # Convert to string with proper formatting
        ET.indent(preset, space="\t")
        xml_str = ET.tostring(preset, encoding="unicode")

        # Remove space before /> to match TotalMix format exactly
        xml_str = xml_str.replace(" />", "/>")

        return xml_str + "\n"

    def get_import_instructions(self) -> str:
        return """To import into RME TotalMix FX:

1. Open TotalMix FX
2. Click the Room EQ button (or press F8)
3. Click Options (gear icon)
4. Select "Load Preset..."
5. Navigate to the exported .tmreq file
6. Click Open

The EQ settings will be loaded. Enable Room EQ to hear the correction."""


# =============================================================================
# REW Text Export Format
# =============================================================================


class REWTextExport(ExportFormat):
    """Export format compatible with Room EQ Wizard and many other applications."""

    def get_format_type(self) -> ExportFormatType:
        return ExportFormatType.REW_TEXT

    def get_file_extension(self) -> str:
        return ".txt"

    def get_display_name(self) -> str:
        return "REW Text Format"

    def _filter_type_str(self, filter_type: FilterType) -> str:
        """Convert filter type to REW string."""
        if filter_type == FilterType.LOW_SHELF:
            return "LS"
        elif filter_type == FilterType.HIGH_SHELF:
            return "HS"
        else:
            return "PK"

    def _format_bands(self, settings: EQSettings | None, channel: str = "") -> str:
        """Format bands for one channel."""
        if not settings or not settings.bands:
            return ""

        lines = []
        if channel:
            lines.append(f"# {channel} Channel")

        for i, band in enumerate(settings.bands, 1):
            if band.enabled:
                type_str = self._filter_type_str(band.filter_type)
                lines.append(
                    f"Filter {i:2d}: ON  {type_str}       "
                    f"Fc {band.frequency:7.1f} Hz  "
                    f"Gain {band.gain:6.1f} dB  "
                    f"Q {band.q:5.2f}"
                )

        return "\n".join(lines)

    def export(
        self,
        left_settings: EQSettings | None = None,
        right_settings: EQSettings | None = None,
    ) -> str:
        """Generate REW-compatible text format."""
        lines = [
            "Filter Settings file",
            "",
            "Room EQ Wizard V5.20",
            "",
            "Equaliser: Generic",
            "",
        ]

        # If we have both channels, output them separately
        if left_settings and right_settings:
            left_content = self._format_bands(left_settings, "Left")
            right_content = self._format_bands(right_settings, "Right")
            if left_content:
                lines.append(left_content)
                lines.append("")
            if right_content:
                lines.append(right_content)
        elif left_settings:
            content = self._format_bands(left_settings)
            if content:
                lines.append(content)
        elif right_settings:
            content = self._format_bands(right_settings)
            if content:
                lines.append(content)

        return "\n".join(lines) + "\n"

    def get_import_instructions(self) -> str:
        return """The REW text format is compatible with many applications:

- Room EQ Wizard (REW): File > Import > Filter Settings
- Equalizer APO (Windows): Copy filters to config.txt
- Many audio plugins and DSP software

Consult your software's documentation for specific import instructions."""


# =============================================================================
# Manual Display Export Format
# =============================================================================


class ManualDisplayExport(ExportFormat):
    """Human-readable format for manual entry into any EQ interface."""

    def get_format_type(self) -> ExportFormatType:
        return ExportFormatType.MANUAL

    def get_file_extension(self) -> str:
        return ".txt"

    def get_display_name(self) -> str:
        return "Manual Entry Format"

    def _format_channel(self, settings: EQSettings | None, channel: str) -> str:
        """Format a single channel as a table."""
        if not settings or not settings.bands:
            return f"{channel}: No corrections needed\n"

        lines = [
            f"{channel} Channel EQ Settings",
            "=" * 50,
            "",
            "Band | Type      | Frequency | Gain    | Q",
            "-----|-----------|-----------|---------|------",
        ]

        for i, band in enumerate(settings.bands, 1):
            if band.enabled:
                type_name = band.filter_type.value.replace("_", " ").title()
                lines.append(
                    f" {i:2d}  | {type_name:9s} | {band.frequency:7.0f} Hz | "
                    f"{band.gain:+5.1f} dB | {band.q:.1f}"
                )

        return "\n".join(lines)

    def export(
        self,
        left_settings: EQSettings | None = None,
        right_settings: EQSettings | None = None,
    ) -> str:
        """Generate human-readable table format."""
        sections = [
            "EQ CORRECTION SETTINGS",
            "Generated by EQilibrium",
            "",
            "Enter these values into your audio interface's parametric EQ.",
            "",
        ]

        if left_settings:
            sections.append(self._format_channel(left_settings, "Left"))
            sections.append("")

        if right_settings:
            sections.append(self._format_channel(right_settings, "Right"))
            sections.append("")

        if not left_settings and not right_settings:
            sections.append("No EQ corrections generated.")
            sections.append("")

        sections.extend([
            "",
            "TIPS:",
            "- Start with the largest corrections first",
            "- Use parametric/peaking EQ type for most bands",
            "- Listen and adjust if needed",
        ])

        return "\n".join(sections) + "\n"

    def get_import_instructions(self) -> str:
        return """To apply these corrections manually:

1. Open your audio interface's EQ settings
2. Add parametric EQ bands for each filter listed
3. Set the frequency, gain, and Q values as shown
4. Enable the EQ and listen to the result

Tip: Start with the largest corrections (highest gain magnitude) first."""


# =============================================================================
# Export Format Factory
# =============================================================================


def get_export_format(
    format_type: ExportFormatType,
    profile: InterfaceProfile | None = None,
) -> ExportFormat:
    """
    Get an export format instance.

    Args:
        format_type: The type of export format
        profile: Interface profile (used for format-specific settings)

    Returns:
        ExportFormat instance
    """
    if format_type == ExportFormatType.RME_TMREQ:
        return RMETotalMixExport()
    elif format_type == ExportFormatType.REW_TEXT:
        return REWTextExport()
    elif format_type == ExportFormatType.MANUAL:
        return ManualDisplayExport()
    else:
        raise ValueError(f"Unknown export format: {format_type}")


def get_available_formats(
    profile: InterfaceProfile | None = None,
) -> list[ExportFormat]:
    """
    Get all available export formats for a profile.

    Args:
        profile: Interface profile (defaults to RME)

    Returns:
        List of available ExportFormat instances
    """
    if profile is None:
        profile = DEFAULT_PROFILE

    return [get_export_format(fmt, profile) for fmt in profile.export_formats]


def export_to_file(
    filepath: Path | str,
    left_settings: EQSettings | None = None,
    right_settings: EQSettings | None = None,
    format_type: ExportFormatType = ExportFormatType.RME_TMREQ,
    profile: InterfaceProfile | None = None,
) -> None:
    """
    Export EQ settings to a file.

    Args:
        filepath: Output file path
        left_settings: Left channel EQ settings
        right_settings: Right channel EQ settings
        format_type: Export format to use
        profile: Interface profile for format-specific settings
    """
    export_format = get_export_format(format_type, profile)
    export_format.export_to_file(filepath, left_settings, right_settings)
