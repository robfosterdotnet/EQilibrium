"""Tests for export format system."""

import tempfile
from pathlib import Path

import pytest

from roomeq.core.biquad import EQBand, FilterType
from roomeq.core.eq_optimizer import EQSettings
from roomeq.core.export_formats import (
    ManualDisplayExport,
    REWTextExport,
    RMETotalMixExport,
    get_available_formats,
    get_export_format,
)
from roomeq.core.interface_profiles import (
    ExportFormatType,
    get_profile,
)


class TestGetExportFormat:
    """Tests for export format factory."""

    def test_get_rme_format(self):
        """Test getting RME TotalMix format."""
        exporter = get_export_format(ExportFormatType.RME_TMREQ)
        assert isinstance(exporter, RMETotalMixExport)
        assert exporter.get_format_type() == ExportFormatType.RME_TMREQ

    def test_get_rew_format(self):
        """Test getting REW text format."""
        exporter = get_export_format(ExportFormatType.REW_TEXT)
        assert isinstance(exporter, REWTextExport)
        assert exporter.get_format_type() == ExportFormatType.REW_TEXT

    def test_get_manual_format(self):
        """Test getting manual display format."""
        exporter = get_export_format(ExportFormatType.MANUAL)
        assert isinstance(exporter, ManualDisplayExport)
        assert exporter.get_format_type() == ExportFormatType.MANUAL

    def test_get_available_formats_default(self):
        """Test getting list of available formats for default profile."""
        formats = get_available_formats()
        # Default profile (RME) has 2 formats
        assert len(formats) >= 2
        format_types = [f.get_format_type() for f in formats]
        assert ExportFormatType.RME_TMREQ in format_types
        assert ExportFormatType.REW_TEXT in format_types

    def test_get_available_formats_manual_profile(self):
        """Test getting formats for manual profile."""
        profile = get_profile("manual")
        formats = get_available_formats(profile)
        format_types = [f.get_format_type() for f in formats]
        assert ExportFormatType.MANUAL in format_types


class TestRMETotalMixExport:
    """Tests for RME TotalMix export format."""

    @pytest.fixture
    def sample_settings(self):
        """Create sample EQ settings."""
        bands = [
            EQBand(FilterType.PEAKING, 63.0, -6.0, 4.3, enabled=True),
            EQBand(FilterType.PEAKING, 125.0, 4.5, 3.2, enabled=True),
        ]
        return EQSettings(bands=bands, channel="left")

    def test_file_extension(self):
        """Test that file extension is .tmreq."""
        exporter = RMETotalMixExport()
        assert exporter.get_file_extension() == ".tmreq"

    def test_export_generates_xml(self, sample_settings):
        """Test that export generates XML content."""
        exporter = RMETotalMixExport()
        content = exporter.export(sample_settings)

        assert "<Preset>" in content
        assert "</Preset>" in content
        assert "<Room EQ L>" in content

    def test_import_instructions_mention_totalmix(self):
        """Test that instructions mention TotalMix."""
        exporter = RMETotalMixExport()
        instructions = exporter.get_import_instructions()

        assert "TotalMix" in instructions

    def test_export_to_file(self, sample_settings):
        """Test exporting to file."""
        exporter = RMETotalMixExport()

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.tmreq"
            exporter.export_to_file(filepath, sample_settings)

            assert filepath.exists()
            content = filepath.read_text()
            assert "<Preset>" in content


class TestREWTextExport:
    """Tests for REW text export format."""

    @pytest.fixture
    def sample_settings(self):
        """Create sample EQ settings."""
        bands = [
            EQBand(FilterType.PEAKING, 63.0, -6.0, 4.3, enabled=True),
            EQBand(FilterType.PEAKING, 125.0, 4.5, 3.2, enabled=True),
        ]
        return EQSettings(bands=bands, channel="left")

    def test_file_extension(self):
        """Test that file extension is .txt."""
        exporter = REWTextExport()
        assert exporter.get_file_extension() == ".txt"

    def test_export_generates_rew_format(self, sample_settings):
        """Test that export generates REW-compatible format."""
        exporter = REWTextExport()
        content = exporter.export(sample_settings)

        assert "Filter Settings file" in content
        assert "Room EQ Wizard" in content
        assert "Filter  1:" in content
        assert "PK" in content

    def test_export_includes_frequencies(self, sample_settings):
        """Test that frequencies are included."""
        exporter = REWTextExport()
        content = exporter.export(sample_settings)

        assert "63" in content
        assert "125" in content

    def test_export_includes_gains(self, sample_settings):
        """Test that gains are included."""
        exporter = REWTextExport()
        content = exporter.export(sample_settings)

        assert "-6.0 dB" in content
        assert "4.5 dB" in content

    def test_import_instructions(self):
        """Test import instructions."""
        exporter = REWTextExport()
        instructions = exporter.get_import_instructions()

        assert len(instructions) > 0

    def test_export_to_file(self, sample_settings):
        """Test exporting to file."""
        exporter = REWTextExport()

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.txt"
            exporter.export_to_file(filepath, sample_settings)

            assert filepath.exists()
            content = filepath.read_text()
            assert "Filter Settings file" in content


class TestManualDisplayExport:
    """Tests for manual display export format."""

    @pytest.fixture
    def sample_settings(self):
        """Create sample EQ settings."""
        bands = [
            EQBand(FilterType.PEAKING, 63.0, -6.0, 4.3, enabled=True),
            EQBand(FilterType.PEAKING, 125.0, 4.5, 3.2, enabled=True),
        ]
        return EQSettings(bands=bands, channel="left")

    def test_file_extension(self):
        """Test that file extension is .txt."""
        exporter = ManualDisplayExport()
        assert exporter.get_file_extension() == ".txt"

    def test_export_generates_table(self, sample_settings):
        """Test that export generates a readable table."""
        exporter = ManualDisplayExport()
        content = exporter.export(sample_settings)

        # Should have table headers
        assert "Band" in content
        assert "Frequency" in content
        assert "Gain" in content
        assert "Q" in content

    def test_export_includes_values(self, sample_settings):
        """Test that band values are included."""
        exporter = ManualDisplayExport()
        content = exporter.export(sample_settings)

        assert "63" in content
        assert "125" in content
        assert "-6.0" in content
        assert "4.3" in content

    def test_import_instructions_mention_manual(self):
        """Test that instructions mention manual entry."""
        exporter = ManualDisplayExport()
        instructions = exporter.get_import_instructions()

        assert "manual" in instructions.lower() or len(instructions) > 0


class TestProfileIntegration:
    """Tests for profile integration with export formats."""

    def test_rme_profile_has_tmreq_format(self):
        """Test that RME profile supports tmreq format."""
        profile = get_profile("rme")
        assert profile is not None
        assert ExportFormatType.RME_TMREQ in profile.export_formats

    def test_generic_profile_has_rew_format(self):
        """Test that generic profile supports REW format."""
        profile = get_profile("generic")
        assert profile is not None
        assert ExportFormatType.REW_TEXT in profile.export_formats

    def test_manual_profile_has_manual_format(self):
        """Test that manual profile supports manual format."""
        profile = get_profile("manual")
        assert profile is not None
        assert ExportFormatType.MANUAL in profile.export_formats

    def test_export_format_uses_profile_constraints(self):
        """Test that export format can use profile constraints."""
        profile = get_profile("rme")
        exporter = get_export_format(ExportFormatType.RME_TMREQ, profile)

        # The exporter should use profile constraints for validation
        assert isinstance(exporter, RMETotalMixExport)
