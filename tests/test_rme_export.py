"""Tests for RME TotalMix export format."""

import tempfile
from pathlib import Path

import pytest

from roomeq.core.biquad import EQBand, FilterType
from roomeq.core.eq_optimizer import EQSettings
from roomeq.core.rme_export import (
    export_to_file,
    generate_tmreq_format,
    get_totalmix_import_instructions,
)


class TestGenerateTmreqFormat:
    """Tests for TotalMix XML format generation."""

    @pytest.fixture
    def sample_left_settings(self):
        """Create sample left channel EQ settings."""
        bands = [
            EQBand(FilterType.PEAKING, 63.0, -6.0, 4.3, enabled=True),
            EQBand(FilterType.PEAKING, 125.0, 4.5, 3.2, enabled=True),
            EQBand(FilterType.PEAKING, 250.0, -2.5, 2.5, enabled=True),
        ]
        return EQSettings(bands=bands, channel="left")

    @pytest.fixture
    def sample_right_settings(self):
        """Create sample right channel EQ settings."""
        bands = [
            EQBand(FilterType.PEAKING, 80.0, -4.0, 3.0, enabled=True),
            EQBand(FilterType.PEAKING, 160.0, 2.0, 4.0, enabled=True),
        ]
        return EQSettings(bands=bands, channel="right")

    def test_generates_xml_structure(self, sample_left_settings):
        """Test that output has valid XML structure."""
        content = generate_tmreq_format(sample_left_settings)

        assert "<Preset>" in content
        assert "</Preset>" in content
        assert "<Room EQ L>" in content
        assert "</Room EQ L>" in content
        assert "<Room EQ R>" in content
        assert "</Room EQ R>" in content
        assert "<Params>" in content
        assert "</Params>" in content

    def test_includes_all_nine_bands(self, sample_left_settings):
        """Test that all 9 bands are included."""
        content = generate_tmreq_format(sample_left_settings)

        for i in range(1, 10):
            assert f'REQ Band{i} Freq' in content
            assert f'REQ Band{i} Q' in content
            assert f'REQ Band{i} Gain' in content

    def test_includes_band_types(self, sample_left_settings):
        """Test that band type parameters are included."""
        content = generate_tmreq_format(sample_left_settings)

        assert 'REQ Band1Type' in content
        assert 'REQ Band8 Type' in content
        assert 'REQ Band9 Type' in content

    def test_includes_delay_and_gain(self, sample_left_settings):
        """Test that delay and channel gain are included."""
        content = generate_tmreq_format(sample_left_settings)

        assert 'REQ Delay' in content
        assert 'Chan Gain' in content

    def test_values_have_trailing_comma(self, sample_left_settings):
        """Test that values have trailing comma like TotalMix format."""
        content = generate_tmreq_format(sample_left_settings)

        # All v="..." attributes should end with comma
        assert 'v="0.00,"' in content

    def test_active_band_values(self, sample_left_settings):
        """Test that active band values are included correctly."""
        content = generate_tmreq_format(sample_left_settings)

        # Band 1: 63 Hz, -6 dB, Q 4.3
        assert 'v="63.00,"' in content
        assert 'v="-6.00,"' in content
        assert 'v="4.30,"' in content

    def test_unused_bands_have_defaults(self, sample_left_settings):
        """Test that unused bands have default values."""
        content = generate_tmreq_format(sample_left_settings)

        # Band 4 should be at default frequency 200 Hz with 0 gain
        # Note: some bands will show these defaults
        assert 'v="200.00,"' in content  # Default for band 4

    def test_stereo_settings(self, sample_left_settings, sample_right_settings):
        """Test that both channels are included."""
        content = generate_tmreq_format(sample_left_settings, sample_right_settings)

        # Should have both channel sections
        assert "<Room EQ L>" in content
        assert "<Room EQ R>" in content

        # Left channel has 63 Hz
        # Right channel has 80 Hz
        assert 'v="63.00,"' in content
        assert 'v="80.00,"' in content

    def test_custom_delay(self, sample_left_settings):
        """Test custom delay value."""
        content = generate_tmreq_format(sample_left_settings, delay=5.5)

        assert 'v="5.50,"' in content

    def test_custom_channel_gain(self, sample_left_settings):
        """Test custom channel gain value."""
        content = generate_tmreq_format(sample_left_settings, channel_gain=-3.0)

        assert 'v="-3.00,"' in content

    def test_empty_settings(self):
        """Test export with no settings (defaults only)."""
        content = generate_tmreq_format()

        assert "<Preset>" in content
        assert "<Room EQ L>" in content
        assert "<Room EQ R>" in content
        # Should have default frequencies
        assert 'v="50.00,"' in content  # Band 1 default


class TestExportToFile:
    """Tests for file export."""

    @pytest.fixture
    def sample_settings(self):
        """Create sample EQ settings."""
        bands = [
            EQBand(FilterType.PEAKING, 100.0, -6.0, 2.0, enabled=True),
        ]
        return EQSettings(bands=bands, channel="left")

    def test_creates_file(self, sample_settings):
        """Test that export creates a file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.tmreq"
            export_to_file(filepath, sample_settings)

            assert filepath.exists()

    def test_file_content_valid(self, sample_settings):
        """Test that exported file has valid content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.tmreq"
            export_to_file(filepath, sample_settings)

            content = filepath.read_text()
            assert "<Preset>" in content
            assert 'v="100.00,"' in content

    def test_both_channels(self, sample_settings):
        """Test exporting both channels in single file."""
        right_bands = [EQBand(FilterType.PEAKING, 200.0, -4.0, 3.0, enabled=True)]
        right_settings = EQSettings(bands=right_bands, channel="right")

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.tmreq"
            export_to_file(filepath, sample_settings, right_settings)

            content = filepath.read_text()
            assert "<Room EQ L>" in content
            assert "<Room EQ R>" in content
            assert 'v="100.00,"' in content  # Left
            assert 'v="200.00,"' in content  # Right

    def test_invalid_settings_raise_error(self):
        """Test that invalid settings raise error."""
        # Band with out-of-range frequency
        bands = [EQBand(FilterType.PEAKING, 10.0, 6.0, 2.0)]  # Below min
        settings = EQSettings(bands=bands, channel="left")

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.tmreq"
            with pytest.raises(ValueError, match="Invalid"):
                export_to_file(filepath, settings)


class TestGetTotalMixInstructions:
    """Tests for TotalMix import instructions."""

    def test_returns_string(self):
        """Test that instructions are returned as string."""
        instructions = get_totalmix_import_instructions()

        assert isinstance(instructions, str)
        assert len(instructions) > 0

    def test_mentions_totalmix(self):
        """Test that instructions mention TotalMix."""
        instructions = get_totalmix_import_instructions()

        assert "TotalMix" in instructions

    def test_mentions_tmreq(self):
        """Test that instructions mention .tmreq format."""
        instructions = get_totalmix_import_instructions()

        assert ".tmreq" in instructions

    def test_mentions_load_preset(self):
        """Test that instructions mention Load Preset."""
        instructions = get_totalmix_import_instructions()

        assert "Load Preset" in instructions
