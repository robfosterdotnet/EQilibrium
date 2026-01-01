"""Listening position setup wizard page."""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QRadioButton,
    QVBoxLayout,
    QWizardPage,
)

from roomeq.ui.widgets import MeasurementUnits, PositionDiagram


class ListeningPositionPage(QWizardPage):
    """Listening position setup page.

    Shows:
    - Visual diagram of measurement positions
    - Selection for number of positions (5/7/9)
    - Explanation of measurement process
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setTitle("Measurement Positions")
        self.setSubTitle(
            "Configure how many positions will be measured around your listening spot."
        )

        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the page UI."""
        layout = QHBoxLayout()
        layout.setSpacing(20)

        # Left side: Position diagram
        left_layout = QVBoxLayout()

        self.position_diagram = PositionDiagram()
        left_layout.addWidget(self.position_diagram)

        # Position legend
        legend_label = QLabel(
            "The numbered positions show where you'll place the microphone "
            "relative to your normal listening position (center)."
        )
        legend_label.setWordWrap(True)
        legend_label.setStyleSheet("color: gray;")
        left_layout.addWidget(legend_label)

        layout.addLayout(left_layout, stretch=2)

        # Right side: Configuration options
        right_layout = QVBoxLayout()

        # Number of positions
        positions_group = QGroupBox("Number of Positions")
        positions_layout = QVBoxLayout()

        self.position_group = QButtonGroup()

        self.radio_5 = QRadioButton("5 positions (Quick)")
        self.radio_5.setToolTip("Faster measurement, less spatial averaging")
        self.position_group.addButton(self.radio_5, 5)

        self.radio_7 = QRadioButton("7 positions (Recommended)")
        self.radio_7.setToolTip("Good balance of accuracy and time")
        self.radio_7.setChecked(True)
        self.position_group.addButton(self.radio_7, 7)

        self.radio_9 = QRadioButton("9 positions (Thorough)")
        self.radio_9.setToolTip("Most accurate, takes longer")
        self.position_group.addButton(self.radio_9, 9)

        self.position_group.idToggled.connect(self._on_position_count_changed)

        positions_layout.addWidget(self.radio_5)
        positions_layout.addWidget(self.radio_7)
        positions_layout.addWidget(self.radio_9)

        positions_group.setLayout(positions_layout)
        right_layout.addWidget(positions_group)

        # Unit system toggle
        units_group = QGroupBox("Distance Units")
        units_layout = QVBoxLayout()

        self.imperial_checkbox = QCheckBox("Use imperial units (feet/inches)")
        self.imperial_checkbox.setToolTip(
            "Switch between metric (cm) and imperial (feet/inches) for position descriptions"
        )
        self.imperial_checkbox.toggled.connect(self._on_units_changed)
        units_layout.addWidget(self.imperial_checkbox)

        units_group.setLayout(units_layout)
        right_layout.addWidget(units_group)

        # Measurement mode
        mode_group = QGroupBox("Measurement Mode")
        mode_layout = QVBoxLayout()

        self.mode_group = QButtonGroup()

        self.radio_quick = QRadioButton("Quick Test (Single Measurement)")
        self.radio_quick.setToolTip(
            "One sweep with both speakers - quick sanity check before full analysis"
        )
        self.mode_group.addButton(self.radio_quick, 0)

        self.radio_full = QRadioButton("Full Analysis (L/R Separate)")
        self.radio_full.setToolTip(
            "Measures each speaker separately - more accurate per-channel EQ"
        )
        self.radio_full.setChecked(True)
        self.mode_group.addButton(self.radio_full, 1)

        self.mode_group.idToggled.connect(self._on_mode_changed)

        mode_layout.addWidget(self.radio_quick)
        mode_layout.addWidget(self.radio_full)

        mode_group.setLayout(mode_layout)
        right_layout.addWidget(mode_group)

        # Time estimate
        self.time_label = QLabel()
        self._update_time_estimate()
        self.time_label.setStyleSheet("font-size: 12px; color: #555;")
        right_layout.addWidget(self.time_label)

        # Measurement process explanation
        process_group = QGroupBox("What to Expect")
        process_layout = QVBoxLayout()

        process_text = QLabel(
            "<p>The measurement process:</p>"
            "<ol>"
            "<li><b>Left speaker first:</b> You'll measure all positions "
            "while only the left speaker plays.</li>"
            "<li><b>Right speaker second:</b> Then repeat all positions "
            "for the right speaker.</li>"
            "</ol>"
            "<p>For each position, a 5-second sweep tone plays while the "
            "microphone records the room response.</p>"
            "<p><b>Total measurements:</b> <span id='total'></span></p>"
        )
        process_text.setWordWrap(True)
        process_text.setTextFormat(Qt.TextFormat.RichText)
        process_layout.addWidget(process_text)

        process_group.setLayout(process_layout)
        right_layout.addWidget(process_group)

        right_layout.addStretch()

        layout.addLayout(right_layout, stretch=1)

        self.setLayout(layout)

    def _on_position_count_changed(self, button_id: int, checked: bool) -> None:
        """Handle position count change."""
        if checked:
            self.position_diagram.set_num_positions(button_id)
            self._update_time_estimate()

    def _on_mode_changed(self, button_id: int, checked: bool) -> None:
        """Handle measurement mode change."""
        if checked:
            is_quick = button_id == 0
            # Disable position count in quick mode (only uses center position)
            self.radio_5.setEnabled(not is_quick)
            self.radio_7.setEnabled(not is_quick)
            self.radio_9.setEnabled(not is_quick)
            # Update diagram to show just 1 position in quick mode
            if is_quick:
                self.position_diagram.set_num_positions(1)
            else:
                self.position_diagram.set_num_positions(self.get_num_positions())
            self._update_time_estimate()

    def _update_time_estimate(self) -> None:
        """Update the time estimate label."""
        num_positions = self.get_num_positions()
        is_quick = self.get_measurement_mode() == "quick"

        # Estimate: ~30 seconds per measurement (5s sweep + mic repositioning)
        if is_quick:
            total_measurements = 1  # Single measurement for quick test
            self.time_label.setText(
                "<b>Estimated time:</b> ~10 seconds (1 quick test measurement)"
            )
        else:
            total_measurements = num_positions * 2  # L + R channels
            time_minutes = total_measurements * 0.5  # 30 seconds each
            self.time_label.setText(
                f"<b>Estimated time:</b> {time_minutes:.0f} minutes "
                f"({total_measurements} measurements, L/R separate)"
            )

    def get_num_positions(self) -> int:
        """Get the selected number of measurement positions."""
        return int(self.position_group.checkedId())

    def get_measurement_mode(self) -> str:
        """Get the selected measurement mode.

        Returns:
            'quick' for both speakers simultaneously, 'full' for L/R separate
        """
        return "quick" if self.mode_group.checkedId() == 0 else "full"

    def _on_units_changed(self, checked: bool) -> None:
        """Handle unit system change."""
        # This will be used by the measurement page
        pass

    def get_units(self) -> MeasurementUnits:
        """Get the selected measurement unit system."""
        if self.imperial_checkbox.isChecked():
            return MeasurementUnits.IMPERIAL
        return MeasurementUnits.METRIC

    def initializePage(self) -> None:  # noqa: N802
        """Initialize when page is shown."""
        # Reset diagram
        self.position_diagram.reset()
        self.position_diagram.set_num_positions(self.get_num_positions())
