"""Analysis wizard page."""

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QSlider,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWizardPage,
)

from roomeq.core.analysis import AnalysisResult, RoomProblem, analyze_response
from roomeq.core.averaging import average_frequency_responses
from roomeq.core.measurement import MeasurementChannel
from roomeq.ui.widgets import ProblemHighlightPlot


class AnalysisPage(QWizardPage):
    """Analysis page showing measurement results.

    Shows:
    - Frequency response graph
    - Detected room problems
    - Overall deviation metric
    - Smoothing controls
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setTitle("Analysis Results")
        self.setSubTitle(
            "Review your room's frequency response and detected problems."
        )

        # Store analysis results per channel
        self._left_result: AnalysisResult | None = None
        self._right_result: AnalysisResult | None = None
        self._combined_result: AnalysisResult | None = None
        self._current_result: AnalysisResult | None = None
        self._problems: list[RoomProblem] = []

        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the page UI."""
        layout = QVBoxLayout()
        layout.setSpacing(15)

        # Channel selector
        channel_layout = QHBoxLayout()
        channel_label = QLabel("Channel:")
        self.channel_combo = QComboBox()
        self.channel_combo.addItems(["Left", "Right", "Combined (Average)"])
        self.channel_combo.currentIndexChanged.connect(self._on_channel_changed)
        channel_layout.addWidget(channel_label)
        channel_layout.addWidget(self.channel_combo)
        channel_layout.addStretch()
        layout.addLayout(channel_layout)

        # Frequency response plot
        plot_group = QGroupBox("Frequency Response")
        plot_layout = QVBoxLayout()

        self.freq_plot = ProblemHighlightPlot()
        plot_layout.addWidget(self.freq_plot)

        # Smoothing control
        smooth_layout = QHBoxLayout()
        smooth_label = QLabel("Smoothing:")
        self.smooth_slider = QSlider(Qt.Orientation.Horizontal)
        self.smooth_slider.setMinimum(1)
        self.smooth_slider.setMaximum(6)  # 1/48, 1/24, 1/12, 1/6, 1/3, 1 octave
        self.smooth_slider.setValue(2)  # Default 1/24 octave
        self.smooth_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.smooth_slider.valueChanged.connect(self._on_smoothing_changed)
        self.smooth_value_label = QLabel("1/24 octave")
        smooth_layout.addWidget(smooth_label)
        smooth_layout.addWidget(self.smooth_slider)
        smooth_layout.addWidget(self.smooth_value_label)
        plot_layout.addLayout(smooth_layout)

        plot_group.setLayout(plot_layout)
        layout.addWidget(plot_group)

        # Statistics and problems side by side
        stats_problems_layout = QHBoxLayout()

        # Statistics
        stats_group = QGroupBox("Statistics")
        stats_layout = QVBoxLayout()

        self.stats_table = QTableWidget(4, 2)
        self.stats_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.stats_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.stats_table.verticalHeader().setVisible(False)
        self.stats_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)

        # Populate with placeholder data
        stats_data = [
            ("RMS Deviation", "5.2 dB"),
            ("Max Peak", "+8.5 dB @ 63 Hz"),
            ("Max Dip", "-6.2 dB @ 125 Hz"),
            ("Variance", "4.8 dB"),
        ]
        for row, (metric, value) in enumerate(stats_data):
            self.stats_table.setItem(row, 0, QTableWidgetItem(metric))
            self.stats_table.setItem(row, 1, QTableWidgetItem(value))

        stats_layout.addWidget(self.stats_table)
        stats_group.setLayout(stats_layout)
        stats_problems_layout.addWidget(stats_group)

        # Problems table
        problems_group = QGroupBox("Detected Room Problems")
        problems_layout = QVBoxLayout()

        self.problems_table = QTableWidget()
        self.problems_table.setColumnCount(4)
        self.problems_table.setHorizontalHeaderLabels(
            ["Type", "Frequency", "Magnitude", "Severity"]
        )
        self.problems_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.problems_table.verticalHeader().setVisible(False)
        self.problems_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.problems_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.problems_table.itemSelectionChanged.connect(self._on_problem_selected)

        problems_layout.addWidget(self.problems_table)
        problems_group.setLayout(problems_layout)
        stats_problems_layout.addWidget(problems_group)

        layout.addLayout(stats_problems_layout)

        self.setLayout(layout)

    def _get_smoothing_octave(self) -> float:
        """Convert slider value to smoothing octave fraction."""
        smoothing_map = {
            1: 1/48,
            2: 1/24,
            3: 1/12,
            4: 1/6,
            5: 1/3,
            6: 1.0,
        }
        return smoothing_map.get(self.smooth_slider.value(), 1/24)

    def _on_channel_changed(self, index: int) -> None:
        """Handle channel selection change."""
        if index == 0:
            self._current_result = self._left_result
        elif index == 1:
            self._current_result = self._right_result
        else:
            self._current_result = self._combined_result
        self._update_plot()
        self._populate_problems_table()

    def _on_smoothing_changed(self, value: int) -> None:
        """Handle smoothing slider change."""
        smoothing_labels = {
            1: "1/48 octave",
            2: "1/24 octave",
            3: "1/12 octave",
            4: "1/6 octave",
            5: "1/3 octave",
            6: "1 octave",
        }
        self.smooth_value_label.setText(smoothing_labels.get(value, ""))
        # Re-run analysis with new smoothing
        self._run_analysis()
        self._update_plot()
        self._populate_problems_table()

    def _on_problem_selected(self) -> None:
        """Highlight selected problem on plot."""
        # Could zoom to selected problem frequency
        pass

    def _run_analysis(self) -> None:
        """Run analysis on measurement data from previous page."""
        wizard = self.wizard()
        if not wizard:
            return

        measurement_page = wizard.page(wizard.PAGE_MEASUREMENT)
        if not measurement_page or not hasattr(measurement_page, 'get_session'):
            return

        session = measurement_page.get_session()
        if not session:
            return

        smoothing = self._get_smoothing_octave()

        # Analyze left channel
        left_measurements = session.get_channel_measurements(MeasurementChannel.LEFT)
        if left_measurements:
            left_avg_freq, left_avg_db = average_frequency_responses(
                [m.frequencies for m in left_measurements],
                [m.magnitude_db for m in left_measurements],
            )
            self._left_result = analyze_response(
                left_avg_freq, left_avg_db, smoothing_octave=smoothing
            )

        # Analyze right channel
        right_measurements = session.get_channel_measurements(MeasurementChannel.RIGHT)
        if right_measurements:
            right_avg_freq, right_avg_db = average_frequency_responses(
                [m.frequencies for m in right_measurements],
                [m.magnitude_db for m in right_measurements],
            )
            self._right_result = analyze_response(
                right_avg_freq, right_avg_db, smoothing_octave=smoothing
            )

        # Combined average of both channels
        all_measurements = left_measurements + right_measurements
        if all_measurements:
            combined_freq, combined_db = average_frequency_responses(
                [m.frequencies for m in all_measurements],
                [m.magnitude_db for m in all_measurements],
            )
            self._combined_result = analyze_response(
                combined_freq, combined_db, smoothing_octave=smoothing
            )

        # Set current result based on channel selection
        index = self.channel_combo.currentIndex()
        if index == 0:
            self._current_result = self._left_result
        elif index == 1:
            self._current_result = self._right_result
        else:
            self._current_result = self._combined_result

    def _update_plot(self) -> None:
        """Update the frequency response plot."""
        self.freq_plot.clear_traces()
        self.freq_plot.clear_problem_markers()

        if self._current_result is None:
            return

        result = self._current_result

        # Plot smoothed response
        self.freq_plot.add_trace(
            "Response",
            result.frequencies,
            result.smoothed_response_db,
            color='b',
            width=2
        )

        # Plot target
        self.freq_plot.add_trace(
            "Target",
            result.frequencies,
            result.target_db,
            color='gray',
            width=1
        )

        # Highlight problems
        for problem in result.problems:
            self.freq_plot.highlight_problem(
                problem.frequency,
                problem.magnitude,
                is_peak=problem.is_peak,
                severity=problem.severity.value
            )

        # Update statistics
        self._update_stats()

    def _update_stats(self) -> None:
        """Update the statistics table."""
        if self._current_result is None:
            return

        result = self._current_result

        # Find max peak and dip
        peaks = [p for p in result.problems if p.is_peak]
        dips = [p for p in result.problems if not p.is_peak]

        max_peak_str = "None"
        if peaks:
            max_peak = max(peaks, key=lambda p: p.magnitude)
            max_peak_str = f"+{max_peak.magnitude:.1f} dB @ {max_peak.frequency:.0f} Hz"

        max_dip_str = "None"
        if dips:
            max_dip = min(dips, key=lambda p: p.magnitude)
            max_dip_str = f"{max_dip.magnitude:.1f} dB @ {max_dip.frequency:.0f} Hz"

        # Calculate variance
        variance = float(np.std(result.deviation_db))

        stats_data = [
            ("RMS Deviation", f"{result.rms_deviation:.1f} dB"),
            ("Max Peak", max_peak_str),
            ("Max Dip", max_dip_str),
            ("Variance", f"{variance:.1f} dB"),
        ]
        for row, (metric, value) in enumerate(stats_data):
            self.stats_table.setItem(row, 0, QTableWidgetItem(metric))
            self.stats_table.setItem(row, 1, QTableWidgetItem(value))

    def _populate_problems_table(self) -> None:
        """Populate the problems table with detected issues."""
        if self._current_result is None:
            self.problems_table.setRowCount(0)
            return

        problems = self._current_result.problems
        self._problems = problems

        self.problems_table.setRowCount(len(problems))
        for row, problem in enumerate(problems):
            ptype = "Peak" if problem.is_peak else "Dip"
            freq = f"{problem.frequency:.0f} Hz"
            mag = f"{problem.magnitude:+.1f} dB"
            severity = problem.severity.value.capitalize()

            self.problems_table.setItem(row, 0, QTableWidgetItem(ptype))
            self.problems_table.setItem(row, 1, QTableWidgetItem(freq))
            self.problems_table.setItem(row, 2, QTableWidgetItem(mag))

            severity_item = QTableWidgetItem(severity)
            if severity == "Severe":
                severity_item.setForeground(Qt.GlobalColor.red)
            elif severity == "Moderate":
                severity_item.setForeground(Qt.GlobalColor.darkYellow)
            self.problems_table.setItem(row, 3, severity_item)

    def initializePage(self) -> None:  # noqa: N802
        """Initialize when page is shown."""
        # Run analysis on measurements from previous page
        self._run_analysis()
        self._update_plot()
        self._populate_problems_table()

    def get_problems(self) -> list[RoomProblem]:
        """Get the list of detected problems."""
        return self._problems

    def get_analysis_result(self) -> AnalysisResult | None:
        """Get the current analysis result."""
        return self._current_result

    def get_left_result(self) -> AnalysisResult | None:
        """Get the left channel analysis result."""
        return self._left_result

    def get_right_result(self) -> AnalysisResult | None:
        """Get the right channel analysis result."""
        return self._right_result
