"""Export wizard page."""

import threading
from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWizardPage,
)

from roomeq.core.analysis import AnalysisResult
from roomeq.core.eq_optimizer import OptimizationResult, optimize_eq
from roomeq.core.rme_export import export_to_file
from roomeq.ui.widgets import ComparisonPlot


class ExportPage(QWizardPage):
    """Export page for saving EQ settings.

    Shows:
    - Generated 9-band parametric EQ
    - Before/after comparison graph
    - Export buttons for different channels
    - TotalMix import instructions
    - AI-powered analysis explanation
    """

    # Signal for AI explanation completion
    _explanation_ready = pyqtSignal(str)
    _explanation_error = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setTitle("Export EQ Settings")
        self.setSubTitle(
            "Review the generated correction EQ and export for RME TotalMix."
        )

        self._exported_files: list[Path] = []
        self._left_optimization: OptimizationResult | None = None
        self._right_optimization: OptimizationResult | None = None
        self._left_analysis: AnalysisResult | None = None
        self._right_analysis: AnalysisResult | None = None

        # Connect signals for thread-safe UI updates
        self._explanation_ready.connect(self._on_explanation_ready)
        self._explanation_error.connect(self._on_explanation_error)

        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the page UI."""
        layout = QVBoxLayout()
        layout.setSpacing(15)

        # Before/after comparison plot
        plot_group = QGroupBox("Before/After Comparison")
        plot_layout = QVBoxLayout()

        self.comparison_plot = ComparisonPlot()
        plot_layout.addWidget(self.comparison_plot)

        # Improvement metric
        self.improvement_label = QLabel("Estimated improvement: 4.8 dB RMS reduction")
        self.improvement_label.setStyleSheet("font-weight: bold; color: green;")
        plot_layout.addWidget(self.improvement_label)

        plot_group.setLayout(plot_layout)
        layout.addWidget(plot_group)

        # EQ bands and export side by side
        bottom_layout = QHBoxLayout()

        # Generated EQ bands table
        eq_group = QGroupBox("Generated EQ Bands")
        eq_layout = QVBoxLayout()

        self.eq_table = QTableWidget()
        self.eq_table.setColumnCount(5)
        self.eq_table.setHorizontalHeaderLabels(["Band", "Frequency", "Gain", "Q", "Type"])
        self.eq_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.eq_table.verticalHeader().setVisible(False)
        self.eq_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)

        eq_layout.addWidget(self.eq_table)

        # Channel selector
        channel_layout = QHBoxLayout()
        self.show_left = QCheckBox("Left Channel")
        self.show_left.setChecked(True)
        self.show_right = QCheckBox("Right Channel")
        self.show_right.setChecked(True)
        channel_layout.addWidget(self.show_left)
        channel_layout.addWidget(self.show_right)
        channel_layout.addStretch()
        eq_layout.addLayout(channel_layout)

        eq_group.setLayout(eq_layout)
        bottom_layout.addWidget(eq_group, stretch=2)

        # Export section
        export_group = QGroupBox("Export to TotalMix")
        export_layout = QVBoxLayout()

        # Export button
        self.export_btn = QPushButton("Export Room EQ Preset...")
        self.export_btn.setStyleSheet(
            "font-weight: bold; font-size: 14px; padding: 10px; "
            "background-color: #3498db; color: white;"
        )
        self.export_btn.clicked.connect(self._export_preset)
        export_layout.addWidget(self.export_btn)

        export_layout.addSpacing(10)

        # Instructions
        instructions = QLabel(
            "Exports a .tmreq preset file that can be\n"
            "loaded directly in TotalMix FX Room EQ.\n\n"
            "Both left and right channel corrections\n"
            "are included in a single file."
        )
        instructions.setStyleSheet("color: #888; font-size: 11px;")
        export_layout.addWidget(instructions)

        export_layout.addSpacing(10)

        # Export status
        self.export_status = QLabel("")
        self.export_status.setWordWrap(True)
        self.export_status.setStyleSheet("color: gray;")
        export_layout.addWidget(self.export_status)

        export_layout.addStretch()

        export_group.setLayout(export_layout)
        bottom_layout.addWidget(export_group, stretch=1)

        layout.addLayout(bottom_layout)

        # AI Analysis Explanation section
        ai_group = QGroupBox("AI Analysis Explanation")
        ai_layout = QVBoxLayout()

        # Button row
        ai_button_layout = QHBoxLayout()
        self.explain_btn = QPushButton("🤖 Explain My Room Analysis")
        self.explain_btn.setStyleSheet(
            "font-size: 14px; padding: 10px; background-color: #9b59b6; color: white;"
        )
        self.explain_btn.clicked.connect(self._generate_explanation)
        ai_button_layout.addWidget(self.explain_btn)
        ai_button_layout.addStretch()
        ai_layout.addLayout(ai_button_layout)

        # Status label
        self.ai_status_label = QLabel("")
        self.ai_status_label.setStyleSheet("color: gray; font-style: italic;")
        ai_layout.addWidget(self.ai_status_label)

        # Explanation text area
        self.explanation_text = QTextEdit()
        self.explanation_text.setReadOnly(True)
        self.explanation_text.setMinimumHeight(200)
        self.explanation_text.setPlaceholderText(
            "Click the button above to get a detailed AI-powered explanation of your "
            "room analysis results and recommended EQ corrections. The explanation will "
            "help you understand what each correction does and why it's recommended."
        )
        ai_layout.addWidget(self.explanation_text)

        ai_group.setLayout(ai_layout)
        layout.addWidget(ai_group)

        self.setLayout(layout)

    def _run_optimization(self) -> None:
        """Run EQ optimization on analysis results from previous page."""
        wizard = self.wizard()
        if not wizard:
            return

        analysis_page = wizard.page(wizard.PAGE_ANALYSIS)
        if not analysis_page:
            return

        # Get analysis results
        if hasattr(analysis_page, 'get_left_result'):
            self._left_analysis = analysis_page.get_left_result()
        if hasattr(analysis_page, 'get_right_result'):
            self._right_analysis = analysis_page.get_right_result()

        # Run optimization for each channel
        if self._left_analysis is not None:
            self._left_optimization = optimize_eq(
                self._left_analysis.frequencies,
                self._left_analysis.smoothed_response_db,
                self._left_analysis.target_db,
                self._left_analysis.problems,
                max_bands=9
            )

        if self._right_analysis is not None:
            self._right_optimization = optimize_eq(
                self._right_analysis.frequencies,
                self._right_analysis.smoothed_response_db,
                self._right_analysis.target_db,
                self._right_analysis.problems,
                max_bands=9
            )

    def _populate_eq_table(self) -> None:
        """Populate the EQ bands table."""
        # Use left channel optimization for display (both are typically similar)
        optimization = self._left_optimization

        if optimization is None:
            # Show placeholder if no optimization available
            self.eq_table.setRowCount(9)
            for row in range(9):
                self.eq_table.setItem(row, 0, QTableWidgetItem(str(row + 1)))
                for col in range(1, 5):
                    item = QTableWidgetItem("---")
                    item.setForeground(Qt.GlobalColor.gray)
                    self.eq_table.setItem(row, col, item)
            return

        # Pad bands to 9
        bands = list(optimization.settings.bands)
        while len(bands) < 9:
            bands.append(None)

        self.eq_table.setRowCount(9)
        for row, band in enumerate(bands[:9]):
            self.eq_table.setItem(row, 0, QTableWidgetItem(str(row + 1)))

            if band is not None:
                freq_str = f"{band.frequency:.0f} Hz"
                gain_str = f"{band.gain:+.1f} dB"
                q_str = f"{band.q:.2f}"
                type_str = "PK"

                self.eq_table.setItem(row, 1, QTableWidgetItem(freq_str))
                self.eq_table.setItem(row, 2, QTableWidgetItem(gain_str))
                self.eq_table.setItem(row, 3, QTableWidgetItem(q_str))
                self.eq_table.setItem(row, 4, QTableWidgetItem(type_str))
            else:
                for col in range(1, 5):
                    item = QTableWidgetItem("---" if col < 4 else "OFF")
                    item.setForeground(Qt.GlobalColor.gray)
                    self.eq_table.setItem(row, col, item)

    def _update_comparison_plot(self) -> None:
        """Update the before/after comparison plot."""
        from roomeq.core.biquad import calculate_combined_response

        # Use left channel for display
        analysis = self._left_analysis
        optimization = self._left_optimization

        if analysis is None:
            return

        frequencies = analysis.frequencies
        original = analysis.smoothed_response_db
        target = analysis.target_db

        # Clip original to reasonable display range
        display_min, display_max = -30.0, 30.0
        original_clipped = np.clip(original, display_min, display_max)

        # Calculate corrected response
        if optimization is not None and optimization.settings.bands:
            eq_curve = calculate_combined_response(
                optimization.settings.bands,
                frequencies,
                sample_rate=48000  # Standard sample rate for display
            )
            corrected = original + eq_curve
            # Clip for display
            corrected_clipped = np.clip(corrected, display_min, display_max)
            eq_curve_clipped = np.clip(eq_curve, display_min, display_max)
        else:
            corrected_clipped = original_clipped
            eq_curve_clipped = np.zeros_like(frequencies)

        # Set fixed Y-axis range for readability
        self.comparison_plot.plot.set_y_range(display_min, display_max)

        self.comparison_plot.set_original(frequencies, original_clipped)
        self.comparison_plot.set_target(frequencies, target)
        self.comparison_plot.set_corrected(frequencies, corrected_clipped)
        self.comparison_plot.set_eq_curve(frequencies, eq_curve_clipped)

        # Update improvement label - calculate on clipped data for realistic estimate
        if optimization is not None:
            # Use only the 200-8000 Hz range for a meaningful improvement metric
            freq_mask = (frequencies >= 200) & (frequencies <= 8000)
            if np.any(freq_mask):
                orig_masked = original_clipped[freq_mask]
                corr_masked = corrected_clipped[freq_mask]
                tgt_masked = target[freq_mask]
                original_rms = float(np.sqrt(np.mean((orig_masked - tgt_masked) ** 2)))
                corrected_rms = float(np.sqrt(np.mean((corr_masked - tgt_masked) ** 2)))
                improvement = original_rms - corrected_rms
            else:
                improvement = 0.0

            self.improvement_label.setText(
                f"Estimated improvement: {improvement:.1f} dB RMS reduction (200-8000 Hz)"
            )
            if improvement > 0:
                self.improvement_label.setStyleSheet("font-weight: bold; color: green;")
            else:
                self.improvement_label.setStyleSheet("font-weight: bold; color: orange;")
        else:
            self.improvement_label.setText("No optimization available")
            self.improvement_label.setStyleSheet("font-weight: bold; color: gray;")

    def _export_preset(self) -> None:
        """Export EQ settings as TotalMix preset."""
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export TotalMix Room EQ Preset",
            "room_eq.tmreq",
            "TotalMix Preset (*.tmreq);;All Files (*)"
        )

        if filename:
            filepath = Path(filename)
            # Ensure .tmreq extension
            if filepath.suffix.lower() != ".tmreq":
                filepath = filepath.with_suffix(".tmreq")

            # Get settings from optimizations
            left_settings = self._left_optimization.settings if self._left_optimization else None
            right_settings = self._right_optimization.settings if self._right_optimization else None

            export_to_file(filepath, left_settings, right_settings)

            self._exported_files.append(filepath)
            self._update_export_status()

            QMessageBox.information(
                self,
                "Export Complete",
                f"Exported: {filepath.name}\n\n"
                f"Location: {filepath.parent}\n\n"
                "To import in TotalMix:\n"
                "1. Open Room EQ panel\n"
                "2. Click Preset → Load Preset...\n"
                "3. Select the .tmreq file"
            )

    def _update_export_status(self) -> None:
        """Update the export status label."""
        if self._exported_files:
            files_str = "\n".join(f"✓ {f.name}" for f in self._exported_files)
            self.export_status.setText(f"Exported files:\n{files_str}")
            self.export_status.setStyleSheet("color: green;")

    def initializePage(self) -> None:  # noqa: N802
        """Initialize when page is shown."""
        # Run optimization on analysis results
        self._run_optimization()
        self._populate_eq_table()
        self._update_comparison_plot()
        self._exported_files.clear()
        self.export_status.setText("")

    def validatePage(self) -> bool:  # noqa: N802
        """Validate before finishing wizard."""
        if not self._exported_files:
            reply = QMessageBox.question(
                self,
                "No Files Exported",
                "You haven't exported any EQ files yet.\n\n"
                "Are you sure you want to finish without exporting?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            return bool(reply == QMessageBox.StandardButton.Yes)
        return True

    def _generate_explanation(self) -> None:
        """Generate AI explanation in background thread."""
        self.explain_btn.setEnabled(False)
        self.ai_status_label.setText("Generating explanation... This may take a moment.")
        self.ai_status_label.setStyleSheet("color: #9b59b6; font-style: italic;")
        self.explanation_text.clear()

        def generate():
            try:
                from roomeq.core.ai_agent import explain_analysis_sync

                explanation = explain_analysis_sync(
                    self._left_analysis,
                    self._right_analysis,
                    self._left_optimization,
                    self._right_optimization,
                )
                self._explanation_ready.emit(explanation)
            except Exception as e:
                self._explanation_error.emit(str(e))

        thread = threading.Thread(target=generate, daemon=True)
        thread.start()

    def _on_explanation_ready(self, explanation: str) -> None:
        """Handle successful explanation generation."""
        self.explain_btn.setEnabled(True)
        self.ai_status_label.setText("Explanation generated successfully!")
        self.ai_status_label.setStyleSheet("color: green; font-style: italic;")
        self.explanation_text.setMarkdown(explanation)

    def _on_explanation_error(self, error: str) -> None:
        """Handle explanation generation error."""
        self.explain_btn.setEnabled(True)
        self.ai_status_label.setText(f"Error: {error}")
        self.ai_status_label.setStyleSheet("color: red; font-style: italic;")
        self.explanation_text.setPlainText(
            f"Failed to generate explanation:\n\n{error}\n\n"
            "Please check your Azure OpenAI configuration in the .env file."
        )
