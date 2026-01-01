"""Measurement wizard page - main work page."""

import threading
from pathlib import Path

import numpy as np
import sounddevice as sd
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
    QWizardPage,
)

from roomeq.core.measurement import (
    MeasurementChannel,
    MeasurementConfig,
    MeasurementEngine,
    MeasurementResult,
    MeasurementSession,
    get_positions,
)
from roomeq.ui.widgets import MicPositionGuide, MonoLevelMeter, PositionDiagram


class MeasurementPage(QWizardPage):
    """Main measurement page where sweeps are captured.

    Shows:
    - Current position indicator
    - Level meter
    - Start/stop controls
    - Progress tracking
    - Redo option
    """

    # Signals
    measurement_started = pyqtSignal()
    measurement_completed = pyqtSignal(int, str)  # position, channel
    _measurement_finished = pyqtSignal(object)  # Internal signal for thread completion

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setTitle("Room Measurement")
        self.setSubTitle("Capture frequency response at each measurement position.")

        self._num_positions = 7
        self._current_position = 1
        self._current_channel = "left"  # "left", "right", or "both"
        self._measurement_mode = "full"  # "quick" or "full"
        self._completed_measurements: dict[str, set[int]] = {
            "left": set(), "right": set(), "both": set()
        }
        self._is_measuring = False
        self._countdown_timer: QTimer | None = None
        self._countdown_value = 3

        # Measurement engine (created in initializePage when device info is available)
        self._engine: MeasurementEngine | None = None
        self._session: MeasurementSession | None = None
        self._positions = get_positions(7)

        # Connect internal signal for thread-safe measurement completion
        self._measurement_finished.connect(self._on_measurement_finished)

        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the page UI."""
        layout = QHBoxLayout()
        layout.setSpacing(20)

        # Left: Position diagram and guide
        left_layout = QVBoxLayout()

        # Channel indicator
        self.channel_label = QLabel("LEFT SPEAKER")
        self.channel_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.channel_label.setStyleSheet(
            "font-size: 18px; font-weight: bold; color: white; "
            "background-color: #4a90d9; padding: 10px; border-radius: 5px;"
        )
        left_layout.addWidget(self.channel_label)

        # Position diagram
        self.position_diagram = PositionDiagram()
        left_layout.addWidget(self.position_diagram)

        # Position guide
        self.position_guide = MicPositionGuide()
        left_layout.addWidget(self.position_guide)

        layout.addLayout(left_layout, stretch=2)

        # Right: Controls and status
        right_layout = QVBoxLayout()

        # Progress section
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout()

        self.progress_label = QLabel("Position 1 of 7")
        self.progress_label.setStyleSheet("font-size: 14px;")
        progress_layout.addWidget(self.progress_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(14)  # 7 positions x 2 channels
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)

        self.overall_label = QLabel("Left: 0/7 | Right: 0/7")
        self.overall_label.setStyleSheet("color: gray;")
        progress_layout.addWidget(self.overall_label)

        progress_group.setLayout(progress_layout)
        right_layout.addWidget(progress_group)

        # Level meter
        meter_group = QGroupBox("Input Level")
        meter_layout = QVBoxLayout()
        self.level_meter = MonoLevelMeter()
        meter_layout.addWidget(self.level_meter)
        meter_group.setLayout(meter_layout)
        right_layout.addWidget(meter_group)

        # Status/countdown area
        self.status_stack = QStackedWidget()

        # Ready state
        ready_widget = QWidget()
        ready_layout = QVBoxLayout()
        ready_label = QLabel("Ready to measure")
        ready_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ready_layout.addWidget(ready_label)
        ready_widget.setLayout(ready_layout)

        # Countdown state
        countdown_widget = QWidget()
        countdown_layout = QVBoxLayout()
        self.countdown_label = QLabel("3")
        self.countdown_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.countdown_label.setStyleSheet(
            "font-size: 72px; font-weight: bold; color: #4a90d9;"
        )
        countdown_layout.addWidget(self.countdown_label)
        countdown_widget.setLayout(countdown_layout)

        # Measuring state
        measuring_widget = QWidget()
        measuring_layout = QVBoxLayout()
        self.measuring_label = QLabel("🔊 Playing sweep...")
        self.measuring_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.measuring_label.setStyleSheet("font-size: 16px; color: green;")
        measuring_layout.addWidget(self.measuring_label)
        measuring_widget.setLayout(measuring_layout)

        self.status_stack.addWidget(ready_widget)      # 0
        self.status_stack.addWidget(countdown_widget)  # 1
        self.status_stack.addWidget(measuring_widget)  # 2

        right_layout.addWidget(self.status_stack)

        # Test speaker button (to check output levels before measuring)
        self.test_speaker_button = QPushButton("🔊 Test Speaker Output")
        self.test_speaker_button.setToolTip(
            "Play a test sweep to check your speaker volume levels"
        )
        self.test_speaker_button.setStyleSheet(
            "font-size: 12px; padding: 10px; background-color: #2196F3; color: white;"
        )
        self.test_speaker_button.clicked.connect(self._test_speaker)
        right_layout.addWidget(self.test_speaker_button)

        # Control buttons
        controls_layout = QHBoxLayout()

        self.measure_button = QPushButton("Start Measurement")
        self.measure_button.setStyleSheet(
            "font-size: 14px; padding: 15px; background-color: #4CAF50; color: white;"
        )
        self.measure_button.clicked.connect(self._start_measurement)
        controls_layout.addWidget(self.measure_button)

        self.redo_button = QPushButton("Redo Last")
        self.redo_button.setEnabled(False)
        self.redo_button.clicked.connect(self._redo_last)
        controls_layout.addWidget(self.redo_button)

        right_layout.addLayout(controls_layout)

        # Session save/load buttons
        session_layout = QHBoxLayout()

        self.save_button = QPushButton("💾 Save Session")
        self.save_button.setToolTip("Save measurements to continue later")
        self.save_button.clicked.connect(self._save_session)
        session_layout.addWidget(self.save_button)

        self.load_button = QPushButton("📂 Load Session")
        self.load_button.setToolTip("Load previously saved measurements")
        self.load_button.clicked.connect(self._load_session)
        session_layout.addWidget(self.load_button)

        right_layout.addLayout(session_layout)

        # Instructions
        self.instruction_label = QLabel(
            "Position the microphone as shown, then click 'Start Measurement'.\n"
            "Keep the room quiet during the sweep."
        )
        self.instruction_label.setWordWrap(True)
        self.instruction_label.setStyleSheet("color: #555; font-size: 11px;")
        right_layout.addWidget(self.instruction_label)

        right_layout.addStretch()

        layout.addLayout(right_layout, stretch=1)

        self.setLayout(layout)

    def set_num_positions(self, num: int) -> None:
        """Set the number of measurement positions."""
        self._num_positions = num
        self._update_progress_max()
        self.position_diagram.set_num_positions(num)
        self._update_ui()

    def set_measurement_mode(self, mode: str) -> None:
        """Set the measurement mode ('quick' or 'full')."""
        self._measurement_mode = mode
        if mode == "quick":
            self._current_channel = "both"
        else:
            self._current_channel = "left"
        self._update_progress_max()
        self._update_ui()

    def _update_progress_max(self) -> None:
        """Update progress bar maximum based on mode."""
        if self._measurement_mode == "quick":
            self.progress_bar.setMaximum(1)  # Single measurement
        else:
            self.progress_bar.setMaximum(self._num_positions * 2)

    def _update_ui(self) -> None:
        """Update all UI elements to reflect current state."""
        # Update channel label
        if self._current_channel == "both":
            self.channel_label.setText("BOTH SPEAKERS")
            self.channel_label.setStyleSheet(
                "font-size: 18px; font-weight: bold; color: white; "
                "background-color: #9b59b6; padding: 10px; border-radius: 5px;"
            )
        elif self._current_channel == "left":
            self.channel_label.setText("LEFT SPEAKER")
            self.channel_label.setStyleSheet(
                "font-size: 18px; font-weight: bold; color: white; "
                "background-color: #4a90d9; padding: 10px; border-radius: 5px;"
            )
        else:
            self.channel_label.setText("RIGHT SPEAKER")
            self.channel_label.setStyleSheet(
                "font-size: 18px; font-weight: bold; color: white; "
                "background-color: #d94a4a; padding: 10px; border-radius: 5px;"
            )

        # Update progress
        if self._measurement_mode == "quick":
            total_done = len(self._completed_measurements["both"])
            self.progress_bar.setValue(total_done)
            self.progress_label.setText("Quick Test — Center Position")
            self.overall_label.setText(
                "Completed" if total_done > 0 else "Ready for single measurement"
            )
        else:
            left_done = len(self._completed_measurements["left"])
            right_done = len(self._completed_measurements["right"])
            total_done = left_done + right_done
            self.progress_bar.setValue(total_done)
            channel_name = self._current_channel.upper()
            pos_text = f"Position {self._current_position} of {self._num_positions}"
            self.progress_label.setText(f"{pos_text} — {channel_name} Speaker")
            self.overall_label.setText(
                f"Completed: {total_done}/{self._num_positions * 2} "
                f"(L: {left_done}, R: {right_done})"
            )

        # Update diagram
        self.position_diagram.set_current_position(self._current_position)
        self.position_diagram.set_current_channel(self._current_channel)
        self.position_diagram.set_completed_positions(
            self._completed_measurements[self._current_channel]
        )

        # Update position guide
        self.position_guide.set_position(self._current_position)

        # Update redo button
        self.redo_button.setEnabled(total_done > 0)

    def _test_speaker(self) -> None:
        """Play a test sweep through the current speaker and show input levels."""
        if self._is_measuring:
            return

        if self._engine is None:
            QMessageBox.warning(
                self, "Error", "No audio device configured. Go back and select a device."
            )
            return

        # Disable buttons during test
        self.test_speaker_button.setEnabled(False)
        self.measure_button.setEnabled(False)
        self.test_speaker_button.setText("🔊 Playing & Monitoring...")

        # Determine which output channel to use
        if self._current_channel == "left":
            output_channel = self._engine.config.output_channel_left
        else:
            output_channel = self._engine.config.output_channel_right

        # Store recording for level display
        self._test_recording: np.ndarray | None = None
        self._test_sample_rate = self._engine.config.sample_rate

        # Play sweep and record in background thread
        def play_and_record():
            try:
                sweep = self._engine.sweep_signal.reshape(-1, 1)
                # Use playrec to play and record simultaneously
                recording = sd.playrec(
                    sweep,
                    samplerate=self._engine.config.sample_rate,
                    device=self._engine.config.device_id,
                    input_mapping=[self._engine.config.input_channel + 1],
                    output_mapping=[output_channel + 1],
                    blocking=True,
                )
                self._test_recording = recording.flatten()
            except Exception:
                self._test_recording = None
            finally:
                # Signal completion back to UI thread
                QTimer.singleShot(0, self._test_speaker_complete)

        # Start level meter update timer
        self._test_level_timer = QTimer()
        self._test_level_timer.timeout.connect(self._update_test_level)
        self._test_level_timer.start(50)  # 20Hz update

        thread = threading.Thread(target=play_and_record, daemon=True)
        thread.start()

    def _update_test_level(self) -> None:
        """Update level meter during test playback."""
        if self._engine is None:
            return

        # Get a quick level reading from the input
        try:
            duration = 0.05
            num_channels = self._engine.config.input_channel + 1
            recording = sd.rec(
                int(duration * self._engine.config.sample_rate),
                samplerate=self._engine.config.sample_rate,
                channels=num_channels,
                device=self._engine.config.device_id,
                blocking=True,
            )
            if recording.ndim > 1:
                channel_data = recording[:, self._engine.config.input_channel]
            else:
                channel_data = recording.flatten()

            rms = np.sqrt(np.mean(channel_data**2))
            if rms > 0:
                rms_db = float(20 * np.log10(rms))
            else:
                rms_db = -100.0

            peak_db = rms_db + 3.0
            self.level_meter.set_level(rms_db, peak_db)
        except Exception:
            pass

    def _test_speaker_complete(self) -> None:
        """Handle test speaker playback completion."""
        # Stop level meter updates
        if hasattr(self, '_test_level_timer') and self._test_level_timer:
            self._test_level_timer.stop()
            self._test_level_timer = None

        # Show peak level from the test
        if hasattr(self, '_test_recording') and self._test_recording is not None:
            peak = np.max(np.abs(self._test_recording))
            if peak > 0:
                peak_db = 20 * np.log10(peak)
                # Update meter one final time with the peak
                self.level_meter.set_level(float(peak_db), float(peak_db))

                # Show result message
                if peak_db > -3:
                    level_status = "⚠️ Too hot! Lower output volume."
                elif peak_db > -12:
                    level_status = "✅ Good levels!"
                elif peak_db > -30:
                    level_status = "⚠️ Low - consider increasing volume."
                else:
                    level_status = "❌ Very low - check connections."

                self.instruction_label.setText(
                    f"Test complete. Peak: {peak_db:.1f} dB\n{level_status}"
                )
            else:
                self.instruction_label.setText(
                    "❌ No signal detected. Check mic connection and input channel."
                )
        else:
            self.instruction_label.setText(
                "Test complete. Check your levels above."
            )

        self.test_speaker_button.setEnabled(True)
        self.measure_button.setEnabled(True)
        self.test_speaker_button.setText("🔊 Test Speaker Output")

    def _start_measurement(self) -> None:
        """Start the measurement countdown and capture."""
        if self._is_measuring:
            return

        self._is_measuring = True
        self.measure_button.setEnabled(False)
        self.test_speaker_button.setEnabled(False)
        self._countdown_value = 3

        # Show countdown
        self.status_stack.setCurrentIndex(1)
        self.countdown_label.setText(str(self._countdown_value))

        # Start countdown timer
        self._countdown_timer = QTimer()
        self._countdown_timer.timeout.connect(self._countdown_tick)
        self._countdown_timer.start(1000)

    def _countdown_tick(self) -> None:
        """Handle countdown timer tick."""
        self._countdown_value -= 1

        if self._countdown_value > 0:
            self.countdown_label.setText(str(self._countdown_value))
        else:
            # Countdown finished, start measurement
            if self._countdown_timer is not None:
                self._countdown_timer.stop()
            self.status_stack.setCurrentIndex(2)
            self._do_measurement()

    def _do_measurement(self) -> None:
        """Perform the actual measurement."""
        self.measurement_started.emit()

        if self._engine is None:
            QMessageBox.warning(
                self, "Error", "Measurement engine not initialized. Go back and select a device."
            )
            self._is_measuring = False
            self.measure_button.setEnabled(True)
            self.status_stack.setCurrentIndex(0)
            return

        # Get current position and channel
        position_idx = self._current_position - 1  # Convert to 0-indexed
        position = self._positions[position_idx]
        if self._current_channel == "left":
            channel = MeasurementChannel.LEFT
        elif self._current_channel == "right":
            channel = MeasurementChannel.RIGHT
        else:
            channel = MeasurementChannel.BOTH

        # Run measurement in background thread to avoid UI freeze
        def run_measurement():
            try:
                result = self._engine.capture(position, channel)
                self._measurement_finished.emit(result)
            except Exception as e:
                self._measurement_finished.emit(e)

        thread = threading.Thread(target=run_measurement, daemon=True)
        thread.start()

    def _on_measurement_finished(self, result: MeasurementResult | Exception) -> None:
        """Handle measurement completion from background thread."""
        if isinstance(result, Exception):
            QMessageBox.warning(
                self, "Measurement Error", f"Measurement failed: {result}"
            )
            self._is_measuring = False
            self.measure_button.setEnabled(True)
            self.status_stack.setCurrentIndex(0)
            return

        # Store result in session
        if self._session is not None:
            if self._measurement_mode == "quick":
                # In quick mode, store same result for both L and R channels
                # Create copies with appropriate channel designation
                from dataclasses import replace
                left_result = replace(result, channel=MeasurementChannel.LEFT)
                right_result = replace(result, channel=MeasurementChannel.RIGHT)
                self._session.add_measurement(left_result)
                self._session.add_measurement(right_result)
            else:
                self._session.add_measurement(result)

        self._measurement_complete()

    def _measurement_complete(self) -> None:
        """Handle measurement completion."""
        # Mark position complete
        self._completed_measurements[self._current_channel].add(self._current_position)

        self.measurement_completed.emit(self._current_position, self._current_channel)

        # Move to next position
        self._advance_position()

        self._is_measuring = False
        self.measure_button.setEnabled(True)
        self.test_speaker_button.setEnabled(True)
        self.status_stack.setCurrentIndex(0)
        self._update_ui()

        # Check if all done
        if self._is_all_complete():
            self.completeChanged.emit()

    def _advance_position(self) -> None:
        """Advance to next measurement position.

        Flow depends on measurement mode:
        - Quick mode: Single measurement only (stays at position 1)
        - Full mode: L then R at each position
        """
        if self._measurement_mode == "quick":
            # Quick mode: single measurement, no advancement needed
            pass
        else:
            # Full mode: L/R per position
            if self._current_channel == "left":
                # Just did left, now do right at same position
                self._current_channel = "right"
            else:
                # Just did right, move to next position and start with left
                self._current_channel = "left"
                if self._current_position < self._num_positions:
                    self._current_position += 1
                # else: all complete

    def _redo_last(self) -> None:
        """Redo the last measurement.

        Goes back one step in the measurement flow.
        """
        if self._measurement_mode == "quick":
            # Quick mode: just clear the single measurement
            self._completed_measurements["both"].clear()
        else:
            # Full mode: L/R per position flow
            if self._current_channel == "right":
                # We're about to do right, so last was left at this position
                if self._current_position in self._completed_measurements["left"]:
                    self._completed_measurements["left"].discard(self._current_position)
                    self._current_channel = "left"
                elif self._current_position > 1:
                    # Go back to previous position's right
                    self._current_position -= 1
                    if self._current_position in self._completed_measurements["right"]:
                        self._completed_measurements["right"].discard(self._current_position)
            else:
                # We're about to do left, so last was right at previous position
                if self._current_position > 1:
                    prev_pos = self._current_position - 1
                    if prev_pos in self._completed_measurements["right"]:
                        self._completed_measurements["right"].discard(prev_pos)
                        self._current_position = prev_pos
                        self._current_channel = "right"
                elif self._current_position == 1 and self._completed_measurements["right"]:
                    # Edge case: at position 1 left, but there are completed rights
                    last_right = max(self._completed_measurements["right"])
                    self._completed_measurements["right"].discard(last_right)
                    self._current_position = last_right
                    self._current_channel = "right"

        self._update_ui()

    def _is_all_complete(self) -> bool:
        """Check if all measurements are complete."""
        if self._measurement_mode == "quick":
            return len(self._completed_measurements["both"]) >= 1  # Single measurement
        else:
            return bool(
                len(self._completed_measurements["left"]) >= self._num_positions and
                len(self._completed_measurements["right"]) >= self._num_positions
            )

    def isComplete(self) -> bool:  # noqa: N802
        """Check if page is complete (all measurements done)."""
        return self._is_all_complete()

    def get_session(self) -> MeasurementSession | None:
        """Get the measurement session with all captured results."""
        return self._session

    def initializePage(self) -> None:  # noqa: N802
        """Initialize when page is shown."""
        wizard = self.wizard()
        if wizard:
            # Get num_positions and measurement mode from listening position page
            listening_page = wizard.page(wizard.PAGE_LISTENING_POSITION)
            if listening_page:
                if hasattr(listening_page, 'get_measurement_mode'):
                    mode = listening_page.get_measurement_mode()
                    self._measurement_mode = mode

                if hasattr(listening_page, 'get_num_positions'):
                    if self._measurement_mode == "quick":
                        # Quick mode: single position (center)
                        num_pos = 1
                    else:
                        num_pos = listening_page.get_num_positions()
                    self._num_positions = num_pos
                    self._positions = get_positions(num_pos)
                    self.position_diagram.set_num_positions(num_pos)

            # Get device configuration from device setup page
            device_page = wizard.page(wizard.PAGE_DEVICE_SETUP)
            if device_page and hasattr(device_page, 'get_device'):
                device = device_page.get_device()
                if device is not None:
                    input_channel = device_page.get_input_channel()
                    output_left, output_right = device_page.get_output_channels()

                    # Create measurement config
                    config = MeasurementConfig(
                        device_id=device.id,
                        input_channel=input_channel,
                        output_channel_left=output_left,
                        output_channel_right=output_right,
                        sample_rate=int(device.default_sample_rate),
                    )

                    # Create measurement engine and session
                    self._engine = MeasurementEngine(config)
                    self._session = MeasurementSession(config, self._num_positions)

        # Reset state based on mode
        self._current_position = 1
        if self._measurement_mode == "quick":
            self._current_channel = "both"
        else:
            self._current_channel = "left"
        self._completed_measurements = {"left": set(), "right": set(), "both": set()}
        self._is_measuring = False

        self._update_progress_max()
        self._update_ui()
        self.status_stack.setCurrentIndex(0)

    def _save_session(self) -> None:
        """Save the current measurement session to a file."""
        if self._session is None:
            QMessageBox.warning(self, "No Session", "No measurements to save.")
            return

        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save Measurement Session",
            "roomeq_session.json",
            "JSON Files (*.json);;All Files (*)"
        )

        if filename:
            try:
                self._session.save(Path(filename))
                QMessageBox.information(
                    self,
                    "Session Saved",
                    f"Measurements saved to:\n{filename}"
                )
            except Exception as e:
                QMessageBox.warning(self, "Save Error", f"Failed to save: {e}")

    def _load_session(self) -> None:
        """Load a measurement session from a file."""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Load Measurement Session",
            "",
            "JSON Files (*.json);;All Files (*)"
        )

        if filename:
            try:
                self._session = MeasurementSession.load(Path(filename))

                # Update completed measurements from loaded session
                self._completed_measurements = {"left": set(), "right": set(), "both": set()}
                for channel in [MeasurementChannel.LEFT, MeasurementChannel.RIGHT]:
                    channel_key = channel.value
                    for pos_id in self._session.measurements[channel]:
                        # Position IDs in session are 0-indexed, UI uses 1-indexed
                        self._completed_measurements[channel_key].add(pos_id + 1)

                # Detect measurement mode from loaded data
                left_count = len(self._session.measurements[MeasurementChannel.LEFT])
                right_count = len(self._session.measurements[MeasurementChannel.RIGHT])

                # Quick mode detection: if we have exactly 1 L and 1 R at position 0,
                # this is likely a quick capture (both speakers, single position)
                left_positions = set(self._session.measurements[MeasurementChannel.LEFT].keys())
                right_positions = set(self._session.measurements[MeasurementChannel.RIGHT].keys())
                is_quick_capture = (
                    left_count == 1 and right_count == 1 and
                    left_positions == {0} and right_positions == {0}
                )

                if is_quick_capture:
                    # Quick mode: single position
                    self._measurement_mode = "quick"
                    self._current_channel = "both"
                    self._num_positions = 1
                    self._positions = get_positions(1)
                    # Mark as complete in "both" tracking
                    self._completed_measurements["both"].add(1)
                else:
                    # Full mode
                    self._measurement_mode = "full"
                    self._current_channel = "left"
                    self._num_positions = len(self._session.positions)
                    self._positions = self._session.positions

                # If all measurements are done, mark as complete
                if self._is_all_complete():
                    self._current_position = self._num_positions
                    if self._measurement_mode == "full":
                        self._current_channel = "right"  # End state for full mode

                self._update_progress_max()
                self._update_ui()

                # Debug info
                left_count = len(self._completed_measurements["left"])
                right_count = len(self._completed_measurements["right"])
                is_complete = self._is_all_complete()

                QMessageBox.information(
                    self,
                    "Session Loaded",
                    f"Loaded {self._session.completed_measurements} measurements.\n"
                    f"Mode: {self._measurement_mode}\n"
                    f"Positions: {self._num_positions}\n"
                    f"Left: {left_count}, Right: {right_count}\n"
                    f"Complete: {is_complete}\n"
                    f"Click Next to proceed to Analysis."
                )

                # Force wizard to re-check completion after event loop processes
                QTimer.singleShot(100, self.completeChanged.emit)
            except Exception as e:
                QMessageBox.warning(self, "Load Error", f"Failed to load: {e}")
