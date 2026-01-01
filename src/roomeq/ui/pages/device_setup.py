"""Device setup wizard page."""

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWizardPage,
)

from roomeq.core.audio_device import AudioDevice, AudioDeviceManager
from roomeq.ui.widgets import MonoLevelMeter


class DeviceSetupPage(QWizardPage):
    """Device setup page for selecting audio interface and channels.

    Shows:
    - Audio interface selection
    - Input channel selection (for microphone)
    - Output channel selection (for speakers)
    - Test button with level meter
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setTitle("Audio Device Setup")
        self.setSubTitle(
            "Select your audio interface and configure input/output channels."
        )

        self._device_manager = AudioDeviceManager()
        self._selected_device: AudioDevice | None = None
        self._test_timer: QTimer | None = None

        self._setup_ui()
        self._refresh_devices()

    def _setup_ui(self) -> None:
        """Set up the page UI."""
        layout = QVBoxLayout()
        layout.setSpacing(15)

        # Device selection
        device_group = QGroupBox("Audio Interface")
        device_layout = QFormLayout()

        self.device_combo = QComboBox()
        self.device_combo.currentIndexChanged.connect(self._on_device_changed)
        device_layout.addRow("Device:", self.device_combo)

        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self._refresh_devices)
        device_layout.addRow("", self.refresh_button)

        self.device_info_label = QLabel("No device selected")
        self.device_info_label.setStyleSheet("color: gray;")
        device_layout.addRow("Info:", self.device_info_label)

        device_group.setLayout(device_layout)
        layout.addWidget(device_group)

        # Channel configuration
        channels_group = QGroupBox("Channel Configuration")
        channels_layout = QFormLayout()

        self.input_channel_spin = QSpinBox()
        self.input_channel_spin.setMinimum(1)
        self.input_channel_spin.setMaximum(32)
        self.input_channel_spin.setValue(1)
        self.input_channel_spin.setToolTip("Channel where your measurement mic is connected")
        channels_layout.addRow("Microphone Input Channel:", self.input_channel_spin)

        self.output_left_spin = QSpinBox()
        self.output_left_spin.setMinimum(1)
        self.output_left_spin.setMaximum(32)
        self.output_left_spin.setValue(1)
        self.output_left_spin.setToolTip("Output channel for left speaker")
        channels_layout.addRow("Left Speaker Output:", self.output_left_spin)

        self.output_right_spin = QSpinBox()
        self.output_right_spin.setMinimum(1)
        self.output_right_spin.setMaximum(32)
        self.output_right_spin.setValue(2)
        self.output_right_spin.setToolTip("Output channel for right speaker")
        channels_layout.addRow("Right Speaker Output:", self.output_right_spin)

        channels_group.setLayout(channels_layout)
        layout.addWidget(channels_group)

        # Test section
        test_group = QGroupBox("Input Level Test")
        test_layout = QHBoxLayout()

        # Level meter
        self.level_meter = MonoLevelMeter()
        test_layout.addWidget(self.level_meter)

        # Test controls
        test_controls = QVBoxLayout()
        self.test_button = QPushButton("Start Test")
        self.test_button.setCheckable(True)
        self.test_button.toggled.connect(self._toggle_test)
        test_controls.addWidget(self.test_button)

        test_instructions = QLabel(
            "Click 'Start Test' and speak into the microphone or "
            "tap it gently. The level meter should respond."
        )
        test_instructions.setWordWrap(True)
        test_instructions.setStyleSheet("color: gray; font-size: 11px;")
        test_controls.addWidget(test_instructions)
        test_controls.addStretch()

        test_layout.addLayout(test_controls)
        test_group.setLayout(test_layout)
        layout.addWidget(test_group)

        layout.addStretch()

        self.setLayout(layout)

    def _refresh_devices(self) -> None:
        """Refresh the list of available audio devices."""
        self.device_combo.clear()

        # Clear the cached device list to force re-enumeration
        self._device_manager.refresh()
        devices = self._device_manager.get_devices()

        # Filter to devices that have both inputs and outputs (most useful for measurement)
        full_devices = [d for d in devices if d.has_inputs and d.has_outputs]
        input_only = [d for d in devices if d.has_inputs and not d.has_outputs]
        output_only = [d for d in devices if d.has_outputs and not d.has_inputs]

        # Add full devices first (most useful)
        for device in sorted(full_devices, key=lambda d: d.name):
            self.device_combo.addItem(device.name, device)

        # Add separator if we have other device types
        if full_devices and (input_only or output_only):
            self.device_combo.insertSeparator(len(full_devices))

        # Add input-only devices
        for device in sorted(input_only, key=lambda d: d.name):
            self.device_combo.addItem(f"{device.name} (input only)", device)

        # Add output-only devices
        for device in sorted(output_only, key=lambda d: d.name):
            self.device_combo.addItem(f"{device.name} (output only)", device)

        if not devices:
            self.device_combo.addItem("No devices found", None)

    def _on_device_changed(self, index: int) -> None:
        """Handle device selection change."""
        data = self.device_combo.currentData()
        self._selected_device = data if isinstance(data, AudioDevice) else None

        if self._selected_device:
            info_text = (
                f"Sample Rate: {int(self._selected_device.default_sample_rate)} Hz | "
                f"Inputs: {self._selected_device.max_input_channels} | "
                f"Outputs: {self._selected_device.max_output_channels}"
            )
            self.device_info_label.setText(info_text)
            # Green if device has both inputs and outputs, orange otherwise
            has_both = self._selected_device.has_inputs and self._selected_device.has_outputs
            style = "color: green;" if has_both else "color: orange;"
            self.device_info_label.setStyleSheet(style)

            # Update channel spinbox limits
            self.input_channel_spin.setMaximum(self._selected_device.max_input_channels)
            self.output_left_spin.setMaximum(self._selected_device.max_output_channels)
            self.output_right_spin.setMaximum(self._selected_device.max_output_channels)
        else:
            self.device_info_label.setText("No device selected")
            self.device_info_label.setStyleSheet("color: gray;")

        self.completeChanged.emit()

    def _toggle_test(self, checked: bool) -> None:
        """Toggle input level test."""
        if checked:
            self.test_button.setText("Stop Test")
            self._start_level_test()
        else:
            self.test_button.setText("Start Test")
            self._stop_level_test()

    def _start_level_test(self) -> None:
        """Start monitoring input level."""
        if not self._selected_device:
            QMessageBox.warning(self, "No Device", "Please select an audio device first.")
            self.test_button.setChecked(False)
            return

        # Create a timer to simulate level updates
        # In real implementation, this would connect to audio device monitoring
        self._test_timer = QTimer()
        self._test_timer.timeout.connect(self._update_test_level)
        self._test_timer.start(50)  # 20 Hz update rate

    def _stop_level_test(self) -> None:
        """Stop monitoring input level."""
        if self._test_timer:
            self._test_timer.stop()
            self._test_timer = None

        self.level_meter.set_level(-60, -60)

    def _update_test_level(self) -> None:
        """Update the level meter with current input level."""
        if not self._selected_device:
            return

        # Get actual level from audio device
        input_channel = self.input_channel_spin.value() - 1  # Convert to 0-indexed
        rms_db = self._device_manager.get_input_level(
            self._selected_device,
            channel=input_channel,
            duration=0.05,  # 50ms measurement
        )

        # Estimate peak as slightly above RMS (typical for speech/music)
        peak_db = rms_db + 3.0

        self.level_meter.set_level(rms_db, peak_db)

    def isComplete(self) -> bool:  # noqa: N802
        """Check if device setup is complete."""
        return self._selected_device is not None

    def get_device(self) -> AudioDevice | None:
        """Get the selected audio device."""
        return self._selected_device

    def get_input_channel(self) -> int:
        """Get the selected input channel (0-indexed)."""
        return int(self.input_channel_spin.value()) - 1

    def get_output_channels(self) -> tuple[int, int]:
        """Get the selected output channels (0-indexed)."""
        return (
            self.output_left_spin.value() - 1,
            self.output_right_spin.value() - 1,
        )

    def cleanupPage(self) -> None:  # noqa: N802
        """Clean up when leaving page."""
        self._stop_level_test()
        self.test_button.setChecked(False)
