"""Welcome wizard page."""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWizardPage,
)


class WelcomePage(QWizardPage):
    """Welcome page with introduction and checklist.

    Shows:
    - Brief intro explaining what the wizard does
    - Checklist of requirements (RME connected, mic ready, etc.)
    - Optional: Load existing project
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setTitle("Welcome to RoomEQ")
        self.setSubTitle(
            "This wizard will guide you through measuring and correcting "
            "your room's frequency response."
        )

        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the page UI."""
        layout = QVBoxLayout()
        layout.setSpacing(20)

        # Introduction text
        intro_label = QLabel(
            "<p>RoomEQ helps you achieve accurate sound reproduction by:</p>"
            "<ol>"
            "<li>Measuring your room's frequency response at multiple positions</li>"
            "<li>Analyzing peaks and dips caused by room acoustics</li>"
            "<li>Generating correction EQ settings for your RME interface</li>"
            "</ol>"
            "<p>The process takes about 15-20 minutes and requires minimal interaction.</p>"
        )
        intro_label.setWordWrap(True)
        intro_label.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(intro_label)

        # Requirements checklist
        checklist_group = QGroupBox("Before You Begin")
        checklist_layout = QVBoxLayout()

        self.check_rme = QCheckBox("RME audio interface is connected and powered on")
        self.check_mic = QCheckBox("Measurement microphone is connected to an input")
        self.check_speakers = QCheckBox("Studio monitors are connected and at listening volume")
        self.check_quiet = QCheckBox("Room is quiet (no fans, AC, traffic noise)")
        self.check_position = QCheckBox("You know where your primary listening position is")

        for checkbox in [self.check_rme, self.check_mic, self.check_speakers,
                         self.check_quiet, self.check_position]:
            checkbox.toggled.connect(self._update_complete_status)
            checklist_layout.addWidget(checkbox)

        checklist_group.setLayout(checklist_layout)
        layout.addWidget(checklist_group)

        # Load existing project option
        project_layout = QHBoxLayout()
        project_label = QLabel("Or load an existing measurement project:")
        self.load_button = QPushButton("Load Project...")
        self.load_button.clicked.connect(self._load_project)
        project_layout.addWidget(project_label)
        project_layout.addWidget(self.load_button)
        project_layout.addStretch()
        layout.addLayout(project_layout)

        layout.addStretch()

        # Tips
        tips_label = QLabel(
            "<p><b>Tips for best results:</b></p>"
            "<ul>"
            "<li>Use a flat measurement microphone if possible</li>"
            "<li>Keep the microphone still during each measurement</li>"
            "<li>Avoid moving around the room during measurements</li>"
            "</ul>"
        )
        tips_label.setWordWrap(True)
        tips_label.setTextFormat(Qt.TextFormat.RichText)
        tips_label.setStyleSheet("color: #555;")
        layout.addWidget(tips_label)

        self.setLayout(layout)

        # Register fields for wizard access
        self.registerField("checklist_complete", self.check_rme)

    def _update_complete_status(self) -> None:
        """Update page completion status based on checklist."""
        self.completeChanged.emit()

    def isComplete(self) -> bool:  # noqa: N802
        """Check if all checklist items are checked."""
        return all([
            self.check_rme.isChecked(),
            self.check_mic.isChecked(),
            self.check_speakers.isChecked(),
            self.check_quiet.isChecked(),
            self.check_position.isChecked(),
        ])

    def _load_project(self) -> None:
        """Open dialog to load existing project."""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Load RoomEQ Project",
            "",
            "RoomEQ Projects (*.roomeq);;All Files (*)"
        )
        if filename:
            # TODO: Load project file
            # For now, just mark checklist complete
            self.check_rme.setChecked(True)
            self.check_mic.setChecked(True)
            self.check_speakers.setChecked(True)
            self.check_quiet.setChecked(True)
            self.check_position.setChecked(True)
