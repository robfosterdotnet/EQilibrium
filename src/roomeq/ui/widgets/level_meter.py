"""Real-time audio level meter widget."""

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QLinearGradient, QPainter, QPen
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget


class LevelMeter(QWidget):
    """Visual audio level meter with peak hold - horizontal orientation.

    Displays RMS level with peak hold indicator.
    Color gradient: green -> yellow -> red for increasing levels.
    """

    # Signal emitted when level is updated
    level_updated = pyqtSignal(float, float)  # (rms_db, peak_db)

    def __init__(self, parent=None):
        super().__init__(parent)

        # Level values in dB
        self._rms_db = -60.0
        self._peak_db = -60.0
        self._peak_hold_db = -60.0

        # Configuration
        self._min_db = -60.0
        self._max_db = 0.0
        self._clip_db = -3.0  # Red zone threshold
        self._warn_db = -12.0  # Yellow zone threshold

        # Peak hold decay
        self._peak_hold_time_ms = 1000  # Hold peak for 1 second
        self._peak_hold_timer = QTimer()
        self._peak_hold_timer.timeout.connect(self._decay_peak_hold)
        self._peak_hold_timer.setInterval(50)
        self._peak_hold_timer.start()

        # Visual settings - horizontal bar
        self._bar_height = 20

        self.setMinimumSize(150, 50)
        self.setMaximumHeight(60)

    def set_level(self, rms_db: float, peak_db: float) -> None:
        """Update the meter levels.

        Args:
            rms_db: RMS level in dB
            peak_db: Peak level in dB
        """
        self._rms_db = max(self._min_db, min(self._max_db, rms_db))
        self._peak_db = max(self._min_db, min(self._max_db, peak_db))

        # Update peak hold
        if self._peak_db > self._peak_hold_db:
            self._peak_hold_db = self._peak_db

        self.level_updated.emit(self._rms_db, self._peak_db)
        self.update()

    def _decay_peak_hold(self) -> None:
        """Gradually decay the peak hold indicator."""
        decay_rate = 20.0  # dB per second
        decay_per_tick = decay_rate * 0.05  # 50ms interval

        if self._peak_hold_db > self._peak_db:
            self._peak_hold_db -= decay_per_tick
            if self._peak_hold_db < self._peak_db:
                self._peak_hold_db = self._peak_db
            self.update()

    def _db_to_position(self, db: float) -> float:
        """Convert dB value to normalized position (0-1)."""
        db = max(self._min_db, min(self._max_db, db))
        return float((db - self._min_db) / (self._max_db - self._min_db))

    def paintEvent(self, event) -> None:  # noqa: N802
        """Draw the horizontal level meter."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Calculate dimensions
        width = self.width()

        # Bar dimensions - horizontal
        bar_x = 5
        bar_width = width - 10
        bar_y = 5
        bar_height = self._bar_height

        # Draw background with rounded corners
        painter.setBrush(QColor(30, 30, 30))
        painter.setPen(QPen(QColor(50, 50, 50), 1))
        painter.drawRoundedRect(bar_x, bar_y, bar_width, bar_height, 3, 3)

        # Create horizontal gradient for the meter
        gradient = QLinearGradient(bar_x, 0, bar_x + bar_width, 0)
        gradient.setColorAt(0.0, QColor(0, 200, 0))       # Green at left
        gradient.setColorAt(0.6, QColor(0, 200, 0))       # Green up to -24dB
        gradient.setColorAt(0.8, QColor(255, 255, 0))     # Yellow around -12dB
        gradient.setColorAt(0.95, QColor(255, 100, 0))    # Orange near 0dB
        gradient.setColorAt(1.0, QColor(255, 0, 0))       # Red at 0dB

        # Draw RMS level (filled bar from left)
        rms_pos = self._db_to_position(self._rms_db)
        rms_width = int(bar_width * rms_pos)
        if rms_width > 0:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(gradient))
            painter.drawRoundedRect(bar_x, bar_y, rms_width, bar_height, 3, 3)

        # Draw peak hold indicator (vertical line)
        peak_pos = self._db_to_position(self._peak_hold_db)
        peak_x = int(bar_x + bar_width * peak_pos)
        painter.setPen(QPen(QColor(255, 255, 255), 2))
        painter.drawLine(peak_x, bar_y, peak_x, bar_y + bar_height)

        # Draw scale markers below the bar
        marker_y = bar_y + bar_height + 3
        painter.setPen(QPen(QColor(80, 80, 80), 1))

        # Draw tick marks
        for db in range(-60, 1, 6):
            pos = self._db_to_position(db)
            x = int(bar_x + bar_width * pos)
            painter.drawLine(x, marker_y, x, marker_y + 3)

        # Draw dB labels
        painter.setPen(QColor(120, 120, 120))
        font = painter.font()
        font.setPointSize(9)
        painter.setFont(font)

        label_y = marker_y + 15
        labels = [(-48, "-48"), (-24, "-24"), (-12, "-12"), (-6, "-6"), (0, "0")]
        for db, text in labels:
            pos = self._db_to_position(db)
            x = int(bar_x + bar_width * pos)
            # Center the text on the position
            text_width = painter.fontMetrics().horizontalAdvance(text)
            painter.drawText(x - text_width // 2, label_y, text)


class StereoLevelMeter(QWidget):
    """Stereo level meter with L/R channels - horizontal layout."""

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout()
        layout.setSpacing(2)
        layout.setContentsMargins(0, 0, 0, 0)

        # Left channel
        left_row = QHBoxLayout()
        left_label = QLabel("L")
        left_label.setFixedWidth(15)
        left_label.setStyleSheet("color: #888; font-weight: bold;")
        self.left_meter = LevelMeter()
        left_row.addWidget(left_label)
        left_row.addWidget(self.left_meter)

        # Right channel
        right_row = QHBoxLayout()
        right_label = QLabel("R")
        right_label.setFixedWidth(15)
        right_label.setStyleSheet("color: #888; font-weight: bold;")
        self.right_meter = LevelMeter()
        right_row.addWidget(right_label)
        right_row.addWidget(self.right_meter)

        layout.addLayout(left_row)
        layout.addLayout(right_row)
        self.setLayout(layout)

    def set_levels(
        self,
        left_rms_db: float, left_peak_db: float,
        right_rms_db: float, right_peak_db: float
    ) -> None:
        """Update both channel levels."""
        self.left_meter.set_level(left_rms_db, left_peak_db)
        self.right_meter.set_level(right_rms_db, right_peak_db)


class MonoLevelMeter(QWidget):
    """Mono level meter with numeric display - horizontal layout."""

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout()
        layout.setSpacing(5)
        layout.setContentsMargins(0, 0, 0, 0)

        # Level meter (horizontal)
        self.meter = LevelMeter()
        layout.addWidget(self.meter)

        # Bottom row: numeric display and status
        bottom_row = QHBoxLayout()
        bottom_row.setSpacing(10)

        # Numeric display
        self.level_label = QLabel("-∞ dB")
        self.level_label.setStyleSheet(
            "font-family: monospace; font-size: 14px; font-weight: bold; color: #4fc3f7;"
        )

        # Status indicator
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #888;")

        bottom_row.addWidget(self.level_label)
        bottom_row.addStretch()
        bottom_row.addWidget(self.status_label)

        layout.addLayout(bottom_row)
        self.setLayout(layout)

        # Connect to level updates
        self.meter.level_updated.connect(self._update_label)

    def _update_label(self, rms_db: float, peak_db: float) -> None:
        """Update the numeric display."""
        if rms_db <= -60:
            self.level_label.setText("-∞ dB")
        else:
            self.level_label.setText(f"{rms_db:.1f} dB")

        # Update status based on level
        if peak_db > -3:
            self.status_label.setText("CLIP!")
            self.status_label.setStyleSheet("color: #ff5555; font-weight: bold;")
        elif peak_db > -12:
            self.status_label.setText("Good")
            self.status_label.setStyleSheet("color: #69f0ae;")
        elif rms_db > -40:
            self.status_label.setText("OK")
            self.status_label.setStyleSheet("color: #4fc3f7;")
        else:
            self.status_label.setText("Low")
            self.status_label.setStyleSheet("color: #888;")

    def set_level(self, rms_db: float, peak_db: float) -> None:
        """Update the meter level."""
        self.meter.set_level(rms_db, peak_db)
