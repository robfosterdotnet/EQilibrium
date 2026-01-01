"""Measurement position diagram widget."""

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QBrush, QColor, QFont, QPainter, QPen
from PyQt6.QtWidgets import QWidget


class PositionDiagram(QWidget):
    """Visual diagram showing measurement positions relative to listening position.

    Shows a top-down view of the listening area with:
    - Speaker positions (L/R)
    - Listening position (center)
    - Measurement position markers (numbered)
    - Current position highlighted
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self._num_positions = 7
        self._current_position = 0  # 0 = none selected, 1-9 = position number
        self._completed_positions: set[int] = set()
        self._current_channel = "left"  # or "right"

        # Colors
        self._color_pending = QColor(200, 200, 200)
        self._color_current = QColor(0, 150, 255)
        self._color_completed = QColor(0, 200, 0)
        self._color_speaker_active = QColor(255, 200, 0)
        self._color_speaker_inactive = QColor(100, 100, 100)

        self.setMinimumSize(300, 350)

    def set_num_positions(self, num: int) -> None:
        """Set the number of measurement positions (5, 7, or 9)."""
        self._num_positions = num
        self.update()

    def set_current_position(self, position: int) -> None:
        """Set the currently active measurement position (1-based)."""
        self._current_position = position
        self.update()

    def set_completed_positions(self, positions: set[int]) -> None:
        """Set the set of completed position numbers."""
        self._completed_positions = positions
        self.update()

    def mark_position_complete(self, position: int) -> None:
        """Mark a position as completed."""
        self._completed_positions.add(position)
        self.update()

    def set_current_channel(self, channel: str) -> None:
        """Set the current channel being measured ('left' or 'right')."""
        self._current_channel = channel
        self.update()

    def reset(self) -> None:
        """Reset all progress."""
        self._current_position = 0
        self._completed_positions.clear()
        self.update()

    def _get_position_coords(self, position: int) -> QPointF:
        """Get normalized coordinates for a position (0-1 range).

        Position layout aligned with core/measurement.py STANDARD_POSITIONS:
            1: Center (core id 0)
            2: Left (core id 1)
            3: Right (core id 2)
            4: Front Left - diagonal (core id 3)
            5: Front Right - diagonal (core id 4)
            6: Back (core id 5)
            7: Up (core id 6) - shown at center with marker

        For 5-position: 1-5 (Center, Left, Right, Front Left, Front Right)
        For 9-position: 1-7 + 8 (Back Left) + 9 (Back Right)
        """
        # Base listening position at center
        cx, cy = 0.5, 0.65

        # Relative offsets (scaled for display)
        offset = 0.12

        positions = {
            1: (cx, cy),  # Center
            2: (cx - offset, cy),  # Left
            3: (cx + offset, cy),  # Right
            4: (cx - offset * 0.7, cy - offset),  # Front Left (diagonal)
            5: (cx + offset * 0.7, cy - offset),  # Front Right (diagonal)
            6: (cx, cy + offset),  # Back
            7: (cx, cy - offset * 0.5),  # Up (shown slightly forward, marked with ↑)
            8: (cx - offset * 0.7, cy + offset),  # Back Left (9-pos only)
            9: (cx + offset * 0.7, cy + offset),  # Back Right (9-pos only)
        }

        if position in positions:
            return QPointF(*positions[position])
        return QPointF(cx, cy)

    def paintEvent(self, event) -> None:  # noqa: N802
        """Draw the position diagram."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        width = self.width()
        height = self.height()

        # Draw background
        painter.fillRect(0, 0, width, height, QColor(250, 250, 250))

        # Draw room outline (simple rectangle)
        room_margin = 20
        room_rect = QRectF(
            room_margin, room_margin, width - 2 * room_margin, height - 2 * room_margin
        )
        painter.setPen(QPen(QColor(100, 100, 100), 1))
        painter.drawRect(room_rect)

        # Draw speakers
        self._draw_speakers(painter, width, height)

        # Draw listening position marker
        self._draw_listening_position(painter, width, height)

        # Draw measurement positions
        self._draw_measurement_positions(painter, width, height)

        # Draw legend
        self._draw_legend(painter, width, height)

    def _draw_speakers(self, painter: QPainter, width: int, height: int) -> None:
        """Draw speaker icons at top of diagram."""
        speaker_y = 50
        speaker_width = 40
        speaker_height = 60

        # Left speaker
        left_x = width * 0.25 - speaker_width / 2
        if self._current_channel == "left":
            left_color = self._color_speaker_active
        else:
            left_color = self._color_speaker_inactive
        self._draw_speaker_icon(
            painter, left_x, speaker_y, speaker_width, speaker_height, left_color, "L"
        )

        # Right speaker
        right_x = width * 0.75 - speaker_width / 2
        if self._current_channel == "right":
            right_color = self._color_speaker_active
        else:
            right_color = self._color_speaker_inactive
        self._draw_speaker_icon(
            painter, right_x, speaker_y, speaker_width, speaker_height, right_color, "R"
        )

    def _draw_speaker_icon(
        self, painter: QPainter, x: float, y: float,
        w: float, h: float, color: QColor, label: str
    ) -> None:
        """Draw a speaker icon."""
        painter.setPen(QPen(Qt.GlobalColor.black, 2))
        painter.setBrush(QBrush(color))

        # Draw speaker cabinet
        rect = QRectF(x, y, w, h)
        painter.drawRoundedRect(rect, 5, 5)

        # Draw woofer circle
        woofer_size = min(w, h) * 0.6
        woofer_x = x + (w - woofer_size) / 2
        woofer_y = y + h * 0.5 - woofer_size / 2
        painter.setBrush(QBrush(QColor(50, 50, 50)))
        painter.drawEllipse(QRectF(woofer_x, woofer_y, woofer_size, woofer_size))

        # Draw label
        painter.setPen(QPen(Qt.GlobalColor.white))
        font = QFont()
        font.setBold(True)
        font.setPointSize(12)
        painter.setFont(font)
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, label)

    def _draw_listening_position(self, painter: QPainter, width: int, height: int) -> None:
        """Draw the main listening position marker."""
        cx = width * 0.5
        cy = height * 0.65

        # Draw chair/seat icon (simple circle with arc)
        seat_size = 50
        painter.setPen(QPen(QColor(100, 100, 100), 2))
        painter.setBrush(QBrush(QColor(180, 180, 180)))
        painter.drawEllipse(QPointF(cx, cy), seat_size / 2, seat_size / 2)

        # Draw "you are here" text
        painter.setPen(QPen(QColor(80, 80, 80)))
        font = QFont()
        font.setPointSize(9)
        painter.setFont(font)
        painter.drawText(
            QRectF(cx - 60, cy + seat_size / 2 + 5, 120, 20),
            Qt.AlignmentFlag.AlignCenter,
            "Listening Position"
        )

    def _draw_measurement_positions(self, painter: QPainter, width: int, height: int) -> None:
        """Draw numbered measurement position markers."""
        marker_size = 24

        for pos in range(1, self._num_positions + 1):
            coords = self._get_position_coords(pos)
            x = coords.x() * width
            y = coords.y() * height

            # Determine color
            if pos == self._current_position:
                color = self._color_current
            elif pos in self._completed_positions:
                color = self._color_completed
            else:
                color = self._color_pending

            # Draw marker circle
            painter.setPen(QPen(Qt.GlobalColor.black, 2))
            painter.setBrush(QBrush(color))
            painter.drawEllipse(QPointF(x, y), marker_size / 2, marker_size / 2)

            # Draw position number
            pen_color = (
                Qt.GlobalColor.white if pos == self._current_position
                else Qt.GlobalColor.black
            )
            painter.setPen(QPen(pen_color))
            font = QFont()
            font.setBold(True)
            font.setPointSize(10)
            painter.setFont(font)
            painter.drawText(
                QRectF(x - marker_size / 2, y - marker_size / 2, marker_size, marker_size),
                Qt.AlignmentFlag.AlignCenter,
                str(pos)
            )

    def _draw_legend(self, painter: QPainter, width: int, height: int) -> None:
        """Draw the legend."""
        legend_y = height - 35
        legend_x = 30
        box_size = 12
        spacing = 80

        font = QFont()
        font.setPointSize(9)
        painter.setFont(font)

        items = [
            (self._color_pending, "Pending"),
            (self._color_current, "Current"),
            (self._color_completed, "Done"),
        ]

        for i, (color, label) in enumerate(items):
            x = legend_x + i * spacing
            painter.setBrush(QBrush(color))
            painter.setPen(QPen(Qt.GlobalColor.black, 1))
            painter.drawRect(int(x), int(legend_y), box_size, box_size)

            painter.setPen(QPen(Qt.GlobalColor.black))
            painter.drawText(int(x + box_size + 5), int(legend_y + box_size - 2), label)


class MicPositionGuide(QWidget):
    """Detailed guide showing exactly where to place the microphone.

    Shows a side view illustrating the position offset from the listening position.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self._position_number = 1
        # Position descriptions aligned with core/measurement.py STANDARD_POSITIONS
        # UI uses 1-indexed (position 1 = core position 0)
        self._position_descriptions = {
            1: ("Center", "At ear height, facing speakers"),
            2: ("Left", "Offset 30cm to your left"),
            3: ("Right", "Offset 30cm to your right"),
            4: ("Front Left", "30cm forward, 15cm left (diagonal)"),
            5: ("Front Right", "30cm forward, 15cm right (diagonal)"),
            6: ("Back", "Offset 30cm away from speakers"),
            7: ("Up", "Raise microphone 30cm higher"),
            # 9-position layout adds:
            8: ("Back Left", "30cm back, 15cm left (diagonal)"),
            9: ("Back Right", "30cm back, 15cm right (diagonal)"),
        }

        self.setMinimumSize(200, 100)

    def set_position(self, position: int) -> None:
        """Set the current position to describe."""
        self._position_number = position
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        """Draw the position guide."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        width = self.width()
        height = self.height()

        # Background
        painter.fillRect(0, 0, width, height, QColor(240, 248, 255))
        painter.setPen(QPen(QColor(100, 149, 237), 2))
        painter.drawRect(0, 0, width - 1, height - 1)

        # Get position info
        name, description = self._position_descriptions.get(
            self._position_number,
            ("Unknown", "")
        )

        # Draw position name
        font = QFont()
        font.setBold(True)
        font.setPointSize(14)
        painter.setFont(font)
        painter.setPen(QPen(QColor(0, 100, 200)))
        painter.drawText(
            QRectF(10, 10, width - 20, 30),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            f"Position {self._position_number}: {name}"
        )

        # Draw description
        font.setBold(False)
        font.setPointSize(11)
        painter.setFont(font)
        painter.setPen(QPen(Qt.GlobalColor.black))
        painter.drawText(
            QRectF(10, 45, width - 20, 50),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop | Qt.TextFlag.TextWordWrap,
            description
        )
