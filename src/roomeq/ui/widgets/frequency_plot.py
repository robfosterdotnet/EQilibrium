"""Frequency response plot widget using pyqtgraph - Modern Dark Theme."""

import numpy as np
import pyqtgraph as pg
from numpy.typing import NDArray
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont, QLinearGradient
from PyQt6.QtWidgets import QCheckBox, QHBoxLayout, QVBoxLayout, QWidget

# Configure pyqtgraph for modern look
pg.setConfigOptions(antialias=True, background='#1a1a2e', foreground='#eaeaea')


class FrequencyPlot(QWidget):
    """Modern frequency response plot widget.

    Dark theme with gradient fills and smooth curves.
    """

    # Modern color palette
    COLORS = {
        'background': '#1a1a2e',
        'grid': '#2d2d44',
        'text': '#eaeaea',
        'accent': '#4fc3f7',
        'original': '#ff6b6b',
        'target': '#888888',
        'corrected': '#69f0ae',
        'eq_curve': '#bb86fc',
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self._traces = {}

    def _setup_ui(self) -> None:
        """Set up the plot widget with modern styling."""
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        # Create pyqtgraph plot widget
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground(self.COLORS['background'])

        # Style the plot
        self._style_plot()

        layout.addWidget(self.plot_widget)

        # Add legend with modern styling
        self.legend = self.plot_widget.addLegend(
            offset=(10, 10),
            brush=pg.mkBrush(self.COLORS['background'] + 'cc'),
            pen=pg.mkPen(self.COLORS['grid']),
        )
        self.legend.setLabelTextColor(self.COLORS['text'])

        self.setLayout(layout)
        self.setMinimumSize(600, 300)

    def _style_plot(self) -> None:
        """Apply modern styling to the plot."""
        plot_item = self.plot_widget.getPlotItem()
        if plot_item is None:
            return

        # Configure axes with log scale for frequency
        self.plot_widget.setLogMode(x=True, y=False)

        # Style grid
        self.plot_widget.showGrid(x=True, y=True, alpha=0.15)

        # Style axes
        axis_pen = pg.mkPen(color=self.COLORS['grid'], width=1)
        axis_font = QFont('SF Pro Display', 10)

        for axis_name in ['bottom', 'left']:
            axis = plot_item.getAxis(axis_name)
            axis.setPen(axis_pen)
            axis.setTextPen(pg.mkPen(self.COLORS['text']))
            axis.setTickFont(axis_font)

        # Labels
        label_style = {'color': self.COLORS['text'], 'font-size': '12px'}
        self.plot_widget.setLabel('bottom', 'Frequency (Hz)', **label_style)
        self.plot_widget.setLabel('left', 'Level (dB)', **label_style)

        # Set ranges
        self.plot_widget.setXRange(20, 20000)
        self.plot_widget.setYRange(-30, 30)

        # Add subtle frequency band markers
        self._add_frequency_bands()

    def _add_frequency_bands(self) -> None:
        """Add subtle frequency band indicators."""
        # Key frequencies with labels
        bands = [
            (20, 'Sub'),
            (60, ''),
            (250, 'Low'),
            (500, ''),
            (1000, 'Mid'),
            (2000, ''),
            (4000, 'High'),
            (8000, ''),
            (16000, 'Air'),
        ]

        for freq, label in bands:
            # Vertical line
            line = pg.InfiniteLine(
                pos=freq,
                angle=90,
                pen=pg.mkPen(color=self.COLORS['grid'], width=1, style=Qt.PenStyle.DotLine)
            )
            self.plot_widget.addItem(line)

    def add_trace(
        self,
        name: str,
        frequencies: NDArray[np.float64],
        magnitude_db: NDArray[np.float64],
        color: str = 'b',
        width: int = 2,
        fill: bool = False,
        fill_alpha: float = 0.3,
    ) -> None:
        """Add or update a frequency response trace with modern styling."""
        # Remove existing trace if present
        if name in self._traces:
            for item in self._traces[name]:
                self.plot_widget.removeItem(item)

        items = []

        # Map simple colors to our palette
        color_map = {
            'r': self.COLORS['original'],
            'g': self.COLORS['corrected'],
            'b': self.COLORS['accent'],
            'gray': self.COLORS['target'],
        }
        actual_color = color_map.get(color, color)

        # Create pen with glow effect
        pen = pg.mkPen(color=actual_color, width=width)

        # Main trace
        trace = self.plot_widget.plot(
            frequencies, magnitude_db,
            pen=pen,
            name=name,
        )
        items.append(trace)

        # Optional fill under curve
        if fill:
            fill_color = QColor(actual_color)
            fill_color.setAlphaF(fill_alpha)
            fill_brush = pg.mkBrush(fill_color)

            fill_curve = pg.PlotCurveItem(
                frequencies, magnitude_db,
                pen=pg.mkPen(None),
                brush=fill_brush,
                fillLevel=-30,
            )
            self.plot_widget.addItem(fill_curve)
            items.append(fill_curve)

        self._traces[name] = items

    def remove_trace(self, name: str) -> None:
        """Remove a trace by name."""
        if name in self._traces:
            for item in self._traces[name]:
                self.plot_widget.removeItem(item)
            del self._traces[name]

    def clear_traces(self) -> None:
        """Remove all traces."""
        for name in list(self._traces.keys()):
            self.remove_trace(name)

    def set_y_range(self, min_db: float, max_db: float) -> None:
        """Set the Y-axis range."""
        self.plot_widget.disableAutoRange(axis='y')
        self.plot_widget.setYRange(min_db, max_db)

    def set_x_range(self, min_hz: float, max_hz: float) -> None:
        """Set the X-axis range in Hz."""
        self.plot_widget.setXRange(min_hz, max_hz)


class ComparisonPlot(QWidget):
    """Modern frequency comparison plot with toggle controls."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the comparison plot with modern styling."""
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Create the main plot
        self.plot = FrequencyPlot()
        layout.addWidget(self.plot)

        # Add styled control checkboxes
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(20)

        checkbox_style = """
            QCheckBox {
                color: #eaeaea;
                font-size: 12px;
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border-radius: 3px;
                border: 2px solid #444;
            }
            QCheckBox::indicator:checked {
                background-color: %s;
                border-color: %s;
            }
        """

        self.show_original = QCheckBox("Original")
        self.show_original.setChecked(True)
        self.show_original.setStyleSheet(
            checkbox_style % (FrequencyPlot.COLORS['original'], FrequencyPlot.COLORS['original'])
        )
        self.show_original.toggled.connect(self._update_visibility)

        self.show_target = QCheckBox("Target")
        self.show_target.setChecked(True)
        self.show_target.setStyleSheet(
            checkbox_style % (FrequencyPlot.COLORS['target'], FrequencyPlot.COLORS['target'])
        )
        self.show_target.toggled.connect(self._update_visibility)

        self.show_corrected = QCheckBox("Corrected")
        self.show_corrected.setChecked(True)
        self.show_corrected.setStyleSheet(
            checkbox_style % (FrequencyPlot.COLORS['corrected'], FrequencyPlot.COLORS['corrected'])
        )
        self.show_corrected.toggled.connect(self._update_visibility)

        self.show_eq = QCheckBox("EQ Curve")
        self.show_eq.setChecked(False)
        self.show_eq.setStyleSheet(
            checkbox_style % (FrequencyPlot.COLORS['eq_curve'], FrequencyPlot.COLORS['eq_curve'])
        )
        self.show_eq.toggled.connect(self._update_visibility)

        controls_layout.addWidget(self.show_original)
        controls_layout.addWidget(self.show_target)
        controls_layout.addWidget(self.show_corrected)
        controls_layout.addWidget(self.show_eq)
        controls_layout.addStretch()

        layout.addLayout(controls_layout)
        self.setLayout(layout)

        # Store data for visibility toggling
        self._data: dict[str, tuple[NDArray, NDArray] | None] = {
            'original': None,
            'target': None,
            'corrected': None,
            'eq': None,
        }

    def set_original(
        self,
        frequencies: NDArray[np.float64],
        magnitude_db: NDArray[np.float64]
    ) -> None:
        """Set the original (measured) response."""
        self._data['original'] = (frequencies, magnitude_db)
        if self.show_original.isChecked():
            self.plot.add_trace(
                'Original', frequencies, magnitude_db,
                color=FrequencyPlot.COLORS['original'],
                width=2,
                fill=True,
                fill_alpha=0.15
            )

    def set_target(
        self,
        frequencies: NDArray[np.float64],
        magnitude_db: NDArray[np.float64]
    ) -> None:
        """Set the target response."""
        self._data['target'] = (frequencies, magnitude_db)
        if self.show_target.isChecked():
            self.plot.add_trace(
                'Target', frequencies, magnitude_db,
                color=FrequencyPlot.COLORS['target'],
                width=2
            )

    def set_corrected(
        self,
        frequencies: NDArray[np.float64],
        magnitude_db: NDArray[np.float64]
    ) -> None:
        """Set the corrected response."""
        self._data['corrected'] = (frequencies, magnitude_db)
        if self.show_corrected.isChecked():
            self.plot.add_trace(
                'Corrected', frequencies, magnitude_db,
                color=FrequencyPlot.COLORS['corrected'],
                width=2,
                fill=True,
                fill_alpha=0.15
            )

    def set_eq_curve(
        self,
        frequencies: NDArray[np.float64],
        magnitude_db: NDArray[np.float64]
    ) -> None:
        """Set the EQ correction curve."""
        self._data['eq'] = (frequencies, magnitude_db)
        if self.show_eq.isChecked():
            self.plot.add_trace(
                'EQ Curve', frequencies, magnitude_db,
                color=FrequencyPlot.COLORS['eq_curve'],
                width=2
            )

    def _update_visibility(self) -> None:
        """Update trace visibility based on checkboxes."""
        traces_config = [
            ('Original', 'original', self.show_original, FrequencyPlot.COLORS['original'], True),
            ('Target', 'target', self.show_target, FrequencyPlot.COLORS['target'], False),
            ('Corrected', 'corrected', self.show_corrected, FrequencyPlot.COLORS['corrected'], True),
            ('EQ Curve', 'eq', self.show_eq, FrequencyPlot.COLORS['eq_curve'], False),
        ]

        for trace_name, data_key, checkbox, color, fill in traces_config:
            if checkbox.isChecked() and self._data[data_key] is not None:
                freqs, mags = self._data[data_key]
                self.plot.add_trace(
                    trace_name, freqs, mags,
                    color=color,
                    width=2,
                    fill=fill,
                    fill_alpha=0.15
                )
            else:
                self.plot.remove_trace(trace_name)


class ProblemHighlightPlot(FrequencyPlot):
    """Frequency plot that highlights detected problems with modern styling."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._problem_markers = []

    def highlight_problem(
        self,
        frequency: float,
        magnitude_db: float,
        is_peak: bool = True,
        severity: str = 'moderate'
    ) -> None:
        """Add a marker highlighting a room problem."""
        # Modern color scheme based on severity
        colors = {
            'minor': '#ffd93d',      # Yellow
            'moderate': '#ff9f43',   # Orange
            'severe': '#ee5a5a',     # Red
        }
        color = colors.get(severity, colors['moderate'])

        # Use circle markers with glow effect
        symbol = 'o'
        size = 14 if severity == 'severe' else 12 if severity == 'moderate' else 10

        # Create scatter plot item for marker
        scatter = pg.ScatterPlotItem(
            x=[frequency],
            y=[magnitude_db],
            symbol=symbol,
            size=size,
            brush=pg.mkBrush(color),
            pen=pg.mkPen('#ffffff', width=2)
        )
        self.plot_widget.addItem(scatter)
        self._problem_markers.append(scatter)

        # Add styled text label
        arrow = '▼' if is_peak else '▲'
        text = pg.TextItem(
            text=f'{arrow} {frequency:.0f}Hz',
            anchor=(0.5, 1.3 if is_peak else -0.3),
            color=color
        )
        text.setFont(QFont('SF Pro Display', 9, QFont.Weight.Bold))
        text.setPos(frequency, magnitude_db)
        self.plot_widget.addItem(text)
        self._problem_markers.append(text)

    def clear_problem_markers(self) -> None:
        """Remove all problem markers."""
        for item in self._problem_markers:
            self.plot_widget.removeItem(item)
        self._problem_markers.clear()
