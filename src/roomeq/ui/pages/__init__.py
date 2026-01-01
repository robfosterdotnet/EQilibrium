"""Wizard page modules."""

from roomeq.ui.pages.analysis import AnalysisPage
from roomeq.ui.pages.device_setup import DeviceSetupPage
from roomeq.ui.pages.export import ExportPage
from roomeq.ui.pages.listening_position import ListeningPositionPage
from roomeq.ui.pages.measurement import MeasurementPage
from roomeq.ui.pages.welcome import WelcomePage

__all__ = [
    "WelcomePage",
    "DeviceSetupPage",
    "ListeningPositionPage",
    "MeasurementPage",
    "AnalysisPage",
    "ExportPage",
]
