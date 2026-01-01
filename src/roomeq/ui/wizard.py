"""Main wizard for room measurement and correction."""

from PyQt6.QtWidgets import QWizard

from roomeq.ui.pages import (
    AnalysisPage,
    DeviceSetupPage,
    ExportPage,
    ListeningPositionPage,
    MeasurementPage,
    WelcomePage,
)


class RoomEQWizard(QWizard):
    """Main wizard for room measurement and correction workflow.

    Guides users through:
    1. Welcome - Introduction and checklist
    2. Device Setup - Audio interface and channel configuration
    3. Listening Position - Number of measurement positions
    4. Measurement - Capture sweeps at each position
    5. Analysis - Review frequency response and problems
    6. Export - Save EQ settings in various formats
    """

    # Page IDs
    PAGE_WELCOME = 0
    PAGE_DEVICE_SETUP = 1
    PAGE_LISTENING_POSITION = 2
    PAGE_MEASUREMENT = 3
    PAGE_ANALYSIS = 4
    PAGE_EXPORT = 5

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("EQilibrium - Room Correction Wizard")
        self.setWizardStyle(QWizard.WizardStyle.ModernStyle)
        self.setMinimumSize(900, 700)

        # Set wizard options
        self.setOption(QWizard.WizardOption.NoBackButtonOnStartPage)
        self.setOption(QWizard.WizardOption.HaveHelpButton, False)

        # Create and add pages
        self._welcome_page: WelcomePage = WelcomePage()
        self._device_page: DeviceSetupPage = DeviceSetupPage()
        self._position_page: ListeningPositionPage = ListeningPositionPage()
        self._measurement_page: MeasurementPage = MeasurementPage()
        self._analysis_page: AnalysisPage = AnalysisPage()
        self._export_page: ExportPage = ExportPage()

        self.setPage(self.PAGE_WELCOME, self._welcome_page)
        self.setPage(self.PAGE_DEVICE_SETUP, self._device_page)
        self.setPage(self.PAGE_LISTENING_POSITION, self._position_page)
        self.setPage(self.PAGE_MEASUREMENT, self._measurement_page)
        self.setPage(self.PAGE_ANALYSIS, self._analysis_page)
        self.setPage(self.PAGE_EXPORT, self._export_page)

        # Customize button text
        self.setButtonText(QWizard.WizardButton.NextButton, "Next →")
        self.setButtonText(QWizard.WizardButton.BackButton, "← Back")
        self.setButtonText(QWizard.WizardButton.FinishButton, "Finish")
        self.setButtonText(QWizard.WizardButton.CancelButton, "Cancel")

    def get_device_page(self) -> DeviceSetupPage:
        """Get the device setup page."""
        return self._device_page

    def get_position_page(self) -> ListeningPositionPage:
        """Get the listening position page."""
        return self._position_page

    def get_measurement_page(self) -> MeasurementPage:
        """Get the measurement page."""
        return self._measurement_page

    def get_analysis_page(self) -> AnalysisPage:
        """Get the analysis page."""
        return self._analysis_page

    def get_export_page(self) -> ExportPage:
        """Get the export page."""
        return self._export_page
