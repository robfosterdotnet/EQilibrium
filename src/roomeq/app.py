"""Main application entry point."""

import sys


def main():
    """Launch the RoomEQ application."""
    # Import here to avoid slow startup for --help etc.
    from PyQt6.QtWidgets import QApplication

    from roomeq.ui.wizard import RoomEQWizard

    app = QApplication(sys.argv)
    app.setApplicationName("RoomEQ")
    app.setApplicationVersion("0.1.0")

    wizard = RoomEQWizard()
    wizard.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
