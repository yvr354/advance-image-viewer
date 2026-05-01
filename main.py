import sys
import os

# Ensure src is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon
from src.ui.main_window import MainWindow
from src.core.config import Config
from src.ui.theme import get_stylesheet


APP_NAME    = "VyuhaAI Image Viewer"
APP_VERSION = "1.0.0"
APP_ORG     = "VyuhaAI"
LOGO_PATH   = os.path.join(os.path.dirname(__file__), "resources", "icons", "logo.png")


def main():
    os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "1"
    # Use hardware OpenGL if GPU available, otherwise fall back to software (CPU)
    os.environ.setdefault("QT_OPENGL", "desktop")
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(APP_VERSION)
    app.setOrganizationName(APP_ORG)
    app.setStyleSheet(get_stylesheet())

    if os.path.exists(LOGO_PATH):
        app.setWindowIcon(QIcon(LOGO_PATH))

    config = Config()
    config.load()

    window = MainWindow(config)

    # Open file passed as argument (e.g. double-click on image)
    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        window.open_image_with_context(sys.argv[1])

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
