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


def main():
    os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "1"
    app = QApplication(sys.argv)
    app.setApplicationName("YVR Advanced Image Viewer")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("YVR")
    app.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps)
    app.setStyleSheet(get_stylesheet())

    config = Config()
    config.load()

    window = MainWindow(config)
    window.show()

    # Open file passed as argument (e.g. double-click on image)
    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        window.open_image(sys.argv[1])

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
