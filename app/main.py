"""
Main entry point for the synthogan-2d application.
"""

import os
import sys

from PySide6.QtWidgets import QApplication

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from ui.main_window import MainWindow


# class MainWindow(QMainWindow):
#     """Главное окно приложения."""
    
#     def __init__(self):
#         super().__init__()
#         self.ui = Ui_MainWindow()
#         self.ui.setupUi(self)


def main():
    """Main application entry point."""
    app = QApplication(sys.argv)
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
