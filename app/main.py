"""
Main entry point for the synthogan-2d application.
"""

import sys
from PySide6.QtWidgets import QApplication
from ui.template_ui import Ui_MainWindow
from ui.main_window import MainWindow
from PySide6.QtWidgets import QMainWindow


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
