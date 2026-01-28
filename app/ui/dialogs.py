"""
Dialog windows for user interactions.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtWidgets import QFileDialog, QMessageBox, QWidget


class Dialogs:
    """Collection of dialog windows."""

    @staticmethod
    def save_mask_dialog(parent: Optional[QWidget] = None) -> Optional[str]:
        filepath, _ = QFileDialog.getSaveFileName(
            parent,
            "Save mask",
            "",
            "NumPy files (*.npy);;All files (*.*)",
        )
        return filepath or None

    @staticmethod
    def open_mask_dialog(parent: Optional[QWidget] = None) -> Optional[str]:
        filepath, _ = QFileDialog.getOpenFileName(
            parent,
            "Open mask",
            "",
            "NumPy files (*.npy);;All files (*.*)",
        )
        return filepath or None

    @staticmethod
    def info(parent: Optional[QWidget], title: str, message: str) -> None:
        QMessageBox.information(parent, title, message)
