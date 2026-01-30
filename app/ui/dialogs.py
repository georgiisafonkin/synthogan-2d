"""
Dialog windows for user interactions.
"""

from __future__ import annotations

from typing import Optional, Tuple

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QFileDialog,
    QMessageBox,
    QWidget,
)


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

    @staticmethod
    def fault_params_dialog(
        parent: Optional[QWidget],
        *,
        throw: float,
        sigma_cross: float,
        along_power: float,
    ) -> Optional[Tuple[float, float, str, float]]:
        dialog = QDialog(parent)
        dialog.setWindowTitle("Fault parameters")
        layout = QFormLayout(dialog)

        throw_spin = QDoubleSpinBox(dialog)
        throw_spin.setRange(0.0, 10000.0)
        throw_spin.setValue(float(throw))

        sigma_spin = QDoubleSpinBox(dialog)
        sigma_spin.setRange(0.0, 10000.0)
        sigma_spin.setValue(float(sigma_cross))

        side_combo = QComboBox(dialog)
        side_combo.addItems(["left", "right"])

        power_spin = QDoubleSpinBox(dialog)
        power_spin.setRange(0.5, 5.0)
        power_spin.setValue(float(along_power))

        layout.addRow("Throw", throw_spin)
        layout.addRow("Sigma cross", sigma_spin)
        layout.addRow("Uplift side", side_combo)
        layout.addRow("Along power", power_spin)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=dialog,
        )
        layout.addRow(buttons)

        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None

        return (
            float(throw_spin.value()),
            float(sigma_spin.value()),
            str(side_combo.currentText()),
            float(power_spin.value()),
        )

    @staticmethod
    def deformation_amplitude_dialog(
        parent: Optional[QWidget],
        *,
        amplitude: float,
    ) -> Optional[float]:
        dialog = QDialog(parent)
        dialog.setWindowTitle("Deformation amplitude")
        layout = QFormLayout(dialog)

        amp_spin = QDoubleSpinBox(dialog)
        amp_spin.setRange(0.0, 10000.0)
        amp_spin.setValue(float(amplitude))
        layout.addRow("Amplitude", amp_spin)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=dialog,
        )
        layout.addRow(buttons)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None

        return float(amp_spin.value())
