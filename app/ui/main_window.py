"""
Main window UI component.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from PySide6.QtWidgets import QMainWindow

from .canvas_widget import CanvasWidget
from .dialogs import Dialogs
from .template_ui import Ui_MainWindow

Point = Tuple[int, int]


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.canvas = CanvasWidget(self.ui.centralwidget)
        self._replace_canvas()
        self._setup_controls()

        self._strokes: Dict[str, List[List[Point]]] = {
            "horizons": [],
            "faults": [],
            "distortions": [],
        }
        self._drawing_mode = "off"

    def _replace_canvas(self) -> None:
        layout = self.ui.elementsvVerticalLayout
        index = layout.indexOf(self.ui.canvasGraphicsView)
        layout.removeWidget(self.ui.canvasGraphicsView)
        self.ui.canvasGraphicsView.setParent(None)
        layout.insertWidget(index, self.canvas)

    def _setup_controls(self) -> None:
        self.ui.widthSpinBox.setRange(64, 4096)
        self.ui.heightSpinBox.setRange(64, 4096)
        self.ui.widthSpinBox.setValue(512)
        self.ui.heightSpinBox.setValue(512)

        self.ui.opacitySpinBox.setRange(0, 100)
        self.ui.opacitySpinBox.setValue(50)

        self.ui.canvasSizeButton.clicked.connect(self._apply_canvas_size)

        self.ui.horizonsDrawButton.clicked.connect(lambda: self._set_drawing_mode("horizons"))
        self.ui.riftsDrawButton.clicked.connect(lambda: self._set_drawing_mode("faults"))
        self.ui.distortionDrawButton.clicked.connect(lambda: self._set_drawing_mode("distortions"))

        self.ui.horizonsAddButton.clicked.connect(lambda: self._add_stroke("horizons"))
        self.ui.riftsAddButton.clicked.connect(lambda: self._add_stroke("faults"))
        self.ui.distortionCompressButton.clicked.connect(lambda: self._add_stroke("distortions"))

        self.ui.horizonsClearButton.clicked.connect(lambda: self._clear_strokes("horizons"))
        self.ui.riftsClearButton.clicked.connect(lambda: self._clear_strokes("faults"))
        self.ui.distortionStretchButton.clicked.connect(lambda: self._clear_strokes("distortions"))

        self.ui.horizonsCreateButton.clicked.connect(lambda: self._show_placeholder("Auto horizons not implemented in UI-only mode."))
        self.ui.riftsCreateButton.clicked.connect(lambda: self._show_placeholder("Auto faults not implemented in UI-only mode."))
        self.ui.distortionApplyButton.clicked.connect(lambda: self._show_placeholder("Auto distortions not implemented in UI-only mode."))

        self.ui.horizonsSaveButton.clicked.connect(lambda: self._save_placeholder("horizons"))
        self.ui.riftsSaveButton.clicked.connect(lambda: self._save_placeholder("faults"))
        self.ui.distortionSaveButton.clicked.connect(lambda: self._save_placeholder("distortions"))

        self.ui.saveMaskButton.clicked.connect(self._save_mask_placeholder)
        self.ui.GANSeismicButton.clicked.connect(lambda: self._show_placeholder("GAN inference not available in UI-only mode."))

        self.ui.opacitySpinBox.valueChanged.connect(self._update_opacity)

        self._apply_canvas_size()

    def _apply_canvas_size(self) -> None:
        width = int(self.ui.widthSpinBox.value())
        height = int(self.ui.heightSpinBox.value())
        self.canvas.set_canvas_size(width, height)
        self.canvas.setFixedSize(width, height)
        self.statusBar().showMessage(f"Canvas size set to {width}x{height}.", 3000)

    def _set_drawing_mode(self, mode: str) -> None:
        self._drawing_mode = mode
        self.canvas.set_drawing_mode(mode)
        self.statusBar().showMessage(f"Drawing mode: {mode}.", 3000)

    def _add_stroke(self, element_type: str) -> None:
        if not self.canvas.commit_path():
            self.statusBar().showMessage("No points to add.", 3000)
            return
        self._strokes[element_type].append([])
        self.statusBar().showMessage(f"Added {element_type} stroke.", 3000)

    def _clear_strokes(self, element_type: str) -> None:
        self._strokes[element_type].clear()
        if self._drawing_mode == element_type:
            self.canvas.clear_points()
        self.canvas.clear_committed_paths()
        self.statusBar().showMessage(f"Cleared {element_type} strokes.", 3000)

    def _save_placeholder(self, element_type: str) -> None:
        filepath = Dialogs.save_mask_dialog(self)
        if filepath:
            Dialogs.info(self, "Save", f"Saved {element_type} placeholder to {filepath}.")

    def _save_mask_placeholder(self) -> None:
        filepath = Dialogs.save_mask_dialog(self)
        if filepath:
            Dialogs.info(self, "Save", f"Saved mask placeholder to {filepath}.")

    def _update_opacity(self, value: int) -> None:
        opacity = max(0.0, min(1.0, value / 100.0))
        self.canvas.set_seismic_opacity(opacity)

    def _show_placeholder(self, message: str) -> None:
        self.statusBar().showMessage(message, 4000)
