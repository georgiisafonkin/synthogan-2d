"""
Main window UI component.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, cast

from PySide6.QtWidgets import QMainWindow

from core.params import HorizonParams
import numpy as np
from core.labeling import horizons_to_layer_labels
from core.faults import FaultGenParams, FaultSpec, fault_lines_from_specs, generate_and_apply_faults

from .canvas_widget import CanvasWidget
from .dialogs import Dialogs
from .template_ui import Ui_MainWindow
from core.horizons import generate_horizons

Point = Tuple[float, float]


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
        self._last_horizons = []
        self._last_mask: Optional[np.ndarray] = None
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

        self.ui.horizonsCreateButton.clicked.connect(self._generate_horizons_auto)
        self.ui.riftsCreateButton.clicked.connect(self._generate_faults_auto)
        self.ui.distortionApplyButton.clicked.connect(lambda: self._show_placeholder("Auto distortions not implemented in UI-only mode."))

        self.ui.horizonsSaveButton.clicked.connect(lambda: self._save_placeholder("horizons"))
        self.ui.riftsSaveButton.clicked.connect(lambda: self._save_placeholder("faults"))
        self.ui.distortionSaveButton.clicked.connect(lambda: self._save_placeholder("distortions"))

        self.ui.saveMaskButton.clicked.connect(self._save_mask_placeholder)
        self.ui.GANSeismicButton.clicked.connect(lambda: self._show_placeholder("GAN inference not available in UI-only mode."))

        self.ui.opacitySpinBox.valueChanged.connect(self._update_opacity)

        self.ui.horizonsCheckBox.toggled.connect(self._update_manual_controls)
        self.ui.riftsCheckBox.toggled.connect(self._update_manual_controls)
        self.ui.distortionCheckBox.toggled.connect(self._update_manual_controls)
        self._update_manual_controls()

        self._apply_canvas_size()

    def _update_manual_controls(self) -> None:
        horizons_enabled = self.ui.horizonsCheckBox.isChecked()
        rifts_enabled = self.ui.riftsCheckBox.isChecked()
        distortions_enabled = self.ui.distortionCheckBox.isChecked()

        self.ui.horizonsDrawButton.setEnabled(horizons_enabled)
        self.ui.horizonsAddButton.setEnabled(horizons_enabled)

        self.ui.riftsDrawButton.setEnabled(rifts_enabled)
        self.ui.riftsAddButton.setEnabled(rifts_enabled)

        self.ui.distortionDrawButton.setEnabled(distortions_enabled)
        self.ui.distortionCompressButton.setEnabled(distortions_enabled)

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
        points = self.canvas.get_points()
        if len(points) < 2:
            self.statusBar().showMessage("No points to add.", 3000)
            return
        self.canvas.commit_path(tag=element_type)
        self._strokes[element_type].append(points)
        if element_type == "horizons":
            self._last_horizons = list(self._strokes["horizons"])
        self.statusBar().showMessage(f"Added {element_type} stroke.", 3000)

    def _clear_strokes(self, element_type: str) -> None:
        self._strokes[element_type].clear()
        if self._drawing_mode == element_type:
            self.canvas.clear_points()
        self.canvas.clear_committed_paths(tag=element_type)
        if element_type == "horizons":
            self.canvas.clear_mask()
        if element_type == "horizons":
            self._last_horizons = []
        self.statusBar().showMessage(f"Cleared {element_type} strokes.", 3000)

    def _save_placeholder(self, element_type: str) -> None:
        self._set_drawing_mode("off")

    def _save_mask_placeholder(self) -> None:
        if not self._last_horizons:
            Dialogs.info(self, "Mask", "Generate horizons before building the mask.")
            self._set_drawing_mode("off")
            return
        width = int(self.ui.widthSpinBox.value())
        height = int(self.ui.heightSpinBox.value())
        W = max(2.0, float(width))
        H = max(2.0, float(height))
        nx = max(2, width)
        ny = max(2, height)
        labels = horizons_to_layer_labels(
            horizons=self._last_horizons,
            W=W,
            H=H,
            nx=nx,
            ny=ny,
        )
        self._last_mask = labels
        self.canvas.set_mask(labels.astype(np.uint8))
        self._set_drawing_mode("off")

    def _update_opacity(self, value: int) -> None:
        opacity = max(0.0, min(1.0, value / 100.0))
        self.canvas.set_seismic_opacity(opacity)

    def _show_placeholder(self, message: str) -> None:
        self.statusBar().showMessage(message, 4000)

    def _generate_horizons_auto(self) -> None:
        width = int(self.ui.widthSpinBox.value())
        height = int(self.ui.heightSpinBox.value())
        num_horizons = int(self.ui.horizonsSpinBox.value())

        if num_horizons <= 0:
            self.statusBar().showMessage("Set horizons count > 0.", 3000)
            return

        # --- stable geometry (matches the “good looking” check.py ratios) ---
        W = max(2, width)
        H = max(2, height)

        # stable sampling along X (don’t tie 1:1 to pixel width)
        nx = int(max(200, min(600, round(W * 0.5))))

        avg = H / float(num_horizons + 1)

        # keep layers thick enough to survive deformation
        min_thickness = max(3.0, 0.45 * avg)
        max_thickness = max(min_thickness + 2.0, 1.15 * avg)

        # deformation tied to thickness (prevents “too steep”)
        deformation_amplitude = max(2.0, min(0.35 * min_thickness, 0.75 * min_thickness))

        # enforce non-crossing / spacing
        min_gap = max(2.0, 0.6 * min_thickness)

        # --- generate ---
        # If your generate_horizons uses HorizonParams (new API):
        params = HorizonParams(
            W=W,
            H=H,
            num_horizons=num_horizons,
            nx=nx,
            min_thickness=min_thickness,
            max_thickness=max_thickness,
            deformation_amplitude=deformation_amplitude,
            min_gap=min_gap,
            # seed=42,  # optional, if you want reproducible output
        )
        horizons = generate_horizons(params)
        self._last_horizons = horizons

        # --- draw to canvas ---
        self.canvas.clear_points()
        self.canvas.clear_committed_paths()
        for horizon in horizons:
            self.canvas.add_path_from_points(horizon, tag="horizons")

        self.statusBar().showMessage(
            f"Generated {len(horizons)} horizons. nx={nx}, min_gap={min_gap:.1f}, amp={deformation_amplitude:.1f}",
            3000,
        )

    def _generate_faults_auto(self) -> None:
        width = int(self.ui.widthSpinBox.value())
        height = int(self.ui.heightSpinBox.value())
        num_faults = int(self.ui.riftsAmountSpinBox.value())

        if num_faults <= 0:
            self.statusBar().showMessage("Set faults count > 0.", 3000)
            return

        length_from = float(self.ui.lengthFromSpinBox.value())
        length_to = float(self.ui.lengthToSpinBox.value())
        angle_from = float(self.ui.anglesFromSpinBox.value())
        angle_to = float(self.ui.anglesToSpinBox.value())
        amp_from = float(self.ui.amplitudeFromSpinBox.value())
        amp_to = float(self.ui.amplitudeToSpinBox.value())

        min_len, max_len = (min(length_from, length_to), max(length_from, length_to))
        min_angle, max_angle = (min(angle_from, angle_to), max(angle_from, angle_to))
        min_amp, max_amp = (min(amp_from, amp_to), max(amp_from, amp_to))
        if max_len <= 1.0:
            self.statusBar().showMessage("Fault length must be > 1.", 3000)
            return

        W = max(2.0, float(width))
        H = max(2.0, float(height))

        # gen = FaultGenParams(
        #     num_faults=num_faults,
        #     x_range=(0.1 * W, 0.9 * W),
        #     y_range=(0.1 * H, 0.9 * H),
        #     length_range=(min_len, max_len),
        #     angle_range_deg=(min_angle, max_angle),
        #     throw_range=(min_amp, max_amp),
        #     sigma_cross=None,
        #     along_power_range=(1.0, 2.5),
        #     uplift_side="random",
        #     min_fault_separation=max(5.0, 0.1 * min_len),
        #     max_tries_per_fault=2000,
        # )

        gen = FaultGenParams(
            num_faults=3,
            x_range=(0.15 * W, 0.85 * W),
            y_range=(0.15 * H, 0.85 * H),
            length_range=(0.25 * H, 1.00 * H),
            angle_range_deg=(50.0, 85.0),
            throw_range=(15.0, 60.0),
            sigma_cross=50.0,
            along_power_range=(1.0, 2.0),
            uplift_side="random",
            min_fault_separation=30.0,
            max_tries_per_fault=2000,
        )


        if not self._last_horizons:
            self.statusBar().showMessage("Generate horizons first.", 3000)
            return

        faulted_horizons, specs = generate_and_apply_faults(
            horizons=self._last_horizons,
            W=W,
            H=H,
            gen=gen,
            return_specs=True,
        )
        specs = cast(List[FaultSpec], specs)
        lines = fault_lines_from_specs(specs)

        self.canvas.clear_points()
        self.canvas.clear_committed_paths()
        for horizon in faulted_horizons:
            self.canvas.add_path_from_points(horizon, tag="horizons")
        # for p1, p2 in lines:
        #     self.canvas.add_path_from_points([p1, p2], tag="faults")

        self._last_horizons = faulted_horizons
        self._last_mask = self.canvas.export_mask(width, height, include_current=False)
        self.statusBar().showMessage(f"Generated {len(lines)} faults.", 3000)