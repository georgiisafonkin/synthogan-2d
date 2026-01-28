"""
Canvas widget for displaying and editing mask and seismic overlays.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

import numpy as np
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QBrush, QColor, QImage, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsPathItem,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
)

Point = Tuple[int, int]


class CanvasWidget(QGraphicsView):
    """Widget for rendering seismic data on canvas with drawing support."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHint(QPainter.Antialiasing, True)

        self._mask_item: Optional[QGraphicsPixmapItem] = None
        self._seismic_item: Optional[QGraphicsPixmapItem] = None
        self._path_item: Optional[QGraphicsPathItem] = None
        self._point_items: List[QGraphicsEllipseItem] = []
        self._committed_paths: List[QGraphicsPathItem] = []
        self._points: List[Point] = []
        self._drawing_mode: str = "off"

        self._pen_path = QPen(QColor(0, 180, 255), 2)
        self._pen_points = QPen(QColor(255, 160, 0), 2)
        self._brush_points = QBrush(QColor(255, 160, 0))
        self.setMouseTracking(True)

    def set_canvas_size(self, width: int, height: int) -> None:
        self._scene.setSceneRect(0, 0, width, height)

    def set_mask(self, mask: np.ndarray) -> None:
        pixmap = self._numpy_to_pixmap(mask)
        if self._mask_item is None:
            self._mask_item = self._scene.addPixmap(pixmap)
            self._mask_item.setZValue(0)
        else:
            self._mask_item.setPixmap(pixmap)

    def set_seismic_overlay(self, seismic: Optional[np.ndarray], opacity: float = 0.5) -> None:
        if seismic is None:
            if self._seismic_item is not None:
                self._scene.removeItem(self._seismic_item)
                self._seismic_item = None
            return
        pixmap = self._numpy_to_pixmap(seismic)
        if self._seismic_item is None:
            self._seismic_item = self._scene.addPixmap(pixmap)
            self._seismic_item.setZValue(1)
        else:
            self._seismic_item.setPixmap(pixmap)
        self._seismic_item.setOpacity(opacity)

    def set_seismic_opacity(self, opacity: float) -> None:
        if self._seismic_item is not None:
            self._seismic_item.setOpacity(opacity)

    def set_drawing_mode(self, mode: str) -> None:
        self._drawing_mode = mode

    def get_points(self) -> List[Point]:
        return list(self._points)

    def take_points(self) -> List[Point]:
        points = list(self._points)
        self.clear_points()
        return points

    def commit_path(self) -> bool:
        if len(self._points) < 2:
            return False
        path = self._build_smooth_path(self._points)
        item = self._scene.addPath(path, self._pen_path)
        item.setZValue(2)
        self._committed_paths.append(item)
        self.clear_points()
        return True

    def clear_points(self) -> None:
        for item in self._point_items:
            self._scene.removeItem(item)
        self._point_items.clear()
        self._points.clear()
        if self._path_item is not None:
            self._scene.removeItem(self._path_item)
            self._path_item = None

    def clear_committed_paths(self) -> None:
        for item in self._committed_paths:
            self._scene.removeItem(item)
        self._committed_paths.clear()

    def mousePressEvent(self, event) -> None:
        if self._drawing_mode != "off" and event.button() == Qt.LeftButton:
            pos = self.mapToScene(event.pos())
            self._add_point(int(pos.x()), int(pos.y()))
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._drawing_mode != "off" and event.buttons() & Qt.LeftButton:
            pos = self.mapToScene(event.pos())
            self._add_point(int(pos.x()), int(pos.y()), allow_duplicates=False)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def _add_point(self, x: int, y: int, allow_duplicates: bool = True) -> None:
        if not allow_duplicates and self._points:
            last = self._points[-1]
            if abs(last[0] - x) < 2 and abs(last[1] - y) < 2:
                return
        self._points.append((x, y))
        radius = 2
        point_item = self._scene.addEllipse(
            x - radius,
            y - radius,
            radius * 2,
            radius * 2,
            self._pen_points,
            self._brush_points,
        )
        point_item.setZValue(3)
        self._point_items.append(point_item)
        self._update_path()

    def _update_path(self) -> None:
        if len(self._points) < 2:
            return
        path = self._build_smooth_path(self._points)
        if self._path_item is None:
            self._path_item = self._scene.addPath(path, self._pen_path)
            self._path_item.setZValue(2)
        else:
            self._path_item.setPath(path)

    @staticmethod
    def _build_smooth_path(points: Iterable[Point]) -> QPainterPath:
        pts = [QPointF(p[0], p[1]) for p in points]
        path = QPainterPath(pts[0])
        if len(pts) == 2:
            path.lineTo(pts[1])
            return path

        # Catmull-Rom style smoothing
        for i in range(1, len(pts) - 1):
            p0 = pts[i - 1]
            p1 = pts[i]
            p2 = pts[i + 1]
            control1 = QPointF(p1.x() + (p2.x() - p0.x()) / 6.0, p1.y() + (p2.y() - p0.y()) / 6.0)
            control2 = QPointF(p2.x() - (p2.x() - p1.x()) / 6.0, p2.y() - (p2.y() - p1.y()) / 6.0)
            path.cubicTo(control1, control2, p2)
        return path

    @staticmethod
    def _normalize_to_uint8(data: np.ndarray) -> np.ndarray:
        if data.dtype == np.uint8:
            return data
        data = data.astype(np.float32)
        min_val = float(np.min(data))
        max_val = float(np.max(data))
        if max_val - min_val < 1e-6:
            return np.zeros_like(data, dtype=np.uint8)
        scaled = (data - min_val) / (max_val - min_val)
        return (scaled * 255.0).clip(0, 255).astype(np.uint8)

    def _numpy_to_pixmap(self, data: np.ndarray) -> QPixmap:
        if data.ndim != 2:
            raise ValueError("Only 2D arrays are supported for display.")
        img = self._normalize_to_uint8(data)
        height, width = img.shape
        qimage = QImage(img.data, width, height, width, QImage.Format_Grayscale8)
        qimage = qimage.copy()
        return QPixmap.fromImage(qimage)
