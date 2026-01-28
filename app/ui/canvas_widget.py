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

    def export_paths(self) -> List[QPainterPath]:
        paths = [item.path() for item in self._committed_paths]
        if len(self._points) >= 2:
            paths.append(self._build_smooth_path(self._points))
        return paths

    def export_mask(self, width: int, height: int, include_current: bool = True) -> np.ndarray:
        image = QImage(width, height, QImage.Format_Grayscale8)
        image.fill(0)
        painter = QPainter(image)
        painter.setRenderHint(QPainter.Antialiasing, True)
        pen = QPen(QColor(255, 255, 255), 2)
        painter.setPen(pen)

        for item in self._committed_paths:
            painter.drawPath(item.path())
        if include_current and len(self._points) >= 2:
            painter.drawPath(self._build_smooth_path(self._points))
        painter.end()

        ptr = image.bits()
        ptr.setsize(height * image.bytesPerLine())
        array = np.frombuffer(ptr, dtype=np.uint8).reshape((height, image.bytesPerLine()))
        return array[:, :width].copy()

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

    def _build_smooth_path(self, points: Iterable[Point]) -> QPainterPath:
        pts = [QPointF(p[0], p[1]) for p in points]
        pts = self._extend_points_to_edges(pts)
        path = QPainterPath(pts[0])
        if len(pts) == 2:
            path.lineTo(pts[1])
            return path

        # Cardinal spline smoothing that passes through every point.
        tension = 0.2
        scale = (1.0 - tension) / 6.0
        for i in range(len(pts) - 1):
            p0 = pts[i - 1] if i > 0 else pts[i]
            p1 = pts[i]
            p2 = pts[i + 1]
            p3 = pts[i + 2] if i + 2 < len(pts) else pts[i + 1]
            control1 = QPointF(p1.x() + (p2.x() - p0.x()) * scale, p1.y() + (p2.y() - p0.y()) * scale)
            control2 = QPointF(p2.x() - (p3.x() - p1.x()) * scale, p2.y() - (p3.y() - p1.y()) * scale)
            path.cubicTo(control1, control2, p2)
        return path

    def _extend_points_to_edges(self, pts: List[QPointF]) -> List[QPointF]:
        if len(pts) < 2:
            return pts
        rect = self._scene.sceneRect()
        width = rect.width()
        height = rect.height()
        if width <= 0 or height <= 0:
            return pts

        first = self._ray_to_rect(pts[0], pts[1], width, height, reverse=True)
        last = self._ray_to_rect(pts[-1], pts[-2], width, height, reverse=True)
        extended = [first] + pts[1:-1] + [last]
        return extended

    @staticmethod
    def _ray_to_rect(start: QPointF, next_pt: QPointF, width: float, height: float, reverse: bool) -> QPointF:
        dx = start.x() - next_pt.x() if reverse else next_pt.x() - start.x()
        dy = start.y() - next_pt.y() if reverse else next_pt.y() - start.y()
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return start

        candidates = []
        if abs(dx) > 1e-6:
            t = (0.0 - start.x()) / dx
            y = start.y() + t * dy
            if t >= 0 and 0.0 <= y <= height - 1:
                candidates.append((t, QPointF(0.0, y)))
            t = ((width - 1) - start.x()) / dx
            y = start.y() + t * dy
            if t >= 0 and 0.0 <= y <= height - 1:
                candidates.append((t, QPointF(width - 1, y)))

        if abs(dy) > 1e-6:
            t = (0.0 - start.y()) / dy
            x = start.x() + t * dx
            if t >= 0 and 0.0 <= x <= width - 1:
                candidates.append((t, QPointF(x, 0.0)))
            t = ((height - 1) - start.y()) / dy
            x = start.x() + t * dx
            if t >= 0 and 0.0 <= x <= width - 1:
                candidates.append((t, QPointF(x, height - 1)))

        if not candidates:
            return start
        _, point = min(candidates, key=lambda item: item[0])
        return point

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
