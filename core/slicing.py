from __future__ import annotations
from typing import Tuple, Optional
import numpy as np


def slice_mask(
    labels: np.ndarray,
    *,
    tile: int = 128,
    overlap: float = 0.5,          # доля перекрытия [0 .. <1)
    pad_mode: str = "constant",
    pad_value: int = 0,
    drop_last: bool = False,
    return_coords: bool = False,
    dtype: Optional[np.dtype] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Режет 2D маску (ny, nx) на перекрывающиеся патчи (N, tile, tile).

    overlap:
        0.0  -> без перекрытия
        0.5  -> 50% перекрытие
        0.75 -> 75% перекрытие
    """

    if labels.ndim != 2:
        raise ValueError("labels must be 2D array (ny, nx)")

    if not (0.0 <= overlap < 1.0):
        raise ValueError("overlap must be in [0, 1)")

    tile = int(tile)
    if tile <= 0:
        raise ValueError("tile must be > 0")

    # вычисляем шаг автоматически
    stride = int(round(tile * (1.0 - overlap)))
    stride = max(1, stride)  # защита от overlap≈1

    arr = labels if dtype is None else labels.astype(dtype, copy=False)
    ny, nx = arr.shape

    def start_positions(L: int) -> np.ndarray:
        if drop_last:
            if L < tile:
                return np.array([], dtype=np.int32)
            return np.arange(0, L - tile + 1, stride, dtype=np.int32)

        if L <= tile:
            return np.array([0], dtype=np.int32)

        pos = list(range(0, L - tile + 1, stride))
        last = L - tile
        if pos[-1] != last:
            pos.append(last)
        return np.asarray(pos, dtype=np.int32)

    if not drop_last:
        pad_y = max(0, tile - ny)
        pad_x = max(0, tile - nx)
        arr = np.pad(
            arr,
            ((0, pad_y), (0, pad_x)),
            mode=pad_mode,
            constant_values=(pad_value if pad_mode == "constant" else 0),
        )

    H, W = arr.shape
    ys = start_positions(H)
    xs = start_positions(W)

    if ys.size == 0 or xs.size == 0:
        raise ValueError("No tiles produced")

    patches = np.empty((ys.size * xs.size, tile, tile), dtype=arr.dtype)
    coords = np.empty((ys.size * xs.size, 2), dtype=np.int32) if return_coords else None

    k = 0
    for y0 in ys:
        for x0 in xs:
            patches[k] = arr[y0:y0 + tile, x0:x0 + tile]
            if return_coords:
                coords[k] = (y0, x0)
            k += 1

    return patches, coords
