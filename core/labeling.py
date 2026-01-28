from __future__ import annotations

from typing import List, Tuple, Optional
import numpy as np

Point = Tuple[float, float]
Horizon = List[Point]
Horizons = List[Horizon]


def horizons_to_layer_labels(
    horizons: Horizons,
    W: float,
    H: float,
    nx: int,
    ny: int,
    *,
    top_value: int = 0,
    start_layer_value: int = 1,
    dtype=np.int16,
    nan_policy: str = "interpolate",      # "interpolate" | "nearest" | "raise"
    resample: str = "auto",               # "auto" | "never"
    x_grid: Optional[np.ndarray] = None,  # если None -> linspace(0,W,nx)
) -> np.ndarray:
    """
    Строит 2D маску слоёв (ny, nx) из горизонтов.

    Важно:
    - После разломов горизонты могут содержать дополнительные точки и NaN-разрывы.
      Тогда включается ресемплинг на регулярную сетку X длины nx.

    resample:
      - "auto": если длины горизонтов != nx или X не совпадает с регулярной сеткой — ресемплим
      - "never": требуем строго nx точек 
    """
    n_h = len(horizons)
    if n_h < 1:
        raise ValueError("Нужен хотя бы один горизонт")
    if H == 0:
        raise ValueError("H must be non-zero")

    if x_grid is None:
        x_grid = np.linspace(0.0, float(W), int(nx), dtype=float)
    else:
        x_grid = np.asarray(x_grid, dtype=float)
        if x_grid.shape != (nx,):
            raise ValueError("x_grid must have shape (nx,)")

    # Определяем, нужно ли ресемплить
    need_resample = False
    if resample == "auto":
        for h in horizons:
            if len(h) != nx:
                need_resample = True
                break
        if not need_resample:
            # даже если длина nx, X могут быть не на сетке (редко для base, но возможно)
            xs0 = np.array([p[0] for p in horizons[0]], dtype=float)
            if xs0.shape != (nx,) or not np.allclose(xs0, x_grid, atol=1e-6, rtol=0.0):
                need_resample = True
    elif resample == "never":
        need_resample = False
    else:
        raise ValueError("resample must be 'auto' or 'never'")

    if (not need_resample) and (not all(len(h) == nx for h in horizons)):
        raise ValueError("Каждый горизонт должен иметь ровно nx точек")

    # ---- Собираем Y (n_h, nx) ----
    if need_resample:
        Y = np.vstack([_resample_horizon_to_grid(h, x_grid, nan_policy=nan_policy) for h in horizons]).astype(float)
    else:
        Y = np.array([[p[1] for p in h] for h in horizons], dtype=float)

        if not np.isfinite(Y).all():
            if nan_policy == "raise":
                bad = np.argwhere(~np.isfinite(Y))
                raise ValueError(f"horizons contain NaN/Inf at indices (h,x) like {bad[:5].tolist()} ...")
            Y = _fill_nonfinite_along_x(Y, policy=nan_policy)

    # сортировка по каждой колонке X
    Y_sorted = np.sort(Y, axis=0)

    # Y -> индексы по вертикали (0..ny-1)
    idx = np.rint((Y_sorted / float(H)) * (ny - 1)).astype(int)
    idx = np.clip(idx, 0, ny - 1)

    labels = np.full((ny, nx), int(top_value), dtype=dtype)

    for j in range(nx):
        prev = 0
        for k in range(n_h):
            cut = int(idx[k, j])
            if cut > prev:
                labels[prev:cut, j] = int(start_layer_value + k)
            prev = max(prev, cut)

        if prev < ny:
            labels[prev:ny, j] = int(start_layer_value + n_h)

    # верх всегда top_value
    for j in range(nx):
        top_cut = int(idx[0, j])
        if top_cut > 0:
            labels[0:top_cut, j] = int(top_value)

    return labels


def _resample_horizon_to_grid(h: Horizon, x_grid: np.ndarray, nan_policy: str) -> np.ndarray:
    """
    Приводит горизонт (полилинию) к y(x_grid).
    - Убирает NaN-разрывы
    - Сортирует по X
    - Делает линейную интерполяцию на x_grid
    - За пределами диапазона заполняет краевыми значениями
    """
    xs = np.array([p[0] for p in h], dtype=float)
    ys = np.array([p[1] for p in h], dtype=float)

    good = np.isfinite(xs) & np.isfinite(ys)
    xs = xs[good]
    ys = ys[good]

    if xs.size < 2:
        # деградация: константа (или нули)
        fill = float(ys[0]) if ys.size == 1 else 0.0
        return np.full_like(x_grid, fill, dtype=float)

    # сортировка по X
    order = np.argsort(xs)
    xs = xs[order]
    ys = ys[order]

    # удаляем дубликаты X (np.interp требует монотонный xp)
    # берём последнее значение для каждого X
    uniq_x, idx_last = np.unique(xs, return_index=True)
    # np.unique return_index даёт первые индексы; чтобы взять последние — делаем так:
    # проще: агрегировать через проход
    if uniq_x.size != xs.size:
        x_u = []
        y_u = []
        i = 0
        n = xs.size
        while i < n:
            x0 = xs[i]
            j = i
            while j < n and xs[j] == x0:
                j += 1
            x_u.append(x0)
            y_u.append(ys[j - 1])
            i = j
        xs = np.array(x_u, dtype=float)
        ys = np.array(y_u, dtype=float)

    # интерполяция на сетку
    y_grid = np.interp(x_grid, xs, ys)

    if nan_policy == "raise":
        # np.interp не даёт NaN, поэтому тут ничего
        return y_grid
    elif nan_policy in ("interpolate", "nearest"):
        # тоже ничего не надо: значения уже конечные
        return y_grid
    else:
        raise ValueError("nan_policy must be one of: 'interpolate', 'nearest', 'raise'")


def _fill_nonfinite_along_x(Y: np.ndarray, policy: str) -> np.ndarray:
    Y2 = np.array(Y, dtype=float, copy=True)
    n_h, nx = Y2.shape

    for i in range(n_h):
        row = Y2[i]
        good = np.isfinite(row)

        if good.all():
            continue

        if not good.any():
            Y2[i] = 0.0
            continue

        x = np.arange(nx, dtype=float)
        xp = x[good]
        fp = row[good]

        if policy == "interpolate":
            Y2[i] = np.interp(x, xp, fp)
        elif policy == "nearest":
            ins = np.searchsorted(xp, x)
            ins = np.clip(ins, 0, len(xp) - 1)
            left = np.clip(ins - 1, 0, len(xp) - 1)
            right = ins
            dl = np.abs(x - xp[left])
            dr = np.abs(x - xp[right])
            nearest_idx = np.where(dr < dl, right, left)
            Y2[i] = fp[nearest_idx]
        else:
            raise ValueError("nan_policy must be one of: 'interpolate', 'nearest', 'raise'")

    return Y2
