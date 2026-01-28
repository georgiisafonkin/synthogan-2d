from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional, Union, Literal
import numpy as np

Point = Tuple[float, float]
Horizon = List[Point]
Horizons = List[Horizon]
Side = Literal["left", "right"]


# ===========================
# 1) Параметры ОДНОГО разлома 
# ===========================
@dataclass(frozen=True)
class FaultParams:
    center: Tuple[float, float]      # (x,y) центр
    length: float                    # длина отрезка
    angle_deg: float                
    uplift_side: Side                # какую сторону смещаем
    throw: float                     # амплитуда вертикального смещения (по Y)
    sigma_cross: Optional[float] = None   # ширина затухания поперёк (если None -> 0.12*length)
    along_power: float = 1.0              # степень затухания вдоль (1..3 типично)


@dataclass(frozen=True)
class FaultSpec:
    params: FaultParams
    angle_rad: float
    ux: float
    uy: float
    nx: float
    ny: float
    half_len: float
    p1: Point
    p2: Point


# ============================================================
# 2) Параметры генерации НАБОРА разломов (ranges, количества, защиты)
# ============================================================
@dataclass(frozen=True)
class FaultGenParams:
    num_faults: int

    # диапазоны для центров (можно задавать в долях W/H)
    x_range: Tuple[float, float]     # например (0.15*W, 0.85*W)
    y_range: Tuple[float, float]     # например (0.15*H, 0.85*H)

    # диапазоны для геометрии
    length_range: Tuple[float, float]    # например (0.25*H, 1.00*H)
    angle_range_deg: Tuple[float, float] # например (70, 90) для крутых

    # диапазоны для смещения/затуханий
    throw_range: Tuple[float, float]     # например (20, 80)
    sigma_cross: Optional[float] = None  # если None -> 0.12*length
    along_power_range: Tuple[float, float] = (1.0, 2.5)

    # какая сторона смещается (можно фиксировать или случайно)
    uplift_side: Union[Side, Literal["random"]] = "random"

    # защиты от пересечений и "слишком близко"
    min_fault_separation: float = 30.0
    max_tries_per_fault: int = 2000


# ===============
# 3) Геометрия 
# ================
def _basis(angle_rad: float) -> Tuple[float, float, float, float]:
    ux = float(np.cos(angle_rad))
    uy = float(np.sin(angle_rad))
    nx = -uy
    ny = ux
    return ux, uy, nx, ny


def _build_spec(fp: FaultParams) -> FaultSpec:
    cx, cy = float(fp.center[0]), float(fp.center[1])
    angle_rad = float(np.deg2rad(fp.angle_deg))
    ux, uy, nx, ny = _basis(angle_rad)
    half_len = 0.5 * float(fp.length)

    p1 = (cx - half_len * ux, cy - half_len * uy)
    p2 = (cx + half_len * ux, cy + half_len * uy)

    return FaultSpec(
        params=fp,
        angle_rad=angle_rad,
        ux=ux, uy=uy, nx=nx, ny=ny,
        half_len=half_len,
        p1=(float(p1[0]), float(p1[1])),
        p2=(float(p2[0]), float(p2[1])),
    )


def fault_lines_from_specs(specs: List[FaultSpec]) -> List[Tuple[Point, Point]]:
    return [(s.p1, s.p2) for s in specs]


def _ccw(A, B, C) -> bool:
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


def _segments_intersect(p1, p2, p3, p4) -> bool:
    return (_ccw(p1, p3, p4) != _ccw(p2, p3, p4)) and (_ccw(p1, p2, p3) != _ccw(p1, p2, p4))


def _dist_point_to_segment(P, A, B) -> float:
    P = np.asarray(P, float)
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    AB = B - A
    denom = float(np.dot(AB, AB))
    if denom < 1e-12:
        return float(np.linalg.norm(P - A))
    t = float(np.dot(P - A, AB) / denom)
    t = max(0.0, min(1.0, t))
    Q = A + t * AB
    return float(np.linalg.norm(P - Q))


def _clip_segment_to_box(p1: Point, p2: Point, W: float, H: float) -> Optional[Tuple[Point, Point]]:
    x0, y0 = float(p1[0]), float(p1[1])
    x1, y1 = float(p2[0]), float(p2[1])
    dx = x1 - x0
    dy = y1 - y0
    eps = 1e-12
    ts: List[float] = []

    if abs(dx) > eps:
        for Xb in (0.0, W):
            t = (Xb - x0) / dx
            y = y0 + t * dy
            if 0.0 <= t <= 1.0 and 0.0 <= y <= H:
                ts.append(float(t))
    if abs(dy) > eps:
        for Yb in (0.0, H):
            t = (Yb - y0) / dy
            x = x0 + t * dx
            if 0.0 <= t <= 1.0 and 0.0 <= x <= W:
                ts.append(float(t))

    if 0.0 <= x0 <= W and 0.0 <= y0 <= H:
        ts.append(0.0)
    if 0.0 <= x1 <= W and 0.0 <= y1 <= H:
        ts.append(1.0)

    if len(ts) < 2:
        return None

    tmin, tmax = min(ts), max(ts)
    q1 = (x0 + tmin * dx, y0 + tmin * dy)
    q2 = (x0 + tmax * dx, y0 + tmax * dy)
    return (float(q1[0]), float(q1[1])), (float(q2[0]), float(q2[1]))


def _too_close(seg_new: Tuple[Point, Point], seg_old: Tuple[Point, Point], min_sep: float) -> bool:
    a1, a2 = seg_new
    b1, b2 = seg_old

    if _segments_intersect(a1, a2, b1, b2):
        return True

    d = min(
        _dist_point_to_segment(a1, b1, b2),
        _dist_point_to_segment(a2, b1, b2),
        _dist_point_to_segment(b1, a1, a2),
        _dist_point_to_segment(b2, a1, a2),
    )
    return d < float(min_sep)


# ============================================================
# 4) Генерация набора разломов с защитами (не пересекаться / min distance)
# ============================================================
def generate_faults_random(
    W: float,
    H: float,
    gen: FaultGenParams,
    seed: Optional[int] = None
) -> List[FaultParams]:
    rng = np.random.default_rng(seed)

    faults: List[FaultParams] = []
    segs: List[Tuple[Point, Point]] = []

    for k in range(int(gen.num_faults)):
        placed = False

        for _try in range(int(gen.max_tries_per_fault)):
            cx = float(rng.uniform(*gen.x_range))
            cy = float(rng.uniform(*gen.y_range))

            length = float(rng.uniform(*gen.length_range))
            angle_deg = float(rng.uniform(*gen.angle_range_deg))

            throw = float(rng.uniform(*gen.throw_range))
            along_power = float(rng.uniform(*gen.along_power_range))

            if gen.uplift_side == "random":
                uplift_side: Side = "left" if rng.random() < 0.5 else "right"
            else:
                uplift_side = gen.uplift_side  # type: ignore[assignment]

            fp = FaultParams(
                center=(cx, cy),
                length=length,
                angle_deg=angle_deg,
                uplift_side=uplift_side,
                throw=throw,
                sigma_cross=gen.sigma_cross,
                along_power=along_power,
            )

            spec = _build_spec(fp)
            clipped = _clip_segment_to_box(spec.p1, spec.p2, float(W), float(H))
            if clipped is None:
                continue

            bad = False
            for old in segs:
                if _too_close(clipped, old, float(gen.min_fault_separation)):
                    bad = True
                    break

            if bad:
                continue

            faults.append(fp)
            segs.append(clipped)
            placed = True
            break

        if not placed:
            # если не удалось поставить очередной разлом — прекращаем, но не падаем
            break

    return faults


# ============================================================
# 5) Применение разломов (разрыв линий + enforce half-plane)
# ============================================================
def _side_mask(d: float, uplift_side: Side) -> float:
    if uplift_side == "left":
        return 1.0 if d > 0.0 else 0.0
    return 1.0 if d < 0.0 else 0.0


def _length_taper_pow(s: float, half_len: float, power: float) -> float:
    if half_len <= 0.0:
        return 0.0
    a = abs(s) / half_len
    if a >= 1.0:
        return 0.0
    q = 1.0 - a
    q = float(np.clip(q, 0.0, 1.0))
    # "косинусный" купол + степень
    base = 0.5 * (1.0 - float(np.cos(np.pi * q)))
    return float(base ** max(1e-6, float(power)))


def _cross_taper(d: float, sigma: float) -> float:
    sigma = max(1e-9, float(sigma))
    r = abs(d) / sigma
    return float(np.exp(-0.5 * r * r))


def _project_to_fault_line(x: float, y: float, cx: float, cy: float, nx: float, ny: float) -> Tuple[float, float]:
    d = (x - cx) * nx + (y - cy) * ny
    return (x - d * nx, y - d * ny)


def _enforce_halfplane(
    x: float,
    y: float,
    cx: float,
    cy: float,
    nx: float,
    ny: float,
    uplift_side: Side,
    eps: float,
) -> Tuple[float, float]:
    d = (x - cx) * nx + (y - cy) * ny
    want_left = (uplift_side == "left")
    ok = (d > 0.0) if want_left else (d < 0.0)
    if ok:
        return x, y

    xp, yp = _project_to_fault_line(x, y, cx, cy, nx, ny)
    sign = +1.0 if want_left else -1.0
    return (xp + sign * eps * nx, yp + sign * eps * ny)


def _segment_intersection_with_fault(
    p0: Point,
    p1: Point,
    cx: float,
    cy: float,
    ux: float,
    uy: float,
    nx: float,
    ny: float,
    half_len: float,
) -> Optional[Tuple[float, float]]:
    x0, y0 = float(p0[0]), float(p0[1])
    x1, y1 = float(p1[0]), float(p1[1])

    d0 = (x0 - cx) * nx + (y0 - cy) * ny
    d1 = (x1 - cx) * nx + (y1 - cy) * ny

    if not (np.isfinite(d0) and np.isfinite(d1)):
        return None
    if d0 * d1 > 0.0:
        return None
    if abs(d0) < 1e-12 and abs(d1) < 1e-12:
        return None

    denom = (d0 - d1)
    if abs(denom) < 1e-18:
        return None

    t = d0 / denom
    if t < 0.0 or t > 1.0:
        return None

    xi = x0 + t * (x1 - x0)
    yi = y0 + t * (y1 - y0)

    si = (xi - cx) * ux + (yi - cy) * uy
    if abs(si) > half_len:
        return None

    return float(xi), float(yi)


def apply_faults(
    horizons: Horizons,
    W: float,
    H: float,
    faults: Union[FaultParams, List[FaultParams]],
    return_specs: bool = False,
) -> Union[Horizons, Tuple[Horizons, List[FaultSpec]]]:
    if not horizons:
        return ([], []) if return_specs else []

    faults_in: List[FaultParams] = faults if isinstance(faults, list) else [faults]
    specs: List[FaultSpec] = [_build_spec(fp) for fp in faults_in]

    out: Horizons = [[(float(x), float(y)) for (x, y) in h] for h in horizons]

    for spec in specs:
        fp = spec.params
        cx, cy = float(fp.center[0]), float(fp.center[1])
        ux, uy, nx, ny = spec.ux, spec.uy, spec.nx, spec.ny
        half_len = spec.half_len

        sigma_cross = float(fp.sigma_cross) if fp.sigma_cross is not None else max(1e-6, 0.12 * float(fp.length))
        eps = max(1e-6, 1e-4 * float(fp.length))

        for hi, horizon in enumerate(out):
            if len(horizon) < 2:
                continue

            new_h: Horizon = []
            prev = horizon[0]

            # первая точка
            x0, y0 = float(prev[0]), float(prev[1])
            if np.isfinite(x0) and np.isfinite(y0):
                s0 = (x0 - cx) * ux + (y0 - cy) * uy
                d0 = (x0 - cx) * nx + (y0 - cy) * ny
                w0 = _length_taper_pow(s0, half_len, fp.along_power)

                if w0 > 0.0 and _side_mask(d0, fp.uplift_side) == 1.0:
                    dy0 = float(fp.throw) * w0 * _cross_taper(d0, sigma_cross)
                    x0m, y0m = _enforce_halfplane(x0, y0 + dy0, cx, cy, nx, ny, fp.uplift_side, eps)
                    new_h.append((float(x0m), float(y0m)))
                else:
                    new_h.append((float(x0), float(y0)))
            else:
                new_h.append(prev)

            # остальные точки
            for i in range(1, len(horizon)):
                cur = horizon[i]
                x1, y1 = float(cur[0]), float(cur[1])

                if not (np.isfinite(prev[0]) and np.isfinite(prev[1]) and np.isfinite(x1) and np.isfinite(y1)):
                    new_h.append(cur)
                    prev = cur
                    continue

                inter = _segment_intersection_with_fault(prev, cur, cx, cy, ux, uy, nx, ny, half_len)
                if inter is not None:
                    xi, yi = inter
                    si = (xi - cx) * ux + (yi - cy) * uy
                    wI = _length_taper_pow(si, half_len, fp.along_power)
                    shift_on_fault = float(fp.throw) * wI  # на линии поперечный taper = 1

                    d_prev = (float(prev[0]) - cx) * nx + (float(prev[1]) - cy) * ny
                    d_cur = (x1 - cx) * nx + (y1 - cy) * ny
                    prev_uplift = _side_mask(d_prev, fp.uplift_side)
                    cur_uplift = _side_mask(d_cur, fp.uplift_side)

                    p_base = (float(xi), float(yi))

                    xs, ys = _project_to_fault_line(float(xi), float(yi + shift_on_fault), cx, cy, nx, ny)
                    p_shift = (float(xs), float(ys))

                    if prev_uplift == 1.0 and cur_uplift == 0.0:
                        new_h.append(p_shift)
                        new_h.append((float("nan"), float("nan")))
                        new_h.append(p_base)
                    elif prev_uplift == 0.0 and cur_uplift == 1.0:
                        new_h.append(p_base)
                        new_h.append((float("nan"), float("nan")))
                        new_h.append(p_shift)
                    else:
                        new_h.append(p_base)
                        new_h.append((float("nan"), float("nan")))
                        new_h.append(p_shift)

                s1 = (x1 - cx) * ux + (y1 - cy) * uy
                d1 = (x1 - cx) * nx + (y1 - cy) * ny
                w1 = _length_taper_pow(s1, half_len, fp.along_power)

                if w1 > 0.0 and _side_mask(d1, fp.uplift_side) == 1.0:
                    dy1 = float(fp.throw) * w1 * _cross_taper(d1, sigma_cross)
                    x1m, y1m = _enforce_halfplane(x1, y1 + dy1, cx, cy, nx, ny, fp.uplift_side, eps)
                    new_h.append((float(x1m), float(y1m)))
                else:
                    new_h.append((float(x1), float(y1)))

                prev = cur

            out[hi] = new_h

    return (out, specs) if return_specs else out

def generate_and_apply_faults(
    horizons: Horizons,
    W: float,
    H: float,
    gen: FaultGenParams,
    seed: Optional[int] = None,
    return_specs: bool = False,
) -> Union[Horizons, Tuple[Horizons, List[FaultSpec]]]:
    faults = generate_faults_random(W=W, H=H, gen=gen, seed=seed)
    return apply_faults(horizons=horizons, W=W, H=H, faults=faults, return_specs=return_specs)
