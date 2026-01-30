"""
Horizon generation and manipulation utilities.
"""


import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter, gaussian_filter1d

from core.params import HorizonParams

Point = Tuple[float, float]
Horizon = List[Point]


# ----------------------------
# 1) Halton (2D, bases 2 and 3)
# ----------------------------
def halton_sequence_2d(nbpts: int) -> np.ndarray:
    h = np.empty((nbpts, 2), dtype=float)
    bases = [2, 3]

    for i, b in enumerate(bases):
        for j in range(nbpts):
            f = 1.0
            r = 0.0
            k = j + 1
            while k > 0:
                f /= b
                r += f * (k % b)
                k //= b
            h[j, i] = r
    return h


# ----------------------------
# 2) Plane fitting / dipping plane
# ----------------------------
def _fit_plane_lsq(xyz: np.ndarray) -> Tuple[float, float, float]:
    rows = xyz.shape[0]
    g = np.ones((rows, 3), dtype=float)
    g[:, 0] = xyz[:, 0]
    g[:, 1] = xyz[:, 1]
    z = xyz[:, 2]
    (a, b, c), *_ = np.linalg.lstsq(g, z, rcond=None)
    return float(a), float(b), float(c)


def _create_dipping_plane(azimuth_deg: float, dip_deg: float,
                          grid_shape_x: int, grid_shape_y: int) -> np.ndarray:
    xyz1 = np.array([grid_shape_x / 2.0, grid_shape_y / 2.0, 0.0], dtype=float)

    strike_angle = azimuth_deg + 90.0
    if strike_angle > 360.0:
        strike_angle -= 360.0
    if strike_angle > 180.0:
        strike_angle -= 180.0

    strike_rad = math.radians(strike_angle)
    distance = min(grid_shape_x, grid_shape_y) / 4.0

    x2 = distance * math.cos(strike_rad) + grid_shape_x / 2.0
    y2 = distance * math.sin(strike_rad) + grid_shape_y / 2.0
    xyz2 = np.array([x2, y2, 0.0], dtype=float)

    dip_angle = dip_deg
    if dip_angle > 360.0:
        dip_angle -= 360.0
    if dip_angle > 180.0:
        dip_angle -= 180.0

    dip_rad = math.radians(dip_angle)
    az_rad = math.radians(azimuth_deg)

    dip_elev = distance * math.sin(dip_rad) * math.sqrt(2.0)

    x3 = distance * math.cos(az_rad) + grid_shape_x / 2.0
    y3 = distance * math.sin(az_rad) + grid_shape_y / 2.0
    xyz3 = np.array([x3, y3, dip_elev], dtype=float)

    xyz = np.vstack((xyz1, xyz2, xyz3))
    a, b, c = _fit_plane_lsq(xyz)

    z = np.zeros((grid_shape_x, grid_shape_y), dtype=float)
    for i in range(grid_shape_x):
        for j in range(grid_shape_y):
            z[i, j] = a * i + b * j + c
    return z


# ----------------------------
# 3) Улучшенная генерация base deformation: rng + NaN fill
# ----------------------------
def generate_base_deformation(nx: int, ny: int,
                              deformation_amplitude: float,
                              initial: bool = True,
                              rng: Optional[np.random.Generator] = None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()

    if initial:
        npts = int(rng.uniform(10, 70) + 0.5)  # оставил твой диапазон
    else:
        npts = int(rng.uniform(3, 5) + 0.5)

    halton_points = halton_sequence_2d(npts + 4)

    xx = halton_points[-npts:] * 1.3
    xx -= 0.15

    x = xx[:, 0] * nx
    y = xx[:, 1] * ny

    z = rng.random(npts)
    z = z - z.mean()

    z_std = float(z.std())
    if z_std < 1e-12:
        z_std = 1.0

    if initial:
        z = z * (deformation_amplitude * 2.0 / z_std)
    else:
        z = z * (deformation_amplitude * 0.5 / z_std)

    x = np.hstack((x, [0, 0, nx, nx]))
    y = np.hstack((y, [0, ny, 0, ny]))
    z = np.hstack((z, [0, 0, 0, 0]))

    xi = np.linspace(0, nx, nx)
    yi = np.linspace(0, ny, ny)

    zi = griddata(
        np.column_stack((x, y)),
        z,
        (xi[:, None], yi[None, :]),
        method="cubic"
    )

    # Улучшение: заполнение NaN через nearest
    if np.isnan(zi).any():
        zi2 = griddata(
            np.column_stack((x, y)),
            z,
            (xi[:, None], yi[None, :]),
            method="nearest"
        )
        zi = np.where(np.isnan(zi), zi2, zi)

    return zi


def generate_perlin_like_noise(nx: int, ny: int, octave: float = 2.0,
                               rng: Optional[np.random.Generator] = None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    base_noise = rng.uniform(-1.0, 1.0, (nx, ny))
    return gaussian_filter(base_noise, sigma=octave)


def vertical_weight(y_norm: float) -> float:
    return float(np.exp(-y_norm * 3.0))


def _enforce_no_crossing(Y: np.ndarray, min_gap: float, smooth_sigma: float = 6.0) -> Tuple[np.ndarray, np.ndarray]:
    Yf = Y.copy()
    order = np.argsort(Yf.mean(axis=1))  # top->bottom
    Yf = Yf[order, :]

    for i in range(1, Yf.shape[0]):
        deficit = (Yf[i - 1, :] + min_gap) - Yf[i, :]
        deficit = np.maximum(deficit, 0.0)

        if smooth_sigma and smooth_sigma > 0:
            deficit = gaussian_filter1d(deficit, sigma=smooth_sigma)

        Yf[i, :] += deficit

    return Yf, order


# ----------------------------
# generate_horizons, но с улучшениями (dataclass, rng, NaN-fill, no-crossing)
# ----------------------------
def generate_horizons(params: HorizonParams) -> List[Horizon]:
    # RNG (улучшение)
    if params.seed is None:
        rng = np.random.default_rng()
        random.seed()
    else:
        rng = np.random.default_rng(params.seed)
        random.seed(params.seed)

    W = float(params.W)
    H = float(params.H)
    num_h = int(params.num_horizons)
    nx = int(params.nx)

    min_gap = float(params.min_gap) if params.min_gap is not None else float(params.min_thickness)

    # толщины rng ? 
    thicknesses = rng.gamma(shape=4.0, scale=2.0, size=num_h)
    tmin, tmax = float(thicknesses.min()), float(thicknesses.max())
    if abs(tmax - tmin) < 1e-12:
        thicknesses = np.full(num_h, (params.min_thickness + params.max_thickness) / 2.0, dtype=float)
    else:
        thicknesses = (thicknesses - tmin) / (tmax - tmin)
        thicknesses = thicknesses * (params.max_thickness - params.min_thickness) + params.min_thickness

    base_ys = np.cumsum(thicknesses)
    base_ys = H * base_ys / base_ys[-1]

    # dips & azimuths (через rng)
    dips = (1.0 - rng.power(100, size=num_h)) * 7.0
    azimuths = rng.uniform(0.0, 360.0, size=num_h)

    # X grid
    x = np.linspace(0.0, W, nx)

    # base deformation (улучшение: rng + NaN-fill внутри)
    base_deformation = generate_base_deformation(
        nx=nx,
        ny=num_h,
        deformation_amplitude=params.deformation_amplitude,
        initial=True,
        rng=rng
    )

    # perlin-like noise (через rng)
    perlin_noise = generate_perlin_like_noise(nx, num_h, octave=2.0, rng=rng)
    perlin_noise *= (params.deformation_amplitude * 0.3)

    # dipping planes
    layer_deformations = np.zeros((nx, num_h), dtype=float)
    for layer_idx in range(num_h):
        dip_val = float(dips[layer_idx])
        if dip_val <= 0:
            continue

        dip_plane = _create_dipping_plane(
            azimuth_deg=float(azimuths[layer_idx]),
            dip_deg=dip_val,
            grid_shape_x=nx,
            grid_shape_y=num_h
        )
        dip_plane = dip_plane - float(dip_plane.min())
        maxv = float(dip_plane.max())
        if maxv > 1e-12:
            dip_plane = dip_plane / maxv * dip_val * 10.0

        layer_deformations[:, layer_idx] = dip_plane[:, layer_idx]

    # сначала собираем Y-матрицу (улучшение)
    Y = np.zeros((num_h, nx), dtype=float)

    for layer_idx, y0 in enumerate(base_ys):
        y0 = float(y0)
        y_norm = (y0 / H) if H != 0 else 0.0
        w = vertical_weight(y_norm)

        y_coords = np.ones(nx, dtype=float) * y0

        if layer_idx < base_deformation.shape[1]:
            y_coords += base_deformation[:, layer_idx]
        else:
            y_coords += base_deformation[:, -1]

        if layer_idx < perlin_noise.shape[1]:
            y_coords += perlin_noise[:, layer_idx] * w
        else:
            y_coords += perlin_noise[:, -1] * w

        if layer_idx < layer_deformations.shape[1]:
            y_coords += layer_deformations[:, layer_idx] * w

        Y[layer_idx, :] = y_coords

    # Улучшение: enforce no crossing / min spacing
    Y_fixed, _order = _enforce_no_crossing(Y, min_gap=min_gap)

    # обратно в список горизонтов (в порядке top->bottom после сортировки)
    horizons: List[Horizon] = []
    for i in range(num_h):
        horizons.append([(float(x[j]), float(Y_fixed[i, j])) for j in range(nx)])

    return horizons
