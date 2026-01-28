"""
Horizon generation and manipulation utilities.
"""


import random
import numpy as np
from typing import List, Tuple
from scipy.interpolate import CubicSpline, griddata
import math

Point = Tuple[float, float]
Horizon = List[Point]

# Halton последовательность для равномерного случайного распределения точек (случай 2D)
def _halton_sequence(dim: int, nbpts: int):
    h = np.empty(nbpts * dim)
    h.fill(np.nan)
    p = np.empty(nbpts)
    p.fill(np.nan)
    p1 = [2, 3]
    lognbpts = math.log(nbpts + 1)
    
    for i in range(dim):
        b = p1[i]
        n = int(math.ceil(lognbpts / math.log(b)))
        for t in range(n):
            p[t] = pow(b, -(t + 1))

        for j in range(nbpts):
            d = j + 1
            sum_ = math.fmod(d, b) * p[0]
            for t in range(1, n):
                d = math.floor(d / b)
                sum_ += math.fmod(d, b) * p[t]
            h[j * dim + i] = sum_
    return h.reshape(nbpts, dim)


def _fit_plane_lsq(xyz):
    rows = xyz.shape[0]
    g = np.ones((rows, 3))
    g[:, 0] = xyz[:, 0]  # X
    g[:, 1] = xyz[:, 1]  # Y
    z = xyz[:, 2]
    
    (a, b, c), _, _, _ = np.linalg.lstsq(g, z, rcond=None)
    return a, b, c


def _create_dipping_plane(azimuth: float, dip: float, grid_shape_x: int, grid_shape_y: int):
    
    # Создаем три точки для определения плоскости
    # Центральная точка на высоте 0
    xyz1 = np.array([grid_shape_x / 2.0, grid_shape_y / 2.0, 0.0])
    
    # Точка в направлении простирания
    strike_angle = azimuth + 90.0
    if strike_angle > 360.0:
        strike_angle -= 360.0
    if strike_angle > 180.0:
        strike_angle -= 180.0
    
    strike_angle = math.radians(strike_angle)
    distance = min(grid_shape_x, grid_shape_y) / 4.0
    x = distance * math.cos(strike_angle) + grid_shape_x / 2.0
    y = distance * math.sin(strike_angle) + grid_shape_y / 2.0
    xyz2 = np.array([x, y, 0.0])
    
    # Точка в направлении максимального падения
    dip_angle = dip * 1.0
    if dip_angle > 360.0:
        dip_angle -= 360.0
    if dip_angle > 180.0:
        dip_angle -= 180.0
    
    dip_angle = math.radians(dip_angle)
    strike_angle = math.radians(azimuth)
    dip_elev = distance * math.sin(dip_angle) * math.sqrt(2.0)
    
    x = distance * math.cos(strike_angle) + grid_shape_x / 2.0
    y = distance * math.sin(strike_angle) + grid_shape_y / 2.0
    xyz3 = np.array([x, y, dip_elev])
    
    # Совмещаем точки
    xyz = np.vstack((xyz1, xyz2, xyz3))
    
    # Подгоняем плоскость
    a, b, c = _fit_plane_lsq(xyz)
    
    # Вычисляем высоты для всей сетки
    z = np.zeros((grid_shape_x, grid_shape_y), dtype=float)
    for i in range(grid_shape_x):
        for j in range(grid_shape_y):
            z[i, j] = a * i + b * j + c
    
    return z


def generate_horizons(
    W: float,
    H: float,
    num_horizons: int,
    nx: int,
    min_thickness: float,
    max_thickness: float,
    deformation_amplitude: float,
    seed: int = None
) -> List[Horizon]:
    
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    # ------------------------------
    # толщины
    # ------------------------------
    # Используем гамма-распределение
    thicknesses = np.random.gamma(4.0, 2, size=num_horizons)
    # Нормализуем к заданному диапазону толщин
    thickness_min, thickness_max = thicknesses.min(), thicknesses.max()
    thicknesses = (thicknesses - thickness_min) / (thickness_max - thickness_min)
    thicknesses = thicknesses * (max_thickness - min_thickness) + min_thickness
    
    base_ys = np.cumsum(thicknesses)
    base_ys = H * base_ys / base_ys[-1] 
    
    # -------------------------
    # наклон горизонтов
    # --------------------------
    # Генерация dips (степенное распределение)
    dips = (1.0 - np.random.power(100, num_horizons)) * 7.0 
    
    # Генерация azimuths
    azimuths = np.random.uniform(0.0, 360.0, size=num_horizons)
    
    # -------------------------------
    # Генерация базового горизонта
    # -------------------------------
    def generate_base_horizon(grid_size_x: int, grid_size_y: int, initial: bool = True):
        # Определяем количество случайных точек 
        number_random_points = int(np.random.uniform(3, 5) + 0.5)
        if initial:
            number_random_points = int(np.random.uniform(25, 100) + 0.5)
        
        # Используем Halton последовательность для равномерного распределения точек
        halton_points = _halton_sequence(2, number_random_points + 4)
        
        # Масштабируем точки к размеру сетки
        xx = halton_points[-number_random_points:] * 1.3
        xx -= 0.15
        
        x = xx[:, 0] * grid_size_x
        y = xx[:, 1] * grid_size_y
        
        # Создаем случайные высоты
        z = np.random.rand(number_random_points)
        z -= z.mean()
        if initial:
            z *= deformation_amplitude * 2.0 / z.std()
        else:
            z *= deformation_amplitude * 0.5 / z.std()
        
        # Добавляем угловые точки с нулевой высотой для стабильности
        x = np.hstack((x, [0, 0, grid_size_x, grid_size_x]))
        y = np.hstack((y, [0, grid_size_y, 0, grid_size_y]))
        z = np.hstack((z, [0, 0, 0, 0]))
        
        # Создаем регулярную сетку
        xi = np.linspace(0, grid_size_x, grid_size_x)
        yi = np.linspace(0, grid_size_y, grid_size_y)
        
        # Интерполяция (кубическая)
        zi = griddata(
            np.column_stack((x, y)),
            z,
            (xi[:, np.newaxis], yi[np.newaxis, :]),
            method='cubic'
        )
        
        return zi
    
    # ------------------------------------------------------------
    #  Генерация Perlin-like шума для естественных деформаций
    # ------------------------------------------------------------
    def generate_perlin_noise(size_x: int, size_y: int, octave: int = 2):
        # Создаем базовый шум
        base_noise = np.random.uniform(-1, 1, (size_x, size_y))
        
        # Применяем сглаживание в зависимости от octave
        from scipy.ndimage import gaussian_filter
        noise = gaussian_filter(base_noise, sigma=octave)
        
        return noise
    
    # ------------------------------------------------------------
    # 5. Горизонтальная сетка
    # ------------------------------------------------------------
    x = np.linspace(0, W, nx)
    
    # Генерация базового деформационного поля
    base_deformation = generate_base_horizon(nx, num_horizons, initial=True)
    
    # Добавление Perlin-шума
    perlin_noise = generate_perlin_noise(nx, num_horizons, octave=2)
    perlin_noise = perlin_noise * (deformation_amplitude * 0.3)
    
    # ------------------------------------------------------------
    # 6. Добавление наклонов для каждого слоя
    # ------------------------------------------------------------
    layer_deformations = np.zeros((nx, num_horizons))
    for layer_idx in range(num_horizons):
        if dips[layer_idx] > 0:  # Добавляем наклон только если dip > 0
            dip_plane = _create_dipping_plane(
                azimuths[layer_idx],
                dips[layer_idx],
                nx,
                num_horizons
            )
            # Нормализуем и масштабируем
            dip_plane = dip_plane - dip_plane.min()
            if dip_plane.max() > 0:
                dip_plane = dip_plane / dip_plane.max() * dips[layer_idx] * 10
            layer_deformations[:, layer_idx] = dip_plane[:, layer_idx] if layer_idx < nx else dip_plane[:, 0]
    
    # ---------------------------
    # 7. Вертикальное затухание
    # ---------------------------
    def vertical_weight(y_norm):
        return np.exp(-y_norm * 3.0)
    
    # ---------------------------
    # 8. Формирование горизонтов
    # ----------------------------
    horizons: List[Horizon] = []
    
    for layer_idx, y0 in enumerate(base_ys):
        y_norm = y0 / H
        
        # Базовые Y-координаты
        y_coords = np.ones(nx) * y0
        
        # Добавляем базовую деформацию
        if layer_idx < base_deformation.shape[1]:
            y_coords += base_deformation[:, layer_idx]
        else:
            y_coords += base_deformation[:, -1]
        
        # Добавляем Perlin-шум с вертикальным затуханием
        if layer_idx < perlin_noise.shape[1]:
            y_coords += perlin_noise[:, layer_idx] * vertical_weight(y_norm)
        else:
            y_coords += perlin_noise[:, -1] * vertical_weight(y_norm)
        
        # Добавляем слоевой наклон
        if dips[layer_idx] > 0 and layer_idx < layer_deformations.shape[1]:
            y_coords += layer_deformations[:, layer_idx] * vertical_weight(y_norm)
        
        # Создаем горизонт как список точек
        horizon_line = []
        for i, xi in enumerate(x):
            horizon_line.append((xi, y_coords[i]))
        
        horizons.append(horizon_line)
        
    return horizons