import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def visualize_npy(data_or_path, title="Data", axis_order="auto"):
    """
    Универсальная визуализация npy массива или numpy array.
    
    Параметры:
    - data_or_path: str/Path или numpy.ndarray
    - title: заголовок окна
    - axis_order: "auto" (по умолчанию), можно задать (0,1,2) для 3D кубов
    """
    # Загружаем, если это путь
    if isinstance(data_or_path, (str, Path)):
        data = np.load(data_or_path)
    else:
        data = data_or_path

    print("Shape:", data.shape, "dtype:", data.dtype)

    # Если 3D куб
    if data.ndim == 3:
        # проверяем канальный вид (C, H, W)
        if data.shape[0] <= 20:  # предполагаем, что это каналы
            C, H, W = data.shape
            data_map = np.argmax(data, axis=0)
            plt.imshow(data_map, cmap="tab20", aspect="auto")
            plt.title(f"{title} (C={C}, visualized as class map)")
        else:  # обычный куб (I, X, Z)
            I, X, Z = data.shape
            mid_slice = I // 2
            plt.imshow(data[mid_slice, :, :], cmap="seismic", aspect="auto")
            plt.title(f"{title} (slice {mid_slice})")
    elif data.ndim == 2:
        plt.imshow(data, cmap="seismic", aspect="auto")
        plt.title(title)
    else:
        print("Массив имеет больше 3 измерений, визуализируем первые два:")
        plt.imshow(data[0, :, :], cmap="seismic", aspect="auto")
        plt.title(title + " (first slice)")

    plt.axis("off")
    plt.show()


visualize_npy("generated_seismic.npy")

#visualize_npy("real_inline_25.npy")