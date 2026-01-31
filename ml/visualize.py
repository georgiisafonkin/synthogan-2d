import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def visualize_npy(data_or_path, title="Data", axis_order="auto", mask_mode="auto"):
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
        # (H, W, 1) -> обычная 2D сейсмика
        if data.shape[-1] == 1:
            plt.imshow(data[:, :, 0], cmap="seismic", aspect="auto")
            plt.title(title)
        # канальный вид (H, W, C)
        elif data.shape[-1] <= 20:
            H, W, C = data.shape
            data_map = np.argmax(data, axis=-1)
            if mask_mode == "auto" and np.isin(data, [0.0, 1.0]).all():
                plt.imshow(data_map, cmap="tab20", aspect="auto")
                plt.title(f"{title} (mask C={C})")
            else:
                plt.imshow(data_map, cmap="tab20", aspect="auto")
                plt.title(f"{title} (C={C}, visualized as class map)")
        # канальный вид (C, H, W)
        elif data.shape[0] <= 20:
            C, H, W = data.shape
            data_map = np.argmax(data, axis=0)
            if mask_mode == "auto" and np.isin(data, [0.0, 1.0]).all():
                plt.imshow(data_map, cmap="tab20", aspect="auto")
                plt.title(f"{title} (mask C={C})")
            else:
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


visualize_npy("generated_seismic_raw_rotated.npy")
visualize_npy("generated_seismic_raw.npy")
visualize_npy("generated_seismic_scales_rotated.npy")
visualize_npy("generated_seismic_scaled.npy")
#visualize_npy("test_masks/mask_inline_25.npy", title="Mask")
