import os
import numpy as np
import tensorflow as tf

# ============================================================
#                ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================

def blend_window_2d(size, overlap, eps=1e-3):
    """2D blend window with non-zero edges to avoid seam artifacts."""
    if overlap <= 0:
        w = np.ones(size, dtype=np.float32)
    else:
        if overlap * 2 > size:
            raise ValueError("overlap must be <= half of patch_size")
        ramp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, overlap, dtype=np.float32))
        ramp = np.clip(ramp, eps, 1.0)
        w = np.ones(size, dtype=np.float32)
        w[:overlap] = ramp
        w[-overlap:] = ramp[::-1]
    return np.outer(w, w).astype(np.float32)[..., None]


def pad_for_tiling(x, patch_size, overlap, mode="reflect"):
    """Pad H,W so that patch grid is uniform for the given stride."""
    if x.ndim != 3:
        raise ValueError("Expected input shape (H, W, C)")
    stride = patch_size - overlap
    if stride <= 0:
        raise ValueError("overlap must be smaller than patch_size")

    h, w = x.shape[:2]

    def target_size(n):
        if n <= patch_size:
            return patch_size
        steps = int(np.ceil((n - patch_size) / stride)) + 1
        return (steps - 1) * stride + patch_size

    nh = target_size(h)
    nw = target_size(w)

    ph0 = (nh - h) // 2
    ph1 = nh - h - ph0
    pw0 = (nw - w) // 2
    pw1 = nw - w - pw0

    x_pad = np.pad(
        x,
        ((ph0, ph1), (pw0, pw1), (0, 0)),
        mode=mode
    )

    return x_pad, (ph0, ph1, pw0, pw1)


def infer_full_with_patches(
    generator,
    full_mask,
    patch_size=128,
    overlap=64,
    batch_size=8
):
    """
    generator : tf.keras.Model
    full_mask : np.ndarray (H, W, C)
    return    : np.ndarray (H, W, 1)
    """

    stride = patch_size - overlap
    window = blend_window_2d(patch_size, overlap)

    # --- pad input ---
    full_mask, pads = pad_for_tiling(full_mask, patch_size, overlap, mode="reflect")
    ph0, ph1, pw0, pw1 = pads
    Hp, Wp = full_mask.shape[:2]

    out_sum = np.zeros((Hp, Wp, 1), dtype=np.float32)
    out_wgt = np.zeros((Hp, Wp, 1), dtype=np.float32)

    patches = []
    coords  = []

    ys = list(range(0, Hp - patch_size + 1, stride))
    xs = list(range(0, Wp - patch_size + 1, stride))

    for y in ys:
        for x in xs:
            patch = full_mask[y:y+patch_size, x:x+patch_size]
            patches.append(patch)
            coords.append((y, x))

            if len(patches) == batch_size:
                batch = np.stack(patches, axis=0)
                preds = generator(batch, training=False).numpy()

                for (yy, xx), p in zip(coords, preds):
                    out_sum[yy:yy+patch_size, xx:xx+patch_size] += p * window
                    out_wgt[yy:yy+patch_size, xx:xx+patch_size] += window

                patches.clear()
                coords.clear()

    # --- last batch ---
    if patches:
        batch = np.stack(patches, axis=0)
        preds = generator(batch, training=False).numpy()

        for (yy, xx), p in zip(coords, preds):
            out_sum[yy:yy+patch_size, xx:xx+patch_size] += p * window
            out_wgt[yy:yy+patch_size, xx:xx+patch_size] += window

    out = out_sum / (out_wgt + 1e-8)

    # --- remove padding ---
    out = out[ph0:Hp-ph1, pw0:Wp-pw1]

    return out


# ============================================================
#                        MAIN
# ============================================================

def main():
    # --------------------------------------------------------
    # 1. ЗАГРУЗКА ГЕНЕРАТОРА
    # --------------------------------------------------------
    generator_path = "generator_128x128.keras"

    generator = tf.keras.models.load_model(
        generator_path,
        compile=False
    )

    print("Generator loaded")
    print("Input shape :", generator.input_shape)
    print("Output shape:", generator.output_shape)

    if generator.input_shape[1] != generator.input_shape[2]:
        raise ValueError("Generator input must be square for patch inference")
    patch_size = int(generator.input_shape[1])
    expected_channels = int(generator.input_shape[-1])

    # --------------------------------------------------------
    # 2. ЗАГРУЗКА И ПОВОРОТ МАСКИ
    # --------------------------------------------------------
    mask_path = os.path.join("test_masks", "mask_inline_25.npy")

    full_mask = np.load(mask_path).astype(np.float32)
    rotate_k = -1  # 90 deg clockwise
    full_mask = np.rot90(full_mask, k=rotate_k, axes=(0, 1))
    print("Mask shape after rotation:", full_mask.shape)
    if full_mask.shape[-1] != expected_channels:
        raise ValueError(
            f"Mask channels ({full_mask.shape[-1]}) != generator channels ({expected_channels})"
        )

    # --------------------------------------------------------
    # 3. ИНФЕРЕНС
    # --------------------------------------------------------
    def run_infer_and_save(mask, suffix):
        generated_rotated = infer_full_with_patches(
            generator=generator,
            full_mask=mask,
            patch_size=patch_size,
            overlap=patch_size // 2,
            batch_size=8
        )

        generated = np.rot90(generated_rotated, k=-rotate_k, axes=(0, 1))
        print("Generated seismic shape:", generated.shape)

        out_path = f"generated_seismic_{suffix}.npy"
        out_rot_path = f"generated_seismic_{suffix}_rotated.npy"
        np.save(out_path, generated)
        np.save(out_rot_path, generated_rotated)

        print("Saved to:", out_path)
        print("Saved rotated to:", out_rot_path)
        print(
            f"Stats ({suffix}):",
            "min =", generated.min(),
            "max =", generated.max(),
            "mean =", generated.mean(),
            "std  =", generated.std()
        )

    # --------------------------------------------------------
    # 4. ИНФЕРЕНС + СОХРАНЕНИЕ
    # --------------------------------------------------------
    run_infer_and_save(full_mask, "raw")

    # Часто входы для GAN нормализуют в [-1, 1]. Попробуем и так.
    if full_mask.min() >= 0.0 and full_mask.max() <= 1.0:
        full_mask_scaled = full_mask * 2.0 - 1.0
        run_infer_and_save(full_mask_scaled, "scaled")


# ============================================================
#                        RUN
# ============================================================

if __name__ == "__main__":
    main()
