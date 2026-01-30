import numpy as np
import tensorflow as tf

# ============================================================
#                ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================

def hann_window_2d(size):
    """2D Hann window for smooth patch blending"""
    w = np.hanning(size)
    return np.outer(w, w).astype(np.float32)[..., None]


def pad_to_multiple(x, mult=16, mode="reflect"):
    """Pad H,W to be divisible by mult"""
    h, w = x.shape[:2]
    nh = int(np.ceil(h / mult) * mult)
    nw = int(np.ceil(w / mult) * mult)

    ph0 = (nh - h) // 2
    ph1 = nh - h - ph0
    pw0 = (nw - w) // 2
    pw1 = nw - w - pw0

    if x.ndim != 3:
        raise ValueError("Expected input shape (H, W, C)")

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
    overlap=32,
    batch_size=8
):
    """
    generator : tf.keras.Model
    full_mask : np.ndarray (H, W, C)
    return    : np.ndarray (H, W, 1)
    """

    stride = patch_size - overlap
    window = hann_window_2d(patch_size)

    # --- pad input ---
    full_mask, pads = pad_to_multiple(full_mask, mult=16)
    ph0, ph1, pw0, pw1 = pads
    Hp, Wp = full_mask.shape[:2]

    out_sum = np.zeros((Hp, Wp, 1), dtype=np.float32)
    out_wgt = np.zeros((Hp, Wp, 1), dtype=np.float32)

    patches = []
    coords  = []

    ys = list(range(0, Hp - patch_size + 1, stride))
    xs = list(range(0, Wp - patch_size + 1, stride))

    if ys[-1] != Hp - patch_size:
        ys.append(Hp - patch_size)
    if xs[-1] != Wp - patch_size:
        xs.append(Wp - patch_size)

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

    # --------------------------------------------------------
    # 2. ЗАГРУЗКА И ПОВОРОТ МАСКИ
    # --------------------------------------------------------
    mask_path = "test_inference_masks/mask_inline_25.npy"

    full_mask = np.load(mask_path).astype(np.float32)

    full_mask = np.rot90(full_mask, k=-1, axes=(0, 1))
    print("Mask shape after rotation:", full_mask.shape)

    # --------------------------------------------------------
    # 3. ИНФЕРЕНС
    # --------------------------------------------------------
    generated = infer_full_with_patches(
        generator=generator,
        full_mask=full_mask,
        patch_size=128,
        overlap=32,
        batch_size=8
    )

    print("Generated seismic shape:", generated.shape)

    # --------------------------------------------------------
    # 4. СОХРАНЕНИЕ РЕЗУЛЬТАТА
    # --------------------------------------------------------
    out_path = "generated_seismic.npy"
    np.save(out_path, generated)

    print("Saved to:", out_path)
    print(
        "Stats:",
        "min =", generated.min(),
        "max =", generated.max(),
        "mean =", generated.mean(),
        "std  =", generated.std()
    )


# ============================================================
#                        RUN
# ============================================================

if __name__ == "__main__":
    main()
