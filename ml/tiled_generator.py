import os
import numpy as np
import tensorflow as tf

def make_flexible_generator(fixed_generator):
    """Клонирует обученный генератор на вход с произвольным HxW."""
    n_channels = fixed_generator.input_shape[-1]
    inputs = tf.keras.Input(shape=(None, None, n_channels))
    flexible = tf.keras.models.clone_model(fixed_generator, input_tensors=inputs)
    flexible.set_weights(fixed_generator.get_weights())
    return flexible

def pad_to_multiple(x, multiple):
    """Паддинг нулями до размера, кратного multiple (симметрично)."""
    h, w = x.shape[:2]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    x_pad = np.pad(
        x,
        ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
        mode="constant",
        constant_values=0.0,
    )
    return x_pad, (pad_top, pad_bottom, pad_left, pad_right)

def unpad(x, pads):
    pad_top, pad_bottom, pad_left, pad_right = pads
    h, w = x.shape[:2]
    y0 = pad_top
    y1 = h - pad_bottom
    x0 = pad_left
    x1 = w - pad_right
    return x[y0:y1, x0:x1]

def blend_window_2d(size, overlap, eps=1e-3):
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

def tile_coords(h, w, patch_size, overlap):
    stride = patch_size - overlap
    ys = list(range(0, max(h - patch_size + 1, 1), stride))
    xs = list(range(0, max(w - patch_size + 1, 1), stride))
    last_y = max(h - patch_size, 0)
    last_x = max(w - patch_size, 0)
    if len(ys) == 0 or ys[-1] != last_y:
        ys.append(last_y)
    if len(xs) == 0 or xs[-1] != last_x:
        xs.append(last_x)
    return ys, xs

def extract_patch_with_local_pad(img, y, x, patch_size, pad_mode="reflect", constant_values=0.0):
    h, w = img.shape[:2]
    y0 = max(y, 0)
    x0 = max(x, 0)
    y1 = min(y + patch_size, h)
    x1 = min(x + patch_size, w)
    patch = img[y0:y1, x0:x1]

    pad_top = 0 if y >= 0 else -y
    pad_left = 0 if x >= 0 else -x
    pad_bottom = patch_size - (y1 - y0) - pad_top
    pad_right = patch_size - (x1 - x0) - pad_left
    pad_spec = ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0))

    if pad_mode == "constant":
        patch = np.pad(patch, pad_spec, mode=pad_mode, constant_values=constant_values)
    else:
        patch = np.pad(patch, pad_spec, mode=pad_mode)
    return patch

def infer_full_adaptive(generator, full_mask, patch_size, overlap, batch_size=8, pad_mode="constant"):
    h, w = full_mask.shape[:2]
    window = blend_window_2d(patch_size, overlap)
    ys, xs = tile_coords(h, w, patch_size, overlap)
    out_sum = np.zeros((h, w, 1), dtype=np.float32)
    out_wgt = np.zeros((h, w, 1), dtype=np.float32)
    patches, coords = [], []

    for y in ys:
        for x in xs:
            patch = extract_patch_with_local_pad(
                full_mask,
                y,
                x,
                patch_size,
                pad_mode=pad_mode,
                constant_values=0.0,
            )
            patches.append(patch)
            coords.append((y, x))
            if len(patches) == batch_size:
                preds = generator(np.stack(patches), training=False).numpy()
                for (yy, xx), p in zip(coords, preds):
                    y_end = min(yy + patch_size, h)
                    x_end = min(xx + patch_size, w)
                    patch_h = y_end - yy
                    patch_w = x_end - xx
                    out_sum[yy:y_end, xx:x_end] += p[:patch_h, :patch_w] * window[:patch_h, :patch_w]
                    out_wgt[yy:y_end, xx:x_end] += window[:patch_h, :patch_w]
                patches.clear()
                coords.clear()

    if patches:
        preds = generator(np.stack(patches), training=False).numpy()
        for (yy, xx), p in zip(coords, preds):
            y_end = min(yy + patch_size, h)
            x_end = min(xx + patch_size, w)
            patch_h = y_end - yy
            patch_w = x_end - xx
            out_sum[yy:y_end, xx:x_end] += p[:patch_h, :patch_w] * window[:patch_h, :patch_w]
            out_wgt[yy:y_end, xx:x_end] += window[:patch_h, :patch_w]

    out = out_sum / (out_wgt + 1e-8)
    return out

def infer_full_tiled_context(
    generator,
    full_mask,
    patch_size,
    inner_size=None,
    overlap_inner=None,
    batch_size=8,
    pad_mode="constant",
):
    """Тайловая генерация с контекстом: подаем 128x128, берем центр."""
    h, w = full_mask.shape[:2]
    if inner_size is None:
        inner_size = patch_size // 2
    if (patch_size - inner_size) % 2 != 0:
        raise ValueError("patch_size - inner_size must be even")
    margin = (patch_size - inner_size) // 2
    if overlap_inner is None:
        overlap_inner = max(1, inner_size // 4)
    window = blend_window_2d(inner_size, overlap_inner)
    ys, xs = tile_coords(h, w, inner_size, overlap_inner)

    out_sum = np.zeros((h, w, 1), dtype=np.float32)
    out_wgt = np.zeros((h, w, 1), dtype=np.float32)
    patches, coords = [], []

    for y in ys:
        for x in xs:
            patch = extract_patch_with_local_pad(
                full_mask,
                y - margin,
                x - margin,
                patch_size,
                pad_mode=pad_mode,
                constant_values=0.0,
            )
            patches.append(patch)
            coords.append((y, x))
            if len(patches) == batch_size:
                preds = generator(np.stack(patches), training=False).numpy()
                for (yy, xx), p in zip(coords, preds):
                    y_end = min(yy + inner_size, h)
                    x_end = min(xx + inner_size, w)
                    patch_h = y_end - yy
                    patch_w = x_end - xx
                    p_center = p[margin:margin + patch_h, margin:margin + patch_w]
                    w_center = window[:patch_h, :patch_w]
                    out_sum[yy:y_end, xx:x_end] += p_center * w_center
                    out_wgt[yy:y_end, xx:x_end] += w_center
                patches.clear()
                coords.clear()

    if patches:
        preds = generator(np.stack(patches), training=False).numpy()
        for (yy, xx), p in zip(coords, preds):
            y_end = min(yy + inner_size, h)
            x_end = min(xx + inner_size, w)
            patch_h = y_end - yy
            patch_w = x_end - xx
            p_center = p[margin:margin + patch_h, margin:margin + patch_w]
            w_center = window[:patch_h, :patch_w]
            out_sum[yy:y_end, xx:x_end] += p_center * w_center
            out_wgt[yy:y_end, xx:x_end] += w_center

    out = out_sum / (out_wgt + 1e-8)
    return out

def infer_full_flexible(generator, full_mask, size_multiple=16):
    """Инференс на полном размере с нулевым паддингом до кратности."""
    full_mask = full_mask.astype(np.float32)
    padded, pads = pad_to_multiple(full_mask, size_multiple)
    pred = generator(padded[None, ...], training=False).numpy()[0]
    return unpad(pred, pads)

def mask_to_onehot(mask, n_channels):
    """Преобразует маску с целыми метками 0..n_channels-1 в one-hot encoding."""
    H, W = mask.shape
    onehot = np.zeros((H, W, n_channels), dtype=np.float32)
    for c in range(n_channels):
        onehot[:, :, c] = (mask == c).astype(np.float32)
    return onehot

def main():
    generator_path = "generator_128x128.keras"
    mask_path = os.path.join("test_masks", "mask.npy")
    
    fixed_generator = tf.keras.models.load_model(generator_path, compile=False)
    generator = make_flexible_generator(fixed_generator)
    n_channels = fixed_generator.input_shape[-1]

    full_mask = np.load(mask_path).astype(np.int16)
    if full_mask.ndim == 2:  # преобразуем single-channel маску в one-hot
        full_mask = mask_to_onehot(full_mask, n_channels=n_channels)
    
    print("Mask shape after one-hot conversion:", full_mask.shape)
    if full_mask.shape[-1] != n_channels:
        raise ValueError("Channel mismatch with generator input")

    # Аккуратный контекстный тайлинг: подаем 128x128, берем центр
    patch_size = fixed_generator.input_shape[1]
    generated = infer_full_tiled_context(
        generator,
        full_mask,
        patch_size=patch_size,
        inner_size=patch_size // 2,
        overlap_inner=patch_size // 4,
        batch_size=8,
        pad_mode="constant",
    )

    out_path = "generated_seismic.npy"
    np.save(out_path, generated)
    print("Saved:", out_path)
    print("Stats:", "min =", generated.min(), "max =", generated.max(),
          "mean =", generated.mean(), "std =", generated.std())

if __name__ == "__main__":
    main()
