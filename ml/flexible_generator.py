import os
import numpy as np
import tensorflow as tf

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

def extract_patch_with_local_pad(img, y, x, patch_size):
    h, w = img.shape[:2]
    patch = np.zeros((patch_size, patch_size, img.shape[2]), dtype=img.dtype)
    y0 = max(y, 0)
    x0 = max(x, 0)
    y1 = min(y + patch_size, h)
    x1 = min(x + patch_size, w)
    patch_y0 = 0 if y >= 0 else -y
    patch_x0 = 0 if x >= 0 else -x
    patch_y1 = patch_y0 + (y1 - y0)
    patch_x1 = patch_x0 + (x1 - x0)
    patch[patch_y0:patch_y1, patch_x0:patch_x1] = img[y0:y1, x0:x1]
    if y1 - y0 < patch_size or x1 - x0 < patch_size:
        patch = np.pad(
            patch,
            ((0, patch_size - (y1 - y0)), (0, patch_size - (x1 - x0)), (0, 0)),
            mode="reflect"
        )
    return patch

def infer_full_adaptive(generator, full_mask, patch_size, overlap, batch_size=8):
    h, w = full_mask.shape[:2]
    window = blend_window_2d(patch_size, overlap)
    ys, xs = tile_coords(h, w, patch_size, overlap)
    out_sum = np.zeros((h, w, 1), dtype=np.float32)
    out_wgt = np.zeros((h, w, 1), dtype=np.float32)
    patches, coords = [], []

    for y in ys:
        for x in xs:
            patch = extract_patch_with_local_pad(full_mask, y, x, patch_size)
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
    
    generator = tf.keras.models.load_model(generator_path, compile=False)
    patch_size = generator.input_shape[1]
    n_channels = generator.input_shape[-1]

    full_mask = np.load(mask_path).astype(np.int16)
    if full_mask.ndim == 2:  # преобразуем single-channel маску в one-hot
        full_mask = mask_to_onehot(full_mask, n_channels=n_channels)
    
    print("Mask shape after one-hot conversion:", full_mask.shape)
    if full_mask.shape[-1] != n_channels:
        raise ValueError("Channel mismatch with generator input")

    # Генерация
    generated = infer_full_adaptive(
        generator,
        full_mask,
        patch_size=patch_size,
        overlap=patch_size // 2,
        batch_size=8
    )

    out_path = "generated_seismic.npy"
    np.save(out_path, generated)
    print("Saved:", out_path)
    print("Stats:", "min =", generated.min(), "max =", generated.max(),
          "mean =", generated.mean(), "std =", generated.std())

if __name__ == "__main__":
    main()
