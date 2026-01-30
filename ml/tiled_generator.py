import os
import numpy as np
import tensorflow as tf


def pad_to_tile(x, tile_size, mode="reflect"):
    """Pad H,W so they are multiples of tile_size."""
    if x.ndim != 3:
        raise ValueError("Expected input shape (H, W, C)")
    h, w = x.shape[:2]
    nh = int(np.ceil(h / tile_size) * tile_size)
    nw = int(np.ceil(w / tile_size) * tile_size)

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


def infer_by_tiles(generator, full_mask, tile_size=128, batch_size=8):
    """
    generator : tf.keras.Model
    full_mask : np.ndarray (H, W, C)
    return    : np.ndarray (H, W, 1)
    """
    full_mask, pads = pad_to_tile(full_mask, tile_size, mode="reflect")
    ph0, ph1, pw0, pw1 = pads
    Hp, Wp = full_mask.shape[:2]

    out = np.zeros((Hp, Wp, 1), dtype=np.float32)

    patches = []
    coords = []

    for y in range(0, Hp, tile_size):
        for x in range(0, Wp, tile_size):
            patch = full_mask[y:y + tile_size, x:x + tile_size]
            patches.append(patch)
            coords.append((y, x))

            if len(patches) == batch_size:
                batch = np.stack(patches, axis=0)
                preds = generator(batch, training=False).numpy()

                for (yy, xx), p in zip(coords, preds):
                    out[yy:yy + tile_size, xx:xx + tile_size] = p

                patches.clear()
                coords.clear()

    if patches:
        batch = np.stack(patches, axis=0)
        preds = generator(batch, training=False).numpy()
        for (yy, xx), p in zip(coords, preds):
            out[yy:yy + tile_size, xx:xx + tile_size] = p

    out = out[ph0:Hp - ph1, pw0:Wp - pw1]
    return out


def main():
    generator_path = "generator_128x128.keras"
    mask_path = os.path.join("test_masks", "mask_inline_25.npy")

    generator = tf.keras.models.load_model(generator_path, compile=False)
    print("Generator loaded")
    print("Input shape :", generator.input_shape)
    print("Output shape:", generator.output_shape)

    if generator.input_shape[1] != generator.input_shape[2]:
        raise ValueError("Generator input must be square for tiling inference")
    tile_size = int(generator.input_shape[1])
    expected_channels = int(generator.input_shape[-1])

    full_mask = np.load(mask_path).astype(np.float32)
    rotate_k = -1  # 90 deg clockwise
    full_mask = np.rot90(full_mask, k=rotate_k, axes=(0, 1))
    print("Mask shape after rotation:", full_mask.shape)

    if full_mask.shape[-1] != expected_channels:
        raise ValueError(
            f"Mask channels ({full_mask.shape[-1]}) != generator channels ({expected_channels})"
        )

    generated_rotated = infer_by_tiles(
        generator=generator,
        full_mask=full_mask,
        tile_size=tile_size,
        batch_size=8
    )

    generated = np.rot90(generated_rotated, k=-rotate_k, axes=(0, 1))
    print("Generated seismic shape:", generated.shape)

    out_path = "generated_seismic_tiled.npy"
    out_rot_path = "generated_seismic_tiled_rotated.npy"
    np.save(out_path, generated)
    np.save(out_rot_path, generated_rotated)

    print("Saved to:", out_path)
    print("Saved rotated to:", out_rot_path)
    print(
        "Stats:",
        "min =", generated.min(),
        "max =", generated.max(),
        "mean =", generated.mean(),
        "std  =", generated.std()
    )


if __name__ == "__main__":
    main()
