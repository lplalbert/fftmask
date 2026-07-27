"""
v11 版本：对 test_bcgd 目录下的图像进行全图水印嵌入（Cb 通道）。

使用 v11 编码器（包含旋转矫正环 r=18）。

用法:
    cd /mnt/lpl/fftmask && python3 embed_test_bcgd_full_v11.py
"""

import os
import sys
import json
import argparse
import hashlib
import numpy as np
import cv2
from tqdm import tqdm
from datetime import datetime

from encode_v11 import WatermarkV11

# ── 参数（与 v11 训练完全一致）────────────────────────────────────────────────
SEED        = 2026
NUM_BITS    = 60
R_WATERMARK = [12, 25]
BITS_LIST   = [15, 45]
K1          = 30000.0
R_RANGE     = 1
R_ROTATION  = 18
ROTATION_CYCLES = 8
ALPHA_EMBED = 0.016
CHANNEL     = "cb"

# ── 图像尺寸和推理 crop 位置 ─────────────────────────────────────────────────
H_IMG, W_IMG = 1080, 1920
BS           = 512

CROP_X0 = W_IMG // 2 - BS // 2   # 704
CROP_Y0 = H_IMG // 2 - BS // 2   # 384

OFFSET_X = (-CROP_X0) % BS   # 320
OFFSET_Y = (-CROP_Y0) % BS   # 128

GLOBAL_WATERMARK_BITS = None


def get_global_watermark_bits():
    global GLOBAL_WATERMARK_BITS
    if GLOBAL_WATERMARK_BITS is None:
        rng = np.random.default_rng(SEED)
        GLOBAL_WATERMARK_BITS = rng.integers(0, 2, size=NUM_BITS).tolist()
    return GLOBAL_WATERMARK_BITS


def tile_by_phase(template, H, W, offset_y=0, offset_x=0):
    """按相位偏移平铺（周期边界）"""
    y_idx = (np.arange(H) + offset_y) % template.shape[0]
    x_idx = (np.arange(W) + offset_x) % template.shape[1]
    return template[y_idx[:, None], x_idx[None, :]]


def embed_watermark_cb(host_bgr, wm_template_512, alpha=0.016, offset_y=0, offset_x=0):
    """
    在 host_bgr 的 Cb 通道嵌入水印，返回带水印的 BGR 图像。
    wm_template_512: 512x512 水印模板（uint8 0-255）
    offset: tiling 的相位偏移
    """
    h, w = host_bgr.shape[:2]

    # BGR → YCrCb
    ycrcb = cv2.cvtColor(host_bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    y, cr, cb = cv2.split(ycrcb)

    # 用指定 offset tiling 水印模板到图像尺寸
    tiled_wm = tile_by_phase(wm_template_512, h, w,
                             offset_y=offset_y, offset_x=offset_x)

    # Cb 通道嵌入
    cb_wm = cb * (1.0 - alpha) + tiled_wm.astype(np.float32) * alpha
    cb_wm = np.clip(cb_wm, 0, 255).astype(np.uint8)

    # 合并回 BGR
    ycrcb_wm = cv2.merge([y.astype(np.uint8), cr.astype(np.uint8), cb_wm])
    return cv2.cvtColor(ycrcb_wm, cv2.COLOR_YCrCb2BGR)


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="v11 全图 Cb 通道水印嵌入")
    parser.add_argument("--input_dir", type=str, default="/mnt/lpl/fftmask/bcgd_train", help="输入图片目录")
    parser.add_argument("--output_dir", type=str, default="/mnt/lpl/fftmask/watermarked_output_cb_v11", help="输出目录")
    parser.add_argument("--num", type=int, default=None, help="最多处理图片数（默认全部）")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # ── 生成 512x512 水印模板（用 WatermarkV11，与 v11 训练一致）──────────────
    bits = get_global_watermark_bits()
    bits_string = "".join(str(b) for b in bits)
    print(f"\nWatermark bits: {bits_string}")

    wm_sys = WatermarkV11(
        L1=512, k1=K1,
        r_watermark=R_WATERMARK, bitsf=BITS_LIST,
        r_rotation=R_ROTATION, rotation_cycles=ROTATION_CYCLES,
        r_range=R_RANGE, n_sectors=NUM_BITS
    )
    Tm_512, M1, _ = wm_sys.generate_template(numbit=np.array(bits))
    print(f"Template: {Tm_512.shape}, range=[{Tm_512.min()}, {Tm_512.max()}], "
          f"M1 nonzero={np.count_nonzero(M1)}")

    # ── 验证：中心 crop 信号是否与训练一致 ────────────────────────────────────
    full_tiled = tile_by_phase(Tm_512, H_IMG, W_IMG, OFFSET_Y, OFFSET_X)
    center_crop = full_tiled[CROP_Y0:CROP_Y0+BS, CROP_X0:CROP_X0+BS]
    train_crop  = Tm_512[0:BS, 0:BS]
    print(f"\nSanity check:")
    print(f"  Center crop == Train crop: {np.allclose(center_crop, train_crop)}")

    # ── 遍历图像嵌入 ─────────────────────────────────────────────────────────
    files = sorted([
        f for f in os.listdir(input_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])
    if args.num is not None:
        files = files[:args.num]
    print(f"\nFound {len(files)} images in {input_dir}")

    mappings = []
    config = {
        "seed":         SEED,
        "channel":      CHANNEL,
        "alpha_embed":  ALPHA_EMBED,
        "num_bits":     NUM_BITS,
        "r_watermark":  R_WATERMARK,
        "bitsf":        BITS_LIST,
        "k1":           K1,
        "r_range":      R_RANGE,
        "r_rotation":   R_ROTATION,
        "rotation_cycles": ROTATION_CYCLES,
        "image_size":   [H_IMG, W_IMG],
        "offset_x":     OFFSET_X,
        "offset_y":     OFFSET_Y,
        "crop_origin":  [CROP_Y0, CROP_X0],
        "block_size":   BS,
        "input_dir":    input_dir,
        "output_dir":   output_dir,
        "version":      "v11",
        "created_at":   datetime.now().isoformat(),
    }

    for fname in tqdm(files, desc="Embedding"):
        host_path = os.path.join(input_dir, fname)
        wm_name   = fname.rsplit(".", 1)[0] + "_wm.png"
        wm_path   = os.path.join(output_dir, wm_name)

        host_bgr = cv2.imread(host_path)
        if host_bgr is None:
            print(f"  [SKIP] {fname}: failed to load")
            continue

        if host_bgr.shape[:2] != (H_IMG, W_IMG):
            host_bgr = cv2.resize(host_bgr, (W_IMG, H_IMG))

        wm_bgr = embed_watermark_cb(host_bgr, Tm_512,
                                     alpha=ALPHA_EMBED,
                                     offset_y=OFFSET_Y,
                                     offset_x=OFFSET_X)
        cv2.imwrite(wm_path, wm_bgr)

        mappings.append({
            "original_file":     fname,
            "watermarked_file":  wm_name,
            "watermarked_path":  wm_path,
            "size":              list(host_bgr.shape[:2]),
            "num_bits":          NUM_BITS,
            "r_watermark":       R_WATERMARK,
            "bitsf":             BITS_LIST,
            "alpha_embed":       ALPHA_EMBED,
            "watermark_bits":    bits,
            "original_sha256":   file_sha256(host_path),
            "watermarked_sha256": file_sha256(wm_path),
        })

    # ── 保存 GT JSON ─────────────────────────────────────────────────────────
    gt_data = {
        "created_at":            datetime.now().isoformat(),
        "config":                config,
        "watermark_bits":        bits,
        "watermark_bits_string": bits_string,
        "count":                len(mappings),
        "mappings":             mappings,
    }
    gt_path = os.path.join(output_dir, "embed_gt_mapping.json")
    with open(gt_path, "w", encoding="utf-8") as f:
        json.dump(gt_data, f, indent=2, ensure_ascii=False)

    print(f"\nDone! {len(mappings)} images embedded.")
    print(f"  Output dir: {output_dir}")
    print(f"  GT JSON   : {gt_path}")


if __name__ == "__main__":
    main()
