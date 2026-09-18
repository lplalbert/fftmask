"""
v17 版本：对图像进行全图水印嵌入（Cb 通道）

使用 v17 编码器（无旋转环，[8,15] 双环）
验证准确率：85.18%

用法:
    cd /data/lpl/fftmask && python embed_v17.py --input_dir <输入目录> --output_dir <输出目录>
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

# ── v17 参数（与训练完全一致）────────────────────────────────────────────────
SEED        = 2026
NUM_BITS    = 60
R_LIST      = [8, 15]         # v17 最佳圆环位置
BITS_LIST   = [20, 40]        # 20bit + 40bit = 60bit
K1          = 30000.0
R_RANGE     = 1
R_ROTATION  = None            # v17 无旋转环
ALPHA_EMBED = 0.016
CHANNEL     = "cb"

# ── 图像尺寸和推理 crop 位置 ─────────────────────────────────────────────────
H_IMG, W_IMG = 1080, 1920   # 图片尺寸（高×宽）
BS           = 512          # tile/block 大小

CROP_X0 = W_IMG // 2 - BS // 2   # 704
CROP_Y0 = H_IMG // 2 - BS // 2   # 384

# offset 使得推理 crop 对齐到 tile origin
OFFSET_X = (-CROP_X0) % BS   # 320
OFFSET_Y = (-CROP_Y0) % BS   # 128

GLOBAL_WATERMARK_BITS = None


def get_global_watermark_bits():
    """生成固定的水印bits"""
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

    Args:
        host_bgr: 原始BGR图像
        wm_template_512: 512x512 水印模板（uint8 0-255）
        alpha: 嵌入强度
        offset_y, offset_x: tiling 的相位偏移

    Returns:
        带水印的BGR图像
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
    """计算文件SHA256"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def visualize_template(wm_template, output_dir):
    """生成模板可视化图"""
    # 1. 灰度模板
    cv2.imwrite(os.path.join(output_dir, "template_gray.png"), wm_template)

    # 2. 频谱圆环图
    h, w = wm_template.shape
    center_x, center_y = w // 2, h // 2

    # 创建圆环可视化（纯黑背景）
    ring_vis = np.zeros((512, 512, 3), dtype=np.uint8)
    # 绘制中心点
    cv2.circle(ring_vis, (center_x, center_y), 3, (255, 255, 255), -1)

    # 绘制水印环（绿色）
    for r in R_LIST:
        cv2.circle(ring_vis, (center_x, center_y), r, (0, 255, 0), 2)
        cv2.putText(ring_vis, f"r={r}", (center_x + r + 5, center_y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # 添加图例
    cv2.putText(ring_vis, "Watermark rings (no rotation)", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imwrite(os.path.join(output_dir, "ring_visualization.png"), ring_vis)

    # 3. 频谱图（FFT）
    template_f32 = wm_template.astype(np.float32)
    fft_result = np.fft.fft2(template_f32)
    fft_shift = np.fft.fftshift(fft_result)
    magnitude = np.log1p(np.abs(fft_shift))
    magnitude = (magnitude / magnitude.max() * 255).astype(np.uint8)

    # 在频谱图上绘制圆环
    spectrum_vis = cv2.cvtColor(magnitude, cv2.COLOR_GRAY2BGR)
    # 绘制水印环（绿色）
    for r in R_LIST:
        cv2.circle(spectrum_vis, (center_x, center_y), r, (0, 255, 0), 2)
        cv2.putText(spectrum_vis, f"r={r}", (center_x + r + 5, center_y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imwrite(os.path.join(output_dir, "spectrum_with_rings.png"), spectrum_vis)

    print(f"  模板可视化已保存到: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="v17 全图 Cb 通道水印嵌入")
    parser.add_argument("--input_dir", type=str,
                       default="/data/xsj/dataset/coco/mini_coco/val",
                       help="输入图片目录")
    parser.add_argument("--output_dir", type=str,
                       default="/data/lpl/fftmask/output/v17_watermarked",
                       help="输出目录")
    parser.add_argument("--num", type=int, default=10,
                       help="最多处理图片数（默认10张）")
    parser.add_argument("--alpha", type=float, default=ALPHA_EMBED,
                       help=f"嵌入强度（默认{ALPHA_EMBED}）")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("v17 水印嵌入工具")
    print("=" * 60)
    print(f"圆环位置: {R_LIST}")
    print(f"Bits配置: {BITS_LIST} (共{sum(BITS_LIST)}bit)")
    print(f"嵌入强度: {args.alpha}")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print("=" * 60)

    # ── 生成 512x512 水印模板 ─────────────────────────────────────────────
    bits = get_global_watermark_bits()
    bits_string = "".join(str(b) for b in bits)
    print(f"\n水印bits: {bits_string}")

    wm_sys = WatermarkV11(
        L1=512, k1=K1,
        r_watermark=R_LIST,
        bitsf=BITS_LIST,
        r_rotation=R_ROTATION,
        r_range=R_RANGE,
        n_sectors=NUM_BITS
    )
    Tm_512, M1, _ = wm_sys.generate_template(numbit=np.array(bits))
    print(f"模板尺寸: {Tm_512.shape}")
    print(f"模板范围: [{Tm_512.min()}, {Tm_512.max()}]")

    # ── 生成模板可视化 ─────────────────────────────────────────────────────
    print("\n生成模板可视化...")
    # 转为灰度
    if len(Tm_512.shape) == 3:
        Tm_gray = cv2.cvtColor(Tm_512, cv2.COLOR_BGR2GRAY)
    else:
        Tm_gray = Tm_512
    visualize_template(Tm_gray, output_dir)

    # ── 验证：中心 crop 信号是否与训练一致 ────────────────────────────────
    full_tiled = tile_by_phase(Tm_512, H_IMG, W_IMG, OFFSET_Y, OFFSET_X)
    center_crop = full_tiled[CROP_Y0:CROP_Y0+BS, CROP_X0:CROP_X0+BS]
    train_crop  = Tm_512[0:BS, 0:BS]
    print(f"\nSanity check:")
    print(f"  Center crop == Train crop: {np.allclose(center_crop, train_crop)}")

    # ── 遍历图像嵌入 ─────────────────────────────────────────────────────
    files = sorted([
        f for f in os.listdir(input_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])
    if args.num is not None:
        files = files[:args.num]
    print(f"\n找到 {len(files)} 张图片")

    mappings = []
    config = {
        "version":      "v17",
        "seed":         SEED,
        "channel":      CHANNEL,
        "alpha_embed":  args.alpha,
        "num_bits":     NUM_BITS,
        "r":            R_LIST,
        "bitsf":        BITS_LIST,
        "k1":           K1,
        "r_range":      R_RANGE,
        "r_rotation":   R_ROTATION,
        "image_size":   [H_IMG, W_IMG],
        "offset_x":     OFFSET_X,
        "offset_y":     OFFSET_Y,
        "crop_origin":  [CROP_Y0, CROP_X0],
        "block_size":   BS,
        "input_dir":    input_dir,
        "output_dir":   output_dir,
        "val_acc":      0.8518,  # 验证准确率
        "created_at":   datetime.now().isoformat(),
    }

    for fname in tqdm(files, desc="嵌入水印"):
        host_path = os.path.join(input_dir, fname)
        wm_name   = fname.rsplit(".", 1)[0] + "_wm.png"
        wm_path   = os.path.join(output_dir, wm_name)

        host_bgr = cv2.imread(host_path)
        if host_bgr is None:
            print(f"  [SKIP] {fname}: 读取失败")
            continue

        # 确保尺寸一致
        if host_bgr.shape[:2] != (H_IMG, W_IMG):
            host_bgr = cv2.resize(host_bgr, (W_IMG, H_IMG))

        # 嵌入水印
        wm_bgr = embed_watermark_cb(host_bgr, Tm_512,
                                     alpha=args.alpha,
                                     offset_y=OFFSET_Y,
                                     offset_x=OFFSET_X)
        cv2.imwrite(wm_path, wm_bgr)

        mappings.append({
            "original_file":     fname,
            "watermarked_file":  wm_name,
            "watermarked_path":  wm_path,
            "size":              list(host_bgr.shape[:2]),
            "num_bits":          NUM_BITS,
            "r":                 R_LIST,
            "bitsf":             BITS_LIST,
            "alpha_embed":       args.alpha,
            "watermark_bits":    bits,
            "original_sha256":   file_sha256(host_path),
            "watermarked_sha256": file_sha256(wm_path),
        })

    # ── 保存 GT JSON ─────────────────────────────────────────────────────
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

    print("\n" + "=" * 60)
    print("完成！")
    print(f"  处理图片数: {len(mappings)}")
    print(f"  输出目录: {output_dir}")
    print(f"  GT JSON: {gt_path}")
    print(f"  模板图: {output_dir}/template_gray.png")
    print(f"  圆环图: {output_dir}/ring_visualization.png")
    print(f"  频谱图: {output_dir}/spectrum_with_rings.png")
    print("=" * 60)


if __name__ == "__main__":
    main()
