"""
v18 镂空水印批量嵌入脚本

特性:
- 镂空模板（可配置 hollow_ratio, M_w, M_b）
- 水印尺寸 = 最短边长的一半
- 水印循环平铺在图片上
- 支持批量处理
- 参数可配置

用法:
    python embed_v18.py --input_dir /path/to/images --output_dir /path/to/output
    python embed_v18.py --input_dir /path/to/images --output_dir /path/to/output --hollow_ratio 0.5 --alpha 0.02
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

from encode_v18 import WatermarkV18


def get_watermark_bits(seed, num_bits):
    """生成水印位"""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=num_bits).tolist()


def tile_watermark(template, H, W):
    """
    将水印模板循环平铺到目标尺寸

    Args:
        template: 水印模板 (h, w)
        H, W: 目标尺寸

    Returns:
        tiled: 平铺后的水印 (H, W)
    """
    h, w = template.shape
    # 计算需要平铺的次数
    repeat_y = (H + h - 1) // h
    repeat_x = (W + w - 1) // w
    # 平铺
    tiled = np.tile(template, (repeat_y, repeat_x))
    # 裁剪到目标尺寸
    return tiled[:H, :W]


def embed_watermark_cb(host_bgr, wm_template, alpha=0.0191):
    """
    在 host_bgr 的 Cb 通道嵌入水印

    Args:
        host_bgr: BGR图像
        wm_template: 水印模板 (uint8)
        alpha: 嵌入强度

    Returns:
        wm_bgr: 带水印的BGR图像
    """
    h, w = host_bgr.shape[:2]

    # BGR → YCrCb
    ycrcb = cv2.cvtColor(host_bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    y, cr, cb = cv2.split(ycrcb)

    # 循环平铺水印到图像尺寸
    tiled_wm = tile_watermark(wm_template, h, w)

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


def main():
    parser = argparse.ArgumentParser(description="v18 镂空水印批量嵌入")
    parser.add_argument("--input_dir", type=str, required=True, help="输入图片目录")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    parser.add_argument("--num", type=int, default=None, help="最多处理图片数（默认全部）")

    # 水印参数
    parser.add_argument("--seed", type=int, default=2026, help="随机种子")
    parser.add_argument("--num_bits", type=int, default=60, help="水印位数")
    parser.add_argument("--r_watermark", type=int, nargs="+", default=[12, 25], help="水印环半径")
    parser.add_argument("--bitsf", type=int, nargs="+", default=[15, 45], help="每环位数")
    parser.add_argument("--k1", type=float, default=30000.0, help="频域幅值强度")
    parser.add_argument("--r_range", type=int, default=1, help="环宽度")

    # 镂空参数
    parser.add_argument("--M_w", type=int, default=255, help="水印高值区域亮度")
    parser.add_argument("--M_b", type=int, default=0, help="水印低值区域亮度")
    parser.add_argument("--hollow_ratio", type=float, default=0.3, help="镂空比例 (0-1)")

    # 嵌入参数
    parser.add_argument("--alpha", type=float, default=0.0191, help="嵌入强度")

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 验证参数
    assert sum(args.bitsf) == args.num_bits, f"sum(bitsf)={sum(args.bitsf)} != num_bits={args.num_bits}"
    assert len(args.r_watermark) == len(args.bitsf), "r_watermark和bitsf长度必须一致"

    # 生成水印位
    bits = get_watermark_bits(args.seed, args.num_bits)
    bits_string = "".join(str(b) for b in bits)
    print(f"\nWatermark bits: {bits_string}")

    # 创建水印编码器
    wm_sys = WatermarkV18(
        L1=512, k1=args.k1,
        r_watermark=args.r_watermark, bitsf=args.bitsf,
        r_range=args.r_range, n_sectors=args.num_bits,
        M_w=args.M_w, M_b=args.M_b, hollow_ratio=args.hollow_ratio
    )

    # 生成镂空模板
    Tm_hollow, M1, _ = wm_sys.generate_template(numbit=np.array(bits), hollow=True)
    print(f"Template: {Tm_hollow.shape}, range=[{Tm_hollow.min()}, {Tm_hollow.max()}]")
    print(f"Hollow ratio: {args.hollow_ratio}, M_w={args.M_w}, M_b={args.M_b}")

    # 计算水印尺寸（最短边长的一半）
    # 获取第一张图片尺寸来确定水印大小
    files = sorted([
        f for f in os.listdir(args.input_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])
    if args.num is not None:
        files = files[:args.num]

    if len(files) == 0:
        print("No images found!")
        return

    # 读取第一张图片获取尺寸
    first_img = cv2.imread(os.path.join(args.input_dir, files[0]))
    if first_img is None:
        print(f"Failed to load first image: {files[0]}")
        return

    h_img, w_img = first_img.shape[:2]
    wm_size = min(h_img, w_img) // 2
    print(f"\nImage size: {h_img}x{w_img}")
    print(f"Watermark size: {wm_size}x{wm_size} (half of shortest edge)")

    # 调整水印模板到目标尺寸
    if wm_size != Tm_hollow.shape[0]:
        Tm_resized = cv2.resize(Tm_hollow, (wm_size, wm_size), interpolation=cv2.INTER_NEAREST)
        print(f"Resized template: {Tm_resized.shape}")
    else:
        Tm_resized = Tm_hollow

    print(f"\nFound {len(files)} images in {args.input_dir}")

    # 遍历图像嵌入
    mappings = []
    config = {
        "seed": args.seed,
        "channel": "cb",
        "alpha_embed": args.alpha,
        "num_bits": args.num_bits,
        "r_watermark": args.r_watermark,
        "bitsf": args.bitsf,
        "k1": args.k1,
        "r_range": args.r_range,
        "M_w": args.M_w,
        "M_b": args.M_b,
        "hollow_ratio": args.hollow_ratio,
        "watermark_size": wm_size,
        "input_dir": args.input_dir,
        "output_dir": args.output_dir,
        "version": "v18_hollow",
        "created_at": datetime.now().isoformat(),
    }

    for fname in tqdm(files, desc="Embedding"):
        host_path = os.path.join(args.input_dir, fname)
        wm_name = fname.rsplit(".", 1)[0] + "_wm.png"
        wm_path = os.path.join(args.output_dir, wm_name)

        host_bgr = cv2.imread(host_path)
        if host_bgr is None:
            print(f"  [SKIP] {fname}: failed to load")
            continue

        # 嵌入水印（自动循环平铺）
        wm_bgr = embed_watermark_cb(host_bgr, Tm_resized, alpha=args.alpha)
        cv2.imwrite(wm_path, wm_bgr)

        mappings.append({
            "original_file": fname,
            "watermarked_file": wm_name,
            "watermarked_path": wm_path,
            "size": list(host_bgr.shape[:2]),
            "num_bits": args.num_bits,
            "r_watermark": args.r_watermark,
            "bitsf": args.bitsf,
            "alpha_embed": args.alpha,
            "watermark_bits": bits,
            "original_sha256": file_sha256(host_path),
            "watermarked_sha256": file_sha256(wm_path),
        })

    # 保存 GT JSON
    gt_data = {
        "created_at": datetime.now().isoformat(),
        "config": config,
        "watermark_bits": bits,
        "watermark_bits_string": bits_string,
        "count": len(mappings),
        "mappings": mappings,
    }
    gt_path = os.path.join(args.output_dir, "embed_gt_mapping.json")
    with open(gt_path, "w", encoding="utf-8") as f:
        json.dump(gt_data, f, indent=2, ensure_ascii=False)

    print(f"\nDone! {len(mappings)} images embedded.")
    print(f"  Output dir: {args.output_dir}")
    print(f"  GT JSON   : {gt_path}")


if __name__ == "__main__":
    main()
