#!/usr/bin/env python3
"""
生成真实屏摄数据采集素材：载体图 × 随机bit序列 → 带水印+信息二维码+四角角标的图片。

流程：
1. 载体图全图分tile嵌入水印（覆盖100%区域）
2. 在四角叠加回字形角标（半透明，不完全遮盖水印）
3. 在左上角叠加信息二维码（半透明，不完全遮盖水印）

用法:
    cd /data/lpl/fftmask
    python generate_real_data.py --version v17 --channel cb --num_sequences 50
    python generate_real_data.py --version v17 --channel cb --num_sequences 3 --preview 3
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from encode_v17 import WatermarkV17
from encode_v18 import WatermarkV18

# ── 版本配置（与训练配置一致）───────────────────────────────────────────────
VERSION_CONFIG = {
    "v17_cb": {
        "encoder_cls": WatermarkV17,
        "encoder_kwargs": dict(L1=512, k1=30000, r_watermark=[12, 25],
                               bitsf=[15, 45], r_range=1, n_sectors=60),
        "channel": "cb",
        "hollow": False,
        "alpha": 0.0191,
    },
    "v17_b": {
        "encoder_cls": WatermarkV17,
        "encoder_kwargs": dict(L1=512, k1=30000, r_watermark=[12, 25],
                               bitsf=[15, 45], r_range=1, n_sectors=60),
        "channel": "b",
        "hollow": False,
        "alpha": 0.0191,
    },
    "v18_cb": {
        "encoder_cls": WatermarkV18,
        "encoder_kwargs": dict(L1=512, k1=30000, r_watermark=[12, 25],
                               bitsf=[15, 45], r_range=1, n_sectors=60,
                               M_w=255, M_b=0, hollow_ratio=0.3),
        "channel": "cb",
        "hollow": True,
        "alpha": 0.0191,
    },
    "v18_b": {
        "encoder_cls": WatermarkV18,
        "encoder_kwargs": dict(L1=512, k1=30000, r_watermark=[12, 25],
                               bitsf=[15, 45], r_range=1, n_sectors=60,
                               M_w=255, M_b=0, hollow_ratio=0.3),
        "channel": "b",
        "hollow": True,
        "alpha": 0.0191,
    },
}

# 信息二维码参数
QR_SIZE = 150
QR_MARGIN = 10
QR_ALPHA = 0.85  # 信息二维码不透明度（保留水印可见性）

# 四角回字形角标参数
CORNER_MARK_PATH = Path(__file__).parent / "watermark_tools" / "qrmark" / "qr_loc_mark.png"
CORNER_SIZE = 120     # 角标显示尺寸（像素）
CORNER_MARGIN = 0     # 距图像边缘的距离（0=紧贴四角）
CORNER_ALPHA = 0.80   # 角标不透明度（保留水印可见性）

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")


# ── 模板生成与嵌入 ───────────────────────────────────────────────────────────

def generate_template(encoder, bits, hollow=False):
    """生成512×512水印模板。"""
    bits_arr = np.asarray(bits, dtype=np.int32)
    if hollow:
        Tm, _, _ = encoder.generate_template(numbit=bits_arr, hollow=True)
    else:
        Tm, _, _ = encoder.generate_template(numbit=bits_arr)
    return Tm


def tile_template(template, target_w, target_h):
    """平铺模板到目标尺寸。"""
    th, tw = template.shape[:2]
    nx = (target_w + tw - 1) // tw
    ny = (target_h + th - 1) // th
    tiled = np.tile(template, (ny, nx))
    start_y = (tiled.shape[0] - target_h) // 2
    start_x = (tiled.shape[1] - target_w) // 2
    return tiled[start_y:start_y + target_h, start_x:start_x + target_w]


def embed_watermark(image_bgr, template_gray, channel, alpha):
    """在图像上嵌入水印（与训练公式一致）。

    dst_channel = host_channel * (1 - alpha) + template * alpha
    """
    t = template_gray.astype(np.float32) / 255.0

    if channel == "cb":
        ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32)
        ycrcb[:, :, 2] = ycrcb[:, :, 2] + alpha * (255.0 * t - ycrcb[:, :, 2])
        result = cv2.cvtColor(
            ycrcb.clip(0, 255).astype(np.uint8), cv2.COLOR_YCrCb2BGR
        )
    else:
        result = image_bgr.copy().astype(np.float32)
        result[:, :, 0] = result[:, :, 0] + alpha * (255.0 * t - result[:, :, 0])
        result = result.clip(0, 255).astype(np.uint8)

    return result


# ── 信息二维码 ───────────────────────────────────────────────────────────────

def generate_qr_image(data: str, size: int = QR_SIZE) -> np.ndarray:
    """生成二维码图像，返回BGR格式（白色背景）。"""
    import qrcode
    qr = qrcode.QRCode(version=1, box_size=10, border=1)
    qr.add_data(data)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white").convert("RGB")
    img = img.resize((size, size))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


# ── 半透明叠加 ───────────────────────────────────────────────────────────────

def overlay_transparent(background, overlay, x, y, alpha=0.85):
    """在背景图上半透明叠加overlay。

    alpha=1.0: 完全覆盖; alpha=0.0: 完全透明（保留背景）。
    overlay的白色区域在叠加后仍显示白色（二维码白底），
    黑色区域会透出背景的水印纹理。
    """
    h, w = overlay.shape[:2]
    H, W = background.shape[:2]

    # 边界裁剪
    x1, y1 = max(x, 0), max(y, 0)
    x2, y2 = min(x + w, W), min(y + h, H)
    ox1, oy1 = x1 - x, y1 - y
    ox2, oy2 = ox1 + (x2 - x1), oy1 + (y2 - y1)

    if x2 <= x1 or y2 <= y1:
        return background

    result = background.copy()
    roi = result[y1:y2, x1:x2].astype(np.float32)
    ol = overlay[oy1:oy2, ox1:ox2].astype(np.float32)

    # alpha混合: result = background * (1-alpha) + overlay * alpha
    blended = roi * (1.0 - alpha) + ol * alpha
    result[y1:y2, x1:x2] = blended.clip(0, 255).astype(np.uint8)
    return result


def overlay_corner_marks(image, mark_img, size=CORNER_SIZE,
                         margin=CORNER_MARGIN, alpha=CORNER_ALPHA):
    """在四角叠加回字形角标（半透明）。"""
    h, w = image.shape[:2]
    mark_resized = cv2.resize(mark_img, (size, size),
                              interpolation=cv2.INTER_AREA)

    positions = [
        (margin, margin),                           # 左上
        (w - size - margin, margin),                # 右上
        (margin, h - size - margin),                # 左下
        (w - size - margin, h - size - margin),     # 右下
    ]
    result = image
    for x, y in positions:
        result = overlay_transparent(result, mark_resized, x, y, alpha)
    return result


# ── 工具函数 ─────────────────────────────────────────────────────────────────

def carrier_short_name(filename: str) -> str:
    """从文件名生成简短标识（确保唯一）。"""
    stem = Path(filename).stem
    if stem.startswith("Snipaste_"):
        parts = stem.split("_")
        if len(parts) >= 3:
            date_part = parts[1].replace("-", "")[4:]
            time_part = parts[2].replace("-", "")
            return f"Snipaste_{date_part}_{time_part}"
        return stem
    return stem[:8]


def load_carrier_images(input_dir, max_size=None):
    """加载载体图列表，按short_name去重（保留第一个）。"""
    all_files = sorted([
        f for f in os.listdir(input_dir)
        if f.lower().endswith(IMAGE_EXTENSIONS)
    ])
    seen = set()
    files = []
    for f in all_files:
        short = carrier_short_name(f)
        if short not in seen:
            seen.add(short)
            files.append(f)
    if max_size and max_size < len(files):
        files = random.sample(files, max_size)
        files.sort()
    return files


def generate_bit_sequences(num_sequences, seed=42):
    """生成随机60-bit序列。"""
    rng = random.Random(seed)
    sequences = {}
    for i in range(num_sequences):
        bits = tuple(rng.randint(0, 1) for _ in range(60))
        sequences[f"bits_{i:03d}"] = bits
    return sequences


def verify_bit_sequences(sequences):
    """验证bit序列分布。"""
    bits_list = list(sequences.values())
    n = len(bits_list)

    bit_positions = list(zip(*bits_list))
    print("\n=== Bit分布验证 ===")
    skewed = []
    for pos, vals in enumerate(bit_positions):
        ones = sum(vals)
        zeros = n - ones
        if ones < n * 0.3 or ones > n * 0.7:
            skewed.append((pos, ones, zeros))
    if skewed:
        print(f"⚠️ {len(skewed)} 个位置偏斜（仅{len(bits_list)}组时属正常）:")
        for pos, ones, zeros in skewed[:3]:
            print(f"  bit[{pos}]: {ones}个1, {zeros}个0")
    else:
        print("✅ 所有bit位置0/1分布均匀")

    def hamming(a, b):
        return sum(x != y for x, y in zip(a, b))

    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            dists.append(hamming(bits_list[i], bits_list[j]))
    print(f"汉明距离: min={min(dists)}, max={max(dists)}, avg={sum(dists)/len(dists):.1f}")


# ── 主流程 ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="生成带水印+信息二维码+四角角标的屏摄素材")
    parser.add_argument("--input_dir", default="/home/lpl2025/lpl/fftmask/train",
                        help="载体图目录")
    parser.add_argument("--output_dir", default="real_data/watermarked",
                        help="输出根目录")
    parser.add_argument("--version", choices=["v17", "v18"], default="v17")
    parser.add_argument("--channel", choices=["b", "cb"], default="cb")
    parser.add_argument("--num_sequences", type=int, default=50,
                        help="随机bit序列数量")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_carriers", type=int, default=None,
                        help="最多使用几张载体图（调试用）")
    parser.add_argument("--preview", type=int, default=0,
                        help="只生成前N张载体图 × 前N组bit（快速预览）")
    parser.add_argument("--qr_alpha", type=float, default=QR_ALPHA,
                        help="信息二维码不透明度")
    parser.add_argument("--corner_size", type=int, default=CORNER_SIZE,
                        help="四角角标尺寸（像素）")
    parser.add_argument("--corner_alpha", type=float, default=CORNER_ALPHA,
                        help="四角角标不透明度")
    parser.add_argument("--no_corners", action="store_true",
                        help="不添加四角角标")
    parser.add_argument("--checklist_days", type=int, default=3,
                        help="拍照清单天数")
    args = parser.parse_args()

    version_key = f"{args.version}_{args.channel}"
    config = VERSION_CONFIG[version_key]
    print(f"版本: {version_key}")
    print(f"  通道: {config['channel']}, alpha: {config['alpha']}, hollow: {config['hollow']}")

    # 加载角标
    corner_mark = None
    if not args.no_corners:
        if CORNER_MARK_PATH.exists():
            corner_mark = cv2.imread(str(CORNER_MARK_PATH))
            print(f"  四角角标: {CORNER_MARK_PATH} ({corner_mark.shape[1]}x{corner_mark.shape[0]})")
        else:
            print(f"  ⚠️ 角标文件不存在: {CORNER_MARK_PATH}，跳过四角角标")

    # 加载载体图
    carrier_files = load_carrier_images(args.input_dir, args.max_carriers)
    if args.preview > 0:
        carrier_files = carrier_files[:args.preview]
    print(f"载体图: {len(carrier_files)} 张 (from {args.input_dir})")

    # 生成bit序列
    num_seq = args.preview if args.preview > 0 else args.num_sequences
    bit_sequences = generate_bit_sequences(num_seq, args.seed)
    print(f"Bit序列: {len(bit_sequences)} 组")
    verify_bit_sequences(bit_sequences)

    # 创建编码器
    encoder = config["encoder_cls"](**config["encoder_kwargs"])

    # 输出目录
    wm_dir = Path(args.output_dir) / version_key
    wm_dir.mkdir(parents=True, exist_ok=True)
    bits_json_path = Path("real_data/bit_sequences.json")
    bits_json_path.parent.mkdir(parents=True, exist_ok=True)

    # 保存bit序列
    with open(bits_json_path, "w") as f:
        json.dump({k: list(v) for k, v in bit_sequences.items()}, f, indent=2)
    print(f"\nBit序列已保存: {bits_json_path}")

    # 生成水印图
    total = len(bit_sequences) * len(carrier_files)
    print(f"\n开始生成 {total} 张水印图...")
    print(f"  流程: 全图嵌入水印 → 四角角标(α={args.corner_alpha}) → 信息二维码(α={args.qr_alpha})")

    all_items = []
    for bits_id, bits in tqdm(bit_sequences.items(), desc="Bit序列"):
        bits_dir = wm_dir / bits_id
        bits_dir.mkdir(parents=True, exist_ok=True)

        # 生成模板（每组bit只需生成一次）
        template_512 = generate_template(encoder, bits, hollow=config["hollow"])

        # 信息二维码内容（每组bit只需生成一次）
        bit_string = "".join(str(b) for b in bits)
        seq_index = int(bits_id.split("_")[1])
        qr_data = f"{args.version}|{args.channel}|{seq_index:02d}|{bit_string}"
        qr_img = generate_qr_image(qr_data, size=QR_SIZE)

        for carrier_file in carrier_files:
            short_name = carrier_short_name(carrier_file)

            # 读取载体图
            img_path = os.path.join(args.input_dir, carrier_file)
            image = cv2.imread(img_path)
            if image is None:
                print(f"  ⚠️ 无法读取: {carrier_file}")
                continue

            h, w = image.shape[:2]

            # ── 步骤1: 全图嵌入水印（覆盖100%区域）──
            tiled = tile_template(template_512, w, h)
            watermarked = embed_watermark(image, tiled, config["channel"], config["alpha"])

            # ── 步骤2: 四角叠加回字形角标（半透明）──
            if corner_mark is not None:
                watermarked = overlay_corner_marks(
                    watermarked, corner_mark,
                    size=args.corner_size, margin=CORNER_MARGIN,
                    alpha=args.corner_alpha
                )

            # ── 步骤3: 左上角叠加信息二维码（半透明，放在角标下方避免重合）──
            qr_y = args.corner_size + 5  # 紧贴左上角标下方
            watermarked = overlay_transparent(
                watermarked, qr_img,
                QR_MARGIN, qr_y,
                alpha=args.qr_alpha
            )

            # 保存
            out_path = bits_dir / f"{short_name}.png"
            cv2.imwrite(str(out_path), watermarked)

            all_items.append(f"{bits_id}/{short_name}.png")

    print(f"\n✅ 已生成 {len(all_items)} 张水印图到 {wm_dir}/")

    # 保存全部条目列表
    all_items_path = Path("real_data/all_items.txt")
    with open(all_items_path, "w") as f:
        for item in all_items:
            f.write(item + "\n")

    # 生成拍照清单（打乱顺序，按天分批）
    random.seed(args.seed + 999)
    shuffled = all_items.copy()
    random.shuffle(shuffled)

    batch_size = 2000
    for day_idx in range(args.checklist_days):
        start = day_idx * batch_size
        end = min(start + batch_size, len(shuffled))
        batch = shuffled[start:end]
        if not batch:
            break
        checklist_path = Path(f"real_data/day{day_idx + 1}_checklist.txt")
        with open(checklist_path, "w") as f:
            for item in batch:
                f.write(item + "\n")
        print(f"  Day {day_idx + 1}: {len(batch)} 张 -> {checklist_path}")

    print(f"\n全部完成！共 {len(all_items)} 张水印图")


if __name__ == "__main__":
    main()
