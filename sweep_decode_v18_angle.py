"""
v18 版本：搜索最佳水印尺寸和旋转角度（两阶段搜索）

第一阶段：粗搜（步长10°）
第二阶段：在最佳角度±10°范围内细搜（步长0.2°）

用法:
    python sweep_decode_v18_angle.py \
        --input_dir /path/to/images \
        --bits_file /path/to/embed_gt_mapping.json \
        --model_path /path/to/best_model.pth \
        --best_ratio 0.55 \
        --coarse_step 10 --fine_step 0.2
"""

import argparse
import os
import json
import cv2
import numpy as np
import torch
from collections import defaultdict
from tqdm import tqdm

from watermark_decoder_v17 import WatermarkDecoderV17


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def list_images(input_path, input_dir, max_images):
    if input_path:
        return [input_path]
    if not input_dir:
        raise ValueError("Either --input or --input_dir must be provided")
    names = sorted(f for f in os.listdir(input_dir) if f.lower().endswith(IMAGE_EXTENSIONS))
    if max_images > 0:
        names = names[:max_images]
    return [os.path.join(input_dir, name) for name in names]


def extract_channel(image_bgr, channel):
    """Extract specified channel from BGR image."""
    if channel == "y":
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    elif channel == "cb":
        ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
        return ycrcb[:, :, 2]
    elif channel == "cr":
        ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
        return ycrcb[:, :, 1]
    else:
        raise ValueError(f"Unknown channel: {channel}")


def rotate_image(image, angle, border_value=0):
    """
    旋转图片

    Args:
        image: 输入图片
        angle: 旋转角度（度）
        border_value: 边界填充值

    Returns:
        rotated: 旋转后的图片
    """
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # 计算旋转后的图片大小，保持所有内容可见
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)

    # 调整旋转矩阵
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2

    rotated = cv2.warpAffine(image, M, (new_w, new_h),
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=border_value)
    return rotated


def resize_to_wm512(image, wm_size, crop_size=512):
    """
    将图片resize，使水印tile大小变成crop_size
    """
    h, w = image.shape[:2]
    resize_ratio = crop_size / wm_size
    new_w = max(crop_size, int(w * resize_ratio))
    new_h = max(crop_size, int(h * resize_ratio))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, resize_ratio


def make_crop_positions(h, w, crop_size, num_crops, seed, mode="center"):
    """Generate crop positions for an image."""
    positions = []
    if mode == "center":
        # 只取中心
        cy, cx = h // 2, w // 2
        positions.append((cy - crop_size // 2, cx - crop_size // 2))
    elif mode == "five_point":
        # Center + 4 corners
        cy, cx = h // 2, w // 2
        half = crop_size // 2
        positions.append((cy - half, cx - half))  # center
        positions.append((0, 0))  # top-left
        positions.append((0, w - crop_size))  # top-right
        positions.append((h - crop_size, 0))  # bottom-left
        positions.append((h - crop_size, w - crop_size))  # bottom-right
    else:
        # Random
        rng = np.random.default_rng(seed)
        for _ in range(num_crops):
            y = rng.integers(0, max(1, h - crop_size + 1))
            x = rng.integers(0, max(1, w - crop_size + 1))
            positions.append((y, x))
    return positions


def load_bits_from_json(json_path):
    """Load watermark bits from embed_gt_mapping.json."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("watermark_bits", [])


def bits_to_string(bits):
    return "".join(str(int(b)) for b in bits)


def format_percent(value):
    if value is None:
        return "N/A"
    return f"{value:.2f}"


def print_table(headers, rows):
    widths = [len(header) for header in headers]
    for row in rows:
        widths = [max(width, len(str(value))) for width, value in zip(widths, row)]

    def fmt(values):
        return "| " + " | ".join(str(value).ljust(width) for value, width in zip(values, widths)) + " |"

    print(fmt(headers))
    print("| " + " | ".join("-" * width for width in widths) + " |")
    for row in rows:
        print(fmt(row))


def main():
    parser = argparse.ArgumentParser(
        description="v18: Sweep watermark size and rotation angle"
    )
    parser.add_argument("--input", type=str, default=None, help="Single input image path")
    parser.add_argument("--input_dir", type=str, default=None, help="Directory of input images")
    parser.add_argument("--bits_file", type=str, default=None,
                        help="Ground-truth bits JSON (embed_gt_mapping.json)")
    parser.add_argument("--model_path", type=str,
                        default="/data/lpl/fftmask/output/v18_hollow_pair/best_model.pth")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--num_bits", type=int, default=60)
    parser.add_argument("--r", type=int, nargs="+", default=[12, 25])
    parser.add_argument("--bitsf", type=int, nargs="+", default=[15, 45])
    parser.add_argument("--angle_bins", type=int, default=180)
    parser.add_argument("--crop_size", type=int, default=512)
    parser.add_argument("--num_crops", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    # 尺寸搜索参数
    parser.add_argument("--best_ratio", type=float, default=0.55,
                        help="Best ratio from previous sweep")
    parser.add_argument("--size_range", type=float, default=0.05,
                        help="Search range around best_ratio (±)")
    parser.add_argument("--size_step", type=float, default=0.01,
                        help="Size search step")
    # 角度搜索参数
    parser.add_argument("--angle_range", type=float, nargs=2, default=[0, 360],
                        help="Angle search range [min, max]")
    parser.add_argument("--angle_step", type=float, default=0.2,
                        help="Angle search step (degrees)")
    # 其他
    parser.add_argument("--channels", type=str, nargs="+", default=["cb"],
                        choices=["y", "cb", "cr"])
    parser.add_argument("--show_all", action="store_true")
    parser.add_argument("--mode", type=str, default="five_point",
                        choices=["center", "five_point", "random"])
    args = parser.parse_args()

    # Load GT bits
    gt_bits = None
    if args.bits_file:
        gt_bits = load_bits_from_json(args.bits_file)
        if len(gt_bits) != args.num_bits:
            raise ValueError(f"bits_file length ({len(gt_bits)}) must equal --num_bits ({args.num_bits})")
        print(f"GT bits: {bits_to_string(gt_bits)}")

    # List images
    image_paths = list_images(args.input, args.input_dir, args.max_images)
    if not image_paths:
        raise ValueError("No input images found")
    print(f"Found {len(image_paths)} images")

    # Load model
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = WatermarkDecoderV17(
        n_sectors=args.num_bits,
        bits=args.bitsf,
        angle_bins=args.angle_bins,
        radius_bins=12,
        ring_positions_init=[float(r) for r in args.r]
    )

    state_dict = torch.load(args.model_path, map_location=device, weights_only=False)
    if isinstance(state_dict, dict) and 'model' in state_dict:
        state_dict = state_dict['model']
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    if 'ring_positions' in state_dict:
        del state_dict['ring_positions']

    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    print(f"Model loaded: {args.model_path}")
    print(f"Device: {device}")

    # 构建搜索空间
    # 尺寸：best_ratio ± size_range
    size_ratios = []
    r = args.best_ratio - args.size_range
    while r <= args.best_ratio + args.size_range + 1e-12:
        size_ratios.append(r)
        r += args.size_step

    # 角度
    angles = []
    a = args.angle_range[0]
    while a <= args.angle_range[1] + 1e-12:
        angles.append(a)
        a += args.angle_step

    print(f"Size search: ratio [{size_ratios[0]:.2f}, {size_ratios[-1]:.2f}], step={args.size_step}")
    print(f"Angle search: [{angles[0]:.0f}, {angles[-1]:.0f}], step={args.angle_step}°")
    print(f"Total combinations: {len(size_ratios)} × {len(angles)} = {len(size_ratios) * len(angles)}")

    # Sweep
    all_rows = []

    for image_idx, image_path in enumerate(tqdm(image_paths, desc="Images")):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            print(f"WARNING: Failed to read {image_path}")
            continue

        image_name = os.path.basename(image_path)
        h_orig, w_orig = image.shape[:2]
        short_side = min(h_orig, w_orig)

        for channel in args.channels:
            best_score = -1.0
            best_row = None

            for ratio in size_ratios:
                wm_size = int(short_side * ratio)
                if wm_size < args.crop_size:
                    continue

                # resize图片使水印tile=512
                resized, resize_ratio = resize_to_wm512(image, wm_size)

                for angle in angles:
                    # 旋转图片
                    if angle != 0:
                        rotated = rotate_image(resized, angle)
                    else:
                        rotated = resized

                    # 提取通道
                    channel_image = extract_channel(rotated, channel)

                    h, w = channel_image.shape[:2]
                    if h < args.crop_size or w < args.crop_size:
                        continue

                    # 生成crop位置
                    positions = make_crop_positions(
                        h, w, args.crop_size, args.num_crops,
                        args.seed + image_idx, args.mode
                    )

                    # 解码
                    predictions = []
                    valid_positions = []
                    for y, x in positions:
                        if y + args.crop_size <= h and x + args.crop_size <= w:
                            crop = channel_image[y:y + args.crop_size, x:x + args.crop_size].copy()
                            predictions.append(crop)
                            valid_positions.append((y, x))

                    if not predictions:
                        continue

                    # Batch decode
                    preds_list = []
                    with torch.no_grad():
                        for start in range(0, len(predictions), args.batch_size):
                            batch = np.stack(predictions[start:start + args.batch_size], axis=0)
                            tensor = torch.from_numpy(batch).to(device=device, dtype=torch.float32)
                            tensor = tensor.unsqueeze(1).div_(127.5).sub_(1.0)
                            output, _, _ = model(tensor)
                            pred = (output > 0.5).long().cpu().numpy()
                            preds_list.extend(pred)

                    predictions = np.asarray(preds_list, dtype=np.int64)

                    # Vote
                    vote_bits = (np.mean(predictions, axis=0) >= 0.5).astype(np.int64)
                    vote_text = bits_to_string(vote_bits)

                    crop_acc = None
                    vote_acc = None
                    if gt_bits is not None:
                        gt = np.array(gt_bits, dtype=np.int64)
                        crop_accs = np.mean(predictions == gt[None, :], axis=1)
                        crop_acc = float(np.mean(crop_accs) * 100.0)
                        vote_acc = float(np.mean(vote_bits == gt) * 100.0)

                    row = {
                        "image": image_name,
                        "channel": channel,
                        "ratio": ratio,
                        "wm_size": wm_size,
                        "angle": angle,
                        "crop_acc": crop_acc,
                        "vote_acc": vote_acc,
                        "vote_bits": vote_text,
                    }
                    all_rows.append(row)

                    if args.show_all:
                        print(f"  {image_name} | {channel} | ratio={ratio:.2f} | "
                              f"angle={angle:3.0f}° | crop={format_percent(crop_acc)} | "
                              f"vote={format_percent(vote_acc)}")

                    # 更新最佳
                    if vote_acc is not None and vote_acc > best_score:
                        best_score = vote_acc
                        best_row = row

            # 打印每个图片的最佳结果
            if best_row:
                print(f"  {image_name}: best_ratio={best_row['ratio']:.2f}, "
                      f"best_angle={best_row['angle']:.0f}°, vote_acc={best_row['vote_acc']:.2f}%")

    # Aggregate results
    groups = defaultdict(list)
    for row in all_rows:
        key = (row["channel"], row["ratio"], row["angle"])
        groups[key].append(row)

    aggregate_rows = []
    for (channel, ratio, angle), items in groups.items():
        vote_values = [item["vote_acc"] for item in items if item["vote_acc"] is not None]
        aggregate_rows.append([
            channel.upper(),
            f"{ratio:.2f}",
            f"{angle:.0f}°",
            len(items),
            format_percent(float(np.mean(vote_values)) if vote_values else None),
        ])

    # Sort by vote_acc
    aggregate_rows.sort(key=lambda x: -1.0 if x[4] == "N/A" else float(x[4]), reverse=True)

    print(f"\n{'='*80}")
    print(f"v18 Sweep Results (Size + Angle)")
    print(f"Model: {args.model_path}")
    print(f"Images: {len(image_paths)}")
    print(f"Channels: {args.channels}")
    print(f"{'='*80}")

    headers = ["Channel", "Ratio", "Angle", "Images", "Vote Acc (%)"]
    print_table(headers, aggregate_rows[:20])  # Top20

    # Best per image
    best_by_image = {}
    for row in all_rows:
        image = row["image"]
        score = row["vote_acc"] if row["vote_acc"] is not None else -1.0
        if image not in best_by_image or score > best_by_image[image][0]:
            best_by_image[image] = (score, row)

    print(f"\n{'='*80}")
    print("Best per image:")
    print(f"{'='*80}")
    for image, (score, row) in sorted(best_by_image.items()):
        print(f"  {image}: ratio={row['ratio']:.2f}, angle={row['angle']:.0f}°, "
              f"vote={format_percent(row['vote_acc'])}%")

    # Save results
    output_dir = os.path.dirname(args.model_path) if os.path.isfile(args.model_path) else "."
    results_path = os.path.join(output_dir, "sweep_results_v18_angle.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({
            "model_path": args.model_path,
            "images": len(image_paths),
            "channels": args.channels,
            "best_ratio": args.best_ratio,
            "size_range": args.size_range,
            "angle_range": args.angle_range,
            "angle_step": args.angle_step,
            "results": all_rows,
            "aggregate": [{"channel": r[0], "ratio": r[1], "angle": r[2],
                           "images": r[3], "vote_acc": r[4]} for r in aggregate_rows],
        }, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {results_path}")


if __name__ == "__main__":
    main()
