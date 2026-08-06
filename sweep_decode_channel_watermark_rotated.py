"""
v1 版本：针对旋转后图片的角度+尺寸联合搜索解码。

特点：
1. 5个裁剪框：中心1个 + 周围4个靠中心（避免旋转黑边）
2. 搜索角度：0° ~ +5°，共6个角度
3. 对每个角度，再搜索框的大小

用法:
    python sweep_decode_channel_watermark_rotated.py \
        --input_dir /mnt/lpl/fftmask/xiaomi_test_data_watermarked_rotated/4096_3072 \
        --bits_file /mnt/lpl/fftmask/xiaomi_test_data_watermarked/bits.txt \
        --model_path /mnt/lpl/fftmask/output/v1_valnoise/20260629_005346/models/best_cb_decoder.pth
"""

import argparse
import os
import json
import cv2
import numpy as np
import torch
from collections import defaultdict
from tqdm import tqdm

from watermark_decoder3 import AdvancedWatermarkDecoder


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


def make_crop_positions_closer(h, w, crop_size, seed):
    """
    生成5个裁剪位置，周围4个靠近中心（避免旋转黑边）。

    对于旋转后的图片，边缘可能有黑色填充区域，所以：
    - 中心：正中央
    - 上/下/左/右：从边缘向中心偏移 offset 距离
    """
    positions = []
    cy, cx = h // 2, w // 2
    half = crop_size // 2

    # 中心
    positions.append((cy - half, cx - half))

    # 偏移量：使用图片尺寸的1/6，确保远离边缘
    offset_y = max(h // 6, crop_size)
    offset_x = max(w // 6, crop_size)

    # 上方（靠近中心）
    top_y = cy - offset_y - half
    top_y = max(0, top_y)
    positions.append((top_y, cx - half))

    # 下方（靠近中心）
    bottom_y = cy + offset_y - half
    bottom_y = min(h - crop_size, bottom_y)
    positions.append((bottom_y, cx - half))

    # 左侧（靠近中心）
    left_x = cx - offset_x - half
    left_x = max(0, left_x)
    positions.append((cy - half, left_x))

    # 右侧（靠近中心）
    right_x = cx + offset_x - half
    right_x = min(w - crop_size, right_x)
    positions.append((cy - half, right_x))

    return positions


def rotate_image_cv2(image_bgr, angle):
    """旋转图片，使用反射填充避免黑边"""
    if abs(angle) < 0.01:
        return image_bgr
    h, w = image_bgr.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image_bgr, M, (w, h), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT)
    return rotated


def parse_bits_file(path):
    """Parse bits file (0/1 text format)."""
    with open(path, "r", encoding="utf-8") as f:
        bits = [int(ch) for ch in f.read().strip() if ch in "01"]
    if not bits:
        raise ValueError(f"No 0/1 bits found in {path}")
    return np.array(bits, dtype=np.int64)


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


def build_candidates(width, height, args):
    """Build candidate tile sizes for sweeping."""
    candidates = []
    short_side = min(width, height)

    if args.tile_sizes:
        for tile_size in args.tile_sizes:
            candidates.append({
                "label": f"tile={tile_size:g}",
                "crop_size": int(tile_size),
            })
    else:
        ratio = args.min_tile_ratio
        while ratio <= args.max_tile_ratio + 1e-12:
            tile_size = max(args.crop_size, int(round(short_side * ratio)))
            candidates.append({
                "label": f"tile={tile_size}({ratio:.2f}S)",
                "crop_size": tile_size,
            })
            ratio += args.tile_step_ratio

    # Deduplicate
    seen = set()
    unique = []
    for item in candidates:
        key = item["crop_size"]
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def main():
    parser = argparse.ArgumentParser(
        description="v1: 角度+尺寸联合搜索解码（针对旋转后图片）"
    )
    parser.add_argument("--input", type=str, default=None, help="Single input image path")
    parser.add_argument("--input_dir", type=str, default=None, help="Directory of input images")
    parser.add_argument("--bits_file", type=str, default=None,
                        help="Ground-truth bits TXT file")
    parser.add_argument("--model_path", type=str,
                        default="/mnt/lpl/fftmask/output/v1_valnoise/20260629_005346/models/best_cb_decoder.pth",
                        help="Path to decoder model")
    parser.add_argument("--device", type=str, default="2", help="CUDA_VISIBLE_DEVICES value")
    parser.add_argument("--num_bits", type=int, default=60, help="Number of watermark bits")
    parser.add_argument("--r", type=int, nargs="+", default=[12, 25], help="Watermark ring radii")
    parser.add_argument("--bitsf", type=int, nargs="+", default=[15, 45],
                        help="Number of bits assigned to each radius")
    parser.add_argument("--ring_width", type=int, default=5, help="Decoder ring half width")
    parser.add_argument("--crop_size", type=int, default=512, help="Crop size decoded by the model")
    parser.add_argument("--num_crops", type=int, default=5, help="Number of crops per image")
    parser.add_argument("--batch_size", type=int, default=64, help="Decoder batch size")
    parser.add_argument("--max_images", type=int, default=0, help="Limit images; 0 means all")
    parser.add_argument("--seed", type=int, default=2026, help="Random crop seed")
    parser.add_argument("--tile_sizes", type=float, nargs="*", default=None,
                        help="Explicit candidate tile sizes")
    parser.add_argument("--min_tile_ratio", type=float, default=0.45, help="Sweep start ratio")
    parser.add_argument("--max_tile_ratio", type=float, default=1.0, help="Sweep end ratio")
    parser.add_argument("--tile_step_ratio", type=float, default=0.01, help="Sweep step ratio")
    parser.add_argument("--channels", type=str, nargs="+", default=["cb"],
                        choices=["y", "cb", "cr"], help="Channels to sweep")
    parser.add_argument("--angle_start", type=float, default=-5.0, help="Start angle for sweep")
    parser.add_argument("--angle_end", type=float, default=5.0, help="End angle for sweep")
    parser.add_argument("--angle_step", type=float, default=1.0, help="Angle step for sweep")
    parser.add_argument("--show_all", action="store_true", help="Print every result")
    args = parser.parse_args()

    # 生成角度列表
    angles = []
    angle = args.angle_start
    while angle <= args.angle_end + 1e-9:
        angles.append(round(angle, 1))
        angle += args.angle_step
    print(f"Angle sweep: {angles}")

    # Load GT bits
    gt_bits = None
    if args.bits_file:
        gt_bits = parse_bits_file(args.bits_file)
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

    model = AdvancedWatermarkDecoder(
        n_sectors=args.num_bits,
        rings=[(r - args.ring_width, r + args.ring_width) for r in args.r],
        bits=args.bitsf,
    )
    state_dict = torch.load(args.model_path, map_location=device, weights_only=False)
    if isinstance(state_dict, dict) and 'model' in state_dict:
        state_dict = state_dict['model']
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    print(f"Model loaded: {args.model_path}")
    print(f"Device: {device}")

    # Sweep
    all_rows = []

    for image_idx, image_path in enumerate(tqdm(image_paths, desc="Images")):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            print(f"WARNING: Failed to read {image_path}")
            continue

        image_name = os.path.basename(image_path)
        h_orig, w_orig = image.shape[:2]
        candidates = build_candidates(w_orig, h_orig, args)

        for channel in args.channels:
            for candidate in candidates:
                wm_size = candidate["crop_size"]
                # 确保wm_size不超过图片尺寸
                if wm_size > min(h_orig, w_orig):
                    wm_size = min(h_orig, w_orig)

                # 对每个角度进行搜索
                for angle in angles:
                    # 旋转图片
                    rotated = rotate_image_cv2(image, angle)

                    h, w = rotated.shape[:2]
                    # 在旋转后的图上裁剪wm_size x wm_size，靠近中心
                    positions = make_crop_positions_closer(h, w, wm_size, args.seed + image_idx)

                    # 提取通道并裁剪，然后resize到512x512
                    channel_image = extract_channel(rotated, channel)
                    crops = []
                    valid_positions = []
                    for y, x in positions:
                        if y + wm_size <= h and x + wm_size <= w:
                            crop_large = channel_image[y:y + wm_size, x:x + wm_size].copy()
                            # resize到512x512
                            crop = cv2.resize(crop_large, (args.crop_size, args.crop_size), interpolation=cv2.INTER_AREA)
                            crops.append(crop)
                            valid_positions.append((y, x))

                    if not crops:
                        continue

                    # Decode
                    predictions = []
                    with torch.no_grad():
                        for start in range(0, len(crops), args.batch_size):
                            batch = np.stack(crops[start:start + args.batch_size], axis=0)
                            tensor = torch.from_numpy(batch).to(device=device, dtype=torch.float32)
                            tensor = tensor.unsqueeze(1).div_(127.5).sub_(1.0)
                            output, _, _ = model(tensor)
                            pred = (output > 0.5).long().cpu().numpy()
                            predictions.extend(pred)

                    predictions = np.asarray(predictions, dtype=np.int64)

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
                        "candidate": candidate["label"],
                        "crop_size": wm_size,
                        "angle": angle,
                        "crop_acc": crop_acc,
                        "vote_acc": vote_acc,
                        "vote_bits": vote_text,
                    }
                    all_rows.append(row)

                    if args.show_all:
                        print(f"  {image_name} | {channel} | {candidate['label']} | "
                              f"angle={angle:+.1f}° | crop={format_percent(crop_acc)} | vote={format_percent(vote_acc)}")

    # Aggregate results - 按 (channel, candidate, angle) 分组
    groups = defaultdict(list)
    for row in all_rows:
        key = (row["channel"], row["candidate"], row["angle"])
        groups[key].append(row)

    aggregate_rows = []
    for (channel, candidate, angle), items in groups.items():
        crop_values = [item["crop_acc"] for item in items if item["crop_acc"] is not None]
        vote_values = [item["vote_acc"] for item in items if item["vote_acc"] is not None]
        aggregate_rows.append([
            channel.upper(),
            candidate,
            f"{angle:+.1f}°",
            len(items),
            format_percent(float(np.mean(crop_values)) if crop_values else None),
            format_percent(float(np.mean(vote_values)) if vote_values else None),
        ])

    # Sort by vote_acc
    aggregate_rows.sort(key=lambda x: -1.0 if x[5] == "N/A" else float(x[5]), reverse=True)

    print(f"\n{'='*100}")
    print(f"v1 Rotated Image Sweep Results (Angle + Size)")
    print(f"Model: {args.model_path}")
    print(f"Images: {len(image_paths)}")
    print(f"Channels: {args.channels}")
    print(f"Angles: {angles}")
    print(f"{'='*100}")

    headers = ["Channel", "Candidate", "Angle", "Images", "Crop Acc (%)", "Vote Acc (%)"]
    print_table(headers, aggregate_rows)

    # Best per image (across all angles and sizes)
    best_by_image = {}
    for row in all_rows:
        image = row["image"]
        score = row["vote_acc"] if row["vote_acc"] is not None else -1.0
        if image not in best_by_image or score > best_by_image[image][0]:
            best_by_image[image] = (score, row)

    print(f"\n{'='*100}")
    print("Best per image (across all angles and sizes):")
    print(f"{'='*100}")
    for image, (score, row) in sorted(best_by_image.items()):
        print(f"  {image}: {row['channel']} | {row['candidate']} | "
              f"angle={row['angle']:+.1f}° | vote={format_percent(row['vote_acc'])}%")

    # Best angle summary
    angle_groups = defaultdict(list)
    for row in all_rows:
        if row["vote_acc"] is not None:
            angle_groups[row["angle"]].append(row["vote_acc"])

    print(f"\n{'='*100}")
    print("Average accuracy by angle:")
    print(f"{'='*100}")
    angle_summary = []
    for angle, accs in sorted(angle_groups.items()):
        avg = np.mean(accs)
        angle_summary.append((angle, avg))
    for angle, avg in sorted(angle_summary, key=lambda x: -x[1]):
        print(f"  {angle:+.1f}°: {avg:.2f}%")

    # Save results
    output_dir = os.path.dirname(args.model_path) if os.path.isfile(args.model_path) else "."
    results_path = os.path.join(output_dir, "sweep_rotated_results_v1.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({
            "model_path": args.model_path,
            "images": len(image_paths),
            "channels": args.channels,
            "angles": angles,
            "results": all_rows,
            "aggregate": [{"channel": r[0], "candidate": r[1], "angle": r[2],
                           "images": r[3], "crop_acc": r[4], "vote_acc": r[5]} for r in aggregate_rows],
        }, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {results_path}")


if __name__ == "__main__":
    main()
