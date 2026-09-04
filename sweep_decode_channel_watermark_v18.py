"""
v18 版本：Sweep candidate watermark scales with batched in-process decoding.

使用 v18 解码器（2环：r=12, r=25，无旋转矫正环）。

用法:
    python sweep_decode_channel_watermark_v18.py \
        --input_dir /path/to/images \
        --bits_file /path/to/embed_gt_mapping.json \
        --model_path /path/to/best_model.pth
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


def make_crop_positions(h, w, crop_size, num_crops, seed, mode="five_point"):
    """Generate crop positions for an image."""
    positions = []
    if mode == "five_point":
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
            y = rng.integers(0, h - crop_size + 1)
            x = rng.integers(0, w - crop_size + 1)
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
        description="v18: Sweep candidate watermark scales with batched in-process decoding."
    )
    parser.add_argument("--input", type=str, default=None, help="Single input image path")
    parser.add_argument("--input_dir", type=str, default=None, help="Directory of input images")
    parser.add_argument("--bits_file", type=str, default=None,
                        help="Ground-truth bits JSON (embed_gt_mapping.json)")
    parser.add_argument("--model_path", type=str,
                        default="/data/lpl/fftmask/output/v18_hollow_pair/best_model.pth",
                        help="Path to v18 decoder model")
    parser.add_argument("--device", type=str, default="0", help="CUDA_VISIBLE_DEVICES value")
    parser.add_argument("--num_bits", type=int, default=60, help="Number of watermark bits")
    parser.add_argument("--r", type=int, nargs="+", default=[12, 25], help="Watermark ring radii")
    parser.add_argument("--bitsf", type=int, nargs="+", default=[15, 45],
                        help="Number of bits assigned to each radius")
    parser.add_argument("--angle_bins", type=int, default=180, help="Angle bins for decoder")
    parser.add_argument("--ring_width", type=int, default=5, help="Decoder ring half width")
    parser.add_argument("--crop_size", type=int, default=512, help="Crop size decoded by the model")
    parser.add_argument("--sample_mode", type=str, default="five_point",
                        choices=["five_point", "random"], help="Sampling mode")
    parser.add_argument("--num_crops", type=int, default=6, help="Random crops per image")
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
    parser.add_argument("--show_all", action="store_true", help="Print every result")
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

    # v18 模型 (2 rings: r=12, r=25, no rotation ring)
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

    # 移除ring_positions，保留模型初始化时的值
    if 'ring_positions' in state_dict:
        del state_dict['ring_positions']

    model.load_state_dict(state_dict, strict=False)
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
                tile_size = candidate["crop_size"]
                short_side = min(h_orig, w_orig)
                scale = tile_size / short_side
                new_w = max(args.crop_size, int(round(w_orig * scale)))
                new_h = max(args.crop_size, int(round(h_orig * scale)))
                resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

                h, w = resized.shape[:2]
                positions = make_crop_positions(
                    h, w, args.crop_size, args.num_crops,
                    args.seed + image_idx, args.sample_mode
                )

                # Extract channel and crop
                channel_image = extract_channel(resized, channel)
                crops = []
                valid_positions = []
                for y, x in positions:
                    if y + args.crop_size <= h and x + args.crop_size <= w:
                        crop = channel_image[y:y + args.crop_size, x:x + args.crop_size].copy()
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
                    "crop_size": tile_size,
                    "crop_acc": crop_acc,
                    "vote_acc": vote_acc,
                    "vote_bits": vote_text,
                }
                all_rows.append(row)

                if args.show_all:
                    print(f"  {image_name} | {channel} | {candidate['label']} | "
                          f"crop={format_percent(crop_acc)} | vote={format_percent(vote_acc)}")

    # Aggregate results
    groups = defaultdict(list)
    for row in all_rows:
        key = (row["channel"], row["candidate"])
        groups[key].append(row)

    aggregate_rows = []
    for (channel, candidate), items in groups.items():
        crop_values = [item["crop_acc"] for item in items if item["crop_acc"] is not None]
        vote_values = [item["vote_acc"] for item in items if item["vote_acc"] is not None]
        aggregate_rows.append([
            channel.upper(),
            candidate,
            len(items),
            format_percent(float(np.mean(crop_values)) if crop_values else None),
            format_percent(float(np.mean(vote_values)) if vote_values else None),
        ])

    # Sort by vote_acc
    aggregate_rows.sort(key=lambda x: -1.0 if x[4] == "N/A" else float(x[4]), reverse=True)

    print(f"\n{'='*80}")
    print(f"v18 Sweep Results")
    print(f"Model: {args.model_path}")
    print(f"Images: {len(image_paths)}")
    print(f"Channels: {args.channels}")
    print(f"{'='*80}")

    headers = ["Channel", "Candidate", "Images", "Crop Acc (%)", "Vote Acc (%)"]
    print_table(headers, aggregate_rows)

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
        print(f"  {image}: {row['channel']} | {row['candidate']} | vote={format_percent(row['vote_acc'])}%")

    # Save results
    output_dir = os.path.dirname(args.model_path) if os.path.isfile(args.model_path) else "."
    results_path = os.path.join(output_dir, "sweep_results_v18.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({
            "model_path": args.model_path,
            "images": len(image_paths),
            "channels": args.channels,
            "results": all_rows,
            "aggregate": [{"channel": r[0], "candidate": r[1], "images": r[2],
                           "crop_acc": r[3], "vote_acc": r[4]} for r in aggregate_rows],
        }, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {results_path}")


if __name__ == "__main__":
    main()
