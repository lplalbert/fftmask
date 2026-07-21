import argparse
import os
import re
import subprocess
import sys
from collections import defaultdict

import cv2
import numpy as np
import torch

from decode_channel_watermark import (
    bits_to_string,
    extract_channel,
    load_model,
    make_crop_positions,
    parse_bits_file,
)


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def parse_resolution(value):
    text = value.lower().replace("*", "x")
    parts = text.split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"Invalid resolution '{value}', expected WIDTHxHEIGHT")
    try:
        width = int(parts[0])
        height = int(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid resolution '{value}', expected integers") from exc
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("Resolution width and height must be positive")
    return width, height


def list_images(input_path, input_dir, max_images):
    if input_path:
        return [input_path]
    if not input_dir:
        raise ValueError("Either --input or --input_dir must be provided")
    names = sorted(f for f in os.listdir(input_dir) if f.lower().endswith(IMAGE_EXTENSIONS))
    if max_images > 0:
        names = names[:max_images]
    return [os.path.join(input_dir, name) for name in names]


def read_image_size(path):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    h, w = image.shape[:2]
    return w, h


def build_candidates(width, height, args):
    candidates = []

    short_side = min(width, height)

    if args.tile_sizes:
        for tile_size in args.tile_sizes:
            candidates.append({
                "label": f"tile={tile_size:g}",
                "crop_size": int(tile_size),
            })
    else:
        if args.min_tile_ratio <= 0 or args.max_tile_ratio <= 0:
            raise ValueError("--min_tile_ratio and --max_tile_ratio must be positive")
        if args.tile_step_ratio <= 0:
            raise ValueError("--tile_step_ratio must be positive")
        if args.min_tile_ratio > args.max_tile_ratio:
            raise ValueError("--min_tile_ratio must be <= --max_tile_ratio")

        ratio = args.min_tile_ratio
        while ratio <= args.max_tile_ratio + 1e-12:
            tile_size = max(args.crop_size, int(round(short_side * ratio)))
            candidates.append({
                "label": f"tile={tile_size}({ratio:.2f}S)",
                "crop_size": tile_size,
            })
            ratio += args.tile_step_ratio

        max_tile_size = max(args.crop_size, int(round(short_side * args.max_tile_ratio)))
        if not candidates or candidates[-1]["crop_size"] != max_tile_size:
            candidates.append({
                "label": f"tile={max_tile_size}({args.max_tile_ratio:.2f}S)",
                "crop_size": max_tile_size,
            })

    # Add min_edge candidates (resize shortest edge, then crop at crop_size)
    if args.min_edges:
        for min_edge in args.min_edges:
            candidates.append({
                "label": f"min_edge={min_edge}",
                "min_edge": min_edge,
                "crop_size": args.crop_size,  # Always crop at 512
            })

    for scale in args.scales:
        tile_size = max(args.crop_size, int(round(short_side * scale)))
        candidates.append({
            "label": f"scale={scale:g}",
            "crop_size": tile_size,
        })

    for resize_w, resize_h in args.resolutions:
        tile_size = max(args.crop_size, min(resize_w, resize_h))
        candidates.append({
            "label": f"resize={resize_w}x{resize_h}",
            "crop_size": tile_size,
        })

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

    # Add min_edge resize candidates
    if args.min_edges:
        for min_edge in args.min_edges:
            short_side = min(width, height)
            if short_side <= min_edge:
                resize_w, resize_h = width, height
            else:
                scale = min_edge / short_side
                resize_w = max(args.crop_size, int(round(width * scale)))
                resize_h = max(args.crop_size, int(round(height * scale)))
            candidates.append({
                "label": f"min_edge={min_edge}",
                "resize_w": resize_w,
                "resize_h": resize_h,
            })

    for scale in args.scales:
        resize_w = max(args.crop_size, int(round(width * scale)))
        resize_h = max(args.crop_size, int(round(height * scale)))
        candidates.append({
            "label": f"scale={scale:g}",
            "resize_w": resize_w,
            "resize_h": resize_h,
        })

    for resize_w, resize_h in args.resolutions:
        candidates.append({
            "label": f"resize={resize_w}x{resize_h}",
            "resize_w": resize_w,
            "resize_h": resize_h,
        })

    seen = set()
    unique = []
    for item in candidates:
        key = (item["label"], item["resize_w"], item["resize_h"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def bits_file_for_channel(args, channel):
    if args.bits_file:
        return args.bits_file
    if args.bits_dir:
        path = os.path.join(args.bits_dir, f"watermark_signal_{channel}.txt")
        if os.path.exists(path):
            return path
    return None


def parse_decoder_output(stdout):
    result = {
        "crop_acc": None,
        "vote_acc": None,
        "vote_exact_num": None,
        "vote_exact_den": None,
        "vote_bits": "",
        "decoded_images": None,
    }

    match = re.search(r"Average crop accuracy:\s*([0-9.]+)%", stdout)
    if match:
        result["crop_acc"] = float(match.group(1))

    match = re.search(r"Average vote accuracy:\s*([0-9.]+)%", stdout)
    if match:
        result["vote_acc"] = float(match.group(1))

    match = re.search(r"Vote exact 60-bit:\s*(\d+)/(\d+)", stdout)
    if match:
        result["vote_exact_num"] = int(match.group(1))
        result["vote_exact_den"] = int(match.group(2))

    match = re.search(r"Decoded images:\s*(\d+)", stdout)
    if match:
        result["decoded_images"] = int(match.group(1))

    for line in stdout.splitlines():
        if not line.startswith("| "):
            continue
        if "Vote Bits" in line or "-----" in line:
            continue
        parts = [part.strip() for part in line.strip().strip("|").split("|")]
        if len(parts) >= 5:
            result["vote_bits"] = parts[4]

    return result


def run_decoder(args, image_path, channel, candidate):
    bits_file = bits_file_for_channel(args, channel)
    cmd = [
        sys.executable,
        args.decoder_script,
        "--input", image_path,
        "--channel", channel,
        "--model_path", args.model_path,
        "--device", args.device,
        "--num_bits", str(args.num_bits),
        "--crop_size", str(args.crop_size),
        "--sample_mode", args.sample_mode,
        "--num_crops", str(args.num_crops),
        "--batch_size", str(args.batch_size),
        "--ring_width", str(args.ring_width),
        "--seed", str(args.seed),
        "--resize_width", str(candidate["resize_w"]),
        "--resize_height", str(candidate["resize_h"]),
    ]
    cmd.extend(["--r"] + [str(v) for v in args.r])
    cmd.extend(["--bitsf"] + [str(v) for v in args.bitsf])
    if bits_file:
        cmd.extend(["--bits_file", bits_file])

    completed = subprocess.run(cmd, check=False, text=True, capture_output=True)
    parsed = parse_decoder_output(completed.stdout)
    parsed.update({
        "returncode": completed.returncode,
        "stderr": completed.stderr.strip(),
        "stdout": completed.stdout,
        "bits_file": bits_file or "",
    })
    return parsed


def decode_crop_channels(model, crop_channels, device, batch_size):
    if not crop_channels:
        return np.empty((0, 0), dtype=np.int64)

    predictions = []
    with torch.no_grad():
        for start in range(0, len(crop_channels), batch_size):
            batch = np.stack(crop_channels[start:start + batch_size], axis=0)
            tensor = torch.from_numpy(batch).to(device=device, dtype=torch.float32)
            tensor = tensor.unsqueeze(1).div_(127.5).sub_(1.0)
            output, _, _ = model(tensor)
            pred = (output > 0.5).long().cpu().numpy()
            predictions.extend(pred)
    return np.asarray(predictions, dtype=np.int64)


def load_bits_for_channel(args, channel, cache):
    bits_file = bits_file_for_channel(args, channel)
    if not bits_file:
        return None
    if bits_file not in cache:
        bits = parse_bits_file(bits_file)
        if len(bits) != args.num_bits:
            raise ValueError(f"bits_file length ({len(bits)}) must equal --num_bits ({args.num_bits})")
        cache[bits_file] = bits
    return cache[bits_file]


def row_from_predictions(args, image_name, channel, candidate, predictions, gt_bits):
    resize_text = f"{candidate.get('crop_size', args.crop_size)}x{candidate.get('crop_size', args.crop_size)}"
    if len(predictions) == 0:
        return {
            "image": image_name,
            "channel": channel,
            "candidate": candidate["label"],
            "resize": resize_text,
            "crop_acc": None,
            "vote_acc": None,
            "vote_exact_num": None,
            "vote_exact_den": None,
            "vote_bits": "",
            "returncode": 1,
            "stderr": "no valid crops",
        }

    vote_bits = (np.mean(predictions, axis=0) >= 0.5).astype(np.int64)
    vote_text = bits_to_string(vote_bits)
    crop_acc = None
    vote_acc = None
    vote_exact_num = None
    vote_exact_den = None

    if gt_bits is not None:
        crop_accs = np.mean(predictions == gt_bits[None, :], axis=1)
        crop_acc = float(np.mean(crop_accs) * 100.0)
        vote_acc = float(np.mean(vote_bits == gt_bits) * 100.0)
        vote_exact_num = int(vote_acc == 100.0)
        vote_exact_den = 1

    return {
        "image": image_name,
        "channel": channel,
        "candidate": candidate["label"],
        "resize": resize_text,
        "crop_acc": crop_acc,
        "vote_acc": vote_acc,
        "vote_exact_num": vote_exact_num,
        "vote_exact_den": vote_exact_den,
        "vote_bits": vote_text,
        "returncode": 0,
        "stderr": "",
    }


def run_batched_decode(args, model, device, image_path, image_idx, channel, candidates, gt_bits):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image_name = os.path.basename(image_path)
    if image is None:
        return [{
            "image": image_name,
            "channel": channel,
            "candidate": "N/A",
            "resize": "N/A",
            "crop_acc": None,
            "vote_acc": None,
            "vote_exact_num": None,
            "vote_exact_den": None,
            "vote_bits": "",
            "returncode": 1,
            "stderr": f"failed to read image: {image_path}",
        }]

    crop_channels = []
    candidate_slices = []
    for candidate in candidates:
        crop_size = args.crop_size  # Always 512
        min_edge = candidate.get("min_edge", None)

        if min_edge:
            # Resize so shortest edge equals min_edge
            h_orig, w_orig = image.shape[:2]
            short_side = min(h_orig, w_orig)
            scale = min_edge / short_side
            new_w = max(crop_size, int(round(w_orig * scale)))
            new_h = max(crop_size, int(round(h_orig * scale)))
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            # Original behavior: resize to candidate's tile_size as shortest edge
            tile_size = candidate.get("crop_size", args.crop_size)
            h_orig, w_orig = image.shape[:2]
            short_side = min(h_orig, w_orig)
            scale = tile_size / short_side
            new_w = max(crop_size, int(round(w_orig * scale)))
            new_h = max(crop_size, int(round(h_orig * scale)))
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        h, w = resized.shape[:2]

        # Random crop positions from the resized image
        positions = make_crop_positions(
            h,
            w,
            crop_size,
            args.num_crops,
            args.seed + image_idx,
            args.sample_mode,
        )
        start = len(crop_channels)
        channel_image = extract_channel(resized, channel)
        for y, x in positions:
            crop = channel_image[y:y + crop_size, x:x + crop_size].copy()
            crop_channels.append(crop)
        end = len(crop_channels)
        candidate_slices.append((candidate, start, end))

    print(
        f"decoding image={image_name}, channel={channel.upper()}, "
        f"candidates={len(candidates)}, crops={len(crop_channels)}, batch_size={args.batch_size}",
        flush=True,
    )
    predictions = decode_crop_channels(model, crop_channels, device, args.batch_size)

    rows = []
    for candidate, start, end in candidate_slices:
        rows.append(row_from_predictions(args, image_name, channel, candidate, predictions[start:end], gt_bits))
    return rows


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


def aggregate_results(rows):
    groups = defaultdict(list)
    for row in rows:
        key = (row["channel"], row["candidate"], row["resize"])
        groups[key].append(row)

    aggregate_rows = []
    for (channel, candidate, resize), items in groups.items():
        crop_values = [item["crop_acc"] for item in items if item["crop_acc"] is not None]
        vote_values = [item["vote_acc"] for item in items if item["vote_acc"] is not None]
        exact_num = sum(item["vote_exact_num"] or 0 for item in items)
        exact_den = sum(item["vote_exact_den"] or 0 for item in items)
        aggregate_rows.append([
            channel.upper(),
            candidate,
            resize,
            len(items),
            format_percent(float(np.mean(crop_values)) if crop_values else None),
            format_percent(float(np.mean(vote_values)) if vote_values else None),
            f"{exact_num}/{exact_den}" if exact_den else "N/A",
        ])

    def sort_key(row):
        vote = -1.0 if row[5] == "N/A" else float(row[5])
        crop = -1.0 if row[4] == "N/A" else float(row[4])
        return (vote, crop)

    return sorted(aggregate_rows, key=sort_key, reverse=True)


def best_rows_by_image(rows):
    best = {}
    for row in rows:
        image = row["image"]
        score = (
            -1.0 if row["vote_acc"] is None else row["vote_acc"],
            -1.0 if row["crop_acc"] is None else row["crop_acc"],
        )
        if image not in best or score > best[image][0]:
            best[image] = (score, row)

    output = []
    for image, (_, row) in sorted(best.items()):
        output.append([
            image,
            row["channel"].upper(),
            row["candidate"],
            row["resize"],
            format_percent(row["crop_acc"]),
            format_percent(row["vote_acc"]),
            row["vote_bits"],
        ])
    return output


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Sweep candidate watermark scales with batched in-process decoding. "
            "Useful when the 512x512 watermark tile may appear larger or smaller in the detected image."
        )
    )
    parser.add_argument("--input", type=str, default=None, help="Single input image path")
    parser.add_argument("--input_dir", type=str, default=None, help="Directory of input images")
    parser.add_argument("--channels", type=str, nargs="+", default=["y", "cb"], choices=["y", "cb", "cr"],
                        help="Channels to sweep")
    parser.add_argument("--bits_file", type=str, default=None,
                        help="Ground-truth bit TXT used for all channels")
    parser.add_argument("--bits_dir", type=str, default=None,
                        help="Directory containing watermark_signal_y.txt / watermark_signal_cb.txt")
    parser.add_argument("--model_path", type=str,
                        default="/mnt/lpl/fftmask/output/v1_valnoise/20260629_005346/models/best_cb_decoder.pth",
                        help="Path to trained decoder model")
    parser.add_argument("--decoder_script", type=str, default="decode_channel_watermark.py",
                        help="Path to decode_channel_watermark.py; only used with --use_subprocess")
    parser.add_argument("--device", type=str, default="0", help="CUDA_VISIBLE_DEVICES value passed to decoder")
    parser.add_argument("--num_bits", type=int, default=60, help="Number of watermark bits")
    parser.add_argument("--r", type=int, nargs="+", default=[12, 25], help="Watermark ring radii")
    parser.add_argument("--bitsf", type=int, nargs="+", default=[15, 45],
                        help="Number of bits assigned to each radius")
    parser.add_argument("--ring_width", type=int, default=5,
                        help="Decoder ring half width, default produces (r-5, r+5)")
    parser.add_argument("--crop_size", type=int, default=512, help="Crop size decoded by the model")
    parser.add_argument("--sample_mode", type=str, default="five_point", choices=["five_point", "random"],
                        help="Sampling mode passed to decode_channel_watermark.py")
    parser.add_argument("--num_crops", type=int, default=6,
                        help="Random crops per image for each candidate when --sample_mode random is used")
    parser.add_argument("--batch_size", type=int, default=64, help="Decoder batch size")
    parser.add_argument("--max_images", type=int, default=0, help="Limit images when --input_dir is used; 0 means all")
    parser.add_argument("--seed", type=int, default=2026, help="Random crop seed")
    parser.add_argument("--tile_sizes", type=float, nargs="*", default=None,
                        help=(
                            "Optional explicit candidate apparent tile sizes in the input image. "
                            "When omitted, candidates are generated from the shortest image side."
                        ))
    parser.add_argument("--min_tile_ratio", type=float, default=0.45,
                        help="Default sweep start: shortest_side * min_tile_ratio; default is 0.45")
    parser.add_argument("--max_tile_ratio", type=float, default=1.0,
                        help="Default sweep end: shortest_side * max_tile_ratio")
    parser.add_argument("--tile_step_ratio", type=float, default=0.01,
                        help="Default sweep step: shortest_side * tile_step_ratio")
    parser.add_argument("--min_edges", type=int, nargs="*", default=None,
                        help="Resize shortest edge to these values before cropping")
    parser.add_argument("--scales", type=float, nargs="*", default=[],
                        help="Extra direct image scale factors to evaluate")
    parser.add_argument("--resolutions", type=parse_resolution, nargs="*", default=[],
                        help="Extra explicit resize resolutions, for example: 1920x1080 2560x1440")
    parser.add_argument("--show_all", action="store_true", help="Print every image/channel/candidate result")
    parser.add_argument("--use_subprocess", action="store_true",
                        help="Use legacy mode that calls decode_channel_watermark.py once per candidate")
    return parser


def main():
    args = build_arg_parser().parse_args()
    if len(args.r) != len(args.bitsf):
        raise ValueError("--r and --bitsf must have the same length")
    if sum(args.bitsf) != args.num_bits:
        raise ValueError(f"--num_bits ({args.num_bits}) must equal sum(--bitsf) ({sum(args.bitsf)})")

    image_paths = list_images(args.input, args.input_dir, args.max_images)
    if not image_paths:
        raise ValueError("No input images found")

    all_rows = []
    if args.use_subprocess:
        total_runs = 0
        for image_path in image_paths:
            width, height = read_image_size(image_path)
            candidates = build_candidates(width, height, args)
            image_name = os.path.basename(image_path)
            for channel in args.channels:
                for candidate in candidates:
                    total_runs += 1
                    print(
                        f"[{total_runs}] decoding image={image_name}, channel={channel.upper()}, "
                        f"{candidate['label']} -> {candidate['resize_w']}x{candidate['resize_h']}",
                        flush=True,
                    )
                    parsed = run_decoder(args, image_path, channel, candidate)
                    resize_text = f"{candidate['resize_w']}x{candidate['resize_h']}"
                    row = {
                        "image": image_name,
                        "channel": channel,
                        "candidate": candidate["label"],
                        "resize": resize_text,
                        "crop_acc": parsed["crop_acc"],
                        "vote_acc": parsed["vote_acc"],
                        "vote_exact_num": parsed["vote_exact_num"],
                        "vote_exact_den": parsed["vote_exact_den"],
                        "vote_bits": parsed["vote_bits"],
                        "returncode": parsed["returncode"],
                        "stderr": parsed["stderr"],
                    }
                    all_rows.append(row)
                    if parsed["returncode"] != 0:
                        print(f"  decoder failed: {parsed['stderr']}")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_model(args, device)
        bits_cache = {}
        print(f"Device: {device}")
        print(f"Model loaded once: {args.model_path}")

        for image_idx, image_path in enumerate(image_paths):
            width, height = read_image_size(image_path)
            candidates = build_candidates(width, height, args)
            for channel in args.channels:
                gt_bits = load_bits_for_channel(args, channel, bits_cache)
                rows = run_batched_decode(
                    args,
                    model,
                    device,
                    image_path,
                    image_idx,
                    channel,
                    candidates,
                    gt_bits,
                )
                all_rows.extend(rows)
                for row in rows:
                    if row["returncode"] != 0:
                        print(f"  decoder failed: {row['stderr']}")

    print("\nAggregate Results")
    print_table(
        ["Channel", "Candidate", "Resize", "Images", "Crop Acc(%)", "Vote Acc(%)", "Vote Exact"],
        aggregate_results(all_rows),
    )

    print("\nBest Candidate Per Image")
    print_table(
        ["Image", "Channel", "Candidate", "Resize", "Crop Acc(%)", "Vote Acc(%)", "Vote Bits"],
        best_rows_by_image(all_rows),
    )

    if args.show_all:
        print("\nAll Results")
        rows = []
        for row in all_rows:
            rows.append([
                row["image"],
                row["channel"].upper(),
                row["candidate"],
                row["resize"],
                format_percent(row["crop_acc"]),
                format_percent(row["vote_acc"]),
                row["vote_bits"],
            ])
        print_table(["Image", "Channel", "Candidate", "Resize", "Crop Acc(%)", "Vote Acc(%)", "Vote Bits"], rows)


if __name__ == "__main__":
    main()
