import argparse
import os

import cv2
import numpy as np
import torch

from watermark_decoder3 import AdvancedWatermarkDecoder


YCRCB_CHANNEL_INDEX = {
    "y": 0,
    "cr": 1,
    "cb": 2,
}


def parse_bits_file(path):
    with open(path, "r", encoding="utf-8") as f:
        bits = [int(ch) for ch in f.read().strip() if ch in "01"]
    if not bits:
        raise ValueError(f"No 0/1 bits found in {path}")
    return np.array(bits, dtype=np.int64)


def bits_to_string(bits):
    return "".join(str(int(v)) for v in bits)


def list_images(input_path=None, input_dir=None, max_images=0):
    if input_path:
        return [input_path]
    if not input_dir:
        raise ValueError("Either --input or --input_dir must be provided")
    names = sorted(
        f for f in os.listdir(input_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
    )
    if max_images > 0:
        names = names[:max_images]
    return [os.path.join(input_dir, name) for name in names]


def extract_channel(image_bgr, channel):
    ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
    return ycrcb[:, :, YCRCB_CHANNEL_INDEX[channel]]


def unique_positions(positions):
    seen = set()
    result = []
    for pos in positions:
        if pos in seen:
            continue
        seen.add(pos)
        result.append(pos)
    return result


def make_five_point_positions(height, width, crop_size):
    if height < crop_size or width < crop_size:
        return []

    max_y = height - crop_size
    max_x = width - crop_size
    positions = [
        (0, 0),
        (0, max_x),
        (max_y, 0),
        (max_y, max_x),
        (max_y // 2, max_x // 2),
    ]
    return unique_positions(positions)


def make_random_positions(height, width, crop_size, num_crops, seed):
    if height < crop_size or width < crop_size:
        return []
    if height == crop_size and width == crop_size:
        return [(0, 0)]

    rng = np.random.default_rng(seed)
    positions = []
    for _ in range(num_crops):
        y = int(rng.integers(0, height - crop_size + 1))
        x = int(rng.integers(0, width - crop_size + 1))
        positions.append((y, x))
    return positions


def make_crop_positions(height, width, crop_size, num_crops, seed, sample_mode):
    if sample_mode == "five_point":
        return make_five_point_positions(height, width, crop_size)
    if sample_mode == "random":
        return make_random_positions(height, width, crop_size, num_crops, seed)
    raise ValueError(f"Unsupported sample_mode: {sample_mode}")


def load_model(args, device):
    model = AdvancedWatermarkDecoder(
        n_sectors=args.num_bits,
        rings=[(r - args.ring_width, r + args.ring_width) for r in args.r],
        bits=args.bitsf,
    )
    state_dict = torch.load(args.model_path, map_location=device)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


def decode_crops(model, crop_channels, device, batch_size):
    predictions = []
    with torch.no_grad():
        for start in range(0, len(crop_channels), batch_size):
            batch = crop_channels[start:start + batch_size]
            tensor = torch.from_numpy(np.stack(batch, axis=0)).to(device=device, dtype=torch.float32)
            tensor = tensor.unsqueeze(1).div_(127.5).sub_(1.0)
            output, _, _ = model(tensor)
            pred = (output > 0.5).long().cpu().numpy()
            predictions.extend(pred)
    return np.asarray(predictions, dtype=np.int64)


def main():
    parser = argparse.ArgumentParser(description="Decode watermark bits from a selected YCrCb channel.")
    parser.add_argument("--input", type=str, default=None, help="Single input image path")
    parser.add_argument("--input_dir", type=str, default=None, help="Directory of input images")
    parser.add_argument("--channel", type=str, default="y", choices=sorted(YCRCB_CHANNEL_INDEX.keys()),
                        help="YCrCb channel used for decoding")
    parser.add_argument("--model_path", type=str,
                        default="/mnt/lpl/fftmask/output/v1_valnoise/20260629_005346/models/best_cb_decoder.pth",
                        help="Path to trained decoder model")
    parser.add_argument("--bits_file", type=str, default=None,
                        help="Optional TXT file with ground-truth watermark bits for accuracy calculation")
    parser.add_argument("--device", type=str, default="0", help="CUDA_VISIBLE_DEVICES value")
    parser.add_argument("--num_bits", type=int, default=60, help="Number of watermark bits")
    parser.add_argument("--r", type=int, nargs="+", default=[12, 25], help="Watermark ring radii")
    parser.add_argument("--bitsf", type=int, nargs="+", default=[15, 45],
                        help="Number of bits assigned to each radius")
    parser.add_argument("--ring_width", type=int, default=5,
                        help="Decoder ring half width, default produces (r-5, r+5)")
    parser.add_argument("--crop_size", type=int, default=512, help="Crop size decoded by the model")
    parser.add_argument("--sample_mode", type=str, default="five_point", choices=["five_point", "random"],
                        help="five_point samples four corners plus center; random uses --num_crops random crops")
    parser.add_argument("--num_crops", type=int, default=6,
                        help="Number of random crops per image when --sample_mode random is used")
    parser.add_argument("--batch_size", type=int, default=64, help="Decoder batch size")
    parser.add_argument("--max_images", type=int, default=0, help="Limit images when --input_dir is used; 0 means all")
    parser.add_argument("--seed", type=int, default=2026, help="Random crop seed")
    parser.add_argument("--resize_width", type=int, default=0,
                        help="Optional width to resize each input before cropping; 0 keeps original")
    parser.add_argument("--resize_height", type=int, default=0,
                        help="Optional height to resize each input before cropping; 0 keeps original")
    args = parser.parse_args()

    if len(args.r) != len(args.bitsf):
        raise ValueError("--r and --bitsf must have the same length")
    if sum(args.bitsf) != args.num_bits:
        raise ValueError(f"--num_bits ({args.num_bits}) must equal sum(--bitsf) ({sum(args.bitsf)})")

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gt_bits = parse_bits_file(args.bits_file) if args.bits_file else None
    if gt_bits is not None and len(gt_bits) != args.num_bits:
        raise ValueError(f"bits_file length ({len(gt_bits)}) must equal --num_bits ({args.num_bits})")

    model = load_model(args, device)
    image_paths = list_images(args.input, args.input_dir, args.max_images)

    print("Channel watermark decode")
    print(f"Device: {device}")
    print(f"Model: {args.model_path}")
    print(f"Channel: {args.channel.upper()}")
    print(f"Images: {len(image_paths)}")
    if gt_bits is not None:
        print(f"Ground truth bits: {bits_to_string(gt_bits)}")
    print("| Image | Crops | Crop Acc(%) | Vote Acc(%) | Vote Bits |")
    print("| ----- | ----- | ----------- | ----------- | --------- |")

    image_vote_accs = []
    image_crop_accs = []
    vote_exact = 0
    decoded_images = 0

    for image_idx, path in enumerate(image_paths):
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            print(f"| {os.path.basename(path)} | 0 | N/A | N/A | failed_to_read |")
            continue

        if args.resize_width > 0 and args.resize_height > 0:
            image = cv2.resize(image, (args.resize_width, args.resize_height), interpolation=cv2.INTER_AREA)

        h, w = image.shape[:2]
        if h < args.crop_size or w < args.crop_size:
            image = cv2.resize(image, (args.crop_size, args.crop_size), interpolation=cv2.INTER_AREA)
            h, w = image.shape[:2]

        positions = make_crop_positions(
            h,
            w,
            args.crop_size,
            args.num_crops,
            args.seed + image_idx,
            args.sample_mode,
        )
        crop_channels = []
        for y, x in positions:
            crop = image[y:y + args.crop_size, x:x + args.crop_size]
            crop_channels.append(extract_channel(crop, args.channel))

        predictions = decode_crops(model, crop_channels, device, args.batch_size)
        vote_bits = (np.mean(predictions, axis=0) >= 0.5).astype(np.int64)
        vote_text = bits_to_string(vote_bits)
        decoded_images += 1

        if gt_bits is None:
            print(f"| {os.path.basename(path)} | {len(predictions)} | N/A | N/A | {vote_text} |")
            continue

        crop_accs = np.mean(predictions == gt_bits[None, :], axis=1)
        vote_acc = float(np.mean(vote_bits == gt_bits))
        crop_acc = float(np.mean(crop_accs))
        image_crop_accs.append(crop_acc)
        image_vote_accs.append(vote_acc)
        vote_exact += int(vote_acc == 1.0)

        print(
            f"| {os.path.basename(path)} | {len(predictions)} | "
            f"{crop_acc * 100:.2f} | {vote_acc * 100:.2f} | {vote_text} |"
        )

    print("\nSummary")
    print(f"Decoded images: {decoded_images}")
    if gt_bits is not None and image_vote_accs:
        print(f"Average crop accuracy: {np.mean(image_crop_accs) * 100:.2f}%")
        print(f"Average vote accuracy: {np.mean(image_vote_accs) * 100:.2f}%")
        print(f"Vote exact 60-bit: {vote_exact}/{len(image_vote_accs)}")


if __name__ == "__main__":
    main()
