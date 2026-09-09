"""
v17/v18 水印解码脚本。

从原图裁剪不同尺寸，resize 到 crop_size 解码，搜索最佳裁剪尺寸。

用法:
    # 搜索最佳裁剪尺寸
    python decode_v17_v18.py \
        --image_dir /path/to/images \
        --bits_file /path/to/bits.json \
        --bits_key b_00 \
        --model_path /path/to/best_model.pth \
        --channel b --sweep_crop

    # 批量搜索
    python decode_v17_v18.py \
        --image_dir /path/to/images \
        --bits_file /path/to/bits.json \
        --model_path /path/to/best_model.pth \
        --channel b --batch --sweep_crop
"""

import argparse
import json
import os
from collections import defaultdict

import cv2
import numpy as np
import torch

from watermark_decoder_v17 import WatermarkDecoderV17


def extract_channel(image_bgr, channel):
    if channel == "b":
        return image_bgr[:, :, 0].copy()
    elif channel == "cb":
        ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
        return ycrcb[:, :, 2].copy()
    elif channel == "cr":
        ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
        return ycrcb[:, :, 1].copy()
    elif channel == "y":
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError(f"Unknown channel: {channel}")


def five_point_crops(h, w, crop_size):
    """返回5点裁剪的 (y, x) 左上角坐标：中心 + 四角。"""
    cy, cx = h // 2, w // 2
    half = crop_size // 2
    return [
        (cy - half, cx - half),       # center
        (0, 0),                        # top-left
        (0, w - crop_size),            # top-right
        (h - crop_size, 0),            # bottom-left
        (h - crop_size, w - crop_size),  # bottom-right
    ]


def decode_single_image(model, image_bgr, channel, device, crop_size=512, crop_from_orig=None):
    """
    对单张图片解码。
    crop_from_orig: 从原图裁剪的尺寸（然后 resize 到 crop_size）。
                    None 表示不裁剪，直接 resize 整张图到 crop_size。
    """
    h, w = image_bgr.shape[:2]

    if crop_from_orig is not None and crop_from_orig < min(h, w):
        # 从原图裁剪 crop_from_orig 大小的区域，5点采样
        positions = five_point_crops(h, w, crop_from_orig)
        ch_img = extract_channel(image_bgr, channel)
        crops = []
        for y, x in positions:
            if y + crop_from_orig <= h and x + crop_from_orig <= w:
                crop = ch_img[y:y + crop_from_orig, x:x + crop_from_orig].copy()
                # resize 到 crop_size
                crop = cv2.resize(crop, (crop_size, crop_size), interpolation=cv2.INTER_AREA)
                crops.append(crop)
    else:
        # 不裁剪，直接 resize 整张图
        ch_img = extract_channel(image_bgr, channel)
        ch_img = cv2.resize(ch_img, (crop_size, crop_size), interpolation=cv2.INTER_AREA)
        crops = [ch_img]

    if not crops:
        return None, None, []

    batch = np.stack(crops, axis=0)
    tensor = torch.from_numpy(batch).to(device=device, dtype=torch.float32)
    tensor = tensor.unsqueeze(1).div_(127.5).sub_(1.0)

    with torch.no_grad():
        output, _, _ = model(tensor)
        pred = (output > 0.5).long().cpu().numpy()

    vote_bits = (np.mean(pred, axis=0) >= 0.5).astype(np.int64)
    vote_text = "".join(str(int(b)) for b in vote_bits)

    return vote_bits, vote_text, pred


def load_gt_bits(bits_file, bits_key):
    with open(bits_file, "r") as f:
        data = json.load(f)
    if bits_key:
        bit_str = data[bits_key]
        return np.array([int(c) for c in bit_str], dtype=np.int64)
    elif "watermark_bits" in data:
        return np.array(data["watermark_bits"], dtype=np.int64)
    else:
        raise ValueError(f"Cannot find bits in {bits_file}, key={bits_key}")


def main():
    parser = argparse.ArgumentParser(description="v17/v18 水印解码")
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--bits_file", required=True)
    parser.add_argument("--bits_key", default=None)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--channel", required=True, choices=["b", "cb", "y", "cr"])
    parser.add_argument("--num_bits", type=int, default=60)
    parser.add_argument("--bitsf", type=int, nargs="+", default=[15, 45])
    parser.add_argument("--r", type=float, nargs="+", default=[12.0, 25.0])
    parser.add_argument("--angle_bins", type=int, default=180)
    parser.add_argument("--crop_size", type=int, default=512, help="送入模型的尺寸")
    parser.add_argument("--sweep_crop", action="store_true", help="搜索最佳裁剪尺寸")
    parser.add_argument("--batch", action="store_true", help="遍历子目录")
    parser.add_argument("--min_ratio", type=float, default=0.38, help="搜索起始比例")
    parser.add_argument("--max_ratio", type=float, default=1.0, help="搜索结束比例")
    parser.add_argument("--step_ratio", type=float, default=0.01, help="搜索步长")
    args = parser.parse_args()

    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WatermarkDecoderV17(
        n_sectors=args.num_bits,
        bits=args.bitsf,
        angle_bins=args.angle_bins,
        radius_bins=12,
        ring_positions_init=args.r,
    )
    state_dict = torch.load(args.model_path, map_location=device, weights_only=False)
    if isinstance(state_dict, dict) and "model" in state_dict:
        state_dict = state_dict["model"]
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    state_dict.pop("ring_positions", None)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device).eval()
    print(f"Model: {args.model_path}")
    print(f"Device: {device}, Channel: {args.channel}, Model input: {args.crop_size}")

    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    def decode_dir(img_dir, gt_bits, gt_text):
        images = []
        for fname in sorted(os.listdir(img_dir)):
            if fname.lower().endswith(exts):
                img = cv2.imread(os.path.join(img_dir, fname), cv2.IMREAD_COLOR)
                if img is not None:
                    images.append((fname, img))
        if not images:
            return []

        h0, w0 = images[0][1].shape[:2]
        short_side = min(h0, w0)
        results = []

        if args.sweep_crop:
            # 构建候选裁剪尺寸（从原图裁剪的大小）
            crop_sizes = []
            ratio = args.min_ratio
            while ratio <= args.max_ratio + 1e-12:
                cs = max(args.crop_size, int(round(short_side * ratio)))
                if cs <= short_side:
                    crop_sizes.append(cs)
                ratio += args.step_ratio
            # 去重保序
            seen = set()
            unique = []
            for cs in crop_sizes:
                if cs not in seen:
                    seen.add(cs)
                    unique.append(cs)
            crop_sizes = unique

            print(f"  原图: {h0}×{w0}, short_side={short_side}")
            print(f"  搜索: {crop_sizes[0]} ~ {crop_sizes[-1]}, 共 {len(crop_sizes)} 个尺寸\n")

            for cs in crop_sizes:
                accs = []
                for fname, img in images:
                    vb, _, _ = decode_single_image(model, img, args.channel, device,
                                                    args.crop_size, crop_from_orig=cs)
                    if vb is not None:
                        accs.append(float(np.mean(vb == gt_bits) * 100.0))
                avg = np.mean(accs) if accs else 0.0
                results.append((cs, avg, len(accs)))
                print(f"    crop={cs:4d} ({cs/short_side:.2f}S)  avg_vote_acc={avg:.2f}%")
        else:
            # 单尺寸
            for fname, img in images:
                vb, vt, _ = decode_single_image(model, img, args.channel, device,
                                                 args.crop_size, crop_from_orig=short_side)
                if vb is None:
                    continue
                acc = float(np.mean(vb == gt_bits) * 100.0)
                results.append((short_side, acc, fname))
                diff = "".join("^" if v != g else " " for v, g in zip(vt, gt_text))
                print(f"    {fname}: {acc:.2f}%  diff: {diff}")

        return results

    if args.batch:
        subdirs = sorted(d for d in os.listdir(args.image_dir)
                         if os.path.isdir(os.path.join(args.image_dir, d)))
        all_by_cs = defaultdict(list)
        for subdir in subdirs:
            key = f"{args.channel}_{subdir}"
            gt_bits = load_gt_bits(args.bits_file, key)
            gt_text = "".join(str(int(b)) for b in gt_bits)
            img_dir = os.path.join(args.image_dir, subdir)
            print(f"\n  {subdir} | GT: {gt_text}")
            results = decode_dir(img_dir, gt_bits, gt_text)
            for cs, acc, _ in results:
                all_by_cs[cs].append(acc)

        if args.sweep_crop and all_by_cs:
            print(f"\n{'='*60}")
            print(f"  汇总（跨 {len(subdirs)} 个子目录）")
            print(f"{'='*60}")
            summary = [(cs, np.mean(accs)) for cs, accs in all_by_cs.items()]
            summary.sort(key=lambda x: x[0])
            best_cs, best_acc = max(summary, key=lambda x: x[1])
            for cs, acc in summary:
                mark = " <-- BEST" if cs == best_cs else ""
                print(f"    crop={cs:4d}  avg_vote_acc={acc:.2f}%{mark}")
            print(f"\n  BEST: crop_size={best_cs}  avg_vote_acc={best_acc:.2f}%")
    else:
        gt_bits = load_gt_bits(args.bits_file, args.bits_key)
        gt_text = "".join(str(int(b)) for b in gt_bits)
        print(f"GT bits: {gt_text}\n")
        decode_dir(args.image_dir, gt_bits, gt_text)


if __name__ == "__main__":
    main()
