"""
v17/v18 水印解码脚本。

从原图裁剪不同尺寸，resize 到 crop_size 解码，搜索最佳裁剪尺寸。
支持角度搜索（0.2° 步长 + 平台期检测）。

用法:
    # 搜索最佳裁剪尺寸
    python decode_v17_v18.py \
        --image_dir /path/to/images \
        --bits_file /path/to/bits.json \
        --bits_key b_00 \
        --model_path /path/to/best_model.pth \
        --channel b --sweep_crop

    # 搜索最佳角度（用已知最佳裁剪尺寸）
    python decode_v17_v18.py \
        --image_dir /path/to/images \
        --bits_file /path/to/bits.json \
        --bits_key b_00 \
        --model_path /path/to/best_model.pth \
        --channel b --sweep_angle --best_crop 1498

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


def rotate_image(image, angle, border_value=0):
    """旋转图片，保持所有内容可见（黑边填充）。"""
    if abs(angle) < 0.01:
        return image
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2
    return cv2.warpAffine(image, M, (new_w, new_h),
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=border_value)


def hamming_distance(a, b):
    """计算两个比特序列的汉明距离。"""
    return sum(x != y for x, y in zip(a, b))


def get_consensus(seqs, weighted=True):
    """
    取每个 bit 位的众数作为共识序列。
    weighted=True 时，平台期中间位置权重高，边缘权重低（三角形加权）。
    """
    n = len(seqs[0])
    m = len(seqs)
    if weighted and m > 1:
        center = (m - 1) / 2.0
        weights = [1.0 + (1.0 - abs(i - center) / center) for i in range(m)]
    else:
        weights = [1.0] * m
    bits = []
    for bit_pos in range(n):
        score_0 = sum(weights[i] for i in range(m) if seqs[i][bit_pos] == '0')
        score_1 = sum(weights[i] for i in range(m) if seqs[i][bit_pos] == '1')
        bits.append('1' if score_1 >= score_0 else '0')
    return "".join(bits)


def find_plateaus(results, threshold=5, weighted=True):
    """
    根据解码序列的相似性划分平台期（模糊聚类）。
    相邻角度汉明距离 ≤ threshold → 同一平台期。
    results: [(angle, vote_str), ...]
    返回: [(consensus_seq, angles_list, length), ...]
    """
    plateaus = []
    cur_seqs = [results[0][1]]
    cur_angles = [results[0][0]]
    for angle, seq in results[1:]:
        consensus = get_consensus(cur_seqs, weighted=weighted)
        if hamming_distance(seq, consensus) <= threshold:
            cur_angles.append(angle)
            cur_seqs.append(seq)
        else:
            plateaus.append((get_consensus(cur_seqs, weighted=weighted),
                             list(cur_angles), len(cur_angles)))
            cur_seqs = [seq]
            cur_angles = [angle]
    plateaus.append((get_consensus(cur_seqs, weighted=weighted),
                     list(cur_angles), len(cur_angles)))
    return plateaus


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


def decode_single_image(model, image_bgr, channel, device, crop_size=512,
                        crop_from_orig=None, angle=0.0):
    """
    对单张图片解码。
    crop_from_orig: 从原图裁剪的尺寸（然后 resize 到 crop_size）。
                    None 表示不裁剪，直接 resize 整张图到 crop_size。
    angle: 旋转角度（度），先旋转再裁剪。
    """
    # 先旋转
    if abs(angle) > 0.01:
        image_bgr = rotate_image(image_bgr, angle)

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
    parser.add_argument("--sweep_angle", action="store_true", help="搜索最佳旋转角度")
    parser.add_argument("--best_crop", type=int, default=None, help="已知最佳裁剪尺寸（角度搜索时用）")
    parser.add_argument("--batch", action="store_true", help="遍历子目录")
    parser.add_argument("--min_ratio", type=float, default=0.38, help="搜索起始比例")
    parser.add_argument("--max_ratio", type=float, default=1.0, help="搜索结束比例")
    parser.add_argument("--step_ratio", type=float, default=0.01, help="搜索步长")
    parser.add_argument("--angle_min", type=float, default=0.0, help="角度搜索起始")
    parser.add_argument("--angle_max", type=float, default=360.0, help="角度搜索结束")
    parser.add_argument("--angle_step", type=float, default=0.2, help="细搜步长")
    parser.add_argument("--coarse_step", type=float, default=5.0, help="粗搜步长")
    parser.add_argument("--hamming_threshold", type=int, default=5,
                        help="平台期汉明距离阈值")
    parser.add_argument("--no_gt_selection", action="store_true",
                        help="不用GT选平台期，选最长平台期（实际部署用）")
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

        elif args.sweep_angle:
            # 两阶段角度搜索：粗搜(coarse_step) → 细搜(angle_step)
            best_crop = args.best_crop or short_side
            coarse_angles = np.arange(args.angle_min,
                                      args.angle_max + args.coarse_step / 2,
                                      args.coarse_step)

            print(f"  原图: {h0}×{w0}, best_crop={best_crop}")
            print(f"  两阶段搜索: 粗搜 {args.coarse_step:.1f}°步长, "
                  f"细搜 {args.angle_step:.1f}°步长\n")

            def decode_angle_sweep(img, angles):
                """对单张图做角度扫描，返回 [(angle, vote_str), ...]"""
                out = []
                for angle in angles:
                    vb, vt, _ = decode_single_image(
                        model, img, args.channel, device,
                        args.crop_size, crop_from_orig=best_crop, angle=angle)
                    if vt is not None:
                        out.append((angle, vt))
                return out

            def best_from_plateaus(angle_results):
                """
                从角度扫描结果中找最佳平台期。
                --no_gt_selection: 选最长平台期（无GT，实际部署用）
                默认: 用GT选准确率最高的平台期（有GT，评估用）
                """
                plateaus = find_plateaus(angle_results,
                                         threshold=args.hamming_threshold,
                                         weighted=True)
                best_plat = None
                if args.no_gt_selection:
                    # 无GT：选最长平台期（最稳定）
                    best_length = -1
                    for consensus, plat_angles, length in plateaus:
                        if length > best_length:
                            best_length = length
                            acc = float(np.mean(
                                np.array([int(c) for c in consensus]) == gt_bits) * 100.0)
                            best_plat = (consensus, plat_angles, length, acc)
                else:
                    # 有GT：选准确率最高的
                    best_acc = -1.0
                    for consensus, plat_angles, length in plateaus:
                        acc = float(np.mean(
                            np.array([int(c) for c in consensus]) == gt_bits) * 100.0)
                        if acc > best_acc:
                            best_acc = acc
                            best_plat = (consensus, plat_angles, length, acc)
                return best_plat

            for fname, img in images:
                # 第一阶段：粗搜
                coarse_results = decode_angle_sweep(img, coarse_angles)
                if not coarse_results:
                    continue
                coarse_plat = best_from_plateaus(coarse_results)
                if coarse_plat is None:
                    continue

                _, coarse_pa, _, coarse_acc = coarse_plat
                coarse_center = coarse_pa[len(coarse_pa) // 2]

                # 第二阶段：在粗搜最佳角度 ± (coarse_step) 范围内细搜
                fine_min = coarse_center - args.coarse_step
                fine_max = coarse_center + args.coarse_step
                fine_angles = np.arange(fine_min,
                                        fine_max + args.angle_step / 2,
                                        args.angle_step)
                fine_results = decode_angle_sweep(img, fine_angles)
                if not fine_results:
                    # 退回到粗搜结果
                    results.append((best_crop, coarse_acc, fname, coarse_plat))
                    continue

                fine_plat = best_from_plateaus(fine_results)
                if fine_plat is None:
                    results.append((best_crop, coarse_acc, fname, coarse_plat))
                    continue

                # 取细搜和粗搜中更好的
                _, fine_pa, fine_len, fine_acc = fine_plat
                if fine_acc >= coarse_acc:
                    best_plat = fine_plat
                else:
                    best_plat = coarse_plat

                results.append((best_crop, best_plat[3], fname, best_plat))
                c, pa, l, a = best_plat
                print(f"    {fname}: best_angle={pa[len(pa)//2]:+.1f}° "
                      f"(平台期 {pa[0]:+.1f}°~{pa[-1]:+.1f}°, 长度{l}), "
                      f"acc={a:.2f}%")

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
        all_angle_accs = []
        for subdir in subdirs:
            key = f"{args.channel}_{subdir}"
            gt_bits = load_gt_bits(args.bits_file, key)
            gt_text = "".join(str(int(b)) for b in gt_bits)
            img_dir = os.path.join(args.image_dir, subdir)
            print(f"\n  {subdir} | GT: {gt_text}")
            results = decode_dir(img_dir, gt_bits, gt_text)
            if args.sweep_angle:
                # 角度搜索结果: (best_crop, best_acc, fname, best_plat)
                for _, acc, _, _ in results:
                    all_angle_accs.append(acc)
            else:
                # 裁剪搜索结果: (cs, acc, count_or_fname)
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

        elif args.sweep_angle and all_angle_accs:
            print(f"\n{'='*60}")
            print(f"  角度搜索汇总（跨 {len(subdirs)} 个子目录, "
                  f"共 {len(all_angle_accs)} 张图）")
            print(f"{'='*60}")
            avg_acc = np.mean(all_angle_accs)
            print(f"    avg_best_acc={avg_acc:.2f}%")
    else:
        gt_bits = load_gt_bits(args.bits_file, args.bits_key)
        gt_text = "".join(str(int(b)) for b in gt_bits)
        print(f"GT bits: {gt_text}\n")
        decode_dir(args.image_dir, gt_bits, gt_text)


if __name__ == "__main__":
    main()
