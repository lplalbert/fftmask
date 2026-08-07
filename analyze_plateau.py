#!/usr/bin/env python3
"""
平台期分析脚本

平台期定义：连续角度解码出相同的比特序列，不管准确率高低，
只要相邻角度解码结果相同就算同一个平台期。

用法:
  python analyze_plateau.py --input wechat1_rotated
  python analyze_plateau.py --input wechat1_rotated --angle-min -10 --angle-max 10 --angle-step 0.5
"""

import os
import sys
import numpy as np
import cv2
import torch
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from watermark_decoder3 import AdvancedWatermarkDecoder
from decode_channel_watermark import extract_channel


def list_images(input_path):
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    if os.path.isfile(input_path):
        return [input_path]
    files = []
    for root, _, fnames in os.walk(input_path):
        for f in sorted(fnames):
            if os.path.splitext(f)[1].lower() in exts:
                files.append(os.path.join(root, f))
    return files


def load_ground_truth(path):
    with open(path, "r") as f:
        line = f.readline().strip()
    return line


def make_crop_positions_closer(h, w, crop_size, seed):
    """生成5个裁剪位置，周围4个靠近中心（避免旋转黑边）"""
    positions = []
    cy, cx = h // 2, w // 2
    half = crop_size // 2
    positions.append((cy - half, cx - half))
    offset_y = max(h // 6, crop_size)
    offset_x = max(w // 6, crop_size)
    top_y = max(0, cy - offset_y - half)
    positions.append((top_y, cx - half))
    bottom_y = min(h - crop_size, cy + offset_y - half)
    positions.append((bottom_y, cx - half))
    left_x = max(0, cx - offset_x - half)
    positions.append((cy - half, left_x))
    right_x = min(w - crop_size, cx + offset_x - half)
    positions.append((cy - half, right_x))
    return positions


def rotate_image_cv2(image_bgr, angle):
    """旋转图片，使用反射填充避免黑边"""
    if abs(angle) < 0.01:
        return image_bgr.copy()
    h, w = image_bgr.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image_bgr, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT)


def decode_at_angle(image, angle, wm_size, model, device):
    """在指定角度解码水印"""
    rotated = rotate_image_cv2(image, angle)
    h2, w2 = rotated.shape[:2]
    positions = make_crop_positions_closer(h2, w2, wm_size, seed=42)
    channel_image = extract_channel(rotated, 'cb')
    crops = []
    for y, x in positions:
        if y + wm_size <= h2 and x + wm_size <= w2:
            crop_large = channel_image[y:y+wm_size, x:x+wm_size].copy()
            crop = cv2.resize(crop_large, (512, 512), interpolation=cv2.INTER_AREA)
            crops.append(crop)
    if not crops:
        return None, None
    tensor = torch.from_numpy(np.stack(crops, axis=0)).to(device=device,
                                                           dtype=torch.float32)
    tensor = tensor.unsqueeze(1).div_(127.5).sub_(1.0)
    with torch.no_grad():
        output, _, _ = model(tensor)
        pred = (output > 0.5).long().cpu().numpy()
    vote = (np.mean(pred, axis=0) >= 0.5).astype(int)
    vote_str = ''.join(str(b) for b in vote)
    return vote_str, pred


def find_plateaus(results):
    """
    根据解码序列划分平台期。
    返回: [(seq, [angles], length), ...]
    """
    plateaus = []
    cur_seq = results[0][1]
    cur_angles = [results[0][0]]
    for angle, seq in results[1:]:
        if seq == cur_seq:
            cur_angles.append(angle)
        else:
            plateaus.append((cur_seq, list(cur_angles), len(cur_angles)))
            cur_seq = seq
            cur_angles = [angle]
    plateaus.append((cur_seq, list(cur_angles), len(cur_angles)))
    return plateaus


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="平台期分析")
    ap.add_argument("--input", required=True, help="输入图片或目录")
    ap.add_argument("--gt", default="xiaomi_test_data_watermarked/bits.txt",
                    help="Ground truth文件")
    ap.add_argument("--model", default=None,
                    help="模型路径（默认自动检测）")
    ap.add_argument("--wm-size", type=int, default=639,
                    help="水印框尺寸（默认639=min(1706,1279)//2）")
    ap.add_argument("--angle-min", type=float, default=-5.0)
    ap.add_argument("--angle-max", type=float, default=5.0)
    ap.add_argument("--angle-step", type=float, default=0.2)
    ap.add_argument("--show-all", action="store_true",
                    help="显示所有角度的解码结果")
    args = ap.parse_args()

    # 模型路径
    if args.model is None:
        model_path = os.path.join(
            os.path.dirname(__file__),
            "output/v1_valnoise/20260629_005346/models/best_cb_decoder.pth")
    else:
        model_path = args.model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AdvancedWatermarkDecoder(n_sectors=60,
                                     rings=[(7, 17), (20, 30)],
                                     bits=[15, 45])
    state = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and 'model' in state:
        state = state['model']
    state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(device).eval()

    gt_str = load_ground_truth(args.gt)
    images = list_images(args.input)
    print(f"Found {len(images)} images")
    print(f"GT:  {gt_str} ({len(gt_str)} bits)")
    print(f"角度范围: {args.angle_min:.1f}° ~ {args.angle_max:.1f}°, "
          f"步长: {args.angle_step:.1f}°\n")

    angles = np.arange(args.angle_min,
                       args.angle_max + args.angle_step / 2,
                       args.angle_step)

    # 统计
    total_images = 0
    has_perfect_plateau = 0
    total_plateaus = 0

    for idx, ip in enumerate(images, 1):
        img = cv2.imread(ip, cv2.IMREAD_COLOR)
        if img is None:
            continue
        total_images += 1

        h, w = img.shape[:2]
        wm_size = min(h, w) // 2

        print("=" * 70)
        print(f"[{idx}/{len(images)}] {os.path.basename(ip)}  "
              f"({w}x{h}, wm_size={wm_size})")
        print("=" * 70)

        # 角度扫描
        results = []
        for angle in angles:
            seq, pred = decode_at_angle(img, angle, wm_size, model, device)
            if seq is not None:
                acc = sum(a == b for a, b in zip(seq, gt_str)) / len(gt_str) * 100
                results.append((angle, seq, acc))

        if args.show_all:
            print(f"\n{'角度':>6}  {'准确率':>6}  解码序列")
            print("-" * 80)
            for angle, seq, acc in results:
                acc_str = f"{acc:5.1f}%"
                if acc == 100:
                    acc_str += " ✓"
                print(f"{angle:+6.1f}°  {acc_str}  {seq}")

        # 找平台期
        plateaus = find_plateaus([(a, s) for a, s, _ in results])

        # 计算每个平台期的准确率
        plateaus_with_acc = []
        for seq, plat_angles, length in plateaus:
            acc = sum(a == b for a, b in zip(seq, gt_str)) / len(gt_str) * 100
            plateaus_with_acc.append((seq, plat_angles, length, acc))

        total_plateaus += len(plateaus_with_acc)

        # 打印平台期汇总（只显示长度>=2的）
        multi = [(s, a, l, acc) for s, a, l, acc in plateaus_with_acc if l >= 2]
        print(f"\n--- 平台期汇总（共{len(plateaus_with_acc)}个，"
              f"其中长度>=2的有{len(multi)}个）---")

        for i, (seq, plat_angles, length, acc) in enumerate(multi, 1):
            acc_str = f"{acc:.1f}%"
            if acc == 100:
                acc_str += " ✓ 完全正确!"
            print(f"  {i}. 长度={length:2d}, "
                  f"角度={plat_angles[0]:+.1f}°~{plat_angles[-1]:+.1f}°, "
                  f"acc={acc_str}")
            print(f"     序列={seq}")

        # 是否有100%准确的平台期
        perfect = [p for p in plateaus_with_acc if p[3] == 100]
        if perfect:
            has_perfect_plateau += 1
            p = perfect[0]
            print(f"\n  >>> 100%准确平台期: "
                  f"{p[1][0]:+.1f}°~{p[1][-1]:+.1f}° (长度{p[2]})")
        else:
            best = max(plateaus_with_acc, key=lambda x: x[3])
            print(f"\n  >>> 最高准确率: {best[3]:.1f}%, "
                  f"角度={best[1][0]:+.1f}°~{best[1][-1]:+.1f}°")

        print()

    # 总结
    print("=" * 70)
    print("总结")
    print("=" * 70)
    print(f"  总图片数: {total_images}")
    print(f"  平台期总数: {total_plateaus}")
    print(f"  平均每张图平台期数: {total_plateaus/max(total_images,1):.1f}")
    print(f"  存在100%准确平台期的图片: {has_perfect_plateau}/{total_images}")
