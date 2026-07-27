"""
v1 旋转测试：测试不同旋转角度下的解码准确率。

用法:
  python inference_rotation_test_v1.py \
    --gt_json /mnt/lpl/fftmask/xiaomi_test_data_watermarked/embed_gt_mapping.json \
    --input_dir /mnt/lpl/fftmask/xiaomi_test_data_watermarked \
    --output_dir /mnt/lpl/fftmask/rotation_test_v1 \
    --model_path /mnt/lpl/fftmask/output/v1_valnoise/20260629_005346/models/best_cb_decoder.pth
"""

import os
import sys
import json
import argparse
import numpy as np
import cv2
import torch
from torchvision import transforms
from tqdm import tqdm

from watermark_decoder3 import AdvancedWatermarkDecoder


def cv2_imread(path):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


TRAIN_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])


def rotate_image(image, angle):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def get_valid_crop_region(h, w, angle, block_size):
    rad = np.radians(abs(angle))
    cos_a = np.cos(rad)
    sin_a = np.sin(rad)
    half_h, half_w = h / 2, w / 2
    margin_x = int(half_w * (1 - cos_a) + half_h * sin_a)
    margin_y = int(half_h * (1 - cos_a) + half_w * sin_a)
    margin_x = min(margin_x, w // 4)
    margin_y = min(margin_y, h // 4)
    return margin_y, h - margin_y, margin_x, w - margin_x


def get_center_crop_in_valid_region(img, block_size, valid_region):
    h, w = img.shape[:2]
    y_min, y_max, x_min, x_max = valid_region
    cy = (y_min + y_max) // 2
    cx = (x_min + x_max) // 2
    y = max(y_min, cy - block_size // 2)
    x = max(x_min, cx - block_size // 2)
    y = min(y, y_max - block_size)
    x = min(x, x_max - block_size)
    return img[y:y + block_size, x:x + block_size], y, x


def get_random_crop_in_valid_region(img, block_size, valid_region, rng):
    y_min, y_max, x_min, x_max = valid_region
    if y_max - y_min < block_size or x_max - x_min < block_size:
        return get_center_crop_in_valid_region(img, block_size, valid_region)
    y = rng.integers(y_min, y_max - block_size + 1)
    x = rng.integers(x_min, x_max - block_size + 1)
    return img[y:y + block_size, x:x + block_size], int(y), int(x)


def extract_cb_tile(tile_bgr):
    ycrcb = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2YCrCb)
    cb_tile = ycrcb[:, :, 2, None]
    return TRAIN_TRANSFORM(cb_tile)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--model_path", type=str,
                        default="/mnt/lpl/fftmask/output/v1_valnoise/20260629_005346/models/best_cb_decoder.pth")
    parser.add_argument("--gt_json", type=str, required=True)
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_bits", type=int, default=60)
    parser.add_argument("--num_crops", type=int, default=5)
    parser.add_argument("--rotation_angles", type=float, nargs="+", default=[0, 1, 2, 3, 4, 5])
    parser.add_argument("--min_edge", type=int, default=1024)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载 v1 模型
    model = AdvancedWatermarkDecoder(
        n_sectors=args.num_bits,
        rings=[(7, 17), (20, 30)],
        bits=[15, 45],
    )
    state_dict = torch.load(args.model_path, map_location=device, weights_only=False)
    if isinstance(state_dict, dict) and 'model' in state_dict:
        state_dict = state_dict['model']
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    print(f"Model: {args.model_path}")

    # 加载 GT
    with open(args.gt_json, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    watermark_bits = np.array(gt_data.get("watermark_bits", []), dtype=np.float32)
    if len(watermark_bits) != args.num_bits:
        raise ValueError(f"GT bits length ({len(watermark_bits)}) != num_bits ({args.num_bits})")

    # 支持两种 GT 格式
    results = gt_data.get("results", {})
    all_mappings = []
    for subfolder, items in results.items():
        for item in items:
            item["subfolder"] = subfolder
            all_mappings.append(item)

    # 收集图片路径
    image_paths = []
    path_to_subfolder = {}
    path_to_size = {}

    for item in all_mappings:
        wm_file = item.get("watermarked")
        subfolder = item.get("subfolder", "")
        if wm_file:
            img_path = os.path.join(args.input_dir, subfolder, wm_file)
            if os.path.exists(img_path):
                image_paths.append(img_path)
                path_to_subfolder[img_path] = subfolder
                path_to_size[img_path] = item.get("size", [0, 0])

    print(f"Images: {len(image_paths)}")
    print(f"Rotation angles: {args.rotation_angles}")

    # 测试每个旋转角度
    all_results = {}
    block_size = 512

    for angle in args.rotation_angles:
        print(f"\n{'='*60}")
        print(f"Testing rotation: {angle}°")
        print(f"{'='*60}")

        subfolder_stats = {}
        rng = np.random.default_rng(42)

        for img_path in tqdm(image_paths, desc=f"Angle {angle}°"):
            image = cv2_imread(img_path)
            if image is None:
                continue

            subfolder = path_to_subfolder.get(img_path, os.path.basename(os.path.dirname(img_path)))
            orig_size = path_to_size.get(img_path, [0, 0])

            # resize 最短边到 min_edge
            h_orig, w_orig = image.shape[:2]
            short_side = min(h_orig, w_orig)
            scale = args.min_edge / short_side
            new_w = int(w_orig * scale)
            new_h = int(h_orig * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

            h, w = image.shape[:2]

            # 旋转
            if angle != 0:
                rotated = rotate_image(image, angle)
            else:
                rotated = image

            # 有效裁剪区域
            valid_region = get_valid_crop_region(h, w, angle, block_size)
            y_min, y_max, x_min, x_max = valid_region

            if y_max - y_min < block_size or x_max - x_min < block_size:
                continue

            if subfolder not in subfolder_stats:
                subfolder_stats[subfolder] = {'center': [], 'random': [], 'sizes': set()}
            subfolder_stats[subfolder]['sizes'].add((orig_size[0], orig_size[1]))

            # 中心裁剪
            center, cy, cx = get_center_crop_in_valid_region(rotated, block_size, valid_region)
            center_tensor = extract_cb_tile(center).unsqueeze(0).to(device)
            with torch.no_grad():
                center_out, _, _ = model(center_tensor)
                center_pred = (center_out > 0.5).float().cpu().numpy().squeeze()
            center_acc = float(np.mean(center_pred == watermark_bits))
            subfolder_stats[subfolder]['center'].append(center_acc)

            # 随机裁剪
            for _ in range(args.num_crops):
                rand_crop, ry, rx = get_random_crop_in_valid_region(rotated, block_size, valid_region, rng)
                rand_tensor = extract_cb_tile(rand_crop).unsqueeze(0).to(device)
                with torch.no_grad():
                    rand_out, _, _ = model(rand_tensor)
                    rand_pred = (rand_out > 0.5).float().cpu().numpy().squeeze()
                rand_acc = float(np.mean(rand_pred == watermark_bits))
                subfolder_stats[subfolder]['random'].append(rand_acc)

        # 汇总
        all_center = []
        all_random = []
        angle_result = {}

        for subfolder, stats in sorted(subfolder_stats.items()):
            avg_c = np.mean(stats['center']) * 100 if stats['center'] else 0
            avg_r = np.mean(stats['random']) * 100 if stats['random'] else 0
            angle_result[subfolder] = {
                "center_acc": avg_c,
                "random_acc": avg_r,
                "num_images": len(stats['center']),
                "sizes": list(stats['sizes']),
            }
            all_center.extend(stats['center'])
            all_random.extend(stats['random'])
            sizes_str = ", ".join([f"{h}x{w}" for h, w in stats['sizes']]) if stats['sizes'] else "?"
            print(f"  {subfolder}: center={avg_c:.2f}%, random={avg_r:.2f}% ({len(stats['center'])} imgs, {sizes_str})")

        avg_center = np.mean(all_center) * 100 if all_center else 0
        avg_random = np.mean(all_random) * 100 if all_random else 0
        angle_result["overall"] = {
            "center_acc": avg_center,
            "random_acc": avg_random,
            "num_images": len(all_center),
        }
        all_results[angle] = angle_result
        print(f"  {'Overall':>20}: center={avg_center:.2f}%, random={avg_random:.2f}%")

    # 打印汇总
    print(f"\n{'='*70}")
    print(f"Rotation Test Summary (v1)")
    print(f"Dataset: {args.input_dir}")
    print(f"Model: {args.model_path}")
    print(f"{'='*70}")

    all_subfolders = set()
    for angle_res in all_results.values():
        all_subfolders.update(k for k in angle_res.keys() if k != "overall")
    all_subfolders = sorted(all_subfolders)

    for subfolder in all_subfolders:
        sizes_set = set()
        for angle, angle_res in all_results.items():
            if subfolder in angle_res and 'sizes' in angle_res[subfolder]:
                sizes_set.update(angle_res[subfolder]['sizes'])
        sizes_str = ", ".join([f"{h}x{w}" for h, w in sorted(sizes_set)]) if sizes_set else "unknown"

        print(f"\n--- {subfolder} (sizes: {sizes_str}) ---")
        print(f"{'Angle':>8} | {'Center Acc':>12} | {'Random Acc':>12} | {'Images':>8}")
        print(f"{'-'*8}-+-{'-'*12}-+-{'-'*12}-+-{'-'*8}")
        for angle, angle_res in all_results.items():
            if subfolder in angle_res:
                res = angle_res[subfolder]
                print(f"{angle:>7.1f}° | {res['center_acc']:>11.2f}% | {res['random_acc']:>11.2f}% | {res['num_images']:>8}")

    print(f"\n--- Overall ---")
    print(f"{'Angle':>8} | {'Center Acc':>12} | {'Random Acc':>12} | {'Images':>8}")
    print(f"{'-'*8}-+-{'-'*12}-+-{'-'*12}-+-{'-'*8}")
    for angle, angle_res in all_results.items():
        if "overall" in angle_res:
            res = angle_res["overall"]
            print(f"{angle:>7.1f}° | {res['center_acc']:>11.2f}% | {res['random_acc']:>11.2f}% | {res['num_images']:>8}")
    print(f"{'='*70}")

    # 保存结果
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, "rotation_test_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({
            "model_path": args.model_path,
            "input_dir": args.input_dir,
            "rotation_angles": args.rotation_angles,
            "results": {str(k): v for k, v in all_results.items()},
        }, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {results_path}")


if __name__ == "__main__":
    main()
