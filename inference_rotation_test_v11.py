"""
v11 旋转测试：测试不同旋转角度下的解码准确率。

旋转后裁剪有效区域，避免填充黑边。

用法:
  python inference_rotation_test_v11.py \
    --gt_json /mnt/lpl/fftmask/xiaomi_test_data_watermarked_v11/embed_gt_mapping.json \
    --input_dir /mnt/lpl/fftmask/xiaomi_test_data_watermarked_v11 \
    --output_dir /mnt/lpl/fftmask/rotation_test_results \
    --model_path /mnt/lpl/fftmask/output/finetune_v11_v3/20260724_095433/models/best_cb_decoder_v11.pth
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

from watermark_decoder_v11 import AdvancedWatermarkDecoderV11


def cv2_imread(path):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


TRAIN_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])


def rotate_image(image, angle):
    """旋转图像，保持尺寸"""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return rotated


def get_valid_crop_region(h, w, angle, block_size):
    """
    计算旋转后不包含黑边的有效裁剪区域。

    旋转后四个角会有黑边，需要缩小裁剪范围。
    返回 (y_min, y_max, x_min, x_max) 表示有效区域的边界。
    """
    # 旋转后，中心区域是安全的
    # 安全区域的大小取决于旋转角度
    # 对于小角度旋转，安全区域近似为:
    # safe_margin = block_size/2 * (1 - cos(angle)) + block_size/2 * sin(angle)
    rad = np.radians(abs(angle))
    cos_a = np.cos(rad)
    sin_a = np.sin(rad)

    # 计算旋转后四个角到中心的距离
    half_h, half_w = h / 2, w / 2

    # 旋转后，原图四个角的新位置
    corners = np.array([
        [-half_w, -half_h],
        [half_w, -half_h],
        [half_w, half_h],
        [-half_w, half_h]
    ])

    # 旋转矩阵
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    rotated_corners = corners @ R.T

    # 有效区域：旋转后图像的内接矩形
    # 保守估计：使用旋转后角点的最小/最大坐标
    min_x = np.min(rotated_corners[:, 0]) + half_w
    max_x = np.max(rotated_corners[:, 0]) + half_w
    min_y = np.min(rotated_corners[:, 1]) + half_h
    max_y = np.max(rotated_corners[:, 1]) + half_h

    # 有效区域需要避开黑边
    # 黑边出现在旋转后的角落
    # 安全边距 = 原图半宽 * (1 - cos(θ)) + 原图半高 * sin(θ)
    margin_x = int(half_w * (1 - cos_a) + half_h * sin_a)
    margin_y = int(half_h * (1 - cos_a) + half_w * sin_a)

    # 确保 margin 不超过图像尺寸
    margin_x = min(margin_x, w // 4)
    margin_y = min(margin_y, h // 4)

    y_min = margin_y
    y_max = h - margin_y
    x_min = margin_x
    x_max = w - margin_x

    return y_min, y_max, x_min, x_max


def get_center_crop_in_valid_region(img, block_size, valid_region):
    """在有效区域内取中心裁剪"""
    h, w = img.shape[:2]
    y_min, y_max, x_min, x_max = valid_region

    # 有效区域的中心
    cy = (y_min + y_max) // 2
    cx = (x_min + x_max) // 2

    # 确保裁剪区域在有效区域内
    y = max(y_min, cy - block_size // 2)
    x = max(x_min, cx - block_size // 2)

    # 确保不超出有效区域
    y = min(y, y_max - block_size)
    x = min(x, x_max - block_size)

    return img[y:y + block_size, x:x + block_size], y, x


def get_random_crop_in_valid_region(img, block_size, valid_region, rng):
    """在有效区域内随机裁剪"""
    h, w = img.shape[:2]
    y_min, y_max, x_min, x_max = valid_region

    # 确保有效区域足够大
    if y_max - y_min < block_size or x_max - x_min < block_size:
        # 如果有效区域太小，回退到中心裁剪
        return get_center_crop_in_valid_region(img, block_size, valid_region)

    y = rng.integers(y_min, y_max - block_size + 1)
    x = rng.integers(x_min, x_max - block_size + 1)
    return img[y:y + block_size, x:x + block_size], int(y), int(x)


def extract_cb_tile(tile_bgr):
    ycrcb = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2YCrCb)
    cb_tile = ycrcb[:, :, 2, None]
    return TRAIN_TRANSFORM(cb_tile)


# 可视化颜色
COLORS = [
    (0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255),
    (255, 0, 255), (255, 255, 0), (0, 128, 255), (128, 0, 255),
    (0, 255, 128), (255, 128, 0), (128, 255, 0), (255, 0, 128),
    (0, 64, 255), (64, 0, 255), (0, 255, 64), (255, 64, 0),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--model_path", type=str,
                        default="/mnt/lpl/fftmask/output/finetune_v11_v3/20260724_095433/models/best_cb_decoder_v11.pth")
    parser.add_argument("--gt_json", type=str, required=True)
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_bits", type=int, default=60)
    parser.add_argument("--num_crops", type=int, default=5, help="Random crops per rotation angle")
    parser.add_argument("--rotation_angles", type=float, nargs="+", default=[0, 1, 2, 3, 4, 5],
                        help="Rotation angles to test")
    parser.add_argument("--min_edge", type=int, default=1024,
                        help="Resize shortest edge to this before cropping")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载 v11 模型
    model = AdvancedWatermarkDecoderV11(
        n_sectors=args.num_bits,
        rings=[(7, 17), (20, 30)],
        bits=[15, 45],
        rotation_ring=(13, 23),
        rotation_patches=36,
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
    mappings = gt_data.get("mappings", [])
    watermark_bits = np.array(gt_data.get("watermark_bits", []), dtype=np.float32)

    if len(watermark_bits) != args.num_bits:
        raise ValueError(f"GT bits length ({len(watermark_bits)}) != num_bits ({args.num_bits})")

    # 收集图片路径
    image_paths = []
    for item in mappings:
        wm_path = item.get("watermarked_path")
        if wm_path and os.path.exists(wm_path):
            image_paths.append(wm_path)
        else:
            wm_file = item.get("watermarked_file")
            subfolder = item.get("subfolder", "")
            if wm_file:
                if subfolder:
                    img_path = os.path.join(args.input_dir, subfolder, wm_file)
                else:
                    img_path = os.path.join(args.input_dir, wm_file)
                if os.path.exists(img_path):
                    image_paths.append(img_path)

    print(f"Images: {len(image_paths)}")
    print(f"Rotation angles: {args.rotation_angles}")

    # 测试每个旋转角度
    all_results = {}
    block_size = 512

    # 从GT中获取subfolder信息和图片尺寸
    path_to_subfolder = {}
    path_to_size = {}
    for item in mappings:
        wm_path = item.get("watermarked_path", "")
        subfolder = item.get("subfolder", "unknown")
        if wm_path:
            path_to_subfolder[wm_path] = subfolder
            path_to_size[wm_path] = item.get("original_size", [0, 0])

    for angle in args.rotation_angles:
        print(f"\n{'='*60}")
        print(f"Testing rotation: {angle}°")
        print(f"{'='*60}")

        # 按子文件夹统计
        subfolder_stats = {}  # {subfolder: {'center': [], 'random': [], 'sizes': []}}
        rng = np.random.default_rng(42)

        for img_path in tqdm(image_paths, desc=f"Angle {angle}°"):
            image = cv2_imread(img_path)
            if image is None:
                continue

            # 获取子文件夹名和原始尺寸
            subfolder = path_to_subfolder.get(img_path, os.path.basename(os.path.dirname(img_path)))
            orig_size = path_to_size.get(img_path, [0, 0])

            # resize 最短边到 min_edge (确保水印tile=512)
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

            # 计算有效裁剪区域
            valid_region = get_valid_crop_region(h, w, angle, block_size)
            y_min, y_max, x_min, x_max = valid_region

            # 检查有效区域是否足够大
            if y_max - y_min < block_size or x_max - x_min < block_size:
                print(f"  WARNING: Valid region too small for {os.path.basename(img_path)}, skipping")
                continue

            # 初始化子文件夹统计
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

        # 汇总统计
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

        # 总体
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
    print(f"Rotation Test Summary")
    print(f"Dataset: {args.input_dir}")
    print(f"Model: {args.model_path}")
    print(f"{'='*70}")

    # 获取所有子文件夹
    all_subfolders = set()
    for angle_res in all_results.values():
        all_subfolders.update(k for k in angle_res.keys() if k != "overall")
    all_subfolders = sorted(all_subfolders)

    # 每个子文件夹的汇总表
    for subfolder in all_subfolders:
        # 获取该子文件夹的图片尺寸
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

    # 总体汇总
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
