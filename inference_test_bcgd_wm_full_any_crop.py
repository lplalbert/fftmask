"""
Inference script for full-image tiled watermark dataset with arbitrary 512x512 crops.

For each watermarked image:
- Extract wm_size x wm_size crop (wm_size = min(H,W) // 2)
- Resize to 512x512 for decoding
- Extract center + N-1 random crops
- Decode each crop with AdvancedWatermarkDecoder
- Aggregate results by majority voting across all crops
- Generate visualization with crop boxes and accuracy

Usage:
  python inference_test_bcgd_wm_full_any_crop.py \
    --gt_json /mnt/lpl/fftmask/xiaomi_test_data_watermarked/embed_gt_mapping.json \
    --input_dir /mnt/lpl/fftmask/xiaomi_test_data_watermarked/4096_3072 \
    --output_dir /mnt/lpl/fftmask/xiaomi_decode_results \
    --num_crops 5
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


def get_center_crop(img, block_size=512):
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    x = cx - block_size // 2
    y = cy - block_size // 2
    return img[y:y + block_size, x:x + block_size], y, x


def get_random_crop(img, block_size=512):
    h, w = img.shape[:2]
    if h < block_size or w < block_size:
        raise ValueError(f"Image {h}x{w} smaller than crop {block_size}x{block_size}")
    y = np.random.randint(0, h - block_size + 1)
    x = np.random.randint(0, w - block_size + 1)
    return img[y:y + block_size, x:x + block_size], y, x


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
    parser = argparse.ArgumentParser(
        description="Decode watermark from images using wm_size = min(H,W)//2 crop"
    )
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--model_path", type=str,
                        default="/data2/fzx/fftmask/output_60_dynamic/models/epoch_281.pth")
    parser.add_argument("--gt_json", type=str,
                        default="/data2/fzx/fftmask_imgs/test_bcgd_wm_template/embed_gt_mapping_template.json")
    parser.add_argument("--input_dir", type=str,
                        default="/data2/fzx/fftmask_imgs/test_bcgd_wm_template")
    parser.add_argument("--output_dir", type=str,
                        default="/data2/fzx/fftmask/capture_inference_results_cb/capture_test_bcgd_wm_template_any_crop")
    parser.add_argument("--num_bits", type=int, default=60)
    parser.add_argument("--num_crops", type=int, default=16,
                        help="Total crops per image, including center crop")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--match_by_original", action="store_true")
    args = parser.parse_args()

    if args.num_crops < 1:
        raise ValueError("--num_crops must be >= 1")

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AdvancedWatermarkDecoder(
        n_sectors=args.num_bits,
        rings=[(r - 5, r + 5) for r in [12, 25]],
        bits=[15, 45],
    )
    state_dict = torch.load(args.model_path, map_location=device, weights_only=True)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    with open(args.gt_json, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    # Support both "mappings" (old format) and "results" (new format)
    mappings = gt_data.get("mappings", [])
    watermark_bits = gt_data.get("watermark_bits", [])

    if not mappings and "results" in gt_data:
        # New format: results is a dict keyed by directory name
        results_dict = gt_data["results"]
        # Find the matching subdirectory
        input_basename = os.path.basename(args.input_dir)
        if input_basename in results_dict:
            mappings = results_dict[input_basename]
        else:
            # Try first available key
            first_key = next(iter(results_dict), None)
            if first_key:
                mappings = results_dict[first_key]
                print(f"  Using results['{first_key}'] (input_dir basename '{input_basename}' not found)")

    if not mappings:
        raise ValueError("GT JSON has no mappings or results")

    actual_images = {}
    for fname in os.listdir(args.input_dir):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            actual_images[fname] = os.path.join(args.input_dir, fname)

    if not actual_images:
        raise ValueError(f"No PNG/JPG images found in: {args.input_dir}")

    image_paths = []
    gt_bits_list = []

    for item in mappings:
        # Support both "watermarked_file" and "watermarked" keys
        wm_file = item.get("watermarked_file") or item.get("watermarked")
        if wm_file:
            img_path = os.path.join(args.input_dir, wm_file)
            if not os.path.exists(img_path):
                # Try finding by original filename
                orig_file = item.get("original_file") or item.get("original")
                if orig_file:
                    stem = os.path.splitext(orig_file)[0]
                    for actual_fname, actual_path in actual_images.items():
                        if stem in actual_fname:
                            img_path = actual_path
                            break
            if os.path.exists(img_path):
                image_paths.append(img_path)
                # Use watermark_bits from item if available, otherwise from top-level
                if "watermark_bits" in item:
                    gt_bits_list.append(item["watermark_bits"])
                elif watermark_bits:
                    gt_bits_list.append(watermark_bits)
                else:
                    print(f"WARNING: No watermark_bits for {wm_file}")
                    continue
            else:
                print(f"WARNING: Missing image: {wm_file}")
        else:
            print(f"WARNING: No watermarked_file in item: {item}")

    if not image_paths:
        raise ValueError("No images matched")

    # 创建可视化目录
    vis_dir = os.path.join(args.output_dir, "crop_vis")
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    # 统计
    results = []
    total_correct = 0
    total_bits = 0
    correct_all = 0
    center_accs = []
    random_accs = []
    voted_accs = []

    block_size = 512

    for img_idx in tqdm(range(len(image_paths)), desc="Decoding"):
        img_path = image_paths[img_idx]
        gt_bits = np.array(gt_bits_list[img_idx], dtype=np.float32)
        fname = os.path.basename(img_path)

        image = cv2_imread(img_path)
        if image is None:
            print(f"WARNING: Failed to load {img_path}")
            continue

        # 计算水印嵌入尺寸 wm_size = min(H,W) // 2
        h, w = image.shape[:2]
        wm_size = min(h, w) // 2
        print(f"  Image {h}x{w}, wm_size={wm_size}")

        vis_img = image.copy()

        # 中心 crop - 先裁剪wm_size x wm_size，再resize到512x512
        center_crop, cy, cx = get_center_crop(image, wm_size)
        center = cv2.resize(center_crop, (block_size, block_size), interpolation=cv2.INTER_AREA)
        center_tensor = extract_cb_tile(center).unsqueeze(0).to(device)
        with torch.no_grad():
            center_out, _, _ = model(center_tensor)
            center_pred = (center_out > 0.5).float().cpu().numpy().squeeze()
        center_acc = float(np.mean(center_pred == gt_bits))
        center_accs.append(center_acc)

        # 画中心框（绿色）
        cv2.rectangle(vis_img, (cx, cy), (cx + wm_size, cy + wm_size), (0, 255, 0), 3)
        cv2.putText(vis_img, f'C {center_acc*100:.0f}%', (cx + 5, cy + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 随机 crop - 在原图上裁剪wm_size x wm_size，再resize到512x512
        all_preds = [center_pred]
        crop_infos = [{"method": "center", "y": cy, "x": cx, "acc": center_acc}]

        for ci in range(args.num_crops - 1):
            try:
                rand_crop_large, ry, rx = get_random_crop(image, wm_size)
                rand_crop = cv2.resize(rand_crop_large, (block_size, block_size), interpolation=cv2.INTER_AREA)
            except ValueError as e:
                print(f"  Warning: {e}, skip random crop {ci+1}")
                continue
            rand_tensor = extract_cb_tile(rand_crop).unsqueeze(0).to(device)
            with torch.no_grad():
                rand_out, _, _ = model(rand_tensor)
                rand_pred = (rand_out > 0.5).float().cpu().numpy().squeeze()
            rand_acc = float(np.mean(rand_pred == gt_bits))
            random_accs.append(rand_acc)
            all_preds.append(rand_pred)
            crop_infos.append({"method": "random", "y": ry, "x": rx, "acc": rand_acc})

            # 画随机框
            color = COLORS[ci % len(COLORS)]
            cv2.rectangle(vis_img, (ry, rx), (ry + wm_size, rx + wm_size), color, 2)
            cv2.putText(vis_img, f'#{ci+1} {rand_acc*100:.0f}%', (ry + 5, rx + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # 投票
        all_preds_arr = np.array(all_preds)
        voted = (np.mean(all_preds_arr, axis=0) > 0.5).astype(float)
        voted_acc = float(np.mean(voted == gt_bits))
        voted_accs.append(voted_acc)
        correct_bits = int(np.sum(voted == gt_bits))
        total_correct += correct_bits
        total_bits += args.num_bits
        if voted_acc == 1.0:
            correct_all += 1

        # 在图上标注投票结果
        cv2.putText(vis_img, f'Vote: {voted_acc*100:.1f}%', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        # 保存可视化
        vis_name = f"vis_{os.path.splitext(fname)[0]}.jpg"
        cv2.imwrite(os.path.join(vis_dir, vis_name), vis_img)

        results.append({
            "file": fname,
            "num_crops": args.num_crops,
            "center_accuracy": center_acc,
            "avg_random_accuracy": float(np.mean([c["acc"] for c in crop_infos if c["method"] == "random"])),
            "voted_accuracy": voted_acc,
            "correct_bits": correct_bits,
            "total_bits": int(args.num_bits),
            "gt": gt_bits.tolist(),
            "voted_pred": voted.tolist(),
            "crop_infos": crop_infos,
        })

    summary = {
        "count": len(results),
        "avg_center_accuracy_pct": np.mean(center_accs) * 100 if center_accs else 0.0,
        "avg_random_accuracy_pct": np.mean(random_accs) * 100 if random_accs else 0.0,
        "avg_voted_accuracy_pct": np.mean(voted_accs) * 100 if voted_accs else 0.0,
        "avg_decode_accuracy_pct": (total_correct / total_bits * 100) if total_bits else 0.0,
        "all_bits_correct": correct_all,
        "total_correct_bits": int(total_correct),
        "total_bits": int(total_bits),
        "dataset": os.path.basename(args.input_dir),
        "gt_json": args.gt_json,
        "input_dir": args.input_dir,
        "model_path": args.model_path,
        "num_crops": args.num_crops,
        "match_by_original": args.match_by_original,
        "watermark_bits_string": gt_data.get("watermark_bits_string", ""),
    }

    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(os.path.join(args.output_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"Dataset : {os.path.basename(args.input_dir)}")
    print(f"Images  : {summary['count']}")
    print(f"Crops   : {args.num_crops} (1 center + {args.num_crops-1} random)")
    print(f"{'='*60}")
    print(f"中心 crop 平均准确率:  {summary['avg_center_accuracy_pct']:.2f}%")
    print(f"随机 crop 平均准确率:  {summary['avg_random_accuracy_pct']:.2f}%")
    print(f"投票后平均准确率:      {summary['avg_voted_accuracy_pct']:.2f}%")
    print(f"{'='*60}")
    print(f"全部 bit 正确: {correct_all}/{summary['count']}")
    print(f"Crop vis: {vis_dir}")
    print(f"Saved   : {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
