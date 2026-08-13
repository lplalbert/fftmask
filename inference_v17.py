"""
v17 版本：Inference script for v17 watermark decoder.

使用 v17 解码器（2环：r=8, r=15，无旋转环）。

Usage:
  python inference_v17.py \
    --gt_json /home/lpl2025/lpl/fftmask/output/v17_xiaomi/embed_gt_mapping.json \
    --input_dir /home/lpl2025/lpl/fftmask/output/v17_xiaomi \
    --output_dir /home/lpl2025/lpl/fftmask/output/v17_xiaomi_decode \
    --num_crops 16
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

from watermark_decoder_v17 import WatermarkDecoderV17


def cv2_imread(path):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


TRAIN_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])


def resize_min_edge(image, min_size=1024):
    """Resize image so shortest edge equals min_size."""
    h, w = image.shape[:2]
    if min(h, w) <= min_size:
        return image
    scale = min_size / min(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--model_path", type=str,
                        default="/home/lpl2025/lpl/fftmask/output/v17_continue/stage2_best.pth")
    parser.add_argument("--gt_json", type=str, required=True)
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_bits", type=int, default=60)
    parser.add_argument("--num_crops", type=int, default=16, help="Total crops per image, including center crop")
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    if args.num_crops < 1:
        raise ValueError("--num_crops must be >= 1")

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # v17 模型 (2 rings: r=8, r=15，无旋转环)
    model = WatermarkDecoderV17(
        n_sectors=args.num_bits,
        bits=[20, 40],  # v17配置：20bit + 40bit = 60bit
        ring_positions_init=[8.0, 15.0],  # v17圆环位置
    )
    state_dict = torch.load(args.model_path, map_location=device, weights_only=False)
    if isinstance(state_dict, dict) and 'model' in state_dict:
        state_dict = state_dict['model']
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    # 检测是否需要ring index重映射（v14 vs v17格式）
    is_v14_format = any(k.startswith('ring_transformers.2.') for k in state_dict.keys())
    if is_v14_format:
        print("检测到v14格式权重，应用ring index重映射")
        mapped_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('ring_transformers.0.'):
                new_k = k.replace('ring_transformers.0.', 'ring_transformers.0.')
                mapped_state_dict[new_k] = v
            elif k.startswith('ring_transformers.1.'):
                new_k = k.replace('ring_transformers.1.', 'ring_transformers.1.')
                mapped_state_dict[new_k] = v
            elif k.startswith('ring_transformers.2.'):
                # v17只有2个ring，跳过第3个
                continue
            else:
                mapped_state_dict[k] = v
        state_dict = mapped_state_dict

    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    with open(args.gt_json, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    mappings = gt_data.get("mappings", [])
    watermark_bits = gt_data.get("watermark_bits", [])

    if not mappings:
        raise ValueError("GT JSON has no mappings")

    # 支持子文件夹结构
    image_paths = []
    gt_bits_list = []

    for item in mappings:
        wm_path = item.get("watermarked_path")
        if wm_path and os.path.exists(wm_path):
            image_paths.append(wm_path)
        else:
            wm_file = item.get("watermarked_file") or item.get("watermarked")
            if wm_file:
                subfolder = item.get("subfolder", "")
                if subfolder:
                    img_path = os.path.join(args.input_dir, subfolder, wm_file)
                else:
                    img_path = os.path.join(args.input_dir, wm_file)
                if os.path.exists(img_path):
                    image_paths.append(img_path)
                else:
                    print(f"WARNING: Missing image: {wm_file}")
                    continue
            else:
                print(f"WARNING: No watermarked_file in item")
                continue

        if "watermark_bits" in item:
            gt_bits_list.append(item["watermark_bits"])
        elif watermark_bits:
            gt_bits_list.append(watermark_bits)
        else:
            print(f"WARNING: No watermark_bits")
            image_paths.pop()
            continue

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

        h, w = image.shape[:2]
        min_side = min(h, w)
        crop_size = min_side // 2  # 最短边的一半
        vis_img = image.copy()

        # 中心 crop（crop_size × crop_size）
        center, cy, cx = get_center_crop(image, crop_size)
        # resize到512×512再解码
        center_512 = cv2.resize(center, (block_size, block_size))
        center_tensor = extract_cb_tile(center_512).unsqueeze(0).to(device)
        with torch.no_grad():
            center_out, _, _ = model(center_tensor)
            center_pred = (center_out > 0.5).float().cpu().numpy().squeeze()
        center_acc = float(np.mean(center_pred == gt_bits))
        center_accs.append(center_acc)

        # 画中心框（绿色）
        cv2.rectangle(vis_img, (cx, cy), (cx + crop_size, cy + crop_size), (0, 255, 0), 3)
        cv2.putText(vis_img, f'C {center_acc*100:.0f}%', (cx + 5, cy + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 随机 crop
        all_preds = [center_pred]
        crop_infos = [{"method": "center", "y": cy, "x": cx, "acc": center_acc}]

        for ci in range(args.num_crops - 1):
            rand_crop, ry, rx = get_random_crop(image, crop_size)
            # resize到512×512再解码
            rand_512 = cv2.resize(rand_crop, (block_size, block_size))
            rand_tensor = extract_cb_tile(rand_512).unsqueeze(0).to(device)
            with torch.no_grad():
                rand_out, _, _ = model(rand_tensor)
                rand_pred = (rand_out > 0.5).float().cpu().numpy().squeeze()
            rand_acc = float(np.mean(rand_pred == gt_bits))
            random_accs.append(rand_acc)
            all_preds.append(rand_pred)
            crop_infos.append({"method": "random", "y": ry, "x": rx, "acc": rand_acc})

            # 画随机框
            color = COLORS[ci % len(COLORS)]
            cv2.rectangle(vis_img, (rx, ry), (rx + crop_size, ry + crop_size), color, 2)
            cv2.putText(vis_img, f'#{ci+1} {rand_acc*100:.0f}%', (rx + 5, ry + 20),
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
        "watermark_bits_string": gt_data.get("watermark_bits_string", ""),
        "version": "v17",
        "rings": [8, 15],
        "bits": [20, 40],
    }

    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(os.path.join(args.output_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"Dataset : {os.path.basename(args.input_dir)}")
    print(f"Images  : {summary['count']}")
    print(f"Crops   : {args.num_crops} (1 center + {args.num_crops-1} random)")
    print(f"Model   : v17 (2 rings: r=8, r=15)")
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
