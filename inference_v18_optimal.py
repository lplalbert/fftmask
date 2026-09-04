"""
v18 最优解码脚本（带尺寸搜索）

当水印尺寸未知时（经过信道后图片可能被resize），遍历不同的tile size，
找到解码准确率最高的那个。

用法:
    python inference_v18_optimal.py \
        --gt_json /path/to/embed_gt_mapping.json \
        --input_dir /path/to/watermarked_images \
        --output_dir /path/to/decode_results \
        --min_ratio 0.45 --max_ratio 1.0 --step_ratio 0.01
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


def resize_to_wm512(image, wm_size, crop_size=512):
    """
    将图片resize，使水印tile大小变成crop_size

    Args:
        image: 原始图片
        wm_size: 水印tile大小（假设）
        crop_size: 目标tile大小（模型输入）

    Returns:
        resized: resize后的图片
        resize_ratio: 实际resize比例
    """
    h, w = image.shape[:2]

    # resize使水印tile变成512
    resize_ratio = crop_size / wm_size

    new_w = max(crop_size, int(w * resize_ratio))
    new_h = max(crop_size, int(h * resize_ratio))

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, resize_ratio


def decode_image(image, model, device, gt_bits, num_crops=5, seed=2026):
    """
    解码一张图片，返回投票准确率和预测bits

    Args:
        image: BGR图片
        model: 解码模型
        device: 设备
        gt_bits: GT bits
        num_crops: crop数量
        seed: 随机种子

    Returns:
        voted_acc: 投票准确率
        voted_bits: 投票后的bits
        crop_accs: 每个crop的准确率列表
    """
    h, w = image.shape[:2]
    block_size = 512

    # 生成crop位置（中心 + 随机）
    positions = []
    # 中心crop
    cy, cx = h // 2, w // 2
    positions.append((cy - block_size // 2, cx - block_size // 2))
    # 随机crop
    rng = np.random.default_rng(seed)
    for _ in range(num_crops - 1):
        y = rng.integers(0, h - block_size + 1)
        x = rng.integers(0, w - block_size + 1)
        positions.append((y, x))

    # 解码所有crop
    predictions = []
    crop_accs = []
    gt = np.array(gt_bits, dtype=np.float32)

    for y, x in positions:
        if y + block_size > h or x + block_size > w:
            continue
        crop = image[y:y + block_size, x:x + block_size]
        tensor = extract_cb_tile(crop).unsqueeze(0).to(device)
        with torch.no_grad():
            output, _, _ = model(tensor)
            pred = (output > 0.5).float().cpu().numpy().squeeze()
        predictions.append(pred)
        crop_accs.append(float(np.mean(pred == gt)))

    if not predictions:
        return 0.0, [], []

    # 投票
    predictions = np.array(predictions)
    voted_bits = (np.mean(predictions, axis=0) > 0.5).astype(float)
    voted_acc = float(np.mean(voted_bits == gt))

    return voted_acc, voted_bits.tolist(), crop_accs


def sweep_image(image, model, device, gt_bits, min_ratio=0.45, max_ratio=1.0,
                step_ratio=0.01, num_crops=5, seed=2026):
    """
    搜索图片的最佳水印尺寸

    Args:
        image: 原始BGR图片
        model: 解码模型
        device: 设备
        gt_bits: GT bits
        min_ratio: 最小比例（wm_size / short_side）
        max_ratio: 最大比例
        step_ratio: 步长
        num_crops: 每个尺寸的crop数量
        seed: 随机种子

    Returns:
        best_ratio: 最佳比例
        best_acc: 最佳准确率
        best_voted_bits: 最佳投票bits
        all_results: 所有尺寸的结果
    """
    h, w = image.shape[:2]
    short_side = min(h, w)

    best_ratio = None
    best_acc = 0.0
    best_voted_bits = []
    all_results = []

    ratio = min_ratio
    while ratio <= max_ratio + 1e-12:
        # 计算假设的水印大小
        wm_size = int(short_side * ratio)
        if wm_size < 512:
            ratio += step_ratio
            continue

        # resize图片，使水印tile变成512
        resized, resize_ratio = resize_to_wm512(image, wm_size)

        # 解码
        voted_acc, voted_bits, crop_accs = decode_image(
            resized, model, device, gt_bits, num_crops=num_crops, seed=seed
        )

        all_results.append({
            "ratio": ratio,
            "wm_size": wm_size,
            "resize_ratio": resize_ratio,
            "voted_acc": voted_acc,
            "avg_crop_acc": float(np.mean(crop_accs)) if crop_accs else 0.0,
        })

        if voted_acc > best_acc:
            best_acc = voted_acc
            best_ratio = ratio
            best_voted_bits = voted_bits

        ratio += step_ratio

    return best_ratio, best_acc, best_voted_bits, all_results


# 可视化颜色
COLORS = [
    (0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255),
    (255, 0, 255), (255, 255, 0), (0, 128, 255), (128, 0, 255),
    (0, 255, 128), (255, 128, 0), (128, 255, 0), (255, 0, 128),
    (0, 64, 255), (64, 0, 255), (0, 255, 64), (255, 64, 0),
]


def main():
    parser = argparse.ArgumentParser(description="v18 最优解码（带尺寸搜索）")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--model_path", type=str,
                        default="/data/lpl/fftmask/output/v18_hollow_pair/best_model.pth")
    parser.add_argument("--gt_json", type=str, required=True)
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_bits", type=int, default=60)
    parser.add_argument("--bits_per_ring", type=int, nargs="+", default=[15, 45])
    parser.add_argument("--ring_positions", type=int, nargs="+", default=[12, 25])
    parser.add_argument("--angle_bins", type=int, default=180)
    # 搜索参数
    parser.add_argument("--min_ratio", type=float, default=0.45, help="Sweep start ratio")
    parser.add_argument("--max_ratio", type=float, default=1.0, help="Sweep end ratio")
    parser.add_argument("--step_ratio", type=float, default=0.01, help="Sweep step ratio")
    # 解码参数
    parser.add_argument("--num_crops", type=int, default=5, help="Crops per size for sweep")
    parser.add_argument("--final_num_crops", type=int, default=16, help="Crops for final decode")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    model = WatermarkDecoderV17(
        n_sectors=args.num_bits,
        bits=args.bits_per_ring,
        angle_bins=args.angle_bins,
        radius_bins=12,
        ring_positions_init=[float(r) for r in args.ring_positions]
    )

    state_dict = torch.load(args.model_path, map_location=device, weights_only=False)
    if isinstance(state_dict, dict) and 'model' in state_dict:
        state_dict = state_dict['model']
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    if 'ring_positions' in state_dict:
        del state_dict['ring_positions']

    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    print(f"Model loaded: {args.model_path}")
    print(f"Device: {device}")

    # 加载GT
    with open(args.gt_json, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    mappings = gt_data.get("mappings", [])
    watermark_bits = gt_data.get("watermark_bits", [])

    if not mappings:
        raise ValueError("GT JSON has no mappings")

    # 收集图片路径和GT bits
    image_paths = []
    gt_bits_list = []

    for item in mappings:
        wm_path = item.get("watermarked_path")
        if wm_path and os.path.exists(wm_path):
            image_paths.append(wm_path)
        else:
            wm_file = item.get("watermarked_file") or item.get("watermarked")
            if wm_file:
                img_path = os.path.join(args.input_dir, wm_file)
                if os.path.exists(img_path):
                    image_paths.append(img_path)
                else:
                    print(f"WARNING: Missing image: {wm_file}")
                    continue
            else:
                continue

        if "watermark_bits" in item:
            gt_bits_list.append(item["watermark_bits"])
        elif watermark_bits:
            gt_bits_list.append(watermark_bits)
        else:
            image_paths.pop()
            continue

    if not image_paths:
        raise ValueError("No images matched")

    # 创建输出目录
    vis_dir = os.path.join(args.output_dir, "crop_vis")
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    print(f"Found {len(image_paths)} images")
    print(f"Sweep: ratio [{args.min_ratio}, {args.max_ratio}], step={args.step_ratio}")

    # 统计
    results = []
    total_correct = 0
    total_bits = 0
    correct_all = 0

    for img_idx in tqdm(range(len(image_paths)), desc="Decoding"):
        img_path = image_paths[img_idx]
        gt_bits = np.array(gt_bits_list[img_idx], dtype=np.float32)
        fname = os.path.basename(img_path)

        image = cv2_imread(img_path)
        if image is None:
            print(f"WARNING: Failed to load {img_path}")
            continue

        h_orig, w_orig = image.shape[:2]
        short_side = min(h_orig, w_orig)

        # 第一步：搜索最佳tile size
        best_ratio, best_acc, best_voted_bits, sweep_results = sweep_image(
            image, model, device, gt_bits,
            min_ratio=args.min_ratio,
            max_ratio=args.max_ratio,
            step_ratio=args.step_ratio,
            num_crops=args.num_crops,
            seed=args.seed + img_idx
        )

        # 第二步：用最佳尺寸进行详细解码（更多crops）
        best_wm_size = int(short_side * best_ratio)
        resized, resize_ratio = resize_to_wm512(image, best_wm_size)
        h, w = resized.shape[:2]
        vis_img = resized.copy()

        # 详细解码
        final_voted_acc, final_voted_bits, final_crop_accs = decode_image(
            resized, model, device, gt_bits,
            num_crops=args.final_num_crops,
            seed=args.seed + img_idx + 10000
        )

        # 可视化
        block_size = 512
        cy, cx = h // 2, w // 2
        # 中心框
        cv2.rectangle(vis_img, (cx - block_size // 2, cy - block_size // 2),
                      (cx + block_size // 2, cy + block_size // 2), (0, 255, 0), 3)

        # 标注
        cv2.putText(vis_img, f'Ratio: {best_ratio:.2f}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.putText(vis_img, f'Vote: {final_voted_acc*100:.1f}%', (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        # 保存可视化
        vis_name = f"vis_{os.path.splitext(fname)[0]}.jpg"
        cv2.imwrite(os.path.join(vis_dir, vis_name), vis_img)

        # 统计
        correct_bits = int(np.sum(np.array(final_voted_bits) == gt_bits))
        total_correct += correct_bits
        total_bits += args.num_bits
        if final_voted_acc == 1.0:
            correct_all += 1

        results.append({
            "file": fname,
            "original_size": f"{w_orig}x{h_orig}",
            "short_side": short_side,
            "best_ratio": best_ratio,
            "best_wm_size": best_wm_size,
            "sweep_voted_acc": best_acc,
            "final_voted_acc": final_voted_acc,
            "final_crop_accs": final_crop_accs,
            "correct_bits": correct_bits,
            "total_bits": int(args.num_bits),
            "gt": gt_bits.tolist(),
            "voted_pred": final_voted_bits,
            "sweep_results": sweep_results,
        })

        print(f"  {fname}: ratio={best_ratio:.2f}, wm_size={best_wm_size}, "
              f"sweep_acc={best_acc*100:.1f}%, final_acc={final_voted_acc*100:.1f}%")

    summary = {
        "count": len(results),
        "avg_voted_accuracy_pct": np.mean([r["final_voted_acc"] for r in results]) * 100 if results else 0.0,
        "avg_decode_accuracy_pct": (total_correct / total_bits * 100) if total_bits else 0.0,
        "all_bits_correct": correct_all,
        "total_correct_bits": int(total_correct),
        "total_bits": int(total_bits),
        "dataset": os.path.basename(args.input_dir),
        "gt_json": args.gt_json,
        "input_dir": args.input_dir,
        "model_path": args.model_path,
        "num_crops": args.final_num_crops,
        "sweep_range": f"[{args.min_ratio}, {args.max_ratio}]",
        "sweep_step": args.step_ratio,
        "watermark_bits_string": gt_data.get("watermark_bits_string", ""),
        "version": "v18_sweep",
    }

    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(os.path.join(args.output_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"Dataset : {os.path.basename(args.input_dir)}")
    print(f"Images  : {summary['count']}")
    print(f"Sweep   : ratio [{args.min_ratio}, {args.max_ratio}], step={args.step_ratio}")
    print(f"Crops   : {args.final_num_crops} (final decode)")
    print(f"Model   : v18 (2 rings: r={args.ring_positions})")
    print(f"{'='*60}")
    print(f"投票后平均准确率:      {summary['avg_voted_accuracy_pct']:.2f}%")
    print(f"{'='*60}")
    print(f"全部 bit 正确: {correct_all}/{summary['count']}")
    print(f"Crop vis: {vis_dir}")
    print(f"Saved   : {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
