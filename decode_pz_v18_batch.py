"""
批量解码 pz/v18 的 b 和 cb 通道图片
支持 B 通道和 Cb 通道
"""

import os
import json
import cv2
import numpy as np
import torch
from pathlib import Path
from watermark_decoder_v17 import WatermarkDecoderV17

BITS_FILE = "/home/lpl2025/lpl/fftmask/0908test_sorted/part/pz_bits.json"
BASE_DIR = "/home/lpl2025/lpl/fftmask/0908test_sorted/part/rectified/pz/v18"

MODELS = {
    "b": "/home/lpl2025/lpl/fftmask/output/v18_b_pair/best_model.pth",
    "cb": "/home/lpl2025/lpl/fftmask/output/v18_hollow_pair/best_model.pth",
}

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def extract_channel(image_bgr, channel):
    """提取指定通道"""
    if channel == "b":
        return image_bgr[:, :, 0]  # BGR 的 B 通道
    elif channel == "cb":
        ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
        return ycrcb[:, :, 2]
    elif channel == "y":
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError(f"Unknown channel: {channel}")


def make_crop_positions(h, w, crop_size, mode="five_point"):
    """生成裁剪位置"""
    positions = []
    if mode == "five_point":
        cy, cx = h // 2, w // 2
        half = crop_size // 2
        positions.append((cy - half, cx - half))  # center
        positions.append((0, 0))  # top-left
        positions.append((0, w - crop_size))  # top-right
        positions.append((h - crop_size, 0))  # bottom-left
        positions.append((h - crop_size, w - crop_size))  # bottom-right
    return positions


def bits_to_string(bits):
    return "".join(str(int(b)) for b in bits)


def decode_directory(model, channel, index_dir, gt_bits, device, crop_size=512, batch_size=64):
    """解码一个目录的所有图片"""
    image_files = sorted([
        f for f in os.listdir(index_dir)
        if f.lower().endswith(IMAGE_EXTENSIONS)
    ])

    if not image_files:
        return None

    results = []
    for img_name in image_files:
        img_path = os.path.join(index_dir, img_name)
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image is None:
            continue

        h, w = image.shape[:2]
        # resize 到合适大小
        scale = crop_size / min(h, w)
        new_w = max(crop_size, int(round(w * scale)))
        new_h = max(crop_size, int(round(h * scale)))
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        h, w = resized.shape[:2]
        channel_image = extract_channel(resized, channel)

        # 5点采样
        positions = make_crop_positions(h, w, crop_size)
        crops = []
        for y, x in positions:
            if y + crop_size <= h and x + crop_size <= w:
                crop = channel_image[y:y + crop_size, x:x + crop_size].copy()
                crops.append(crop)

        if not crops:
            continue

        # 批量解码
        predictions = []
        with torch.no_grad():
            for start in range(0, len(crops), batch_size):
                batch = np.stack(crops[start:start + batch_size], axis=0)
                tensor = torch.from_numpy(batch).to(device=device, dtype=torch.float32)
                tensor = tensor.unsqueeze(1).div_(127.5).sub_(1.0)
                output, _, _ = model(tensor)
                pred = (output > 0.5).long().cpu().numpy()
                predictions.extend(pred)

        predictions = np.asarray(predictions, dtype=np.int64)

        # 投票
        vote_bits = (np.mean(predictions, axis=0) >= 0.5).astype(np.int64)
        vote_text = bits_to_string(vote_bits)

        # 计算准确率
        if gt_bits is not None:
            gt = np.array(gt_bits, dtype=np.int64)
            vote_acc = float(np.mean(vote_bits == gt) * 100.0)
            crop_accs = np.mean(predictions == gt[None, :], axis=1)
            crop_acc = float(np.mean(crop_accs) * 100.0)
        else:
            vote_acc = None
            crop_acc = None

        results.append({
            "image": img_name,
            "vote_bits": vote_text,
            "vote_acc": vote_acc,
            "crop_acc": crop_acc,
        })

    return results


def main():
    # 加载 bits
    with open(BITS_FILE, "r") as f:
        all_bits = json.load(f)

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 加载模型
    models = {}
    for channel, model_path in MODELS.items():
        print(f"Loading {channel} model: {model_path}")
        model = WatermarkDecoderV17(
            n_sectors=60,
            bits=[15, 45],
            angle_bins=180,
            radius_bins=12,
            ring_positions_init=[12.0, 25.0],
        )
        state_dict = torch.load(model_path, map_location=device, weights_only=False)
        if isinstance(state_dict, dict) and 'model' in state_dict:
            state_dict = state_dict['model']
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        if 'ring_positions' in state_dict:
            del state_dict['ring_positions']
        model.load_state_dict(state_dict, strict=False)
        model = model.to(device)
        model.eval()
        models[channel] = model

    # 解码结果
    all_results = {}

    for channel in ["b", "cb"]:
        channel_dir = os.path.join(BASE_DIR, channel)
        if not os.path.isdir(channel_dir):
            print(f"\n目录不存在: {channel_dir}")
            continue

        print(f"\n{'='*80}")
        print(f"Channel: {channel.upper()}")
        print(f"{'='*80}")

        channel_results = {}

        for index in sorted(os.listdir(channel_dir)):
            index_dir = os.path.join(channel_dir, index)
            if not os.path.isdir(index_dir):
                continue

            bits_key = f"{channel}_{index}"
            gt_bits_str = all_bits.get(bits_key)
            if gt_bits_str:
                gt_bits = [int(b) for b in gt_bits_str]
            else:
                gt_bits = None

            results = decode_directory(
                models[channel], channel, index_dir, gt_bits, device
            )

            if results:
                # 统计
                vote_accs = [r["vote_acc"] for r in results if r["vote_acc"] is not None]
                avg_vote_acc = np.mean(vote_accs) if vote_accs else None
                crop_accs = [r["crop_acc"] for r in results if r["crop_acc"] is not None]
                avg_crop_acc = np.mean(crop_accs) if crop_accs else None

                print(f"\n  [{channel}/{index}] {len(results)} 张图片")
                if avg_vote_acc is not None:
                    print(f"    平均 vote_acc: {avg_vote_acc:.2f}%")
                    print(f"    平均 crop_acc: {avg_crop_acc:.2f}%")
                    print(f"    GT bits: {gt_bits_str[:30]}...")

                # 打印每张图片结果
                for r in results:
                    acc_str = f"{r['vote_acc']:.1f}%" if r['vote_acc'] is not None else "N/A"
                    print(f"    {r['image']}: vote_acc={acc_str}")

                channel_results[index] = {
                    "count": len(results),
                    "avg_vote_acc": avg_vote_acc,
                    "avg_crop_acc": avg_crop_acc,
                    "gt_bits": gt_bits_str,
                    "details": results,
                }

        all_results[channel] = channel_results

    # 保存结果
    output_file = "/home/lpl2025/lpl/fftmask/0908test_sorted/part/pz_v18_decode_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {output_file}")

    # 总结
    print(f"\n{'='*80}")
    print("总结")
    print(f"{'='*80}")
    for channel in ["b", "cb"]:
        if channel in all_results:
            accs = [v["avg_vote_acc"] for v in all_results[channel].values() if v["avg_vote_acc"] is not None]
            if accs:
                print(f"  {channel.upper()}: 平均 vote_acc = {np.mean(accs):.2f}% (共 {len(accs)} 个 index)")


if __name__ == "__main__":
    main()
