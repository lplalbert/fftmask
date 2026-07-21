"""
v11 水印解码脚本
支持旋转矫正的水印解码

用法：
    python decode_channel_watermark_v11.py \
        --input image.png \
        --model_path best_cb_decoder_v11.pth \
        --bits_file bits.txt \
        --channel cb
"""
import argparse
import os

import cv2
import numpy as np
import torch

from watermark_decoder_v11 import AdvancedWatermarkDecoderV11


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


def load_model(args, device):
    """加载v11模型"""
    r_watermark = args.r
    ring_width = args.ring_width
    rings = [(ri - ring_width, ri + ring_width) for ri in r_watermark]

    r_rotation = args.r_rotation
    rotation_ring = (r_rotation - ring_width, r_rotation + ring_width)

    model = AdvancedWatermarkDecoderV11(
        n_sectors=args.num_bits,
        rings=rings,
        bits=args.bitsf,
        rotation_ring=rotation_ring,
        rotation_cycles=args.rotation_cycles,
    )
    state_dict = torch.load(args.model_path, map_location=device)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


def decode_crops(model, crop_channels, device, batch_size):
    """解码多个crop"""
    predictions = []
    rotation_angles = []
    with torch.no_grad():
        for start in range(0, len(crop_channels), batch_size):
            batch = crop_channels[start:start + batch_size]
            tensor = torch.from_numpy(np.stack(batch, axis=0)).to(device=device, dtype=torch.float32)
            tensor = tensor.unsqueeze(1).div_(127.5).sub_(1.0)
            output, _, rotation_angle = model(tensor, return_rotation=True)
            pred = (output > 0.5).long().cpu().numpy()
            predictions.extend(pred)
            rotation_angles.extend(rotation_angle.cpu().numpy())
    return np.asarray(predictions, dtype=np.int64), np.asarray(rotation_angles)


def main():
    parser = argparse.ArgumentParser(description="v11 watermark decoder with rotation correction")
    parser.add_argument("--input", type=str, default=None, help="Single input image path")
    parser.add_argument("--input_dir", type=str, default=None, help="Directory of input images")
    parser.add_argument("--channel", type=str, default="cb", choices=sorted(YCRCB_CHANNEL_INDEX.keys()),
                        help="YCrCb channel used for decoding")
    parser.add_argument("--model_path", type=str,
                        default="/mnt/lpl/fftmask/output/finetune_v11/best_cb_decoder_v11.pth",
                        help="Path to the decoder checkpoint")
    parser.add_argument("--bits_file", type=str, default=None,
                        help="Ground-truth bits file for accuracy calculation")
    parser.add_argument("--num_bits", type=int, default=60, help="Number of watermark bits")
    parser.add_argument("--r", type=int, nargs="+", default=[12, 25], help="Watermark ring radii")
    parser.add_argument("--bitsf", type=int, nargs="+", default=[15, 45],
                        help="Number of bits assigned to each radius")
    parser.add_argument("--ring_width", type=int, default=5,
                        help="Decoder ring half width")
    parser.add_argument("--r_rotation", type=int, default=18,
                        help="Rotation correction ring radius")
    parser.add_argument("--rotation_cycles", type=int, default=8,
                        help="Number of cycles in rotation correction ring")
    parser.add_argument("--crop_size", type=int, default=512, help="Crop size decoded by the model")
    parser.add_argument("--num_crops", type=int, default=5, help="Number of random crops per image")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for decoding")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for crop positions")
    parser.add_argument("--sample_mode", type=str, default="five_point",
                        choices=["five_point", "random"],
                        help="Crop sampling mode")
    parser.add_argument("--max_images", type=int, default=0,
                        help="Maximum number of images to process (0 = all)")
    parser.add_argument("--device", type=str, default="0",
                        help="CUDA device index")

    args = parser.parse_args()

    # 验证参数
    if len(args.r) != len(args.bitsf):
        raise ValueError("--r and --bitsf must have the same length")
    if sum(args.bitsf) != args.num_bits:
        raise ValueError(f"--num_bits ({args.num_bits}) must equal sum(--bitsf) ({sum(args.bitsf)})")

    gt_bits = None
    if args.bits_file:
        gt_bits = parse_bits_file(args.bits_file)
        if len(gt_bits) != args.num_bits:
            raise ValueError(f"bits_file length ({len(gt_bits)}) must equal --num_bits ({args.num_bits})")

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args, device)

    image_paths = list_images(args.input, args.input_dir, args.max_images)
    if not image_paths:
        print("No images found.")
        return

    print(f"Loaded model: {args.model_path}")
    print(f"Device: {device}")
    print(f"Images: {len(image_paths)}")
    print(f"Channel: {args.channel}")
    print(f"Watermark rings: {args.r}, bits: {args.bitsf}")
    print(f"Rotation ring: r={args.r_rotation}, cycles={args.rotation_cycles}")
    print()

    # 解码所有图像
    from decode_channel_watermark import make_crop_positions

    total_correct = 0
    total_bits = 0

    for image_path in image_paths:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to read: {image_path}")
            continue

        h, w = image.shape[:2]
        image_name = os.path.basename(image_path)

        # 提取通道
        channel_image = extract_channel(image, args.channel)

        # 生成裁剪位置
        positions = make_crop_positions(h, w, args.crop_size, args.num_crops, args.seed, args.sample_mode)

        # 裁剪
        crop_channels = []
        for y, x in positions:
            crop = channel_image[y:y + args.crop_size, x:x + args.crop_size]
            crop_channels.append(crop)

        # 解码
        predictions, rotation_angles = decode_crops(model, crop_channels, device, args.batch_size)

        # 投票
        vote_bits = (np.mean(predictions, axis=0) >= 0.5).astype(np.int64)
        vote_rotation = np.mean(rotation_angles) * 180  # 转换为度

        # 计算准确率
        if gt_bits is not None:
            crop_accs = [np.mean(pred == gt_bits) for pred in predictions]
            vote_acc = np.mean(vote_bits == gt_bits)
            total_correct += int(vote_acc * args.num_bits)
            total_bits += args.num_bits

            print(f"{image_name}:")
            print(f"  Crop accuracy: {np.mean(crop_accs) * 100:.2f}%")
            print(f"  Vote accuracy: {vote_acc * 100:.2f}%")
            print(f"  Rotation angle: {vote_rotation:.1f}°")
            print(f"  Vote bits: {bits_to_string(vote_bits)}")
        else:
            print(f"{image_name}:")
            print(f"  Vote bits: {bits_to_string(vote_bits)}")
            print(f"  Rotation angle: {vote_rotation:.1f}°")

    if gt_bits is not None and total_bits > 0:
        print(f"\nOverall accuracy: {total_correct / total_bits * 100:.2f}%")


if __name__ == "__main__":
    main()
