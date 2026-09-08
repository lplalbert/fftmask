"""
从图片中提取固定位置的二维码，解码得到bit序列，保存文件名与bit的对应关系

用法:
    python extract_qr_mapping.py --image_dir ./output_images
    python extract_qr_mapping.py --image_dir ./output_images --qr_x 120 --qr_y 50 --qr_size 200 --output mapping.csv
"""

import argparse
import os
import glob
import cv2
import numpy as np


def cv2_imread(file_path):
    """支持中文路径的 cv2.imread"""
    return cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)


def extract_qr_from_image(image_path, qr_x, qr_y, qr_size):
    """从图片固定位置截取二维码并解码"""
    image = cv2_imread(image_path)
    if image is None:
        return None

    h, w = image.shape[:2]
    # 边界检查
    if qr_x + qr_size > w or qr_y + qr_size > h:
        return None

    # 截取二维码区域
    qr_region = image[qr_y:qr_y+qr_size, qr_x:qr_x+qr_size]

    # 使用 OpenCV QRCode 检测器
    detector = cv2.QRCodeDetector()
    data, bbox, straight = detector.detectAndDecode(qr_region)

    if data:
        return data
    return None


def parse_args():
    parser = argparse.ArgumentParser(description="从图片中提取二维码并建立映射")
    parser.add_argument("--image_dir", type=str, required=True,
                        help="图片文件夹路径")
    parser.add_argument("--qr_x", type=int, default=120,
                        help="二维码X位置 (默认: 120)")
    parser.add_argument("--qr_y", type=int, default=0,
                        help="二维码Y位置 (默认: 0)")
    parser.add_argument("--qr_size", type=int, default=200,
                        help="二维码大小 (默认: 200)")
    parser.add_argument("--output", type=str, default="image_bit_mapping.csv",
                        help="输出文件名 (默认: image_bit_mapping.csv)")
    parser.add_argument("--recursive", action="store_true",
                        help="递归搜索子文件夹")
    return parser.parse_args()


def main():
    args = parse_args()

    # 收集图片
    image_files = []
    extensions = ['*.jpg', '*.png', '*.jpeg']
    if args.recursive:
        for ext in extensions:
            image_files.extend(glob.glob(os.path.join(args.image_dir, '**', ext), recursive=True))
    else:
        for ext in extensions:
            image_files.extend(glob.glob(os.path.join(args.image_dir, ext)))

    # 去重（Windows不区分大小写）
    image_files = sorted(list(set(image_files)))
    print(f"找到 {len(image_files)} 张图片")

    # 提取二维码
    mapping = []
    failed = []

    for i, img_path in enumerate(image_files):
        bit_seq = extract_qr_from_image(img_path, args.qr_x, args.qr_y, args.qr_size)
        filename = os.path.basename(img_path)

        if bit_seq:
            mapping.append((filename, bit_seq))
            print(f"[{i+1}/{len(image_files)}] {filename} -> {bit_seq[:20]}...")
        else:
            failed.append(filename)
            print(f"[{i+1}/{len(image_files)}] {filename} -> 解码失败")

    # 保存结果
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write("filename,bits\n")
        for filename, bits in mapping:
            f.write(f"{filename},{bits}\n")

    print("=" * 50)
    print(f"成功: {len(mapping)} 张")
    print(f"失败: {len(failed)} 张")
    print(f"映射已保存到: {args.output}")

    if failed:
        print("\n失败文件列表:")
        for f in failed:
            print(f"  - {f}")


if __name__ == "__main__":
    main()
