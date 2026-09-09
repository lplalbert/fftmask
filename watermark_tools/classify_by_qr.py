#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根据照片中的信息二维码自动分类。

读取二维码内容（格式: v17|cb|01|bits），自动按 版本/通道/序号 分类存放。

用法:
    # 分类单张照片
    python classify_by_qr.py --input photo.jpg --output sorted/

    # 批量分类目录下所有照片
    python classify_by_qr.py --input_dir ./photos --output sorted/

    # 分类后移动（而非复制）
    python classify_by_qr.py --input_dir ./photos --output sorted/ --move

    # 只打印分类结果，不实际移动/复制
    python classify_by_qr.py --input_dir ./photos --dry-run
"""

import argparse
import csv
import json
import re
import shutil
import sys
from pathlib import Path

import cv2


def decode_qr(image):
    """从图片中解码二维码，返回解码结果列表（使用OpenCV内置检测器）"""
    import numpy as np
    detector = cv2.QRCodeDetector()

    # 多种预处理方式
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    candidates = [
        image,
        gray,
        cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1],
        cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)[1],
        cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        cv2.GaussianBlur(gray, (5, 5), 0),
    ]

    for processed in candidates:
        # 尝试多二维码检测
        retval, decoded_info, points, _ = detector.detectAndDecodeMulti(processed)
        if retval and decoded_info:
            decoded = []
            for data in decoded_info:
                if data and '|' in data:
                    decoded.append({"data": data, "type": "QRCODE", "rect": None})
            if decoded:
                return decoded

        # 尝试单个检测
        data, points, _ = detector.detectAndDecode(processed)
        if data and '|' in data:
            return [{"data": data, "type": "QRCODE", "rect": None}]

    return []


def parse_qr_data(data):
    """
    解析二维码数据。
    格式: v17|cb|01|1001000010...
    返回: {version, channel, index, bits} 或 None
    """
    # 优先匹配新格式: version|channel|index|bits
    m = re.match(r"^(v\d+)\|(cb|b)\|(\d{2})\|([01]{60})$", data)
    if m:
        return {
            "version": m.group(1),
            "channel": m.group(2),
            "index": int(m.group(3)),
            "bits": m.group(4),
        }

    # 兼容旧格式: version|channel|bits（无序号）
    m = re.match(r"^(v\d+)\|(cb|b)\|([01]{60})$", data)
    if m:
        return {
            "version": m.group(1),
            "channel": m.group(2),
            "index": None,
            "bits": m.group(3),
        }

    return None


def decode_with_multi_scale(image, scales=(1.0, 0.5, 2.0, 0.25)):
    """多尺度尝试解码二维码"""
    for scale in scales:
        if scale != 1.0:
            h, w = image.shape[:2]
            resized = cv2.resize(image, (int(w * scale), int(h * scale)))
        else:
            resized = image

        results = decode_qr(resized)
        if results:
            return results

    return []


def decode_from_region(image, x_ratio=0.0, y_ratio=0.08, w_ratio=0.15, h_ratio=0.2):
    """
    从图片指定区域裁剪后解码二维码。
    默认裁剪左上角区域（二维码通常在 y=100 附近）。
    """
    h, w = image.shape[:2]
    x = int(w * x_ratio)
    y = int(h * y_ratio)
    rw = int(w * w_ratio)
    rh = int(h * h_ratio)
    region = image[y:y + rh, x:x + rw]

    results = decode_qr(region)
    if results:
        return results

    # 区域解码失败，尝试全图
    return []


def process_single_image(image_path):
    """
    处理单张图片，返回分类信息。
    Returns: {file, version, channel, index, bits, qr_data, error}
    """
    result = {
        "file": str(image_path),
        "version": None,
        "channel": None,
        "index": None,
        "bits": None,
        "qr_data": None,
        "error": None,
    }

    image = cv2.imread(str(image_path))
    if image is None:
        result["error"] = "无法读取图片"
        return result

    # 先从左上角区域解码（二维码在(0,100)位置）
    decoded = decode_from_region(image)

    # 区域解码失败，尝试全图多尺度
    if not decoded:
        decoded = decode_with_multi_scale(image)

    if not decoded:
        result["error"] = "未检测到二维码"
        return result

    # 尝试解析每个检测到的二维码
    for d in decoded:
        parsed = parse_qr_data(d["data"])
        if parsed:
            result.update(parsed)
            result["qr_data"] = d["data"]
            return result

    # 检测到二维码但格式不匹配
    result["error"] = f"二维码格式不匹配: {decoded[0]['data'][:50]}"
    result["qr_data"] = decoded[0]["data"]
    return result


def classify_single(image_path, output_dir, move=False, dry_run=False):
    """分类单张图片，返回分类结果"""
    info = process_single_image(image_path)

    if info["error"]:
        print(f"  ✗ {Path(image_path).name}: {info['error']}")
        return info

    # 构建目标路径: output/v17/cb/02/
    if info["index"] is not None:
        dest_dir = Path(output_dir) / info["version"] / info["channel"] / f"{info['index']:02d}"
    else:
        # 旧格式无序号
        dest_dir = Path(output_dir) / info["version"] / info["channel"] / "unknown"

    dest_path = dest_dir / Path(image_path).name

    print(f"  ✓ {Path(image_path).name} → {info['version']}/{info['channel']}/{info['index']:02d if info['index'] is not None else '?'}")

    if dry_run:
        info["dest"] = str(dest_path)
        return info

    dest_dir.mkdir(parents=True, exist_ok=True)

    if move:
        shutil.move(str(image_path), str(dest_path))
    else:
        shutil.copy2(str(image_path), str(dest_path))

    info["dest"] = str(dest_path)
    return info


def main():
    parser = argparse.ArgumentParser(description="根据二维码自动分类照片")
    parser.add_argument("--input", type=Path, help="单张图片路径")
    parser.add_argument("--input_dir", type=Path, help="图片目录")
    parser.add_argument("--output", type=Path, default=Path("sorted"), help="输出目录（默认: sorted/）")
    parser.add_argument("--move", action="store_true", help="移动文件（默认是复制）")
    parser.add_argument("--dry-run", action="store_true", help="只打印，不实际操作")
    parser.add_argument("--log", type=Path, help="输出分类日志CSV")
    args = parser.parse_args()

    if not args.input and not args.input_dir:
        parser.error("需要指定 --input 或 --input_dir")

    # 收集图片列表
    image_paths = []
    if args.input:
        image_paths.append(args.input)
    if args.input_dir:
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
            image_paths.extend(sorted(args.input_dir.glob(ext)))

    if not image_paths:
        print("没有找到图片")
        sys.exit(1)

    print(f"找到 {len(image_paths)} 张图片")
    if args.dry_run:
        print("[DRY RUN] 只打印分类结果，不实际操作\n")

    # 分类
    results = []
    success = 0
    fail = 0

    for img_path in image_paths:
        info = classify_single(img_path, args.output, move=args.move, dry_run=args.dry_run)
        results.append(info)
        if info["error"]:
            fail += 1
        else:
            success += 1

    # 统计
    print(f"\n分类完成: 成功 {success}, 失败 {fail}")

    # 按版本/通道/序号统计
    from collections import Counter
    groups = Counter()
    for r in results:
        if not r["error"]:
            key = f"{r['version']}/{r['channel']}/{r['index']:02d}" if r['index'] is not None else f"{r['version']}/{r['channel']}/?"
            groups[key] += 1

    if groups:
        print("\n分类统计:")
        for key, count in sorted(groups.items()):
            print(f"  {key}: {count} 张")

    # 保存日志
    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        with open(args.log, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["file", "version", "channel", "index", "bits", "qr_data", "error", "dest"])
            writer.writeheader()
            writer.writerows(results)
        print(f"\n日志已保存: {args.log}")


if __name__ == "__main__":
    main()
