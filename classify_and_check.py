#!/usr/bin/env python3
"""
分类+检查：根据QR码分类图片，并检查分类是否正确

用法:
    python classify_and_check.py --input_dir /path/to/0908test --output /path/to/output
"""

import argparse
import csv
import json
import re
import shutil
import sys
from pathlib import Path
from collections import Counter, defaultdict

import cv2
import numpy as np


def read_image(path):
    """读取图片（支持中文路径）"""
    with open(str(path), 'rb') as f:
        raw = np.frombuffer(f.read(), dtype=np.uint8)
    return cv2.imdecode(raw, cv2.IMREAD_COLOR)


def decode_qr(image):
    """从图片中解码二维码（多尺度多区域多预处理）"""
    detector = cv2.QRCodeDetector()
    h, w = image.shape[:2]

    # 缩放到合理大小加速检测
    scale = min(1.0, 1024 / max(h, w))
    if scale < 1.0:
        small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        small = image

    sh, sw = small.shape[:2]
    regions = [
        (0, 0, sw, sh),           # 全图
        (0, 0, sw // 2, sh // 2), # 左上1/4
        (0, 0, sw // 3, sh // 3), # 左上1/9
        (0, 0, sw // 4, sh // 4), # 左上1/16
    ]

    for x1, y1, x2, y2 in regions:
        crop = small[y1:y2, x1:x2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        # 按优先级尝试不同预处理（参考check_classification.py）
        for processed in [
            crop,                    # 原图（BGR）
            gray,                    # 灰度
            cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1],  # 阈值127
            cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)[1],  # 阈值100
            cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],  # OTSU自适应
            cv2.GaussianBlur(gray, (5, 5), 0),  # 高斯模糊
        ]:
            d, _, _ = detector.detectAndDecode(processed)
            if d and '|' in d:
                return d

    return None


def parse_qr_data(data):
    """解析二维码数据: v17|cb|01|1001000010..."""
    if not data:
        return None
    parts = data.split('|')
    if len(parts) != 4:
        return None
    return {
        'version': parts[0],
        'channel': parts[1],
        'index': parts[2],
        'bits': parts[3],
    }


def classify_images(input_dir, output_dir, move=False, dry_run=False):
    """分类图片到 version/channel/index 目录"""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    image_files = sorted([
        f for f in input_dir.rglob('*')
        if f.is_file() and f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp')
    ])

    print(f"找到 {len(image_files)} 张图片")

    results = []
    success = 0
    fail = 0

    for img_path in image_files:
        img = read_image(img_path)
        if img is None:
            print(f"  ✗ {img_path.name}: 无法读取")
            results.append({'file': img_path.name, 'error': '无法读取'})
            fail += 1
            continue

        qr_data = decode_qr(img)
        parsed = parse_qr_data(qr_data)

        if not parsed:
            print(f"  ✗ {img_path.name}: 未检测到二维码")
            results.append({'file': img_path.name, 'error': '未检测到二维码'})
            fail += 1
            continue

        # 构建目标路径: output/v17/cb/02/
        dest_dir = output_dir / parsed['version'] / parsed['channel'] / parsed['index']
        dest_path = dest_dir / img_path.name

        print(f"  ✓ {img_path.name} → {parsed['version']}/{parsed['channel']}/{parsed['index']}")

        if not dry_run:
            dest_dir.mkdir(parents=True, exist_ok=True)
            if move:
                shutil.move(str(img_path), str(dest_path))
            else:
                shutil.copy2(str(img_path), str(dest_path))

        results.append({
            'file': img_path.name,
            'version': parsed['version'],
            'channel': parsed['channel'],
            'index': parsed['index'],
            'bits': parsed['bits'],
            'qr_data': qr_data,
            'dest': str(dest_path),
            'error': None,
        })
        success += 1

    # 统计
    print(f"\n分类完成: 成功 {success}, 失败 {fail}")

    groups = Counter()
    for r in results:
        if not r.get('error'):
            key = f"{r['version']}/{r['channel']}/{r['index']}"
            groups[key] += 1

    if groups:
        print("\n分类统计:")
        for key, count in sorted(groups.items()):
            print(f"  {key}: {count} 张")

    return results


def check_classification(base_dir):
    """检查已分类目录中图片QR码与目录结构是否一致"""
    base_dir = Path(base_dir)
    results = []
    contradictions = []

    for version_dir in sorted(base_dir.iterdir()):
        if not version_dir.is_dir():
            continue
        version = version_dir.name

        for channel_dir in sorted(version_dir.iterdir()):
            if not channel_dir.is_dir():
                continue
            channel = channel_dir.name

            for index_dir in sorted(channel_dir.iterdir()):
                if not index_dir.is_dir():
                    continue
                index = index_dir.name

                images = sorted([
                    f for f in index_dir.iterdir()
                    if f.is_file() and f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp')
                ])

                for img_path in images:
                    img = read_image(img_path)
                    if img is None:
                        continue

                    qr_data = decode_qr(img)
                    parsed = parse_qr_data(qr_data)

                    entry = {
                        'file': img_path.name,
                        'dir_version': version,
                        'dir_channel': channel,
                        'dir_index': index,
                        'qr_raw': qr_data,
                        'qr_version': parsed['version'] if parsed else None,
                        'qr_channel': parsed['channel'] if parsed else None,
                        'qr_index': parsed['index'] if parsed else None,
                        'bits': parsed['bits'] if parsed else None,
                    }
                    results.append(entry)

                    # 检查是否矛盾
                    if parsed:
                        if parsed['version'] != version or parsed['channel'] != channel or parsed['index'] != index:
                            contradictions.append(entry)
                            print(f"[矛盾] {version}/{channel}/{index}/{img_path.name} → QR: {qr_data}")
                    else:
                        print(f"[未识别] {version}/{channel}/{index}/{img_path.name}")

    return results, contradictions


def main():
    parser = argparse.ArgumentParser(description="根据QR码分类图片并检查")
    parser.add_argument("--input_dir", type=Path, required=True, help="输入目录")
    parser.add_argument("--output", type=Path, required=True, help="输出目录")
    parser.add_argument("--move", action="store_true", help="移动文件（默认复制）")
    parser.add_argument("--dry-run", action="store_true", help="只打印不操作")
    parser.add_argument("--check", action="store_true", help="检查已分类目录")
    parser.add_argument("--log", type=Path, help="输出日志CSV")
    args = parser.parse_args()

    if args.check:
        # 检查模式
        print(f"检查目录: {args.input_dir}")
        results, contradictions = check_classification(args.input_dir)
        print(f"\n总计: {len(results)} 张图片")
        print(f"识别成功: {sum(1 for r in results if r['qr_raw'])} 张")
        print(f"矛盾: {len(contradictions)} 张")

        if args.log:
            args.log.parent.mkdir(parents=True, exist_ok=True)
            with open(args.log, 'w', encoding='utf-8') as f:
                json.dump({
                    'total': len(results),
                    'recognized': sum(1 for r in results if r['qr_raw']),
                    'contradictions': len(contradictions),
                    'details': results,
                    'contradiction_list': contradictions,
                }, f, ensure_ascii=False, indent=2)
            print(f"日志已保存: {args.log}")
    else:
        # 分类模式
        print(f"输入目录: {args.input_dir}")
        print(f"输出目录: {args.output}")
        if args.dry_run:
            print("[DRY RUN] 只打印分类结果\n")

        results = classify_images(args.input_dir, args.output, move=args.move, dry_run=args.dry_run)

        if args.log:
            args.log.parent.mkdir(parents=True, exist_ok=True)
            with open(args.log, 'w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['file', 'version', 'channel', 'index', 'bits', 'qr_data', 'dest', 'error'])
                writer.writeheader()
                writer.writerows(results)
            print(f"日志已保存: {args.log}")


if __name__ == "__main__":
    main()
