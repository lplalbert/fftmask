"""
批量处理：分类 + 矫正 + 提取比特序列（GPU优化版）

步骤：
1. 检测二维码，按版本/通道分类到子目录
2. 批量矫正（full=1920x1080, part=768x768）
3. 提取比特序列到JSON

GPU优化：
- YOLO推理使用大batch_size，充分利用GPU
- 图片预加载和YOLO推理流水线化
- 矫正阶段也用YOLO批量推理
"""

import cv2
import json
import shutil
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import time
import torch
from concurrent.futures import ThreadPoolExecutor
import argparse


MODEL_PATHS = [
    "Qrlocted/20250729_best.pt",
    "../Qrlocted/20250729_best.pt",
    "/data/lpl/fftmask/Qrlocted/20250729_best.pt",
]


def find_model():
    for p in MODEL_PATHS:
        if Path(p).exists():
            return str(p)
    raise FileNotFoundError("找不到YOLO模型")


def read_image(path):
    with open(str(path), 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def decode_qr(img):
    """检测二维码，优先使用缩放后的图片加速"""
    detector = cv2.QRCodeDetector()
    h, w = img.shape[:2]

    # 先缩放到合理大小加速检测
    scale = min(1.0, 1024 / max(h, w))
    if scale < 1.0:
        small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        small = img

    regions = [
        (0, 0, small.shape[1], small.shape[0]),  # 全图
        (0, 0, small.shape[1] // 2, small.shape[0] // 2),  # 左上1/4
    ]

    for x1, y1, x2, y2 in regions:
        crop = small[y1:y2, x1:x2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        for processed in [
            gray,
            cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1],
            cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        ]:
            d, _, _ = detector.detectAndDecode(processed)
            if d and '|' in d:
                parts = d.split('|')
                if len(parts) == 4:
                    return {'version': parts[0], 'channel': parts[1],
                            'index': parts[2], 'bits': parts[3]}
    return None


def load_images_parallel(image_files, max_workers=8):
    """并行加载图片"""
    images = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(read_image, f): f for f in image_files}
        for future in futures:
            f = futures[future]
            try:
                img = future.result()
                if img is not None:
                    images[f.name] = img
            except Exception as e:
                print(f"  加载失败 {f.name}: {e}")
    return images


def classify_images(input_dir, output_base, batch_size=32):
    """分类图片到 version/channel 目录（批量QR检测，递归查找图片）"""
    input_dir = Path(input_dir)
    output_base = Path(output_base)

    # 递归查找所有图片
    image_files = sorted([
        f for f in input_dir.rglob('*')
        if f.is_file() and f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp')
    ])

    print(f"  分类 {len(image_files)} 张图片...")

    # 并行加载图片
    t0 = time.time()
    images = load_images_parallel(image_files)
    print(f"  图片加载: {time.time()-t0:.1f}s, 成功 {len(images)} 张")

    # 批量QR检测
    classified = {}
    failed = []
    detector = cv2.QRCodeDetector()

    t0 = time.time()
    for f in image_files:
        img = images.get(f.name)
        if img is None:
            failed.append(f.name)
            continue

        qr = decode_qr(img)
        if qr:
            v, ch = qr['version'], qr['channel']
            target_dir = output_base / v / ch
            target_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(f), str(target_dir / f.name))

            key = f"{v}/{ch}"
            if key not in classified:
                classified[key] = []
            classified[key].append({'file': f.name, 'qr': qr})
        else:
            failed.append(f.name)

    print(f"  QR检测: {time.time()-t0:.1f}s")

    # 保存未识别的
    if failed:
        unrec_dir = output_base / "_unrecognized"
        unrec_dir.mkdir(exist_ok=True)
        for name in failed:
            src = input_dir / name
            if src.exists():
                shutil.copy2(str(src), str(unrec_dir / name))

    print(f"  分类完成: {sum(len(v) for v in classified.values())} 成功, {len(failed)} 失败")
    for key, items in sorted(classified.items()):
        print(f"    {key}: {len(items)} 张")

    return classified, failed


def filter_by_size(markers, tolerance=0.3):
    if len(markers) <= 4:
        return markers
    avg_sizes = [(m["size"][0] + m["size"][1]) / 2 for m in markers]
    best_count, best_idx = 0, 0
    for i, si in enumerate(avg_sizes):
        count = sum(1 for sj in avg_sizes if abs(si - sj) / si <= tolerance)
        if count > best_count:
            best_count, best_idx = count, i
    ref = avg_sizes[best_idx]
    filtered = [m for m, s in zip(markers, avg_sizes) if abs(s - ref) / ref <= tolerance]
    return filtered if len(filtered) >= 4 else markers


def select_corners(markers):
    filtered = filter_by_size(markers)
    if len(filtered) < 4:
        return None
    sorted_by_y = sorted(filtered, key=lambda m: m["center"][1])
    top = sorted(sorted_by_y[:2], key=lambda m: m["center"][0])
    bot = sorted(sorted_by_y[2:], key=lambda m: m["center"][0])
    return [top[0], top[1], bot[0], bot[1]]


def order_points(pts):
    pts = np.array(pts, dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    ordered = np.zeros((4, 2), dtype="float32")
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]
    ordered[1] = pts[np.argmin(diff)]
    ordered[3] = pts[np.argmax(diff)]
    return ordered


def rectify_image(image, corners, out_w, out_h):
    ordered = order_points([m["center"] for m in corners])
    dst = np.array([[0, 0], [out_w-1, 0], [out_w-1, out_h-1], [0, out_h-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(ordered, dst)
    return cv2.warpPerspective(image, M, (out_w, out_h))


def rectify_directory(input_dir, output_dir, model, out_w, out_h, batch_size=16, imgsz=2048):
    """矫正一个目录的所有图片（GPU批量推理）"""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 递归查找所有图片
    image_files = sorted([
        f for f in input_dir.rglob('*')
        if f.is_file() and f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp')
    ])

    if not image_files:
        return 0, 0

    print(f"    找到 {len(image_files)} 张图片")

    # 并行加载所有图片
    t0 = time.time()
    all_images = load_images_parallel(image_files, max_workers=8)
    print(f"    图片加载: {time.time()-t0:.1f}s ({len(all_images)} 张)")

    ok, fail = 0, 0

    # 批量YOLO推理（大batch充分利用GPU）
    t_yolo = 0
    t_rectify = 0

    for i in range(0, len(image_files), batch_size):
        batch_files = image_files[i:i+batch_size]
        batch_images = []
        batch_valid = []

        for f in batch_files:
            img = all_images.get(f.name)
            if img is not None:
                batch_images.append(img)
                batch_valid.append(f)

        if not batch_images:
            continue

        # YOLO批量推理（GPU）
        t1 = time.time()
        results = model(batch_images, verbose=False, conf=0.3, imgsz=imgsz)
        t_yolo += time.time() - t1

        # 矫正（CPU）
        t1 = time.time()
        for img, res, f in zip(batch_images, results, batch_valid):
            markers = []
            if res.boxes.xywh is not None:
                for anchor, conf_val in zip(res.boxes.xywh, res.boxes.conf):
                    cx, cy, w, h = anchor
                    markers.append({"center": (float(cx), float(cy)),
                                    "size": (float(w), float(h)),
                                    "conf": float(conf_val)})

            corners = select_corners(markers)
            if corners:
                rectified = rectify_image(img, corners, out_w, out_h)
                cv2.imwrite(str(output_dir / f.name), rectified)
                ok += 1
            else:
                fail += 1
        t_rectify += time.time() - t1

    print(f"    YOLO推理: {t_yolo:.1f}s, 矫正写入: {t_rectify:.1f}s")
    return ok, fail


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="输入目录 (包含pz, pzvx等)")
    parser.add_argument("--output", required=True, help="输出目录")
    parser.add_argument("--mode", choices=["full", "part"], required=True)
    parser.add_argument("--batch-size", type=int, default=16, help="YOLO批量推理大小")
    parser.add_argument("--imgsz", type=int, default=2048, help="YOLO输入尺寸")
    parser.add_argument("--workers", type=int, default=8, help="图片加载线程数")
    args = parser.parse_args()

    input_root = Path(args.input)
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    mode = args.mode
    out_w, out_h = (1920, 1080) if mode == "full" else (768, 768)

    # 检查GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"设备: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    model_path = find_model()
    print(f"加载模型: {model_path}")
    model = YOLO(model_path)

    # 预热GPU
    if device == 'cuda':
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        model(dummy, verbose=False, imgsz=640)
        print("GPU预热完成")

    all_bits = {}
    total_ok, total_fail = 0, 0
    t_start = time.time()

    # 处理每个子目录
    sub_dirs = sorted([d for d in input_root.iterdir() if d.is_dir()])
    print(f"\n找到 {len(sub_dirs)} 个子目录: {[d.name for d in sub_dirs]}")

    for sub_dir in sub_dirs:
        print(f"\n{'='*60}")
        print(f"处理: {sub_dir.name} ({mode}模式, 输出{out_w}x{out_h})")
        print(f"{'='*60}")

        # 第1步：分类
        print("\n[1/2] 分类图片...")
        classified, failed = classify_images(sub_dir, output_root / sub_dir.name, args.batch_size)

        # 第2步：矫正（GPU批量推理）
        print("\n[2/2] 矫正图片...")
        for key, items in sorted(classified.items()):
            v, ch = key.split('/')
            src_dir = output_root / sub_dir.name / v / ch
            dst_dir = output_root / f"{sub_dir.name}_rectified" / v / ch

            ok, fail = rectify_directory(src_dir, dst_dir, model, out_w, out_h,
                                        args.batch_size, args.imgsz)
            print(f"  {key}: {ok} 成功, {fail} 失败")
            total_ok += ok
            total_fail += fail

            # 收集比特序列
            for item in items:
                qr = item['qr']
                vv, cch, idx = qr['version'], qr['channel'], qr['index']
                if vv not in all_bits:
                    all_bits[vv] = {}
                if cch not in all_bits[vv]:
                    all_bits[vv][cch] = {}
                all_bits[vv][cch][idx] = qr['bits']

    # 保存比特序列
    bits_file = output_root / "bit_sequences.json"
    with open(bits_file, 'w', encoding='utf-8') as f:
        json.dump(all_bits, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"处理完成!")
    print(f"  总耗时: {elapsed:.1f}s")
    print(f"  矫正成功: {total_ok}, 失败: {total_fail}")
    print(f"  比特序列: {bits_file}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
