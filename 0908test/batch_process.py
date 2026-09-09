"""
批量处理：分类 + 矫正 + 提取比特序列

步骤：
1. 检测二维码，按版本/通道分类到子目录
2. 批量矫正（full=1920x1080, part=768x768）
3. 提取比特序列到JSON
"""

import cv2
import json
import shutil
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import time


MODEL_PATHS = [
    r"C:\Users\24976\Desktop\code\QrMark_Loc\Qrlocted\20250729_best.pt",
    r"C:\Users\24976\Desktop\code\inrsteg-final_v1\Qrlocted\20250729_best.pt",
    "Qrlocted/20250729_best.pt",
    "../Qrlocted/20250729_best.pt",
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
    detector = cv2.QRCodeDetector()
    h, w = img.shape[:2]
    regions = [(0, 0, w, h), (0, 0, w // 2, h // 2), (0, 0, w // 3, h // 3)]

    for x1, y1, x2, y2 in regions:
        crop = img[y1:y2, x1:x2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        for processed in [
            crop, gray,
            cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1],
            cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)[1],
            cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
            cv2.GaussianBlur(gray, (5, 5), 0),
        ]:
            d, _, _ = detector.detectAndDecode(processed)
            if d and '|' in d:
                parts = d.split('|')
                if len(parts) == 4:
                    return {'version': parts[0], 'channel': parts[1],
                            'index': parts[2], 'bits': parts[3]}
    return None


def classify_images(input_dir, output_base):
    """分类图片到 version/channel 目录"""
    input_dir = Path(input_dir)
    output_base = Path(output_base)

    image_files = sorted([
        f for f in input_dir.iterdir()
        if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp')
    ])

    print(f"  分类 {len(image_files)} 张图片...")
    classified = {}
    failed = []

    for f in image_files:
        img = read_image(f)
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


def detect_corners_batch(model, images, conf=0.3, imgsz=2048):
    results = model(images, verbose=False, conf=conf, imgsz=imgsz)
    all_markers = []
    for res in results:
        markers = []
        if res.boxes.xywh is not None:
            for anchor, conf_val in zip(res.boxes.xywh, res.boxes.conf):
                cx, cy, w, h = anchor
                markers.append({"center": (float(cx), float(cy)),
                                "size": (float(w), float(h)),
                                "conf": float(conf_val)})
        all_markers.append(markers)
    return all_markers


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


def rectify_directory(input_dir, output_dir, model, out_w, out_h, batch_size=4):
    """矫正一个目录的所有图片"""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted([
        f for f in input_dir.iterdir()
        if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp')
    ])

    if not image_files:
        return 0, 0

    ok, fail = 0, 0
    for i in range(0, len(image_files), batch_size):
        batch = image_files[i:i+batch_size]
        images = []
        valid = []
        for f in batch:
            img = read_image(f)
            if img is not None:
                images.append(img)
                valid.append(f)

        if not images:
            continue

        all_markers = detect_corners_batch(model, images)

        for img, markers, f in zip(images, all_markers, valid):
            corners = select_corners(markers)
            if corners:
                rectified = rectify_image(img, corners, out_w, out_h)
                cv2.imwrite(str(output_dir / f.name), rectified)
                ok += 1
            else:
                fail += 1

    return ok, fail


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="输入目录 (包含pz, pzvx等)")
    parser.add_argument("--output", required=True, help="输出目录")
    parser.add_argument("--mode", choices=["full", "part"], required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    input_root = Path(args.input)
    output_root = Path(args.output)
    mode = args.mode
    out_w, out_h = (1920, 1080) if mode == "full" else (768, 768)

    model_path = find_model()
    print(f"加载模型: {model_path}")
    model = YOLO(model_path)

    all_bits = {}

    for sub_dir in sorted(input_root.iterdir()):
        if not sub_dir.is_dir():
            continue

        print(f"\n{'='*60}")
        print(f"处理: {sub_dir.name} ({mode}模式, 输出{out_w}x{out_h})")
        print(f"{'='*60}")

        # 第1步：分类
        print("\n[1/2] 分类图片...")
        classified, failed = classify_images(sub_dir, output_root / sub_dir.name)

        # 第2步：矫正
        print("\n[2/2] 矫正图片...")
        for key, items in sorted(classified.items()):
            v, ch = key.split('/')
            src_dir = output_root / sub_dir.name / v / ch
            dst_dir = output_root / f"{sub_dir.name}_rectified" / v / ch

            ok, fail = rectify_directory(src_dir, dst_dir, model, out_w, out_h, args.batch_size)
            print(f"  {key}: {ok} 成功, {fail} 失败")

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
    print(f"\n比特序列已保存: {bits_file}")


if __name__ == "__main__":
    main()
