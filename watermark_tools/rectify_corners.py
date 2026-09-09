#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
四角回字形透视矫正

用YOLO检测所有回字形标记，选取最靠近图片四角的4个做透视变换。
自动排除信息二维码中的回字形（它不在角落）。

用法:
    python rectify_corners.py --image photo.jpg
    python rectify_corners.py --image photo.jpg --output rectified.jpg --debug
    python rectify_corners.py --image photo.jpg --screen-w 1920 --screen-h 1080
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


# 默认模型路径（多个备选位置）
DEFAULT_MODEL_PATHS = [
    "Qrlocted/20250729_best.pt",
    "../Qrlocted/20250729_best.pt",
    r"C:\Users\24976\Desktop\code\QrMark_Loc\Qrlocted\20250729_best.pt",
    r"C:\Users\24976\Desktop\code\inrsteg-final_v1\Qrlocted\20250729_best.pt",
]


def find_model(model_path=None):
    """查找YOLO模型文件"""
    if model_path:
        p = Path(model_path)
        if p.exists():
            return p
        raise FileNotFoundError(f"模型不存在: {model_path}")

    for p in DEFAULT_MODEL_PATHS:
        if Path(p).exists():
            return Path(p)

    raise FileNotFoundError(
        "找不到模型文件，请用 --model 指定路径\n"
        "备选位置: " + ", ".join(DEFAULT_MODEL_PATHS)
    )


def detect_all_markers(model, image, conf=0.3, imgsz=2048):
    """检测图片中所有回字形标记，返回中心点坐标列表"""
    results = model(image, verbose=False, conf=conf, imgsz=imgsz)
    markers = []
    for res in results:
        if res.boxes.xywh is None:
            continue
        for anchor, conf_val in zip(res.boxes.xywh, res.boxes.conf):
            cx, cy, w, h = anchor
            markers.append({
                "center": (float(cx), float(cy)),
                "size": (float(w), float(h)),
                "conf": float(conf_val),
            })
    return markers


def filter_by_size(markers, tolerance=0.3):
    """
    按大小筛选：找出大小最相似的一组标记（至少4个）。

    策略：对每个标记，计算有多少其他标记和它大小接近（容差内）。
    选择"邻居最多"的那个标记作为代表，然后收集它所有大小相近的邻居。

    这样可以自动识别角标（通常4个大小相近）并排除二维码中的小回字形。
    """
    if len(markers) <= 4:
        return markers

    # 计算每个标记的平均尺寸
    avg_sizes = [(m["size"][0] + m["size"][1]) / 2 for m in markers]

    # 对每个标记，找大小相近的邻居数量
    best_count = 0
    best_idx = 0
    for i, si in enumerate(avg_sizes):
        count = 0
        for j, sj in enumerate(avg_sizes):
            if abs(si - sj) / si <= tolerance:
                count += 1
        if count > best_count:
            best_count = count
            best_idx = i

    # 收集与最佳代表大小相近的标记
    ref_size = avg_sizes[best_idx]
    filtered = []
    for m, avg_size in zip(markers, avg_sizes):
        if abs(avg_size - ref_size) / ref_size <= tolerance:
            filtered.append(m)

    # 如果筛选后不足4个，返回全部
    if len(filtered) < 4:
        return markers

    return filtered


def select_corner_markers(markers, image_w, image_h, size_tolerance=0.3):
    """
    选取4个大小相似的标记作为角标，确保形成合理的矩形。

    策略：
    1. 按大小聚类，找出最相似的一组标记
    2. 从所有4点组合中，选最接近矩形的一组
    3. 用凸包+面积最大化来选取最优4点
    """
    from itertools import combinations

    if len(markers) < 4:
        raise RuntimeError(f"检测到 {len(markers)} 个标记，不足4个，无法矫正")

    # 按大小筛选
    filtered = filter_by_size(markers, tolerance=size_tolerance)
    if len(filtered) < 4:
        raise RuntimeError(f"大小筛选后只剩 {len(filtered)} 个标记，不足4个")

    if len(filtered) == 4:
        # 刚好4个，直接用
        points = np.array([m["center"] for m in filtered], dtype="float32")
        return _order_rect(points, filtered)

    # 多于4个，选最接近矩形的组合
    best_score = -1
    best_combo = None

    for combo in combinations(range(len(filtered)), 4):
        pts = np.array([filtered[i]["center"] for i in combo], dtype="float32")
        score = _rect_score(pts)
        if score > best_score:
            best_score = score
            best_combo = combo

    selected = [filtered[i] for i in best_combo]
    points = np.array([m["center"] for m in selected], dtype="float32")
    return _order_rect(points, selected)


def _rect_score(pts):
    """评估4个点接近矩形的程度（0~1，越大越接近矩形）"""
    # 计算凸包面积与外接矩形面积的比值
    hull = cv2.convexHull(pts.reshape(-1, 1, 2).astype(np.float32))
    hull_area = cv2.contourArea(hull)

    # 外接矩形
    rect = cv2.minAreaRect(hull)
    rect_area = rect[1][0] * rect[1][1]

    if rect_area < 1:
        return 0

    # 面积比（接近1说明点在矩形边上）
    area_ratio = hull_area / rect_area

    # 宽高比（接近1说明接近正方形，可选）
    w, h = rect[1]
    aspect = min(w, h) / max(w, h) if max(w, h) > 0 else 0

    # 综合得分
    return area_ratio * 0.7 + aspect * 0.3


def _order_rect(points, markers):
    """将4个点按左上、右上、右下、左下排序"""
    # 用minAreaRect获取正确的角点顺序
    rect = cv2.minAreaRect(points.reshape(-1, 1, 2).astype(np.float32))
    box = cv2.boxPoints(rect)  # 返回4个角点

    # box的顺序是：左下、右下、右上、左上（按OpenCV惯例）
    # 我们需要：左上、右上、右下、左下
    s = box.sum(axis=1)
    d = np.diff(box, axis=1).flatten()

    ordered = np.zeros((4, 2), dtype="float32")
    ordered[0] = box[np.argmin(s)]   # 左上
    ordered[2] = box[np.argmax(s)]   # 右下
    ordered[1] = box[np.argmin(d)]   # 右上
    ordered[3] = box[np.argmax(d)]   # 左下

    # 将排序后的点映射回原始markers
    result = []
    for target in ordered:
        # 找最近的原始点
        dists = [np.linalg.norm(np.array(m["center"]) - target) for m in markers]
        best_idx = int(np.argmin(dists))
        result.append(markers[best_idx])

    return result


def order_points(pts):
    """按 左上、右上、右下、左下 顺序排列四个点"""
    pts = np.array(pts, dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    ordered = np.zeros((4, 2), dtype="float32")
    ordered[0] = pts[np.argmin(s)]   # 左上
    ordered[2] = pts[np.argmax(s)]   # 右下
    ordered[1] = pts[np.argmin(diff)]  # 右上
    ordered[3] = pts[np.argmax(diff)]  # 左下
    return ordered


def rectify_by_corners(image, corners, output_w, output_h):
    """
    根据4个角点做透视变换。

    Args:
        image: 输入图片
        corners: 4个角点坐标 [(x,y), ...]，顺序：左上、右上、左下、右下
        output_w: 输出宽度
        output_h: 输出高度

    Returns:
        矫正后的图片
    """
    ordered = order_points(corners)

    dst = np.array([
        [0, 0],
        [output_w - 1, 0],
        [output_w - 1, output_h - 1],
        [0, output_h - 1],
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(ordered, dst)
    warped = cv2.warpPerspective(image, M, (output_w, output_h))
    return warped


def draw_debug(image, markers, selected, output_path=None):
    """画调试图：所有检测点（绿）、选中的四角点（红）"""
    debug = image.copy()

    # 画所有检测点（绿色）
    for m in markers:
        cx, cy = int(m["center"][0]), int(m["center"][1])
        cv2.circle(debug, (cx, cy), 8, (0, 255, 0), 2)
        cv2.putText(debug, f'{m["conf"]:.2f}', (cx + 10, cy - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # 画选中的四角点（红色，大）
    labels = ["TL", "TR", "BL", "BR"]
    for i, m in enumerate(selected):
        cx, cy = int(m["center"][0]), int(m["center"][1])
        cv2.circle(debug, (cx, cy), 15, (0, 0, 255), 3)
        w, h = m["size"]
        cv2.putText(debug, f'{labels[i]} {w:.0f}x{h:.0f}', (cx - 10, cy - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # 画四边形连线
    pts = np.array([list(m["center"]) for m in selected], dtype=np.int32)
    cv2.polylines(debug, [pts.reshape(-1, 1, 2)], True, (0, 255, 255), 2)

    if output_path:
        cv2.imwrite(str(output_path), debug)
        print(f"调试图已保存: {output_path}")

    return debug


def crop_center_region(image, region_size):
    """裁剪图片中心的 region_size x region_size 区域"""
    h, w = image.shape[:2]
    cx, cy = w // 2, h // 2
    half = region_size // 2
    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = min(w, x1 + region_size)
    y2 = min(h, y1 + region_size)
    return image[y1:y2, x1:x2], (x1, y1)


def main():
    parser = argparse.ArgumentParser(description="四角回字形透视矫正")
    parser.add_argument("--image", required=True, help="输入图片路径")
    parser.add_argument("--output", default=None, help="输出矫正图片路径（默认: 输入文件名_rectified.jpg）")
    parser.add_argument("--model", default=None, help="YOLO模型路径")
    parser.add_argument("--conf", type=float, default=0.3, help="检测置信度阈值")
    parser.add_argument("--imgsz", type=int, default=2048, help="推理图片尺寸")
    parser.add_argument("--screen-w", type=int, default=1920, help="矫正后输出宽度（全屏模式）")
    parser.add_argument("--screen-h", type=int, default=1080, help="矫正后输出高度（全屏模式）")
    parser.add_argument("--local", action="store_true", help="局部模式：只输出768x768区域")
    parser.add_argument("--region_size", type=int, default=768, help="局部模式区域大小")
    parser.add_argument("--debug", action="store_true", help="保存调试图（标记检测结果）")
    parser.add_argument("--size-tolerance", type=float, default=0.3, help="大小筛选容差，默认0.3")
    args = parser.parse_args()

    # 加载模型
    model_path = find_model(args.model)
    print(f"加载模型: {model_path}")
    model = YOLO(str(model_path))

    # 读取图片
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"错误: 图片不存在 {image_path}")
        sys.exit(1)

    image = cv2.imread(str(image_path))
    if image is None:
        print(f"错误: 无法读取图片 {image_path}")
        sys.exit(1)

    h, w = image.shape[:2]
    print(f"图片尺寸: {w}x{h}")

    # 检测所有标记
    markers = detect_all_markers(model, image, conf=args.conf, imgsz=args.imgsz)
    print(f"检测到 {len(markers)} 个回字形标记:")
    for i, m in enumerate(markers):
        cx, cy = m["center"]
        w, h = m["size"]
        print(f"  [{i}] ({cx:.0f}, {cy:.0f}) size={w:.0f}x{h:.0f} conf={m['conf']:.3f}")

    if len(markers) < 4:
        print(f"错误: 标记不足4个，无法矫正")
        sys.exit(1)

    # 选四角标记
    selected = select_corner_markers(markers, w, h, size_tolerance=args.size_tolerance)
    print("选中的四角标记:")
    labels = ["左上", "右上", "左下", "右下"]
    for i, m in enumerate(selected):
        cx, cy = m["center"]
        print(f"  {labels[i]}: ({cx:.0f}, {cy:.0f}) conf={m['conf']:.3f}")

    # 透视矫正
    if args.local:
        # 局部模式：先矫正全屏，再裁剪中心区域
        output_w = args.screen_w
        output_h = args.screen_h
        corners = [m["center"] for m in selected]
        rectified = rectify_by_corners(image, corners, output_w, output_h)

        # 裁剪中心区域
        region_size = args.region_size
        cropped, offset = crop_center_region(rectified, region_size)
        print(f"局部模式: 裁剪中心 {region_size}x{region_size} 区域")
        final_output = cropped
    else:
        # 全屏模式
        output_w = args.screen_w
        output_h = args.screen_h
        corners = [m["center"] for m in selected]
        rectified = rectify_by_corners(image, corners, output_w, output_h)
        final_output = rectified

    # 保存结果
    if args.output:
        output_path = Path(args.output)
    else:
        suffix = "_local" if args.local else "_rectified"
        output_path = image_path.with_name(image_path.stem + suffix + ".jpg")

    cv2.imwrite(str(output_path), final_output)
    print(f"矫正完成: {output_path} ({final_output.shape[1]}x{final_output.shape[0]})")

    # 调试图
    if args.debug:
        debug_img = draw_debug(image, markers, selected)
        # 局部模式：在调试图上画出裁剪区域
        if args.local:
            h, w = debug_img.shape[:2]
            cx, cy = w // 2, h // 2
            half = args.region_size // 2
            x1, y1 = cx - half, cy - half
            x2, y2 = x1 + args.region_size, y1 + args.region_size
            cv2.rectangle(debug_img, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.putText(debug_img, f"crop {args.region_size}x{args.region_size}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        debug_path = image_path.with_name(image_path.stem + "_debug.jpg")
        cv2.imwrite(str(debug_path), debug_img)
        print(f"调试图已保存: {debug_path}")


if __name__ == "__main__":
    main()
