#!/usr/bin/env python3
"""
全屏播放水印图，用于手机拍照采集。

用法:
    python display_for_capture.py --checklist real_data/day1_checklist.txt --wm_dir real_data/watermarked/v17_cb
    python display_for_capture.py --checklist real_data/day1_checklist.txt --wm_dir real_data/watermarked/v17_cb --start 500

操作:
    空格/右箭头  下一张
    左箭头       上一张
    Q/ESC        退出
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


LOG_FILE = Path("real_data/capture_log.txt")


def put_text_outline(img, text, pos, font_scale, color, thickness=2):
    """绘制带黑色描边的白色文字，便于在各种背景上阅读。"""
    x, y = pos
    # 黑色描边
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    # 白色文字
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, color, thickness, cv2.LINE_AA)


def make_display_image(image, info_text, screen_w, screen_h):
    """将图片缩放并居中放到黑色背景上，左上角显示信息文字。"""
    h, w = image.shape[:2]

    # 计算缩放比例（保持宽高比，填满屏幕）
    scale = max(screen_w / w, screen_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 居中裁剪
    start_y = (new_h - screen_h) // 2
    start_x = (new_w - screen_w) // 2
    cropped = resized[start_y:start_y + screen_h, start_x:start_x + screen_w]

    # 如果不是全屏（理论上不会发生），用黑色背景
    if cropped.shape[0] != screen_h or cropped.shape[1] != screen_w:
        canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
        y_off = (screen_h - cropped.shape[0]) // 2
        x_off = (screen_w - cropped.shape[1]) // 2
        canvas[y_off:y_off + cropped.shape[0], x_off:x_off + cropped.shape[1]] = cropped
        cropped = canvas

    # 左上角信息
    y_offset = 40
    for line in info_text:
        put_text_outline(cropped, line, (20, y_offset), 0.8, (255, 255, 255), 2)
        y_offset += 35

    return cropped


def main():
    parser = argparse.ArgumentParser(description="全屏播放水印图用于拍照采集")
    parser.add_argument("--checklist", required=True, help="拍照清单文件")
    parser.add_argument("--wm_dir", required=True, help="水印图根目录")
    parser.add_argument("--start", type=int, default=0, help="从第几张开始")
    parser.add_argument("--log", default=str(LOG_FILE), help="采集日志路径")
    args = parser.parse_args()

    wm_dir = Path(args.wm_dir)
    checklist_path = Path(args.checklist)

    # 读取清单
    with open(checklist_path) as f:
        items = [line.strip() for line in f if line.strip()]

    total = len(items)
    print(f"清单: {checklist_path} ({total} 张)")
    print(f"水印目录: {wm_dir}")
    print(f"从第 {args.start} 张开始")
    print("\n操作: 空格/右箭头=下一张, 左箭头=上一张, Q/ESC=退出\n")

    # 创建全屏窗口
    window_name = "Capture"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # 获取屏幕尺寸
    # 先显示一张获取尺寸
    first_img_path = wm_dir / items[args.start]
    first_img = cv2.imread(str(first_img_path))
    if first_img is None:
        print(f"❌ 无法读取: {first_img_path}")
        sys.exit(1)

    screen_h, screen_w = 1080, 1920  # 默认值，显示后会自适应

    current = args.start
    log_entries = []

    while 0 <= current < total:
        item = items[current]
        img_path = wm_dir / item

        image = cv2.imread(str(img_path))
        if image is None:
            print(f"⚠️ 无法读取: {img_path}, 跳过")
            current += 1
            continue

        # 解析信息
        parts = item.split("/")
        bits_id = parts[0] if len(parts) > 0 else "?"
        carrier = parts[1].replace(".png", "") if len(parts) > 1 else "?"

        info_text = [
            f"[{current + 1}/{total}]  {bits_id}  {carrier}",
            f"SPACE=next  LEFT=prev  Q=quit",
        ]

        display = make_display_image(image, info_text, screen_w, screen_h)
        cv2.imshow(window_name, display)

        # 记录日志
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entries.append(f"{current + 1} | {item} | {timestamp}")

        # 等待按键
        while True:
            key = cv2.waitKey(0) & 0xFFFF

            # 空格或右箭头 → 下一张
            if key in (ord(" "), 0xFF53, 83):  # space, right, right(win)
                current += 1
                break
            # 左箭头 → 上一张
            elif key in (0xFF51, 81):  # left, left(win)
                current = max(0, current - 1)
                break
            # Q 或 ESC → 退出
            elif key in (ord("q"), ord("Q"), 27):
                # 保存日志
                log_path = Path(args.log)
                log_path.parent.mkdir(parents=True, exist_ok=True)
                with open(log_path, "a") as f:
                    for entry in log_entries:
                        f.write(entry + "\n")
                print(f"\n已退出，记录 {len(log_entries)} 条日志到 {log_path}")
                print(f"下次从 --start {current} 继续")
                cv2.destroyAllWindows()
                return

    # 播放完毕
    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        for entry in log_entries:
            f.write(entry + "\n")
    print(f"\n✅ 全部播放完毕，记录 {len(log_entries)} 条日志到 {log_path}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
