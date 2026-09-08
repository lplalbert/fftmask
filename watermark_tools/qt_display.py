"""Qt透明窗口叠加水印（屏幕水印）。

基于test6.py的实现方式：
- 灰色画布(128) + setWindowOpacity(alpha) 控制强度
- OS GPU硬件加速合成，无延迟

支持两种模式：
1. 固定模板模式：一直显示同一个水印模板
2. 播放列表模式（--playlist）：按播放列表切换水印模板
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel

import ctypes


def load_template(version, channel, index, template_dir):
    """加载BGR域平铺模板（灰度画布嵌入）"""
    manifest_path = template_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"错误: 找不到 {manifest_path}")
        sys.exit(1)
    with open(manifest_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    version_records = [r for r in records if r["version"] == version]
    if index >= len(version_records):
        print(f"错误: {version} 没有 index={index} 的模板")
        sys.exit(1)

    rec = version_records[index]
    bgr_key = f"bgr_{channel}_filename"
    filename = rec[bgr_key]
    path = template_dir / version / filename
    print(f"加载模板: {path.name}")
    img = cv2.imread(str(path))
    if img is None:
        print(f"错误: 无法读取 {path}")
        sys.exit(1)
    return img


def load_all_templates(version, channel, template_dir, num_templates=10):
    """加载所有模板，返回 dict: index -> BGR图片"""
    manifest_path = template_dir / "manifest.json"
    if not manifest_path.exists():
        return {}
    with open(manifest_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    version_records = [r for r in records if r["version"] == version]

    templates = {}
    for i, rec in enumerate(version_records[:num_templates]):
        bgr_key = f"bgr_{channel}_filename"
        filename = rec.get(bgr_key)
        if not filename:
            continue
        path = template_dir / version / filename
        img = cv2.imread(str(path))
        if img is not None:
            templates[i] = img

    return templates


class TransparentWindow(QMainWindow):

    def __init__(self, templates, alpha=0.03, interval=5000, template_indices=None):
        """
        Args:
            templates: dict: index -> BGR图片（已嵌入灰度画布）
            alpha: 窗口透明度
            interval: 模板切换间隔ms（播放列表模式用）
            template_indices: list[int] or None，播放列表对应的模板索引序列
        """
        super().__init__()
        self.templates = templates
        self.alpha = alpha
        self.template_indices = template_indices
        self.playlist_index = 0

        # 取第一个模板设置窗口
        first_key = list(templates.keys())[0]
        self.current_key = first_key

        # 设置窗口
        self.setWindowFlags(
            Qt.WindowStaysOnTopHint |
            Qt.FramelessWindowHint |
            Qt.MaximizeUsingFullscreenGeometryHint
        )
        self.setAttribute(Qt.WA_TranslucentBackground, False)
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)
        self.setWindowOpacity(alpha)

        screen = QApplication.primaryScreen().geometry()
        self.screen_w = screen.width()
        self.screen_h = screen.height()
        self.setGeometry(0, 0, self.screen_w, self.screen_h)

        self.label = QLabel(self)
        self.label.setGeometry(0, 0, self.screen_w, self.screen_h)

        self.show_template(first_key)

        # 播放列表模式：定时切换
        if template_indices and len(template_indices) > 1:
            self.timer = QTimer(self)
            self.timer.timeout.connect(self.next_template)
            self.timer.start(interval)

        # 延迟设置点击穿透（等窗口完全初始化）
        QTimer.singleShot(500, self.set_click_through)

        # 定时置顶（防止被其他窗口遮挡）
        self._top_timer = QTimer(self)
        self._top_timer.timeout.connect(self.raise_)
        self._top_timer.start(2000)

        mode = "播放列表" if template_indices else "固定"
        print(f"[QtOverlay] alpha={alpha}, 模式={mode}, 模板数={len(templates)}")
        print(f"[QtOverlay] ESC退出")

    def show_template(self, key):
        """显示指定模板"""
        img = self.templates[key]
        h, w = img.shape[:2]
        self._rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 保持引用
        qimg = QImage(self._rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        self.label.setPixmap(QPixmap.fromImage(qimg))
        self.current_key = key

    def next_template(self):
        """切换到下一个模板（播放列表模式）"""
        if not self.template_indices:
            return
        self.playlist_index = (self.playlist_index + 1) % len(self.template_indices)
        idx = self.template_indices[self.playlist_index]
        if idx in self.templates:
            self.show_template(idx)
            print(f"[QtOverlay] 切换到模板 {idx} ({self.playlist_index + 1}/{len(self.template_indices)})")

    def set_click_through(self):
        hwnd = int(self.winId())
        ex_style = ctypes.windll.user32.GetWindowLongW(hwnd, -20)
        ctypes.windll.user32.SetWindowLongW(
            hwnd, -20,
            ex_style | 0x80000 | 0x20
        )

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()


def main():
    parser = argparse.ArgumentParser(description="Qt透明窗口叠加水印")
    parser.add_argument("--version", choices=["v17", "v18"], default="v17")
    parser.add_argument("--channel", choices=["cb", "b"], default=None, help="通道（默认: v17=cb, v18=b）")
    parser.add_argument("--alpha", type=float, default=0.03, help="水印透明度")
    parser.add_argument("--interval", type=int, default=5000, help="模板切换间隔ms")
    parser.add_argument("--index", type=int, default=0, help="固定模式: 模板索引")
    parser.add_argument("--template_dir", type=Path, help="模板目录")
    parser.add_argument("--playlist", type=Path, help="播放列表JSON（run_both.py生成）")
    args = parser.parse_args()

    template_dir = args.template_dir or Path(__file__).parent / "generated_templates"
    if not template_dir.is_absolute():
        template_dir = Path(__file__).parent / template_dir

    channel = args.channel or ("cb" if args.version == "v17" else "b")

    # 播放列表模式
    template_indices = None
    if args.playlist:
        with open(args.playlist, "r", encoding="utf-8") as f:
            playlist = json.load(f)
        template_indices = playlist["template_indices"]
        used_indices = set(template_indices)
        all_templates = load_all_templates(args.version, channel, template_dir, num_templates=10)
        templates = {i: all_templates[i] for i in used_indices if i in all_templates}
        print(f"[QtOverlay] 播放列表模式: {len(templates)} 个模板")
    else:
        # 固定模式
        img = load_template(args.version, channel, args.index, template_dir)
        templates = {args.index: img}

    app = QApplication(sys.argv)
    window = TransparentWindow(templates, args.alpha, args.interval, template_indices)
    window.show()
    window.raise_()
    window.activateWindow()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
