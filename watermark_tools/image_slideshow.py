"""全屏图片播放脚本（含二维码 + 回字形角标）。

配合qt_display.py使用：先运行本脚本，再运行qt_display.py叠加水印层。

支持两种模式：
1. 固定模板模式：所有图片用同一个QR/水印
2. 播放列表模式（--playlist）：每张图片对应不同模板
"""
import argparse
import json
import sys
from pathlib import Path

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QPainter, QFont, QColor
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel


class SlideshowWindow(QMainWindow):

    def __init__(self, image_paths, qr_pixmaps=None, corner_pixmap=None,
                 show_corners=False, interval=3000, template_indices=None,
                 version="v17", channel="cb", local_mode=False, region_size=768):
        super().__init__()
        self.image_paths = image_paths
        self.qr_pixmaps = qr_pixmaps  # dict: index -> QPixmap
        self.corner_pixmap = corner_pixmap
        self.show_corners = show_corners
        self.template_indices = template_indices  # list[int] or None
        self.version = version
        self.channel = channel
        self.current_index = 0
        self.local_mode = local_mode
        self.region_size = region_size

        self.setWindowState(Qt.WindowFullScreen | Qt.WindowMaximized)
        self.setWindowFlags(Qt.FramelessWindowHint)

        screen = QApplication.primaryScreen().geometry()
        self.screen_w = screen.width()
        self.screen_h = screen.height()
        self.setGeometry(0, 0, self.screen_w, self.screen_h)

        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignCenter)
        self.show_image()

        if len(image_paths) > 1:
            self.timer = QTimer(self)
            self.timer.timeout.connect(self.next_image)
            self.timer.start(interval)

        print(f"[Slideshow] 图片={len(image_paths)}, 间隔={interval}ms")
        print(f"[Slideshow] 二维码={'播放列表模式' if template_indices else '固定模式'}, 四角标={'有' if show_corners else '无'}")
        print(f"[Slideshow] ESC退出, 左右箭头切换")

    def get_current_qr(self):
        """获取当前图片对应的QR码"""
        if self.template_indices is not None:
            idx = self.template_indices[self.current_index]
            return self.qr_pixmaps.get(idx) if self.qr_pixmaps else None
        # 固定模式：只有一个QR
        if self.qr_pixmaps:
            return list(self.qr_pixmaps.values())[0]
        return None

    def show_image(self):
        path = self.image_paths[self.current_index]
        pixmap = QPixmap(str(path))
        pixmap = pixmap.scaled(self.screen_w, self.screen_h,
                               Qt.KeepAspectRatio, Qt.SmoothTransformation)

        # 创建全屏画布
        canvas = QPixmap(self.screen_w, self.screen_h)
        canvas.fill(Qt.black)
        painter = QPainter(canvas)

        # 居中绘制图片
        x = (self.screen_w - pixmap.width()) // 2
        y = (self.screen_h - pixmap.height()) // 2
        painter.drawPixmap(x, y, pixmap)

        # 四角回字形标记
        if self.show_corners and self.corner_pixmap:
            cw = self.corner_pixmap.width()
            ch = self.corner_pixmap.height()

            if self.local_mode:
                # 局部模式：回字形围住中心 region_size x region_size 区域
                rs = self.region_size
                rx = (self.screen_w - rs) // 2
                ry = (self.screen_h - rs) // 2
                painter.drawPixmap(rx, ry, self.corner_pixmap)                    # 左上
                painter.drawPixmap(rx + rs - cw, ry, self.corner_pixmap)          # 右上
                painter.drawPixmap(rx, ry + rs - ch, self.corner_pixmap)          # 左下
                painter.drawPixmap(rx + rs - cw, ry + rs - ch, self.corner_pixmap) # 右下
            else:
                # 全屏模式：紧贴屏幕边缘
                painter.drawPixmap(0, 0, self.corner_pixmap)
                painter.drawPixmap(self.screen_w - cw, 0, self.corner_pixmap)
                painter.drawPixmap(0, self.screen_h - ch, self.corner_pixmap)
                painter.drawPixmap(self.screen_w - cw, self.screen_h - ch, self.corner_pixmap)

        # 信息二维码 + 序号 + 版本/通道
        qr_pixmap = self.get_current_qr()
        if qr_pixmap:
            qr_w = qr_pixmap.width()

            if self.local_mode:
                # 局部模式：QR放在768区域左边
                rs = self.region_size
                rx = (self.screen_w - rs) // 2
                ry = (self.screen_h - rs) // 2
                qr_x = max(0, rx - qr_w - 20)  # 768区域左侧，留20px间距
                qr_y = ry + 10
            else:
                # 全屏模式：QR放在左上角
                qr_x = 0
                qr_y = 100

            painter.drawPixmap(qr_x, qr_y, qr_pixmap)

            # QR下方：序号 + 版本/通道
            qr_bottom = qr_y + qr_pixmap.height()

            # 序号
            font_num = QFont("Consolas", 14, QFont.Bold)
            painter.setFont(font_num)
            painter.setPen(QColor(0, 255, 0))
            counter_text = f"{self.current_index + 1}/{len(self.image_paths)}"
            painter.drawText(qr_x + 5, qr_bottom + 2, qr_w - 10, 24, Qt.AlignLeft, counter_text)

            # 版本/通道
            font_info = QFont("Consolas", 10)
            painter.setFont(font_info)
            painter.setPen(QColor(0, 255, 255))
            if self.template_indices is not None:
                tpl_idx = self.template_indices[self.current_index]
                info_text = f"{self.version}|{self.channel}|tpl{tpl_idx}"
            else:
                info_text = f"{self.version}|{self.channel}"
            painter.drawText(qr_x + 5, qr_bottom + 22, qr_w - 10, 18, Qt.AlignLeft, info_text)

        painter.end()
        self.label.setPixmap(canvas)
        self.label.setGeometry(0, 0, self.screen_w, self.screen_h)

        tpl_info = ""
        if self.template_indices is not None:
            tpl_info = f" [tpl={self.template_indices[self.current_index]}]"
        print(f"[Slideshow] {path.name} ({self.current_index + 1}/{len(self.image_paths)}){tpl_info}")

    def next_image(self):
        self.current_index = (self.current_index + 1) % len(self.image_paths)
        self.show_image()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()
        elif event.key() == Qt.Key_Right:
            self.next_image()
        elif event.key() == Qt.Key_Left:
            self.current_index = (self.current_index - 1) % len(self.image_paths)
            self.show_image()


def load_qr(version, index, channel, template_dir):
    """加载单个QR码"""
    manifest_path = template_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    with open(manifest_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    version_records = [r for r in records if r["version"] == version]
    if index >= len(version_records):
        return None
    rec = version_records[index]
    qr_key = f"qr_{channel}_path"
    qr_rel = rec.get(qr_key, rec.get("qr_path"))
    if not qr_rel:
        return None
    qr_path = template_dir / qr_rel
    if qr_path.exists():
        return QPixmap(str(qr_path)).scaled(150, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    return None


def load_all_qr(version, channel, template_dir, num_templates=10):
    """加载所有模板的QR码，返回 dict: index -> QPixmap"""
    manifest_path = template_dir / "manifest.json"
    if not manifest_path.exists():
        return {}
    with open(manifest_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    version_records = [r for r in records if r["version"] == version]

    pixmaps = {}
    for i, rec in enumerate(version_records[:num_templates]):
        qr_key = f"qr_{channel}_path"
        qr_rel = rec.get(qr_key, rec.get("qr_path"))
        if not qr_rel:
            continue
        qr_path = template_dir / qr_rel
        if qr_path.exists():
            pixmaps[i] = QPixmap(str(qr_path)).scaled(150, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation)

    return pixmaps


def main():
    parser = argparse.ArgumentParser(description="全屏图片播放")
    parser.add_argument("--image_dir", type=Path)
    parser.add_argument("--interval", type=int, default=1000)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--version", choices=["v17", "v18"], default="v17")
    parser.add_argument("--channel", choices=["cb", "b"], default=None, help="水印通道（默认: v17=cb, v18=b）")
    parser.add_argument("--index", type=int, default=0, help="固定模式: 模板索引")
    parser.add_argument("--template_dir", type=Path)
    parser.add_argument("--no_qr", action="store_true")
    parser.add_argument("--corners", action="store_true", help="四角加回字形标记")
    parser.add_argument("--local", action="store_true", help="局部模式：回字形围住768x768区域")
    parser.add_argument("--region_size", type=int, default=768, help="局部模式区域大小")
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
        image_paths = [Path(p) for p in playlist["images"]]
        template_indices = playlist["template_indices"]
        print(f"[Slideshow] 播放列表模式: {len(image_paths)}张图片")
    else:
        # 固定模式
        image_dir = args.image_dir or Path(__file__).parent / "clean_image"
        if not image_dir.is_absolute():
            image_dir = Path(__file__).parent / image_dir

        if args.image:
            image_paths = [args.image]
        else:
            image_paths = sorted(list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg")))
            if not image_paths:
                print(f"错误: {image_dir} 下没有图片")
                sys.exit(1)

    app = QApplication(sys.argv)

    # 加载二维码
    qr_pixmaps = {}
    if not args.no_qr:
        if template_indices is not None:
            # 播放列表模式：加载所有用到的QR
            used_indices = set(template_indices)
            all_qr = load_all_qr(args.version, channel, template_dir, num_templates=10)
            qr_pixmaps = {i: all_qr[i] for i in used_indices if i in all_qr}
            print(f"[Slideshow] 已加载 {len(qr_pixmaps)} 个QR码")
        else:
            # 固定模式：只加载一个
            qr = load_qr(args.version, args.index, channel, template_dir)
            if qr:
                qr_pixmaps[args.index] = qr

    # 回字形角标
    corner_pixmap = None
    if args.corners:
        corner_path = Path(__file__).parent / "qrmark" / "qr_loc_mark.png"
        if corner_path.exists():
            corner_pixmap = QPixmap(str(corner_path)).scaled(
                100, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            print(f"[Slideshow] 回字形角标已加载: {corner_path}")
        else:
            print(f"[Slideshow] 警告: 找不到 {corner_path}")

    window = SlideshowWindow(image_paths, qr_pixmaps, corner_pixmap,
                             args.corners, args.interval, template_indices,
                             args.version, channel, args.local, args.region_size)
    window.showFullScreen()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
