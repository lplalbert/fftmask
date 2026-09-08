"""
带水印覆盖的图片显示脚本
功能：
1. 从 clean_image 文件夹读取图片
2. 按顺序轮换水印模板覆盖到图片上
3. 在图片上显示二维码（包含 bit 信息）和四角定位标记
4. 支持 Cb / B / both 通道嵌入，v17/v18 独立 alpha
5. 全屏定时切换显示
"""

import os
import glob
import json
from pathlib import Path
import numpy as np
import cv2
import qrcode
from PIL import Image, ImageTk, ImageDraw, ImageFont
import tkinter as tk
from screeninfo import get_monitors


class WatermarkDisplay:
    def __init__(self, config):
        self.config = config
        self.image_index = 0
        self.template_index = 0
        self.screen_width = 1920
        self.screen_height = 1080
        self.root = None
        self.label = None

        self.image_files = self.load_image_files()
        self.watermark_templates = self.load_watermark_templates()
        self.qr_mark_image = self.load_qr_mark_image()

        if not self.image_files:
            raise ValueError(f"未找到图片文件，请检查 {config['image_folders']}")
        if not self.watermark_templates:
            raise ValueError(f"未找到水印模板，请检查 {config['watermark_dir']}")

        print(f"加载了 {len(self.image_files)} 张图片")
        print(f"加载了 {len(self.watermark_templates)} 个水印模板")

    # ── 资源加载 ────────────────────────────────────────────────────────────

    def load_image_files(self):
        files = []
        for folder in self.config['image_folders']:
            if os.path.exists(folder):
                for ext in ['*.jpg', '*.png', '*.JPG', '*.jpeg']:
                    files.extend(glob.glob(os.path.join(folder, ext)))
        return sorted(files)

    def load_watermark_templates(self):
        """从 manifest 加载模板（带版本/通道/alpha 元数据），或回退到目录 PNG"""
        watermark_dir = Path(self.config['watermark_dir'])
        manifest_path = Path(self.config.get('manifest') or watermark_dir / 'manifest.json')
        if not manifest_path.is_absolute():
            manifest_path = Path.cwd() / manifest_path
        templates = []
        if manifest_path.exists():
            data = json.loads(manifest_path.read_text(encoding='utf-8'))
            root = manifest_path.parent
            allowed = self.config.get('version', 'both')
            for item in data.get('templates', []):
                if allowed != 'both' and item.get('version') != allowed:
                    continue
                path = root / item['path']
                if path.exists():
                    templates.append({**item, 'path': str(path)})
            return templates
        if not watermark_dir.exists():
            return []
        return [
            {'filename': path.name, 'path': str(path), 'version': 'legacy',
             'bits': '', 'channel': 'cb', 'alpha': self.config['alpha_v17']}
            for path in sorted(watermark_dir.glob('*.png'))
        ]

    def load_qr_mark_image(self):
        qr_path = self.config['qr_image']
        if not qr_path:
            return None
        qr_size = self.config['qr_size']
        if not os.path.exists(qr_path):
            print(f"警告: QR码图片不存在: {qr_path}")
            return None
        qr_image = cv2.imread(qr_path)
        if qr_image is None:
            print(f"警告: 无法读取QR码图片: {qr_path}")
            return None
        return cv2.resize(qr_image, (qr_size, qr_size))

    # ── 水印模板平铺 ───────────────────────────────────────────────────────

    def tile_template(self, template, height, width):
        """按 512×512 周期平铺，居中对齐"""
        template = np.asarray(template)
        th, tw = template.shape[:2]
        y0 = (th - height % th) % th // 2
        x0 = (tw - width % tw) % tw // 2
        tiled = np.tile(template, ((height + th - 1) // th + 1,
                                   (width + tw - 1) // tw + 1))
        return tiled[y0:y0 + height, x0:x0 + width]

    # ── 通道融合 ────────────────────────────────────────────────────────────

    def blend_cb(self, image, template, alpha):
        """YCrCb Cb 通道融合"""
        h, w = image.shape[:2]
        tiled = self.tile_template(template, h, w).astype(np.float32)
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb).astype(np.float32)
        ycrcb[:, :, 2] = ycrcb[:, :, 2] * (1 - alpha) + tiled * alpha
        return cv2.cvtColor(np.clip(ycrcb, 0, 255).astype(np.uint8), cv2.COLOR_YCrCb2BGR)

    def blend_b(self, image, template, alpha):
        """BGR Blue 通道融合"""
        h, w = image.shape[:2]
        tiled = self.tile_template(template, h, w).astype(np.float32)
        result = image.astype(np.float32).copy()
        result[:, :, 0] = result[:, :, 0] * (1 - alpha) + tiled * alpha
        return np.clip(result, 0, 255).astype(np.uint8)

    # ── 信息叠加 ────────────────────────────────────────────────────────────

    def generate_info_qr(self, data):
        """生成包含 bit 信息的二维码"""
        qr = qrcode.QRCode(
            version=2,
            error_correction=qrcode.constants.ERROR_CORRECT_H,
            box_size=8,
            border=2,
        )
        qr.add_data(data)
        qr.make(fit=True)
        qr_img = qr.make_image(fill_color="black", back_color="white")
        return np.array(qr_img.convert('RGB'))

    def add_info_qr(self, image, bits_string, version, channel, position, qr_size):
        """在指定位置贴上包含 bit 信息的二维码，附带版本和通道标注"""
        # 二维码内容：版本 + 通道 + bit 串
        qr_data = f"{version}|{channel}|{bits_string}"
        qr_img = self.generate_info_qr(qr_data)
        qr_img = cv2.resize(qr_img, (qr_size, qr_size))

        # 在二维码下方加文字标注
        label_text = f"{version} {channel} {bits_string[:20]}"
        pil_qr = Image.fromarray(qr_img)
        draw = ImageDraw.Draw(pil_qr)
        try:
            font = ImageFont.truetype("arial.ttf", 14)
        except:
            try:
                font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 14)
            except:
                font = ImageFont.load_default()
        draw.text((4, qr_size - 18), label_text, fill=(128, 128, 128), font=font)
        qr_img = np.array(pil_qr)

        # 贴到图片上
        x, y = position
        h, w = image.shape[:2]
        if x + qr_size <= w and y + qr_size <= h:
            image[y:y + qr_size, x:x + qr_size] = qr_img
        return image

    def add_qr_corners(self, image):
        if self.qr_mark_image is None:
            return image
        h, w = image.shape[:2]
        block_size = self.config['qr_size']
        image[:block_size, :block_size] = self.qr_mark_image
        image[:block_size, w - block_size:] = self.qr_mark_image
        image[h - block_size:, :block_size] = self.qr_mark_image
        image[h - block_size:, w - block_size:] = self.qr_mark_image
        return image

    # ── 主处理流程 ──────────────────────────────────────────────────────────

    def get_alpha_for_version(self, version):
        """v17 直接用 alpha，v18 按镂空比例自动放大：alpha / (1 - hollow_ratio)"""
        base = self.config['alpha']
        if version == 'v18':
            return base / (1 - self.config['hollow_ratio'])
        return base

    def process_image(self, image_path, template_record):
        """用一张模板处理一张干净图片"""
        image = cv2.imread(image_path)
        if image is None:
            print(f"无法读取图片: {image_path}")
            return None, None

        image = cv2.resize(image, (self.screen_width, self.screen_height))
        bits_string = template_record.get('bits', '')
        version = template_record.get('version', 'v17')
        channel = self.config['channel']        # cb 或 b
        alpha = self.get_alpha_for_version(version)
        template_path = template_record['path']

        watermark_template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        if watermark_template is None:
            print(f"无法读取水印模板: {template_path}")
            return image, bits_string

        # 1. 叠加二维码（包含 bit 信息）
        qr_position = self.config['qr_position']
        qr_size = self.config['info_qr_size']
        image = self.add_info_qr(image, bits_string, version, channel,
                                 qr_position, qr_size)

        # 2. 四角定位标记
        image = self.add_qr_corners(image)

        # 3. 嵌入水印到指定通道
        if channel == 'cb':
            image = self.blend_cb(image, watermark_template, alpha)
        else:
            image = self.blend_b(image, watermark_template, alpha)

        return image, bits_string

    # ── 显示逻辑 ────────────────────────────────────────────────────────────

    def display_image_on_screen(self, image, screen_index):
        monitors = get_monitors()
        if screen_index >= len(monitors):
            print(f"屏幕索引 {screen_index} 超出范围")
            return
        monitor = monitors[screen_index]
        img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        img = img.resize((monitor.width, monitor.height), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        if self.label:
            self.label.config(image=photo)
            self.label.image = photo

    def change_image(self, screen_index):
        if self.image_index >= len(self.image_files):
            self.image_index = 0

        image_path = self.image_files[self.image_index]
        template_record = self.watermark_templates[self.template_index]
        self.template_index = (self.template_index + 1) % len(self.watermark_templates)

        processed_image, bits_string = self.process_image(image_path, template_record)

        if processed_image is not None:
            self.display_image_on_screen(processed_image, screen_index)
            version = template_record.get('version', 'legacy')
            channel = self.config['channel']
            print(f"[{self.image_index + 1}/{len(self.image_files)}] "
                  f"图片: {os.path.basename(image_path)} | "
                  f"{version} -> {channel} | "
                  f"bits: {bits_string[:20]}...")

        self.image_index += 1
        self.root.after(self.config['turn_time'], self.change_image, screen_index)

    def move_window_to_screen(self, window, screen_index):
        monitors = get_monitors()
        if screen_index < len(monitors):
            monitor = monitors[screen_index]
            window.geometry(f'{monitor.width}x{monitor.height}+{monitor.x}+{monitor.y}')
            self.screen_width = monitor.width
            self.screen_height = monitor.height

    def run(self):
        self.root = tk.Tk()
        self.move_window_to_screen(self.root, self.config['screen'])
        self.root.overrideredirect(True)
        self.root.bind("<Escape>", lambda e: self.root.quit())
        self.label = tk.Label(self.root)
        self.label.pack(fill=tk.BOTH, expand=True)
        self.change_image(self.config['screen'])
        self.root.mainloop()


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="水印模板全屏显示脚本")

    # 嵌入通道（二选一）
    parser.add_argument("--channel", choices=["cb", "b"], default="cb",
                        help="嵌入通道: cb=YCrCb蓝差通道, b=BGR蓝通道 (默认: cb)")

    # 嵌入强度（等效强度，v18 自动按镂空比例放大）
    parser.add_argument("--alpha", type=float, default=0.016,
                        help="等效嵌入强度 (默认: 0.016)，v18 自动乘以 1/(1-hollow_ratio)")

    # 模板选择
    parser.add_argument("--watermark_dir", type=str,
                        default="./watermark_tools/generated_templates",
                        help="生成模板根目录")
    parser.add_argument("--manifest", type=str, default=None,
                        help="manifest.json路径，默认从watermark_dir读取")
    parser.add_argument("--version", choices=["both", "v17", "v18"], default="both",
                        help="显示哪些版本的模板")

    # 图片来源
    parser.add_argument("--image_dir", type=str, nargs="+",
                        default=["./watermark_tools/clean_image"],
                        help="图片来源文件夹")

    # 二维码（默认显示，包含 bit 信息）
    parser.add_argument("--qr_position", type=int, nargs=2, default=[20, 20],
                        help="二维码位置 (x y)，默认左上角 (20 20)")
    parser.add_argument("--info_qr_size", type=int, default=200,
                        help="信息二维码大小 (默认: 200)")

    # 四角定位标记（默认不显示，需要时手动指定图片路径）
    parser.add_argument("--qr_image", type=str, default=None,
                        help="四角QR码定位标记图片路径，默认不显示")
    parser.add_argument("--qr_size", type=int, default=120,
                        help="四角QR码大小 (默认: 120)")

    # 显示控制
    parser.add_argument("--turn_time", type=int, default=2000,
                        help="切换间隔毫秒 (默认: 2000)")
    parser.add_argument("--screen", type=int, default=0,
                        help="屏幕索引 (默认: 0)")

    args = parser.parse_args()

    config = {
        'channel': args.channel,
        'alpha': args.alpha,
        'hollow_ratio': 0.3,  # v18 镂空比例，自动换算等效强度
        'turn_time': args.turn_time,
        'screen': args.screen,
        'image_folders': args.image_dir,
        'watermark_dir': args.watermark_dir,
        'manifest': args.manifest,
        'version': args.version,
        'qr_position': tuple(args.qr_position),
        'info_qr_size': args.info_qr_size,
        'qr_image': args.qr_image,
        'qr_size': args.qr_size,
    }

    display = WatermarkDisplay(config)
    display.run()


if __name__ == "__main__":
    main()
