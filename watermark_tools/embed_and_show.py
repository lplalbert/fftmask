"""嵌入水印到图片并展示效果。

用法:
    python watermark_tools/embed_and_show.py
    python watermark_tools/embed_and_show.py --image path/to/image.png

效果:
1. 加载干净图片
2. 用训练公式嵌入水印: dst_channel = host_channel * (1-alpha) + template * alpha
3. 在融合后贴上二维码（白色背景，不参与融合）
4. 输出: 原图、水印图、差值图、对比图
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


SPEC = {
    "v17": {"channel": "cb", "alpha": 0.016},
    "v18": {"channel": "b", "alpha": 0.0228},
}


def embed_watermark(image_bgr, template_gray, channel, alpha):
    """在图像上嵌入水印。

    训练公式: dst_channel = host_channel * (1-alpha) + template * alpha
    template_gray: 0或255的灰度图，已平铺到与image相同尺寸
    """
    result = image_bgr.copy().astype(np.float32)
    t = template_gray.astype(np.float32) / 255.0  # 0.0 or 1.0

    if channel == "cb":
        ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32)
        # dst_Cb = host_Cb * (1-alpha) + 255 * alpha (for template=255)
        # dst_Cb = host_Cb * (1-alpha) + 0 * alpha   (for template=0)
        # => dst_Cb = host_Cb + alpha * (255*t - host_Cb)
        ycrcb[:, :, 2] = ycrcb[:, :, 2] + alpha * (255.0 * t - ycrcb[:, :, 2])
        result = cv2.cvtColor(ycrcb.clip(0, 255).astype(np.uint8), cv2.COLOR_YCrCb2BGR).astype(np.float32)
    else:
        # dst_B = host_B * (1-alpha) + 255 * alpha (for template=255)
        result[:, :, 0] = result[:, :, 0] + alpha * (255.0 * t - result[:, :, 0])

    return result.clip(0, 255).astype(np.uint8)


def tile_template(template, target_w, target_h):
    """平铺模板到目标尺寸。"""
    th, tw = template.shape[:2]
    nx = (target_w + tw - 1) // tw
    ny = (target_h + th - 1) // th
    tiled = np.tile(template, (ny, nx))
    start_y = (tiled.shape[0] - target_h) // 2
    start_x = (tiled.shape[1] - target_w) // 2
    return tiled[start_y:start_y + target_h, start_x:start_x + target_w]


def paste_qr(image, qr_bgr, position="top_left"):
    """在图像上贴二维码（直接覆盖，白色背景）。"""
    result = image.copy()
    h, w = qr_bgr.shape[:2]
    H, W = image.shape[:2]
    if position == "top_left":
        y, x = 10, 10
    elif position == "top_right":
        y, x = 10, W - w - 10
    elif position == "bottom_right":
        y, x = H - h - 10, W - w - 10
    else:
        y, x = H - h - 10, 10
    result[y:y+h, x:x+w] = qr_bgr
    return result


def make_diff_image(original, watermarked, scale=20):
    """生成差值图（放大scale倍便于观察）。"""
    diff = cv2.absdiff(original, watermarked)
    return np.clip(diff.astype(np.float32) * scale, 0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description="Embed watermark and show effect")
    parser.add_argument("--image", type=Path, help="Input image (default: first image in clean_image)")
    parser.add_argument("--version", choices=["v17", "v18", "both"], default="both")
    parser.add_argument("--index", type=int, default=0, help="Template index")
    parser.add_argument("--output_dir", type=Path, help="Output directory")
    parser.add_argument("--alpha", type=float, help="Override alpha")
    parser.add_argument("--max_size", type=int, default=1920, help="Max image dimension")
    args = parser.parse_args()

    # 找图片
    if args.image:
        img_path = args.image
    else:
        clean_dir = Path(__file__).parent / "clean_image"
        images = sorted(clean_dir.glob("*.png"))
        if not images:
            images = sorted(clean_dir.glob("*.jpg"))
        img_path = images[0]

    print(f"Input image: {img_path}")
    image = cv2.imread(str(img_path))
    if image is None:
        print(f"Error: cannot read {img_path}")
        sys.exit(1)

    # 缩放
    h, w = image.shape[:2]
    scale = min(1.0, args.max_size / max(h, w))
    if scale < 1.0:
        image = cv2.resize(image, (int(w * scale), int(h * scale)))
    print(f"Image size: {image.shape[1]}x{image.shape[0]}")

    # 输出目录
    output_dir = args.output_dir or Path(__file__).parent / "generated_templates" / "embed_demo"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存原图
    cv2.imwrite(str(output_dir / "original.png"), image)

    template_dir = Path(__file__).parent / "generated_templates"
    manifest_path = template_dir / "manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    versions = ["v17", "v18"] if args.version == "both" else [args.version]

    for version in versions:
        spec = SPEC[version]
        channel = spec["channel"]
        alpha = args.alpha if args.alpha else spec["alpha"]

        # 加载灰度模板
        rec = [r for r in records if r["version"] == version][args.index]
        gray_path = template_dir / rec["path"]
        template = cv2.imread(str(gray_path), cv2.IMREAD_GRAYSCALE)
        tiled = tile_template(template, image.shape[1], image.shape[0])

        # 嵌入水印
        watermarked = embed_watermark(image, tiled, channel, alpha)

        # 加载二维码并贴上（使用对应通道的QR）
        qr_key = f"qr_{channel}_path"
        qr_rel = rec.get(qr_key, rec.get("qr_path"))
        qr_path = template_dir / qr_rel
        qr = cv2.imread(str(qr_path))
        watermarked_with_qr = paste_qr(watermarked, qr, "top_left")

        # 差值图
        diff = make_diff_image(image, watermarked, scale=20)

        # 保存
        prefix = f"{version}_{args.index:02d}_{channel}"
        cv2.imwrite(str(output_dir / f"{prefix}_watermarked.png"), watermarked)
        cv2.imwrite(str(output_dir / f"{prefix}_watermarked_qr.png"), watermarked_with_qr)
        cv2.imwrite(str(output_dir / f"{prefix}_diff_x20.png"), diff)

        # 拼接对比图: 原图 | 水印图 | 差值图
        h, w = image.shape[:2]
        target_h = min(h, 600)
        target_w = int(w * target_h / h)
        orig_resized = cv2.resize(image, (target_w, target_h))
        wm_resized = cv2.resize(watermarked_with_qr, (target_w, target_h))
        diff_resized = cv2.resize(diff, (target_w, target_h))
        comparison = np.hstack([orig_resized, wm_resized, diff_resized])
        cv2.imwrite(str(output_dir / f"{prefix}_comparison.png"), comparison)

        print(f"  {version}: channel={channel}, alpha={alpha}")
        print(f"    -> {prefix}_watermarked.png")
        print(f"    -> {prefix}_watermarked_qr.png (with QR overlay)")
        print(f"    -> {prefix}_diff_x20.png (difference x20)")
        print(f"    -> {prefix}_comparison.png (original | watermarked+QR | diff)")

    print(f"\nOutput directory: {output_dir}")


if __name__ == "__main__":
    main()
