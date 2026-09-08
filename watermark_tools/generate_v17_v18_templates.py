"""Generate deterministic v17/v18 display templates and a manifest.

与test6.py方式一致：
- 灰度画布(128)嵌入水印到Cb/B通道 → BGR图
- setWindowOpacity(alpha) 控制强度
- 无逐像素计算，OS硬件加速合成
"""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import qrcode

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from encode_v17 import WatermarkV17
from encode_v18 import WatermarkV18

V17 = {
    "channel": "cb", "alpha": 0.016, "rings": [12, 25],
    "bits_per_ring": [15, 45], "hollow_ratio": 0.0,
}
V18 = {
    "channel": "b", "alpha": 0.0228, "rings": [12, 25],
    "bits_per_ring": [15, 45], "hollow_ratio": 0.3,
}

SCREEN_W, SCREEN_H = 1920, 1080


def bits_for(seed, index):
    return np.random.default_rng(seed + index).integers(0, 2, 60).tolist()


def generate_v17(size, bits):
    encoder = WatermarkV17(
        L1=size, k1=30000.0, r_watermark=V17["rings"],
        bitsf=V17["bits_per_ring"], r_range=1, n_sectors=60,
    )
    return encoder.generate_template(np.asarray(bits, dtype=np.int32))[0]


def generate_v18(size, bits, hollow_seed):
    encoder = WatermarkV18(
        L1=size, k1=30000.0, r_watermark=V18["rings"],
        bitsf=V18["bits_per_ring"], r_range=1, n_sectors=60,
        M_w=255, M_b=0, hollow_ratio=V18["hollow_ratio"],
    )
    state = np.random.get_state()
    try:
        np.random.seed(hollow_seed)
        return encoder.generate_template(
            np.asarray(bits, dtype=np.int32), hollow=True
        )[0]
    finally:
        np.random.set_state(state)


def write_template(path, template):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), template)


def tile_template(template, target_w, target_h):
    th, tw = template.shape[:2]
    nx = (target_w + tw - 1) // tw
    ny = (target_h + th - 1) // th
    tiled = np.tile(template, (ny, nx))
    start_y = (tiled.shape[0] - target_h) // 2
    start_x = (tiled.shape[1] - target_w) // 2
    return tiled[start_y:start_y + target_h, start_x:start_x + target_w]


def embed_on_gray_canvas(tiled_gray, channel):
    """在灰度画布(128)上嵌入水印，返回BGR图。

    与test6.py一致：画布灰色，水印嵌入目标通道。
    setWindowOpacity(alpha) 时，整个画面以alpha透明度叠加到桌面。
    """
    h, w = tiled_gray.shape[:2]

    if channel == "cb":
        # YCrCb空间：Y=128, Cr=128, Cb=template
        ycrcb = np.full((h, w, 3), 128, dtype=np.uint8)
        ycrcb[:, :, 2] = tiled_gray  # Cb = template (0 or 255)
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    else:
        # B通道：G=R=128, B=template
        canvas = np.full((h, w, 3), 128, dtype=np.uint8)
        canvas[:, :, 0] = tiled_gray  # B = template (0 or 255)
        return canvas


def generate_qr_image(data: str, size: int = 150) -> np.ndarray:
    """生成二维码图像，返回BGR格式（白色背景）。"""
    qr = qrcode.QRCode(version=1, box_size=10, border=1)
    qr.add_data(data)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white").convert("RGB")
    img = img.resize((size, size))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def main():
    parser = argparse.ArgumentParser(description="Generate v17/v18 templates")
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--screen_w", type=int, default=SCREEN_W)
    parser.add_argument("--screen_h", type=int, default=SCREEN_H)
    args = parser.parse_args()

    output_dir = args.output_dir or Path(__file__).parent / "generated_templates"
    v17_dir = output_dir / "v17"
    v18_dir = output_dir / "v18"
    v17_dir.mkdir(parents=True, exist_ok=True)
    v18_dir.mkdir(parents=True, exist_ok=True)

    records = []

    for version, spec, version_dir in [("v17", V17, v17_dir), ("v18", V18, v18_dir)]:
        for index in range(args.count):
            bits = bits_for(args.seed, index)
            if version == "v17":
                template = generate_v17(args.size, bits)
            else:
                template = generate_v18(args.size, bits, args.seed + 10000 + index)
            bit_string = "".join(map(str, bits))

            # 1. 灰度模板 512x512
            filename = f"{version}_{index:02d}_{bit_string}.png"
            path = version_dir / filename
            write_template(path, template)

            # 2. 平铺
            tiled = tile_template(template, args.screen_w, args.screen_h)

            # 3. 灰度画布嵌入（与test6.py一致）
            for ch in ["cb", "b"]:
                bgr = embed_on_gray_canvas(tiled, ch)
                bgr_filename = f"{version}_{index:02d}_{bit_string}_on_{ch}.png"
                bgr_path = version_dir / bgr_filename
                cv2.imwrite(str(bgr_path), bgr)

            # 4. 二维码（按通道分别生成，包含序号）
            qr_paths = {}
            for ch in ["cb", "b"]:
                qr_data = f"{version}|{ch}|{index:02d}|{bit_string}"
                qr_img = generate_qr_image(qr_data, size=150)
                qr_path = version_dir / f"{version}_{index:02d}_{bit_string}_qr_{ch}.png"
                cv2.imwrite(str(qr_path), qr_img)
                qr_paths[ch] = qr_path

            records.append({
                "version": version, "index": index, "filename": filename,
                "path": path.relative_to(output_dir).as_posix(),
                "bgr_cb_filename": f"{version}_{index:02d}_{bit_string}_on_cb.png",
                "bgr_cb_path": (version_dir / f"{version}_{index:02d}_{bit_string}_on_cb.png").relative_to(output_dir).as_posix(),
                "bgr_b_filename": f"{version}_{index:02d}_{bit_string}_on_b.png",
                "bgr_b_path": (version_dir / f"{version}_{index:02d}_{bit_string}_on_b.png").relative_to(output_dir).as_posix(),
                "qr_cb_filename": f"{version}_{index:02d}_{bit_string}_qr_cb.png",
                "qr_cb_path": qr_paths["cb"].relative_to(output_dir).as_posix(),
                "qr_b_filename": f"{version}_{index:02d}_{bit_string}_qr_b.png",
                "qr_b_path": qr_paths["b"].relative_to(output_dir).as_posix(),
                "bits": bit_string, "channel": spec["channel"],
                "alpha": spec["alpha"], "size": args.size,
                "screen_size": f"{args.screen_w}x{args.screen_h}",
                "rings": spec["rings"], "bits_per_ring": spec["bits_per_ring"],
                "hollow_ratio": spec["hollow_ratio"],
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            })

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    csv_path = output_dir / "manifest.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    print(f"Generated {args.count * 2} templates ({args.count} v17 + {args.count} v18)")
    print(f"Each: gray 512x512 + BGR {args.screen_w}x{args.screen_h} (cb & b) + QR")
    print(f"Method: gray canvas + setWindowOpacity (same as test6.py)")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
