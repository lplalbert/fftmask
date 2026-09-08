"""预览所有生成的模板（含二维码）。

生成对比图：灰度模板 + Cb模板 + B模板 + QR码
"""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def create_preview(version_dir, version_name, num_templates=3):
    """为一个版本创建预览图（显示前num_templates个模板）。"""
    rows = []
    for i in range(num_templates):
        # 找到对应的模板文件
        gray_files = sorted(version_dir.glob(f"{version_name}_{i:02d}_*.png"))
        gray_files = [f for f in gray_files if "_on_" not in f.stem and "_qr" not in f.stem]
        if not gray_files:
            continue

        stem = gray_files[0].stem
        gray = cv2.imread(str(version_dir / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
        on_cb = cv2.imread(str(version_dir / f"{stem}_on_cb.png"))
        on_b = cv2.imread(str(version_dir / f"{stem}_on_b.png"))
        qr_cb = cv2.imread(str(version_dir / f"{stem}_qr_cb.png"), cv2.IMREAD_UNCHANGED)
        qr_b = cv2.imread(str(version_dir / f"{stem}_qr_b.png"), cv2.IMREAD_UNCHANGED)

        if any(x is None for x in [gray, on_cb, on_b, qr_cb, qr_b]):
            continue

        # 灰度模板缩放到和BGR模板相同高度（1080的缩略图）
        th = 108
        tw = 192
        gray_thumb = cv2.resize(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), (tw, th))
        cb_thumb = cv2.resize(on_cb, (tw, th))
        b_thumb = cv2.resize(on_b, (tw, th))

        # QR码缩放（cb和b各一个）
        def resize_qr(qr_img):
            qr_h = th
            qr_w = int(qr_img.shape[1] * th / qr_img.shape[0])
            thumb = cv2.resize(qr_img[:, :, :3], (qr_w, qr_h))
            return cv2.resize(thumb, (tw, th))

        qr_cb_thumb = resize_qr(qr_cb)
        qr_b_thumb = resize_qr(qr_b)

        # 水平拼接：灰度 | Cb | B | QR_cb | QR_b
        row = np.hstack([gray_thumb, cb_thumb, b_thumb, qr_cb_thumb, qr_b_thumb])
        rows.append(row)

    if not rows:
        return None

    # 垂直拼接所有行
    return np.vstack(rows)


def main():
    base = Path(__file__).parent / "generated_templates"

    previews = []
    for version in ["v17", "v18"]:
        version_dir = base / version
        preview = create_preview(version_dir, version, num_templates=3)
        if preview is not None:
            previews.append(preview)

    if not previews:
        print("没有找到生成的模板，请先运行 generate_v17_v18_templates.py")
        return

    # 添加标题
    title_h = 30
    title1 = np.zeros((title_h, previews[0].shape[1], 3), dtype=np.uint8)
    cv2.putText(title1, "v17: Gray | Cb | B | QR_cb | QR_b", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    title2 = np.zeros((title_h, previews[0].shape[1], 3), dtype=np.uint8)
    cv2.putText(title2, "v18: Gray | Cb | B | QR_cb | QR_b", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # 合并
    final = np.vstack([title1, previews[0], title2, previews[1]])

    # 保存
    output_path = base / "preview_all_templates.png"
    cv2.imwrite(str(output_path), final)
    print(f"Preview saved to: {output_path}")
    print(f"Size: {final.shape[1]}x{final.shape[0]}")


if __name__ == "__main__":
    main()
