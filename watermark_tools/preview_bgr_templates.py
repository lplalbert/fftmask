"""预览生成的BGR域模板图片。"""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def show_templates():
    base = Path(__file__).parent / "generated_templates"

    rows = []
    for version in ["v17", "v18"]:
        vdir = base / version
        # 找第一组模板
        gray_files = sorted(vdir.glob(f"{version}_00_*.png"))
        gray_files = [f for f in gray_files if "_on_" not in f.stem]
        if not gray_files:
            continue
        stem = gray_files[0].stem
        gray = cv2.imread(str(vdir / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
        on_cb = cv2.imread(str(vdir / f"{stem}_on_cb.png"))
        on_b = cv2.imread(str(vdir / f"{stem}_on_b.png"))

        gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        row = np.hstack([gray_3ch, on_cb, on_b])
        rows.append(row)

    if not rows:
        print("没有找到生成的模板，请先运行 generate_v17_v18_templates.py")
        return

    # 统一尺寸
    h = min(r.shape[0] for r in rows)
    w = min(r.shape[1] for r in rows)
    rows_resized = [cv2.resize(r, (w, h)) for r in rows]

    # 上下拼接
    composite = np.vstack(rows_resized)

    # 缩小显示
    scale = min(1.0, 800 / composite.shape[1], 600 / composite.shape[0])
    display = cv2.resize(composite, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    # 添加文字标注
    y_offset = 30
    cv2.putText(display, "v17: Gray | Cb | B", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    y_offset += h // 2 * scale + 30
    cv2.putText(display, "v18: Gray | Cb | B", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("Templates: Gray | On_Cb | On_B (v17 top, v18 bottom)", display)
    print("按任意键关闭预览窗口...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    show_templates()
