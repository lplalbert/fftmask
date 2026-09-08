"""
预览脚本：生成 v17/v18 模板，嵌入到一张干净图片的 Cb 和 B 通道，输出对比图。

用法:
    python watermark_tools/preview_blend.py
    python watermark_tools/preview_blend.py --image watermark_tools/clean_image/xxx.png --seed 2026 --bits 0
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from encode_v17 import WatermarkV17
from encode_v18 import WatermarkV18


def make_bits(seed, index=0):
    return np.random.default_rng(seed + index).integers(0, 2, 60).tolist()


def gen_v17(size, bits):
    enc = WatermarkV17(L1=size, k1=30000, r_watermark=[12, 25],
                       bitsf=[15, 45], r_range=1, n_sectors=60)
    return enc.generate_template(np.asarray(bits, dtype=np.int32))[0]


def gen_v18(size, bits, seed):
    enc = WatermarkV18(L1=size, k1=30000, r_watermark=[12, 25],
                       bitsf=[15, 45], r_range=1, n_sectors=60,
                       M_w=255, M_b=0, hollow_ratio=0.3)
    state = np.random.get_state()
    try:
        np.random.seed(seed)
        return enc.generate_template(np.asarray(bits, dtype=np.int32), hollow=True)[0]
    finally:
        np.random.set_state(state)


def tile_template(template, h, w):
    th, tw = template.shape[:2]
    y0 = (th - h % th) % th // 2
    x0 = (tw - w % tw) % tw // 2
    tiled = np.tile(template, ((h + th - 1) // th + 1, (w + tw - 1) // tw + 1))
    return tiled[y0:y0 + h, x0:x0 + w]


def blend_cb(image, template, alpha=0.016):
    """Cb 通道融合"""
    h, w = image.shape[:2]
    tiled = tile_template(template, h, w).astype(np.float32)
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    ycrcb[:, :, 2] = ycrcb[:, :, 2] * (1 - alpha) + tiled * alpha
    return cv2.cvtColor(np.clip(ycrcb, 0, 255).astype(np.uint8), cv2.COLOR_YCrCb2BGR)


def blend_b(image, template, alpha=0.0191):
    """B 通道融合"""
    h, w = image.shape[:2]
    tiled = tile_template(template, h, w).astype(np.float32)
    result = image.astype(np.float32).copy()
    result[:, :, 0] = result[:, :, 0] * (1 - alpha) + tiled * alpha
    return np.clip(result, 0, 255).astype(np.uint8)


def diff_image(a, b, scale=20):
    """两张图的差异放大可视化"""
    diff = np.abs(a.astype(np.float32) - b.astype(np.float32))
    diff = np.clip(diff * scale, 0, 255).astype(np.uint8)
    return diff


def add_label(image, text, y=30):
    img = image.copy()
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return img


def main():
    parser = argparse.ArgumentParser(description="预览 v17/v18 模板在 Cb/B 通道的融合效果")
    parser.add_argument("--image", type=str, default=None, help="干净图片路径")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--bits", type=int, default=0, help="使用第几个 bit 序列")
    parser.add_argument("--size", type=int, default=512, help="模板尺寸")
    parser.add_argument("--alpha_v17", type=float, default=0.016, help="v17 嵌入强度")
    parser.add_argument("--alpha_v18", type=float, default=0.0228, help="v18 嵌入强度(等效v17的0.016)")
    parser.add_argument("--output_dir", type=str, default="watermark_tools/preview_output")
    parser.add_argument("--diff_scale", type=int, default=20, help="差异放大倍数")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # --- 1. 生成 bit 序列 ---
    bits = make_bits(args.seed, args.bits)
    bit_str = "".join(map(str, bits))
    print(f"60-bit: {bit_str}")

    # --- 2. 生成模板 (都用 r=[12,25]) ---
    print("生成 v17 模板...")
    tm_v17 = gen_v17(args.size, bits)
    print(f"  v17: {tm_v17.shape}, range=[{tm_v17.min()}, {tm_v17.max()}]")

    print("生成 v18 模板 (镂空)...")
    tm_v18 = gen_v18(args.size, bits, args.seed + 10000 + args.bits)
    print(f"  v18: {tm_v18.shape}, range=[{tm_v18.min()}, {tm_v18.max()}]")

    # --- 3. 模板嵌入到灰色画布上（Cb 和 B 各一张） ---
    gray_canvas = np.full((args.size, args.size, 3), 128, dtype=np.uint8)
    alpha_map = {"v17": args.alpha_v17, "v18": args.alpha_v18}
    for label, tm in [("v17", tm_v17), ("v18", tm_v18)]:
        a = alpha_map[label]
        cb_vis = blend_cb(gray_canvas, tm, a)
        cv2.imwrite(str(out / f"template_{label}_on_cb.png"), cb_vis)
        b_vis = blend_b(gray_canvas, tm, a)
        cv2.imwrite(str(out / f"template_{label}_on_b.png"), b_vis)
        cv2.imwrite(str(out / f"template_{label}_raw.png"), tm)
    print("  模板可视化已保存 (raw / on_cb / on_b)")

    # --- 4. 准备干净图片 ---
    if args.image and Path(args.image).exists():
        host = cv2.imread(args.image)
        print(f"使用图片: {args.image}, 尺寸={host.shape}")
    else:
        candidates = sorted(Path("watermark_tools/clean_image").glob("*.png"))
        if candidates:
            host = cv2.imread(str(candidates[0]))
            print(f"使用 clean_image: {candidates[0].name}, 尺寸={host.shape}")
        else:
            host = np.random.randint(50, 200, (600, 800, 3), dtype=np.uint8)
            print("未找到图片，使用随机噪声")

    h, w = host.shape[:2]
    cv2.imwrite(str(out / "original.png"), host)

    # --- 5. 四种组合：v17/v18 × Cb/B ---
    combos = [
        ("v17", tm_v17, "cb", blend_cb, args.alpha_v17),
        ("v17", tm_v17, "b",  blend_b, args.alpha_v17),
        ("v18", tm_v18, "cb", blend_cb, args.alpha_v18),
        ("v18", tm_v18, "b",  blend_b, args.alpha_v18),
    ]
    results = {}
    for name, tm, ch, fn, a in combos:
        out_img = fn(host, tm, a)
        fname = f"{name}_on_{ch}.png"
        cv2.imwrite(str(out / fname), out_img)
        results[(name, ch)] = out_img
        diff_img = diff_image(host, out_img, args.diff_scale)
        cv2.imwrite(str(out / f"diff_{name}_{ch}_x{args.diff_scale}.png"), diff_img)
        print(f"  {name} -> {ch} 通道 OK")

    # --- 6. v17+ v18 同时嵌入图 ---
    for ch, fn in [("cb", blend_cb), ("b", blend_b)]:
        img = fn(host, tm_v17, args.alpha_v17)
        img = fn(img, tm_v18, args.alpha_v18)
        cv2.imwrite(str(out / f"both_v17+v18_on_{ch}.png"), img)
        cv2.imwrite(str(out / f"diff_both_{ch}_x{args.diff_scale}.png"),
                    diff_image(host, img, args.diff_scale))
    print("  v17+v18 同时嵌入 Cb / B OK")

    # --- 7. 拼接大对比图 ---
    target_h = 400
    target_w = int(w * target_h / h)

    def rsz(img):
        return cv2.resize(img, (target_w, target_h))

    # 模板行: v17 raw / v17 on cb / v17 on b | v18 raw / v18 on cb / v18 on b
    tm_v17_cb = cv2.imread(str(out / "template_v17_on_cb.png"))
    tm_v17_b = cv2.imread(str(out / "template_v17_on_b.png"))
    tm_v18_cb = cv2.imread(str(out / "template_v18_on_cb.png"))
    tm_v18_b = cv2.imread(str(out / "template_v18_on_b.png"))
    # 裸模板转为3通道以便拼接
    tm_v17_3ch = cv2.cvtColor(tm_v17, cv2.COLOR_GRAY2BGR)
    tm_v18_3ch = cv2.cvtColor(tm_v18, cv2.COLOR_GRAY2BGR)
    pad = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    row_tmpl = np.hstack([
        add_label(pad, "templates", y=25),
        add_label(rsz(tm_v17_3ch), "v17 raw", y=25),
        add_label(rsz(tm_v17_cb), "v17 on Cb", y=25),
        add_label(rsz(tm_v17_b), "v17 on B", y=25),
        add_label(rsz(tm_v18_3ch), "v18 raw", y=25),
        add_label(rsz(tm_v18_cb), "v18 on Cb", y=25),
        add_label(rsz(tm_v18_b), "v18 on B", y=25),
    ])

    # 嵌入行: original / v17-cb / v17-b / v18-cb / v18-b / both-cb / both-b
    both_cb = cv2.imread(str(out / "both_v17+v18_on_cb.png"))
    both_b  = cv2.imread(str(out / "both_v17+v18_on_b.png"))
    row_embed = np.hstack([
        add_label(rsz(host), "Original"),
        add_label(rsz(results[("v17","cb")]), "v17 Cb"),
        add_label(rsz(results[("v17","b")]),  "v17 B"),
        add_label(rsz(results[("v18","cb")]), "v18 Cb"),
        add_label(rsz(results[("v18","b")]),  "v18 B"),
        add_label(rsz(both_cb), "v17+18 Cb"),
        add_label(rsz(both_b),  "v17+18 B"),
    ])

    # 差异行
    diff_v17_cb = cv2.imread(str(out / "diff_v17_cb_x20.png"))
    diff_v17_b  = cv2.imread(str(out / "diff_v17_b_x20.png"))
    diff_v18_cb = cv2.imread(str(out / "diff_v18_cb_x20.png"))
    diff_v18_b  = cv2.imread(str(out / "diff_v18_b_x20.png"))
    diff_bc = cv2.imread(str(out / "diff_both_cb_x20.png"))
    diff_bb = cv2.imread(str(out / "diff_both_b_x20.png"))
    # padding 列对齐
    pad = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    row_diff = np.hstack([
        add_label(pad, "diff x20"),
        add_label(rsz(diff_v17_cb), "v17 Cb"),
        add_label(rsz(diff_v17_b),  "v17 B"),
        add_label(rsz(diff_v18_cb), "v18 Cb"),
        add_label(rsz(diff_v18_b),  "v18 B"),
        add_label(rsz(diff_bc), "both Cb"),
        add_label(rsz(diff_bb), "both B"),
    ])

    comparison = np.vstack([row_tmpl, row_embed, row_diff])
    cv2.imwrite(str(out / "comparison_all.png"), comparison)
    print(f"\n对比图: {out.resolve()}/comparison_all.png")

    print(f"\n全部输出在: {out.resolve()}")
    print("  template_v17_raw.png        - v17 裸模板")
    print("  template_v17_on_cb.png      - v17 模板嵌在 Cb 通道")
    print("  template_v17_on_b.png       - v17 模板嵌在 B 通道")
    print("  template_v18_raw.png        - v18 裸模板(镂空)")
    print("  template_v18_on_cb.png      - v18 模板嵌在 Cb 通道")
    print("  template_v18_on_b.png       - v18 模板嵌在 B 通道")
    print("  v17_on_cb.png               - v17 嵌入图片 Cb")
    print("  v17_on_b.png                - v17 嵌入图片 B")
    print("  v18_on_cb.png               - v18 嵌入图片 Cb")
    print("  v18_on_b.png                - v18 嵌入图片 B")
    print("  both_v17+v18_on_cb.png      - v17+v18 同时嵌入 Cb")
    print("  both_v17+v18_on_b.png       - v17+v18 同时嵌入 B")
    print(f"  diff_*_x{args.diff_scale}.png          - 差异放大图")
    print("  comparison_all.png          - 拼接对比大图")


if __name__ == "__main__":
    main()
