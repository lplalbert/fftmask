"""
v17 版本：对图像进行全图水印嵌入（Cb 通道）

水印模板尺寸 = 最短边长 / 2，循环平铺到全图

用法:
    cd /data/lpl/fftmask && python embed_v17_tile.py --input_dir <输入目录> --output_dir <输出目录>
"""

import os
import sys
import json
import argparse
import hashlib
import numpy as np
import cv2
from tqdm import tqdm
from datetime import datetime

from encode_v11 import WatermarkV11

# ── v17 参数（与训练完全一致）────────────────────────────────────────────────
SEED        = 2026
NUM_BITS    = 60
R_LIST      = [8, 15]         # v17 最佳圆环位置
BITS_LIST   = [20, 40]        # 20bit + 40bit = 60bit
K1          = 30000.0
R_RANGE     = 1
R_ROTATION  = None            # v17 无旋转环
ALPHA_EMBED = 0.016
CHANNEL     = "cb"

GLOBAL_WATERMARK_BITS = None


def get_global_watermark_bits():
    """生成固定的水印bits"""
    global GLOBAL_WATERMARK_BITS
    if GLOBAL_WATERMARK_BITS is None:
        rng = np.random.default_rng(SEED)
        GLOBAL_WATERMARK_BITS = rng.integers(0, 2, size=NUM_BITS).tolist()
    return GLOBAL_WATERMARK_BITS


def generate_watermark_template(template_size):
    """
    生成指定尺寸的水印模板

    Args:
        template_size: 模板尺寸（正方形）

    Returns:
        水印模板 (template_size x template_size)
    """
    bits = get_global_watermark_bits()

    wm_sys = WatermarkV11(
        L1=template_size, k1=K1,
        r_watermark=R_LIST,
        bitsf=BITS_LIST,
        r_rotation=R_ROTATION,
        r_range=R_RANGE,
        n_sectors=NUM_BITS
    )
    Tm, M1, _ = wm_sys.generate_template(numbit=np.array(bits))

    # 转为灰度
    if len(Tm.shape) == 3:
        Tm_gray = cv2.cvtColor(Tm, cv2.COLOR_BGR2GRAY)
    else:
        Tm_gray = Tm

    return Tm_gray


def tile_by_phase(template, H, W, offset_y=0, offset_x=0):
    """按相位偏移平铺（周期边界）"""
    y_idx = (np.arange(H) + offset_y) % template.shape[0]
    x_idx = (np.arange(W) + offset_x) % template.shape[1]
    return template[y_idx[:, None], x_idx[None, :]]


def embed_watermark_cb(host_bgr, wm_template, alpha=0.016):
    """
    在 host_bgr 的 Cb 通道嵌入水印，返回带水印的 BGR 图像。
    自动计算offset使中心对齐。

    Args:
        host_bgr: 原始BGR图像
        wm_template: 水印模板（灰度图）
        alpha: 嵌入强度

    Returns:
        带水印的BGR图像
    """
    h, w = host_bgr.shape[:2]
    tw, th = wm_template.shape[:2]

    # 计算中心crop位置
    crop_x0 = w // 2 - tw // 2
    crop_y0 = h // 2 - th // 2

    # 计算offset使得中心crop对齐到tile origin
    offset_x = (-crop_x0) % tw
    offset_y = (-crop_y0) % th

    # BGR → YCrCb
    ycrcb = cv2.cvtColor(host_bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    y, cr, cb = cv2.split(ycrcb)

    # 循环平铺水印模板到图像尺寸
    tiled_wm = tile_by_phase(wm_template, h, w,
                             offset_y=offset_y, offset_x=offset_x)

    # Cb 通道嵌入
    cb_wm = cb * (1.0 - alpha) + tiled_wm.astype(np.float32) * alpha
    cb_wm = np.clip(cb_wm, 0, 255).astype(np.uint8)

    # 合并回 BGR
    ycrcb_wm = cv2.merge([y.astype(np.uint8), cr.astype(np.uint8), cb_wm])
    return cv2.cvtColor(ycrcb_wm, cv2.COLOR_YCrCb2BGR)


def file_sha256(path):
    """计算文件SHA256"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def visualize_template(wm_template, output_dir, r_list):
    """生成模板可视化图"""
    # 1. 灰度模板
    cv2.imwrite(os.path.join(output_dir, "template_gray.png"), wm_template)

    # 2. 频谱圆环图
    h, w = wm_template.shape
    center_x, center_y = w // 2, h // 2

    # 创建圆环可视化
    ring_vis = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.circle(ring_vis, (center_x, center_y), 3, (255, 255, 255), -1)

    # 绘制水印环（绿色）
    for r in r_list:
        cv2.circle(ring_vis, (center_x, center_y), r, (0, 255, 0), 2)
        cv2.putText(ring_vis, f"r={r}", (center_x + r + 5, center_y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.putText(ring_vis, f"Size: {w}x{h}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imwrite(os.path.join(output_dir, "ring_visualization.png"), ring_vis)

    # 3. 频谱图（FFT）
    template_f32 = wm_template.astype(np.float32)
    fft_result = np.fft.fft2(template_f32)
    fft_shift = np.fft.fftshift(fft_result)
    magnitude = np.log1p(np.abs(fft_shift))
    magnitude = (magnitude / magnitude.max() * 255).astype(np.uint8)

    # 在频谱图上绘制圆环
    spectrum_vis = cv2.cvtColor(magnitude, cv2.COLOR_GRAY2BGR)
    for r in r_list:
        cv2.circle(spectrum_vis, (center_x, center_y), r, (0, 255, 0), 2)

    cv2.imwrite(os.path.join(output_dir, "spectrum_with_rings.png"), spectrum_vis)

    print(f"  模板可视化已保存到: {output_dir}")


def process_directory(input_dir, output_dir, alpha=0.016, num=None):
    """处理整个目录"""
    os.makedirs(output_dir, exist_ok=True)

    # 获取图片列表
    files = sorted([
        f for f in os.listdir(input_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
    ])
    if num is not None:
        files = files[:num]

    if not files:
        print(f"未找到图片: {input_dir}")
        return

    # 读取第一张图片获取尺寸
    first_img = cv2.imread(os.path.join(input_dir, files[0]))
    h, w = first_img.shape[:2]
    min_side = min(h, w)
    target_size = min_side // 2  # 最短边的一半

    print(f"\n图像尺寸: {w}x{h}")
    print(f"最短边: {min_side}")
    print(f"目标模板尺寸: {target_size}x{target_size}")

    # 生成512×512水印模板（与训练一致）
    wm_template_512 = generate_watermark_template(512)
    print(f"原始模板(512×512)范围: [{wm_template_512.min()}, {wm_template_512.max()}]")

    # resize到目标尺寸
    wm_template = cv2.resize(wm_template_512, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    print(f"缩放后模板({target_size}×{target_size})范围: [{wm_template.min()}, {wm_template.max()}]")

    # 生成可视化
    visualize_template(wm_template, output_dir, R_LIST)

    # 处理图片
    bits = get_global_watermark_bits()
    bits_string = "".join(str(b) for b in bits)

    mappings = []
    for fname in tqdm(files, desc="嵌入水印"):
        host_path = os.path.join(input_dir, fname)
        wm_name = fname.rsplit(".", 1)[0] + "_wm.png"
        wm_path = os.path.join(output_dir, wm_name)

        host_bgr = cv2.imread(host_path)
        if host_bgr is None:
            print(f"  [SKIP] {fname}: 读取失败")
            continue

        # 嵌入水印
        wm_bgr = embed_watermark_cb(host_bgr, wm_template, alpha=alpha)
        cv2.imwrite(wm_path, wm_bgr)

        mappings.append({
            "original_file": fname,
            "watermarked_file": wm_name,
            "watermarked_path": wm_path,
            "size": list(host_bgr.shape[:2]),
            "template_size": target_size,
            "alpha_embed": alpha,
            "watermark_bits": bits,
        })

    # 保存GT JSON
    gt_data = {
        "created_at": datetime.now().isoformat(),
        "config": {
            "version": "v17",
            "channel": CHANNEL,
            "alpha_embed": alpha,
            "num_bits": NUM_BITS,
            "r": R_LIST,
            "bitsf": BITS_LIST,
            "template_size": target_size,
            "input_dir": input_dir,
            "output_dir": output_dir,
        },
        "watermark_bits": bits,
        "watermark_bits_string": bits_string,
        "count": len(mappings),
        "mappings": mappings,
    }
    gt_path = os.path.join(output_dir, "embed_gt_mapping.json")
    with open(gt_path, "w", encoding="utf-8") as f:
        json.dump(gt_data, f, indent=2, ensure_ascii=False)

    print(f"\n完成！处理 {len(mappings)} 张图片")
    print(f"  输出目录: {output_dir}")
    print(f"  GT JSON: {gt_path}")


def main():
    parser = argparse.ArgumentParser(description="v17 全图 Cb 通道水印嵌入")
    parser.add_argument("--input_dir", type=str,
                       default="/home/lpl2025/lpl/fftmask/xiaomi_test_data/4096_3072",
                       help="输入图片目录")
    parser.add_argument("--output_dir", type=str,
                       default="/home/lpl2025/lpl/fftmask/output/v17_xiaomi",
                       help="输出目录")
    parser.add_argument("--num", type=int, default=None,
                       help="最多处理图片数（默认全部）")
    parser.add_argument("--alpha", type=float, default=ALPHA_EMBED,
                       help=f"嵌入强度（默认{ALPHA_EMBED}）")
    args = parser.parse_args()

    print("=" * 60)
    print("v17 水印嵌入工具（循环平铺版）")
    print("=" * 60)
    print(f"圆环位置: {R_LIST}")
    print(f"Bits配置: {BITS_LIST} (共{sum(BITS_LIST)}bit)")
    print(f"嵌入强度: {args.alpha}")
    print(f"输入目录: {args.input_dir}")
    print(f"输出目录: {args.output_dir}")
    print("=" * 60)

    process_directory(args.input_dir, args.output_dir, args.alpha, args.num)


if __name__ == "__main__":
    main()
