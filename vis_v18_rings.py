"""
v18 可视化：频谱圆环、模板、加水印后的图像
"""
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from encode_v11 import WatermarkV11

def visualize_v18(config_name, ring_positions, bits_per_ring, num_bits_total, test_img_path=None):
    """可视化指定v18配置"""

    # 创建水印系统（无旋转环）
    wm = WatermarkV11(
        L1=512,
        k1=30000,
        r_watermark=ring_positions,
        bitsf=bits_per_ring,
        r_rotation=None,  # 无旋转环
        r_range=1,
        n_sectors=num_bits_total
    )

    # 随机生成水印bits
    np.random.seed(42)
    watermark_bits = np.random.randint(0, 2, size=num_bits_total)

    # 生成模板
    Tm, M1, _ = wm.generate_template(numbit=watermark_bits)

    # 读取测试图像
    if test_img_path is None:
        test_img_path = "/data/xsj/dataset/coco/mini_coco/train/000000043997.jpg"
    image = cv2.imread(test_img_path)
    image = cv2.resize(image, (512, 512))

    # 嵌入水印
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    y_ch, cr_ch, cb_ch = cv2.split(ycrcb)
    Tm_gray = cv2.cvtColor(Tm, cv2.COLOR_BGR2GRAY).astype(np.float32) if len(Tm.shape) == 3 else Tm.astype(np.float32)
    alpha = 0.016
    cb_wm = cb_ch * (1 - alpha) + Tm_gray * alpha
    cb_wm = np.clip(cb_wm, 0, 255).astype(np.uint8)
    ycrcb_wm = cv2.merge([y_ch.astype(np.uint8), cr_ch.astype(np.uint8), cb_wm])
    watermarked = cv2.cvtColor(ycrcb_wm, cv2.COLOR_YCrCb2BGR)

    # 计算频谱
    def get_spectrum(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude = np.log1p(np.abs(fshift))
        return magnitude

    spec_orig = get_spectrum(image)
    spec_wm = get_spectrum(watermarked)

    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f'v18 {config_name} - Rings: {ring_positions}, Bits: {bits_per_ring} (Total: {num_bits_total}bit)',
                 fontsize=16, fontweight='bold')

    # 频谱圆环图
    ax = axes[0, 0]
    h, w = spec_orig.shape
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    R = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)

    ring_vis = np.zeros_like(spec_orig)
    colors = plt.cm.Set1(np.linspace(0, 1, len(ring_positions)))
    for i, r in enumerate(ring_positions):
        mask = (R >= r - 2) & (R <= r + 10)
        ring_vis[mask] = i + 1

    ax.imshow(ring_vis, cmap='Set1', vmin=0, vmax=len(ring_positions))
    ax.set_title('Ring Sampling Regions')
    for i, r in enumerate(ring_positions):
        ax.text(10, 20 + i * 25, f'Ring{i}: r={r}', color='white', fontsize=10,
                bbox=dict(boxstyle='round', facecolor=colors[i][:3], alpha=0.8))
    ax.axis('off')

    # 模板图
    ax = axes[0, 1]
    Tm_display = cv2.cvtColor(Tm, cv2.COLOR_BGR2RGB) if len(Tm.shape) == 3 else Tm
    ax.imshow(Tm_display, cmap='gray')
    ax.set_title('Template (Tm)')
    ax.axis('off')

    # 原图
    ax = axes[1, 0]
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax.set_title('Original Image')
    ax.axis('off')

    # 加水印后的图
    ax = axes[1, 1]
    ax.imshow(cv2.cvtColor(watermarked, cv2.COLOR_BGR2RGB))
    ax.set_title('Watermarked Image')
    ax.axis('off')

    plt.tight_layout()
    output_path = f'/data/lpl/fftmask/output/v18_{config_name}_visualization.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")
    return output_path


if __name__ == '__main__':
    # 4环配置
    visualize_v18(
        config_name='4rings',
        ring_positions=[8, 13, 18, 23],
        bits_per_ring=[20, 20, 20, 20],
        num_bits_total=80
    )

    # 5环配置
    visualize_v18(
        config_name='5rings',
        ring_positions=[8, 13, 18, 23, 28],
        bits_per_ring=[20, 20, 20, 20, 20],
        num_bits_total=100
    )
