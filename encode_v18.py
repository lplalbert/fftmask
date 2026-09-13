"""
v18 水印编码器 — 镂空模板
在 v17 基础上增加镂空模板生成

环结构 (与 v17 相同):
- Ring 1: r=8, 20 bits
- Ring 2: r=15, 40 bits

镂空原理:
- 先生成二值模板 Tm_binary (0/255)
- 对白色区域(环)随机挖掉 hollow_ratio 比例的像素变为0
- 黑色区域(背景)保持0不变
- 镂空模板值只有两种: 0 和 255
"""
import numpy as np
from numpy.fft import ifftshift, ifft2, fftshift, fft2
import cv2
from encode_v17 import WatermarkV17


class WatermarkV18(WatermarkV17):
    """
    v18 水印编码器
    继承 v17，新增镂空模板生成
    """
    def __init__(self, L1=512, k1=30000, r_watermark=[12, 25], bitsf=[15, 45],
                 r_range=1, n_sectors=60,
                 M_w=255, M_b=0, hollow_ratio=0.3):
        """
        Args:
            L1: 模板大小
            k1: 频域幅值强度
            r_watermark: 水印环半径列表
            bitsf: 每个环的位数
            r_range: 环宽度
            n_sectors: 水印总位数
            M_w: 水印高值区域亮度 (白色区域保留时的值)
            M_b: 水印低值区域亮度 (背景/镂空区域的值)
            hollow_ratio: 镂空比例 (0-1)，白色区域被挖掉的比例
        """
        super().__init__(L1, k1, r_watermark, bitsf, r_range, n_sectors)
        self.M_w = M_w
        self.M_b = M_b
        self.hollow_ratio = hollow_ratio

    def generate_hollow_template(self, Tm_binary):
        """
        从二值模板生成镂空模板

        流程:
        1. 复制二值模板 (背景=0, 环=255)
        2. 对白色区域(环)：随机挖掉 hollow_ratio 比例的像素变为0
        3. 黑色区域(背景)保持0不变

        Args:
            Tm_binary: 二值模板 (0/255), uint8

        Returns:
            Tm_hollow: 镂空模板, uint8, 值只有 0 和 255
        """
        Tm_hollow = Tm_binary.copy()
        white_mask = Tm_binary > 127

        if self.hollow_ratio > 0:
            # 对白色像素，随机挖掉 hollow_ratio 比例
            hollow_mask = np.random.rand(*Tm_binary.shape) < self.hollow_ratio
            Tm_hollow[white_mask & hollow_mask] = 0

        return Tm_hollow

    def generate_template(self, numbit=None, hollow=True):
        """
        生成水印模板

        Args:
            numbit: 水印位数组
            hollow: 是否生成镂空模板 (True) 还是返回二值模板 (False)

        Returns:
            Tm: 空域模板 (镂空或二值)
            M1: 频域模板
            numbit: 水印位数组
        """
        # 用父类方法生成二值模板
        Tm_binary, M1, numbit = super().generate_template(numbit)

        if hollow:
            Tm = self.generate_hollow_template(Tm_binary)
        else:
            Tm = Tm_binary

        return Tm, M1, numbit


if __name__ == "__main__":
    import os

    wm = WatermarkV18(
        L1=512, k1=30000,
        r_watermark=[8, 15], bitsf=[20, 40],
        r_range=1, n_sectors=60,
        M_w=255, M_b=0, hollow_ratio=0.3
    )

    os.makedirs("img_encode_v18", exist_ok=True)

    for i in range(3):
        # 二值模板
        Tm_bin, M1, numbit = wm.generate_template(hollow=False)
        cv2.imwrite(f"img_encode_v18/binary_{i}.png", Tm_bin)

        # 镂空模板
        Tm_hollow, _, _ = wm.generate_template(numbit, hollow=True)
        cv2.imwrite(f"img_encode_v18/hollow_{i}.png", Tm_hollow)

        print(f"Template {i}: binary range=[{Tm_bin.min()}, {Tm_bin.max()}], "
              f"hollow range=[{Tm_hollow.min()}, {Tm_hollow.max()}], "
              f"bits={numbit[:10]}...")