"""
v17 水印编码器
仅包含水印环，无旋转矫正环

环结构:
- Ring 1: r=8, 20 bits 水印
- Ring 2: r=15, 40 bits 水印
"""
import numpy as np
from numpy.fft import ifftshift, ifft2, fftshift, fft2
import cv2


class WatermarkV17:
    def __init__(self, L1=512, k1=30000, r_watermark=[8, 15], bitsf=[20, 40],
                 r_range=1, n_sectors=60):
        """
        Args:
            L1: 模板大小
            k1: 频域幅值强度
            r_watermark: 水印环半径列表 [r1, r2]
            bitsf: 每个水印环对应的位数 [bits1, bits2]
            r_range: 环宽度
            n_sectors: 水印总位数
        """
        self.L1 = L1
        self.k1 = k1
        self.r_watermark = r_watermark
        self.bitsf = bitsf
        self.r_range = r_range
        self.n_sectors = n_sectors

        # 验证
        total_bits = sum(bitsf)
        if total_bits != n_sectors:
            raise ValueError(f"sum(bitsf)={total_bits} != n_sectors={n_sectors}")

    def generate_watermark_pattern(self, numbit):
        """
        生成水印环的模式
        """
        M1 = np.zeros((self.L1, self.L1), dtype=np.float32)
        cx, cy = self.L1 // 2, self.L1 // 2
        bit_index = 0

        for radius_idx, r in enumerate(self.r_watermark):
            current_bits = self.bitsf[radius_idx]
            start_angle = 0
            end_angle = np.pi
            radius_angles = np.linspace(start_angle, end_angle, current_bits + 1)

            for bit in range(current_bits):
                val = numbit[bit_index + bit]
                if val == 0:
                    continue

                bit_start_angle = radius_angles[bit]
                bit_end_angle = radius_angles[bit + 1]

                xs, ys = [], []
                for thetar in range(r, r + self.r_range + 1):
                    ri = thetar
                    N = max(2, int(ri * 20))
                    theta_arr = np.linspace(bit_start_angle, bit_end_angle, N)
                    x_arr = cx + np.round(ri * np.cos(theta_arr)).astype(np.int32)
                    y_arr = cy + np.round(ri * np.sin(theta_arr)).astype(np.int32)
                    xs.append(x_arr)
                    ys.append(y_arr)

                xs = np.concatenate(xs)
                ys = np.concatenate(ys)
                mask = (xs >= 0) & (xs < self.L1) & (ys >= 0) & (ys < self.L1)
                xs = xs[mask]
                ys = ys[mask]

                M1[ys, xs] = self.k1
                x_sym = (-xs) % self.L1
                y_sym = (-ys) % self.L1
                M1[y_sym, x_sym] = self.k1

            bit_index += current_bits

        return M1

    def generate_template(self, numbit=None):
        """
        生成完整的水印模板 (仅水印环)

        Args:
            numbit: 水印位数组，如果为None则随机生成

        Returns:
            Tm: 空域水印模板
            M1: 频域模板
            numbit: 水印位数组
        """
        if numbit is None:
            numbit = np.random.randint(0, 2, size=self.n_sectors)
        elif len(numbit) != self.n_sectors:
            raise ValueError(f"numbit length ({len(numbit)}) must equal n_sectors ({self.n_sectors})")

        # 生成水印环
        M1 = self.generate_watermark_pattern(numbit)

        # 逆DFT -> 空域
        spatial = np.real(ifft2(ifftshift(M1)))
        Tm = np.where(spatial < 0, 0, 255).astype(np.uint8)

        return Tm, M1, numbit


if __name__ == "__main__":
    import os

    wm = WatermarkV17(
        L1=512,
        k1=30000,
        r_watermark=[8, 15],
        bitsf=[20, 40],
        r_range=1,
        n_sectors=60
    )

    os.makedirs("img_encode_v17", exist_ok=True)

    for i in range(3):
        Tm, M1, numbit = wm.generate_template()
        cv2.imwrite(f"img_encode_v17/watermark_template_{i}.png", Tm)
        cv2.imwrite(f"img_encode_v17/watermark_spectrum_{i}.png",
                    cv2.normalize(M1, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
        print(f"Generated template {i}, bits: {numbit[:10]}...")