"""
v11 水印编码器
在两个水印环之间加一个旋转矫正环(r=18)，用于检测图像旋转角度

环结构:
- Ring 1: r=12, 15 bits 水印
- Ring 2: r=18, 旋转矫正环 (固定正弦波模式)
- Ring 3: r=25, 45 bits 水印
"""
import numpy as np
from numpy.fft import ifftshift, ifft2, fftshift, fft2
import cv2


class WatermarkV11:
    def __init__(self, L1=512, k1=30000, r_watermark=[12, 25], bitsf=[15, 45],
                 r_rotation=18, rotation_cycles=8, r_range=1, n_sectors=60):
        """
        Args:
            L1: 模板大小
            k1: 频域幅值强度
            r_watermark: 水印环半径列表 [r1, r2]
            bitsf: 每个水印环对应的位数 [bits1, bits2]
            r_rotation: 旋转矫正环半径
            rotation_cycles: 旋转矫正环的正弦波周期数
            r_range: 环宽度
            n_sectors: 水印总位数
        """
        self.L1 = L1
        self.k1 = k1
        self.r_watermark = r_watermark
        self.bitsf = bitsf
        self.r_rotation = r_rotation
        self.rotation_cycles = rotation_cycles
        self.r_range = r_range
        self.n_sectors = n_sectors

        # 验证
        total_bits = sum(bitsf)
        if total_bits != n_sectors:
            raise ValueError(f"sum(bitsf)={total_bits} != n_sectors={n_sectors}")

    def generate_rotation_pattern(self):
        """
        生成旋转矫正环的固定模式
        使用正弦波，旋转角度会导致相位偏移
        """
        pattern = np.zeros((self.L1, self.L1), dtype=np.float32)
        cx, cy = self.L1 // 2, self.L1 // 2

        for thetar in range(self.r_rotation, self.r_rotation + self.r_range + 1):
            ri = thetar
            # 采样点数
            N = max(2, int(ri * 20))
            theta_arr = np.linspace(0, 2 * np.pi, N, endpoint=False)

            # 正弦波模式：rotation_cycles个周期
            # 值在 [-1, 1] 之间
            sin_values = np.cos(self.rotation_cycles * theta_arr)

            # 坐标
            x_arr = cx + np.round(ri * np.cos(theta_arr)).astype(np.int32)
            y_arr = cy + np.round(ri * np.sin(theta_arr)).astype(np.int32)

            # 边界过滤
            mask = (x_arr >= 0) & (x_arr < self.L1) & (y_arr >= 0) & (y_arr < self.L1)
            x_arr = x_arr[mask]
            y_arr = y_arr[mask]
            sin_values = sin_values[mask]

            # 赋值 (映射到 [0, k1])
            pattern[y_arr, x_arr] = self.k1 * (sin_values + 1) / 2

            # 共轭对称
            x_sym = (-x_arr) % self.L1
            y_sym = (-y_arr) % self.L1
            pattern[y_sym, x_sym] = pattern[y_arr, x_arr]

        return pattern

    def generate_watermark_pattern(self, numbit):
        """
        生成水印环的模式 (和encode2.py相同)
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
        生成完整的水印模板 (水印环 + 旋转矫正环)

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

        # 生成旋转矫正环
        rotation_pattern = self.generate_rotation_pattern()

        # 合并
        M1 = M1 + rotation_pattern

        # 逆DFT -> 空域
        spatial = np.real(ifft2(ifftshift(M1)))
        Tm = np.where(spatial < 0, 0, 255).astype(np.uint8)

        return Tm, M1, numbit

    def get_rotation_ring_polar(self, fft_mag, output_shape=(12, 180)):
        """
        从FFT幅度谱中提取旋转矫正环的极坐标表示

        Args:
            fft_mag: FFT幅度谱 (H, W)
            output_shape: (R_bins, T_bins)

        Returns:
            polar: 极坐标表示 (R_bins, T_bins)
        """
        import torch
        import torch.nn.functional as F

        R_bins, T_bins = output_shape
        H, W = fft_mag.shape

        rho = torch.linspace(self.r_rotation - self.r_range, self.r_rotation + self.r_range, R_bins)
        theta = torch.linspace(0, np.pi, T_bins)

        grid_rho, grid_theta = torch.meshgrid(rho, theta, indexing='ij')
        grid_x = grid_rho * torch.cos(grid_theta) / (W / 2)
        grid_y = grid_rho * torch.sin(grid_theta) / (H / 2)

        grid = torch.stack([grid_y, grid_x], dim=-1).unsqueeze(0)
        fft_tensor = torch.from_numpy(fft_mag).unsqueeze(0).unsqueeze(0).float()
        polar = F.grid_sample(fft_tensor, grid, mode='bilinear', align_corners=True)

        return polar.squeeze().numpy()

    def detect_rotation_angle(self, fft_mag):
        """
        检测旋转角度

        Args:
            fft_mag: FFT幅度谱 (H, W)

        Returns:
            angle_deg: 检测到的旋转角度 (度)
        """
        # 提取旋转环的极坐标表示
        polar = self.get_rotation_ring_polar(fft_mag, output_shape=(4, 360))

        # 沿半径方向取均值
        polar_1d = polar.mean(axis=0)  # (360,)

        # 生成参考模式
        theta_ref = np.linspace(0, 2 * np.pi, 360, endpoint=False)
        ref_pattern = np.cos(self.rotation_cycles * theta_ref)

        # 互相关检测偏移
        from scipy.signal import correlate
        corr = correlate(polar_1d, ref_pattern, mode='full')
        center = len(corr) // 2

        # 找到最大相关值的位置
        # 只搜索 [0, 360) 范围
        search_range = 360
        corr_search = corr[center - search_range // 2: center + search_range // 2]
        max_idx = np.argmax(corr_search)

        # 转换为角度
        angle_deg = (max_idx - search_range // 2) * (180 / 360)

        return angle_deg


if __name__ == "__main__":
    import os

    wm = WatermarkV11(
        L1=512,
        k1=30000,
        r_watermark=[12, 25],
        bitsf=[15, 45],
        r_rotation=18,
        rotation_cycles=8,
        r_range=1,
        n_sectors=60
    )

    os.makedirs("img_encode_v11", exist_ok=True)

    for i in range(3):
        Tm, M1, numbit = wm.generate_template()
        cv2.imwrite(f"img_encode_v11/watermark_template_{i}.png", Tm)
        cv2.imwrite(f"img_encode_v11/watermark_spectrum_{i}.png",
                    cv2.normalize(M1, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
        print(f"Generated template {i}, bits: {numbit[:10]}...")
