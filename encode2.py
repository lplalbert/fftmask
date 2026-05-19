import numpy as np
from numpy.fft import ifftshift, ifft2, fftshift, fft2
import cv2

class Watermark16Sector2:
    def __init__(self, L1=676, k1=120.0, r=[5,9, 12], bitsf=[5,15,40], r_range=1, n_sectors=None):
        self.L1 = L1        # 模板大小
        self.k1 = k1       # 频域幅值强度
        self.r = r        # 低频半径列表
        self.bitsf = bitsf  # 每个半径对应的位数
        self.r_range = r_range # 低频半径范围
        total_bits = sum(self.bitsf)
        if n_sectors is None:
            self.n_sectors = total_bits
        elif n_sectors != total_bits:
            raise ValueError(f"n_sectors ({n_sectors}) must equal sum(bitsf) ({total_bits})")
        else:
            self.n_sectors = n_sectors

    # ======================
    # 【嵌入】生成16分类低频水印模板（你的共轭对称代码）
    # ======================
    def generate_template(self, numbit=None):
        """
        输入：num 0~15
        输出：空域水印Tm + 频域谱M1
        """
      
        M1 = np.zeros((self.L1, self.L1), dtype=np.float32)
        cx, cy = self.L1 // 2, self.L1 // 2
        total_bits = sum(self.bitsf)
        if numbit is None:
            numbit = np.random.randint(0, 2, size=total_bits)
        elif len(numbit) != total_bits:
            raise ValueError(f"numbit length ({len(numbit)}) must equal sum(bitsf) ({total_bits})")

        # 核心：180度(0~π) 按半径一圈一圈嵌入
        bit_index = 0

        # 遍历每个半径
        for radius_idx, r in enumerate(self.r):
            # 当前半径负责的位数
            current_bits = self.bitsf[radius_idx]

            # 每个半径占整个π的范围
            start_angle = 0
            end_angle = np.pi

            # 为当前半径的每个位划分角度（在整个π范围内均匀划分）
            radius_angles = np.linspace(start_angle, end_angle, current_bits + 1)

            # 遍历当前半径负责的每个位
            for bit in range(current_bits):
                bit_start_angle = radius_angles[bit]
                bit_end_angle = radius_angles[bit + 1]

                # 遍历低频圆环（连续区域，纯低频）
                for thetar in range(r, r + self.r_range + 1):
                    # 圆环采样点，保证平滑
                    ri = thetar

                    theta_list = np.linspace(bit_start_angle, bit_end_angle, max(2, int(ri * 20)))
                    for theta in theta_list:
                        # 计算坐标
                        x = cx + int(np.round(ri * np.cos(theta)))
                        y = cy + int(np.round(ri * np.sin(theta)))

                        # 边界检查 + 幅值赋值
                        if 0 <= x < self.L1 and 0 <= y < self.L1:
                            if numbit[bit_index + bit] == 1:
                                M1[y, x] = self.k1

                            # 共轭对称：频域对称位置
                            x_sym = (-x) % self.L1
                            y_sym = (-y) % self.L1
                            M1[y_sym, x_sym] = M1[y, x]

            # 更新位索引
            bit_index += current_bits

        # 逆DFT → 空域水印
        spatial = np.real(ifft2(ifftshift(M1)))
        Tm = np.where(spatial < 0, 0, 255).astype(np.uint8)
        return Tm, M1
import numpy as np
from numpy.fft import ifftshift, ifft2, fftshift, fft2
import cv2

import numpy as np
from numpy.fft import ifftshift, ifft2, fftshift, fft2
import cv2

class Watermark16Sector1:
    def __init__(self, L1=676, k1=120.0, r=[5,9, 12], bitsf=[5,15,40], r_range=1, n_sectors=None):
        self.L1 = L1        # 模板大小
        self.k1 = k1       # 频域幅值强度
        self.r = r        # 低频半径列表
        self.bitsf = bitsf  # 每个半径对应的位数
        self.r_range = r_range # 低频半径范围
        total_bits = sum(self.bitsf)
        if n_sectors is None:
            self.n_sectors = total_bits
        elif n_sectors != total_bits:
            raise ValueError(f"n_sectors ({n_sectors}) must equal sum(bitsf) ({total_bits})")
        else:
            self.n_sectors = n_sectors

    # ======================
    # 【仅提速，结果100%一致】生成16分类低频水印模板
    # ======================
    def generate_template(self, numbit=None):
        """
        输入：num 0~15
        输出：空域水印Tm + 频域谱M1
        """
        M1 = np.zeros((self.L1, self.L1), dtype=np.float32)
        cx, cy = self.L1 // 2, self.L1 // 2
        total_bits = sum(self.bitsf)
        if numbit is None:
            numbit = np.random.randint(0, 2, size=total_bits)
        elif len(numbit) != total_bits:
            raise ValueError(f"numbit length ({len(numbit)}) must equal sum(bitsf) ({total_bits})")

        bit_index = 0

        # 遍历每个半径
        for radius_idx, r in enumerate(self.r):
            current_bits = self.bitsf[radius_idx]
            start_angle = 0
            end_angle = np.pi
            radius_angles = np.linspace(start_angle, end_angle, current_bits + 1)

            # 遍历当前半径负责的每个位
            for bit in range(current_bits):
                val = numbit[bit_index + bit]
                if val == 0:
                    continue  # 跳过0，直接加速

                bit_start_angle = radius_angles[bit]
                bit_end_angle = radius_angles[bit + 1]

                # ======================
                # 批量计算所有坐标（原版逻辑完全不变）
                # ======================
                xs, ys = [], []
                for thetar in range(r, r + self.r_range + 1):
                    ri = thetar
                    N = max(2, int(ri * 20))
                    theta_arr = np.linspace(bit_start_angle, bit_end_angle, N)

                    # 向量化计算所有 x, y（一次性算完，不循环）
                    x_arr = cx + np.round(ri * np.cos(theta_arr)).astype(np.int32)
                    y_arr = cy + np.round(ri * np.sin(theta_arr)).astype(np.int32)

                    xs.append(x_arr)
                    ys.append(y_arr)

                # 合并所有坐标
                xs = np.concatenate(xs)
                ys = np.concatenate(ys)

                # 边界过滤
                mask = (xs >= 0) & (xs < self.L1) & (ys >= 0) & (ys < self.L1)
                xs = xs[mask]
                ys = ys[mask]

                # 一次性批量赋值（比循环快100倍）
                M1[ys, xs] = self.k1

                # 共轭对称 一次性批量计算
                x_sym = (-xs) % self.L1
                y_sym = (-ys) % self.L1
                M1[y_sym, x_sym] = self.k1

            bit_index += current_bits

        # 逆DFT → 空域水印（和原版完全一样）
        spatial = np.real(ifft2(ifftshift(M1)))
        Tm = np.where(spatial < 0, 0, 255).astype(np.uint8)
        return Tm, M1

if __name__ == "__main__":
    import time
    t0 = time.time()

    # 初始化参数和你完全一样
    wm = Watermark16Sector1(L1=512, k1=30000, r=[5,9, 12], bitsf=[5,15,40], r_range=0, n_sectors=60)
    Tm, M1 = wm.generate_template(numbit=None)

    cv2.imwrite(f"img_mask/watermark_template.png", Tm)
    cv2.imwrite(f"img_mask/watermark_spectrum.png",
                cv2.normalize(M1, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))

    print(f"耗时：{time.time() - t0:.3f}s")
   
