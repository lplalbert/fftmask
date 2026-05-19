"""
CWT Watermark Method - 基于DFT域的水印嵌入和提取
从 single_block.py 抽取核心功能，支持命令行操作

功能：
- encode: 将8字符字符串转换为64位二进制水印并嵌入图像
- decode: 从图像中提取水印并验证（需要提供原始消息作为GT）
"""

import numpy as np
import cv2
from scipy.signal import wiener
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from typing import Tuple, Optional
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import argparse
import os
import sys


class CWTWatermark:
    """
    CWT Watermark System - 基于DFT域的水印系统
    支持64位水印的嵌入和提取
    """

    def __init__(self, block_size: int = 512):
        """
        初始化水印系统

        Args:
            block_size: 图像块大小，默认512x512
        """
        self.L1 = block_size  # 消息水印模板大小
        # self.R1 = np.array([132, 145, 159, 172])  # 嵌入半径
        self.R1 = np.array([25, 40, 55, 70])  # 【低频核心】小半径 = 频域中心低频
        self.k1 = 20000  # 消息水印嵌入幅度
        self.alpha = 0.35  # 嵌入强度因子
        self.gamma = 0.23  # JND参数
        self.k3 = 1.5  # 提取阈值参数

        # 5x5 Sobel算子
        self.sob1 = np.array([
            [-1, 0, 0, 0, 1],
            [-1, 0, 0, 0, 1],
            [-np.sqrt(2), 0, 0, 0, np.sqrt(2)],
            [-1, 0, 0, 0, 1],
            [-1, 0, 0, 0, 1]
        ])

        self.sob2 = np.array([
            [-1, -1, -np.sqrt(2), -1, -1],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [1, 1, np.sqrt(2), 1, 1]
        ])

        # 5x5加权低通滤波器
        self.h = np.array([
            [1, 1, 1, 1, 1],
            [1, 2, 2, 2, 1],
            [1, 2, 0, 2, 1],
            [1, 2, 2, 2, 1],
            [1, 1, 1, 1, 1]
        ]) / 32.0

    def string_to_bits(self, message: str) -> np.ndarray:
        """
        将8字符字符串转换为64位二进制数组

        Args:
            message: 8字符的字符串

        Returns:
            bits: 64位二进制数组
        """
        if len(message) != 8:
            raise ValueError(f"Message must be exactly 8 characters, got {len(message)}")

        bits = []
        for char in message:
            # 将字符转换为ASCII码，然后转为8位二进制
            ascii_val = ord(char)
            for i in range(7, -1, -1):  # 从高位到低位
                bits.append((ascii_val >> i) & 1)

        return np.array(bits, dtype=np.int32)

    def bits_to_string(self, bits: np.ndarray) -> str:
        """
        将64位二进制数组转换为8字符字符串

        Args:
            bits: 64位二进制数组

        Returns:
            message: 8字符的字符串
        """
        if len(bits) != 64:
            raise ValueError(f"Bits must be exactly 64 bits, got {len(bits)}")

        message = ""
        for i in range(8):
            # 每8位组成一个字符
            char_bits = bits[i*8:(i+1)*8]
            ascii_val = 0
            for j, bit in enumerate(char_bits):
                ascii_val = (ascii_val << 1) | int(bit)
            # 只取有效的ASCII字符（32-126可打印字符）
            if 32 <= ascii_val <= 126:
                message += chr(ascii_val)
            else:
                message += '?'  # 无效字符用?代替

        return message

    def generate_message_watermark_template(self, watermark_bits: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成64位水印模板

        Args:
            watermark_bits: 64位水印序列

        Returns:
            Tm: 水印模板 (L1 x L1)
            M1_shifted: DFT幅度谱（用于可视化）
        """
        if len(watermark_bits) != 64:
            raise ValueError(f"Watermark bits must be 64, current: {len(watermark_bits)}")

        # Step 1: 创建DFT幅度谱零矩阵
        M1 = np.zeros((self.L1, self.L1))

        # Step 2: 在指定半径位置嵌入64位水印
        center = self.L1 / 2
        l = 64  # 固定64位

        for j in range(l):
            # 计算嵌入位置（论文公式12）
            theta = j / l * np.pi
            radius_idx = j % len(self.R1)
            R = self.R1[radius_idx]
            
               
            xi = int(center + 1 + np.floor(R * np.cos(theta)))
            yi = int(center + 1 + np.floor(R * np.sin(theta)))

            # 边界检查
            if 0 <= xi < self.L1 and 0 <= yi < self.L1:
                # 嵌入系数（论文公式13）
                if watermark_bits[j] == 1:
                    M1[xi, yi] = self.k1
                    # DFT幅度谱关于原点对称，需要设置对称位置
                    xi_sym = self.L1 - xi
                    yi_sym = self.L1 - yi
                    if 0 <= xi_sym < self.L1 and 0 <= yi_sym < self.L1:
                        M1[xi_sym, yi_sym] = self.k1
                else:
                    M1[xi, yi] = 0

        # Step 3: 逆DFT
        complex_spectrum = M1.astype(np.complex128)
        M1_shifted = ifftshift(complex_spectrum)
        spatial_domain = ifft2(M1_shifted)
        spatial_real = np.real(spatial_domain)

        # Step 4: 二值化
        Tm = np.where(spatial_real < 0, 0, 255).astype(np.uint8)

        return Tm, np.abs(M1)
    def generate_message_watermark_template2(self,
        W1: np.ndarray = None ):  # 二进制水印
        """
        【低频版】生成平滑的信息水印模板Tm
        改动：离散点 → 中心连续环形条带（低频），空域输出平滑无噪点
        """
        # 1. 初始化二进制序列
        if W1 is None:
            l = 64
            W1 = np.random.randint(0, 2, size=l)
        else:
            l = len(W1)
            assert W1.ndim == 1 and np.isin(W1, [0, 1]).all(), "W1必须是0/1一维数组"
        # print(f"Generated binary watermark sequence W1 (length={l}):\n{W1}")
        # 2. 初始化频域矩阵 + 中心坐标
        M1 = np.zeros((self.L1, self.L1))
        cx, cy = self.L1 /2, self.L1 / 2  # 频域中心坐标

        # 3. 【核心改动】低频环形条带嵌入（不是单点！）
        ring_width = 2  # 环宽度，越大空域越平滑
        theta_width = 0
        for j in range(l):
            theta = j / l * np.pi
            r = self.R1[j % len(self.R1)]
            bit = W1[j]

            # 遍历环宽度，生成连续条带（关键：填充区域，不是单点）
            for dr in range(-ring_width, ring_width + 1):
                current_r = r + dr
                if current_r <= 0:
                    continue
                num_steps = 8  # 环上采样点数，越多空域越平滑 
                for t_step in np.linspace(-theta_width, theta_width, num_steps):
                    theta_j = theta + t_step
                # 计算当前点坐标
                    x = int(cx +1+ np.round(current_r * np.cos(theta_j)))
                    y = int(cy +1+ np.round(current_r * np.sin(theta_j)))

                    # 边界检查
                    if 0 <= x < self.L1 and 0 <= y < self.L1:
                        if bit == 1:
                            M1[x, y] = self.k1 

                            # 【共轭对称】保证空域是实信号+平滑
                            x_sym = self.L1 - x
                            y_sym = self.L1 - y
                            if 0 <= x_sym < self.L1 and 0 <= y_sym < self.L1:
                                M1[x_sym, y_sym] = M1[x, y]
                        else:
                            M1[x, y] = 0
                        

        # 4. 频域移位 + 逆DFT（标准流程）
        complex_spectrum = M1.astype(np.complex128)
        freq_shifted = ifftshift(complex_spectrum)
        spatial_domain = ifft2(freq_shifted)
        spatial_real = np.real(spatial_domain)

        # 5. 二值化生成平滑水印
        Tm = np.where(spatial_real < 0, 0, 255).astype(np.uint8)

        return Tm,  np.abs(M1)

    def calculate_jnd(self, bg: np.ndarray) -> np.ndarray:
        """
        计算简化的JND模型

        Args:
            bg: 灰度图像

        Returns:
            JND: JND阈值图
        """
        h, w = bg.shape

        # Step 1: 计算平均背景亮度 Bg
        Bg = cv2.filter2D(bg.astype(np.float32), -1, self.h)

        # Step 2: 计算亮度掩蔽阈值 PL
        PL = np.zeros_like(Bg)
        mask_dark = Bg < 127
        mask_bright = ~mask_dark

        PL[mask_dark] = 17 * (1 - np.sqrt(Bg[mask_dark]) / 127) + 3
        PL[mask_bright] = 3 * ((Bg[mask_bright] - 127) / 128) + 3

        # Step 3: 计算图像梯度 G
        G1 = cv2.filter2D(bg.astype(np.float32), -1, self.sob1)
        G2 = cv2.filter2D(bg.astype(np.float32), -1, self.sob2)
        G = np.sqrt(G1**2 + G2**2)

        # Step 4: 计算对比度掩蔽阈值 Pc
        lamda = 0.0001 * Bg + 0.115
        k = 0.5 - 0.01 * Bg
        Pc = G * self.gamma * lamda + k

        # Step 5: 计算JND
        JND = PL + Pc - 0.3 * np.minimum(PL, Pc)

        return JND

    def embed(self, host_image: np.ndarray, message: str) -> Tuple[np.ndarray, float, float]:
        """
        嵌入水印到图像（Y通道）

        Args:
            host_image: 宿主图像（BGR格式，512x512）
            message: 8字符的水印消息

        Returns:
            watermarked_image: 含水印图像（BGR）
            psnr: PSNR值（RGB图像）
            ssim: SSIM值（RGB图像）
        """
        print(f"Embedding watermark: '{message}'")

        # 将消息转换为64位二进制
        watermark_bits = self.string_to_bits(message)
        print(f"Watermark bits: {watermark_bits}")

        # 生成水印模板
        Tm, m1 = self.generate_message_watermark_template2(watermark_bits)
        cv2.imwrite("message_watermark_template.png", Tm)  # 保存水印模板图像以供调试
        cv2.imwrite("message_watermark_spectrum.png", cv2.normalize(m1, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))  # 保存水印频谱图像以供调试

        # 确保图像是512x512
        if host_image.shape[:2] != (self.L1, self.L1):
            host_image = cv2.resize(host_image, (self.L1, self.L1))

        # 进行clip和类型转换
        host_image = np.clip(host_image, 0, 255)
        host_image = host_image.astype(np.uint8)

        # 转换为BGR
        if len(host_image.shape) == 2:
            host_bgr = cv2.cvtColor(host_image, cv2.COLOR_GRAY2BGR)
        else:
            host_bgr = host_image.copy()

        # 转换为YUV
        host_yuv = cv2.cvtColor(host_bgr, cv2.COLOR_BGR2YUV)
        host_y = host_yuv[:, :, 0].astype(np.float32)  # 提取Y通道

        # 转换Tm为float
        Tm_float = Tm.astype(np.float32) / 255.0

        # 计算JND
        JND = self.calculate_jnd(host_y)

        # 水印叠加（公式18）在Y通道
        I_lum = host_y + Tm_float * JND * self.alpha

        # 限制到[0, 255]
        I_lum = np.clip(I_lum, 0, 255)
        wm_y = I_lum.astype(np.uint8)

        # 替换YUV中的Y通道
        watermarked_yuv = host_yuv.copy()
        watermarked_yuv[:, :, 0] = wm_y

        # 转换回BGR
        watermarked_image = cv2.cvtColor(watermarked_yuv, cv2.COLOR_YUV2BGR)

        # 确保最终图像在正确范围内
        watermarked_image = np.clip(watermarked_image, 0, 255)
        watermarked_image = watermarked_image.astype(np.uint8)

        # 计算PSNR和SSIM（RGB图像）
        host_rgb = cv2.cvtColor(host_bgr, cv2.COLOR_BGR2RGB)
        watermarked_rgb = cv2.cvtColor(watermarked_image, cv2.COLOR_BGR2RGB)

        psnr = peak_signal_noise_ratio(host_rgb, watermarked_rgb, data_range=255)
        ssim = structural_similarity(host_rgb, watermarked_rgb, data_range=255, channel_axis=2)

        print(f"Embedding completed, PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}")

        return watermarked_image, psnr, ssim

    def extract(self, watermarked_image: np.ndarray, gt_message: Optional[str] = None) -> Tuple[str, float, np.ndarray]:
        """
        从图像中提取水印（Y通道）

        Args:
            watermarked_image: 含水印图像（BGR，512x512）
            gt_message: 原始消息（用于计算准确率）

        Returns:
            extracted_message: 提取的消息字符串
            accuracy: 解码准确率（如果提供了GT消息）
            extracted_bits: 提取的64位二进制数组
        """
        print("Extracting watermark...")

        # 确保尺寸
        if watermarked_image.shape[:2] != (self.L1, self.L1):
            watermarked_image = cv2.resize(watermarked_image, (self.L1, self.L1))

        # 提取Y通道
        print(watermarked_image.shape)
        if len(watermarked_image.shape) == 3:
            watermarked_yuv = cv2.cvtColor(watermarked_image, cv2.COLOR_BGR2YUV)
            Ic = watermarked_yuv[:, :, 0].astype(np.float32)  # Y通道
        else:
            Ic = watermarked_image.astype(np.float32)

        # 使用Wiener滤波器
        I_estimated = wiener(Ic, mysize=(5, 5)).astype(np.float32)

        # 提取水印信号（残差）
        I_n = Ic - I_estimated
        # I_n=    Ic
        cv2.imwrite("extracted_luminance.png", I_n.astype(np.uint8))  # 保存提取的亮度图像以供调试

        # 计算DFT幅度谱
        dft_result = fft2(I_n)
        dft_mag = np.abs(fftshift(dft_result))
        cv2.imwrite("extracted_magnitude2.png", cv2.normalize(dft_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))  # 保存DFT幅度谱图像以供调试
        # 提取64位水印
        extracted_bits = self._extract_bits_from_dft2(dft_mag)
        print(f"Extracted bits: {extracted_bits}")
        # 转换为字符串
        extracted_message = self.bits_to_string(extracted_bits)

        # 计算准确率（如果提供了GT消息）
        accuracy = 0.0
        if gt_message is not None:
            if len(gt_message) != 8:
                print(f"Warning: GT message must be 8 characters, got {len(gt_message)}")
            else:
                gt_bits = self.string_to_bits(gt_message)
                if len(extracted_bits) == 64 and len(gt_bits) == 64:
                    correct = np.sum(gt_bits == extracted_bits)
                    accuracy = correct / 64.0
                    print(f"Extraction completed, Accuracy: {accuracy*100:.2f}% ({correct}/64)")
                else:
                    print("Warning: Bit count mismatch, cannot calculate accuracy")
        else:
            print(f"Extraction completed, Extracted message: '{extracted_message}'")

        return extracted_message, accuracy, extracted_bits

    def _extract_bits_from_dft(self, dft_mag: np.ndarray) -> np.ndarray:
        """
        从DFT幅度谱中提取64位水印
        """
        center = self.L1 / 2
        l = 64  # 固定64位

        # 收集所有幅度值用于统计
        all_magnitudes = []
        for R in self.R1:
            for angle in np.linspace(0, np.pi, 360):
                r = int(R)
                for delta_r in range(-2, 3):
                    dr= int(delta_r+r)
                    theta = angle
                    x = int(center + 1 + dr * np.cos(theta))
                    y = int(center + 1 + dr * np.sin(theta))
                    if 0 <= x < self.L1 and 0 <= y < self.L1:
                        all_magnitudes.append(dft_mag[x, y])

        if not all_magnitudes:
            return np.zeros(64, dtype=np.int32)

        # 计算统计量
        mu_T = np.mean(all_magnitudes)
        sigma_T = np.std(all_magnitudes)

        # 阈值
        threshold = mu_T + self.k3 * sigma_T

        # 提取64位水印
        bits = []
        
        ##jiema
        W1_extract = np.zeros(l, dtype=np.uint8)
        angles = np.linspace(0, np.pi, l)
        ring_width = 2  # 与嵌入时相同的环宽度
        for j in range(l):
            theta = angles[j]
            r = self.R1[j % len(self.R1)]
            # 收集当前环所有点的幅值
            ring_values = []
            for dr in range(-ring_width, ring_width+1):
                cr = r + dr
                if cr <= 0:
                    continue
                x =int(center +1+ int(np.round(cr * np.cos(theta))))
                y = int(center +1+ int(np.round(cr * np.sin(theta))))
                if 0<=x<self.L1 and 0<=y<self.L1:
                    ring_values.append(dft_mag[y, x])

            # 4. 平均幅值判0/1
            if len(ring_values) > 0:
                avg_val = np.mean(ring_values)
                W1_extract[j] = 1 if avg_val >= threshold else 0
            else:
                W1_extract[j] = 0

        return W1_extract

        # for j in range(l):
        #     theta = j / l * np.pi
        #     radius_idx = j % len(self.R1)
        #     R = self.R1[radius_idx]

        #     x0 = int(center + 1 + np.floor(R * np.cos(theta)))
        #     y0 = int(center + 1 + np.floor(R * np.sin(theta)))

        #     max_magnitude = 0
        #     search_radius = 3  # 7x7搜索窗口
        #     for dx in range(-search_radius, search_radius + 1):
        #         for dy in range(-search_radius, search_radius + 1):
        #             x = x0 + dx
        #             y = y0 + dy
        #             if 0 <= x < self.L1 and 0 <= y < self.L1:
        #                 max_magnitude = max(max_magnitude, dft_mag[x, y])

        #     # 决定位值
        #     bit = 1 if max_magnitude >= threshold else 0
        #     bits.append(bit)

        # return np.array(bits, dtype=np.int32)
    def _extract_bits_from_dft2(self, dft_mag: np.ndarray) -> np.ndarray:
        """
        从DFT幅度谱中提取64位水印
        """
        center = self.L1 / 2
        l = 64  # 固定64位

        # 收集所有幅度值用于统计
        all_magnitudes_r = []
        
        for R in self.R1:
            all_magnitudes = []
            # for angle in np.linspace(0, np.pi, 360):
            for angle in np.linspace(0, np.pi, 360):
                r = int(R)
                ind=8
                for delta_r in range(-ind, ind + 1):
                    r= int(r + delta_r)
                    # for delta_theta in range(-5, 6):
                    # theta = angle + delta_theta * np.pi / 180
                    theta = angle
                    x = int(center + 1 + r * np.cos(theta))
                    y = int(center  + 1 + r* np.sin(theta))
                    if 0 <= x < self.L1 and 0 <= y < self.L1:
                        all_magnitudes.append(dft_mag[y, x])

            if not all_magnitudes:
                return np.zeros(64, dtype=np.int32)
            mu_T_r = np.mean(all_magnitudes)
            sigma_T_r = np.std(all_magnitudes)
            all_magnitudes_r.append((mu_T_r, sigma_T_r))

        # 计算统计量
        # valid_mag = dft_mag[dft_mag > 0]
        # mu_T = np.mean(valid_mag)
        # sigma_T = np.std(valid_mag)
        # threshold = mu_T + self.k3 * sigma_T
        # print(f"Calculated threshold parameters: mu_T={mu_T:.2f}, sigma_T={sigma_T:.2f}")
        # mu_T = np.mean(all_magnitudes)
        # sigma_T = np.std(all_magnitudes)
        # print(f"Calculated threshold parameters from collected magnitudes: mu_T={mu_T:.2f}, sigma_T={sigma_T:.2f}")
        # 阈值
        threshold_r=[]
        for mu_T_r, sigma_T_r in all_magnitudes_r:
            threshold_r.append(mu_T_r + self.k3 * sigma_T_r)
        print(f"Calculated threshold parameters for each radius: {threshold_r}")
        # threshold = (mu_T + self.k3 * sigma_T)

        # 提取64位水印
        bits = []
        search_angle_range = 0.06  # 搜索的角度弧度范围
        search_r_range =3     # 搜索的半径偏差范围

        for j in range(l):
            theta_base = j / l * np.pi
            R_base = self.R1[j % len(self.R1)]
            threshold= threshold_r[j % len(threshold_r)]
            max_v_j = 0
            ring_values = []
            # 在极坐标定义的“条带扇区”内寻找最大能量点
            # r 方向搜索
            for dr in range(-search_r_range, search_r_range + 1):
                curr_r = R_base + dr
                
                # theta 方向搜索（离散化采样点，确保覆盖整个条带）
                num_search_steps = 8
                for dt in np.linspace(-search_angle_range, search_angle_range, num_search_steps):
                    curr_theta = theta_base + dt
                    
                    # 坐标转换
                    xi = int(np.round(center + curr_r * np.cos(curr_theta)))
                    yi = int(np.round(center + curr_r * np.sin(curr_theta)))
                    
                    if 0 <= xi < self.L1 and 0 <= yi < self.L1:
                        # 这里的 dft_mag[yi, xi] 对应图像矩阵的 [行, 列]
                        max_v_j = max(max_v_j, dft_mag[yi, xi])
                        ring_values.append(dft_mag[yi, xi])
            mean_v_j = np.mean(ring_values) if ring_values else 0
            # print(f"Bit {j}: max_v_j={max_v_j:.2f}, mean_v_j={mean_v_j:.2f}, threshold={threshold:.2f}")
            # 决定位值（论文公式 26）
            bit = 1 if mean_v_j >= threshold else 0
            bits.append(bit)
        ##jiema
        # W1_extract = np.zeros(l, dtype=np.uint8)
        # # angles = np.linspace(0, np.pi, l)
        # ring_width = 2  # 与嵌入时相同的环宽度
        # for j in range(l):
        #     theta = j / l * np.pi
        #     r = self.R1[j % len(self.R1)]
        #     # 收集当前环所有点的幅值
        #     ring_values = []
        #     for dr in range(-ring_width, ring_width+1):
        #         cr = r + dr
        #         if cr <= 0:
        #             continue
        #         x = int(center + 1 + int(np.round(cr * np.cos(theta))))
        #         y = int(center + 1 + int(np.round(cr * np.sin(theta))))
        #         search_radius = 0
        #         for dx in range(-search_radius, search_radius + 1):
        #             for dy in range(-search_radius, search_radius + 1):
        #                 x_adj = x + dx
        #                 y_adj = y + dy
        #                 if 0 <= x_adj < self.L1 and 0 <= y_adj < self.L1:
        #                     ring_values.append(dft_mag[x_adj, y_adj])
        #             # if 0<=x<self.L1 and 0<=y<self.L1:
        #             #     ring_values.append(dft_mag[x, y])

        #     # 4. 平均幅值判0/1
        #     if len(ring_values) > 0:
        #         mean_val = np.mean(ring_values)
        #         max_val = np.max(ring_values)
        #         print(f"Bit {j}: mean_val={mean_val:.2f}, max_val={max_val:.2f}, threshold={threshold:.2f}")
        #         W1_extract[j] = 1 if mean_val >= threshold else 0
        #     else:
        #         W1_extract[j] = 0

        return bits

def batch_encode(wm: CWTWatermark, input_dir: str, output_dir: str, message: str):
    """
    批量嵌入水印
    
    Args:
        wm: 水印系统实例
        input_dir: 输入图像文件夹
        output_dir: 输出图像文件夹
        message: 8字符的水印消息
    """
    print(f"\n{'='*80}")
    print(f"BATCH ENCODING - Message: '{message}'")
    print(f"{'='*80}")
    
    # 创建输出文件夹
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 获取所有图像文件
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    image_files = [f for f in os.listdir(input_dir) 
                   if f.lower().endswith(image_extensions)]
    image_files.sort()
    
    if not image_files:
        print(f"Error: No images found in {input_dir}")
        return
    
    print(f"Found {len(image_files)} images to process")
    
    # 存储所有PSNR和SSIM值
    psnr_values = []
    ssim_values = []
    
    # 处理每张图像
    for idx, image_file in enumerate(image_files):
        print(f"\n[{idx+1}/{len(image_files)}] Processing: {image_file}")
        
        image_path = os.path.join(input_dir, image_file)
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"  Warning: Failed to load {image_file}, skipping...")
            continue
        
        # 调整图像大小到512x512
        if image.shape[:2] != (512, 512):
            image = cv2.resize(image, (512, 512))
        
        # 嵌入水印
        watermarked, psnr, ssim = wm.embed(image, message)
        
        # 保存结果
        base_name = os.path.splitext(image_file)[0]
        output_path = os.path.join(output_dir, f"{base_name}_watermarked.png")
        cv2.imwrite(output_path, watermarked)
        print(f"  Saved: {output_path}")
        print(f"  PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}")
        
        psnr_values.append(psnr)
        ssim_values.append(ssim)
    
    # 输出统计信息
    if psnr_values:
        avg_psnr = np.mean(psnr_values)
        avg_ssim = np.mean(ssim_values)
        std_psnr = np.std(psnr_values)
        std_ssim = np.std(ssim_values)
        
        print(f"\n{'='*80}")
        print("BATCH ENCODING STATISTICS")
        print(f"{'='*80}")
        print(f"Total images processed: {len(psnr_values)}")
        print(f"Average PSNR: {avg_psnr:.2f} dB (std: {std_psnr:.2f})")
        print(f"Average SSIM: {avg_ssim:.4f} (std: {std_ssim:.4f})")
        print(f"Min PSNR: {np.min(psnr_values):.2f} dB")
        print(f"Max PSNR: {np.max(psnr_values):.2f} dB")
        print(f"Min SSIM: {np.min(ssim_values):.4f}")
        print(f"Max SSIM: {np.max(ssim_values):.4f}")
        print(f"{'='*80}")


def batch_decode(wm: CWTWatermark, input_dir: str, message: str):
    """
    批量提取水印
    
    Args:
        wm: 水印系统实例
        input_dir: 输入图像文件夹（含水印图像）
        message: 原始消息（GT，用于计算准确率）
    """
    print(f"\n{'='*80}")
    print(f"BATCH DECODING - GT Message: '{message}'")
    print(f"{'='*80}")
    
    # 获取所有图像文件
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    image_files = [f for f in os.listdir(input_dir) 
                   if f.lower().endswith(image_extensions)]
    image_files.sort()
    
    if not image_files:
        print(f"Error: No images found in {input_dir}")
        return
    
    print(f"Found {len(image_files)} images to process")
    
    # 存储所有准确率
    accuracy_values = []
    
    # 处理每张图像
    for idx, image_file in enumerate(image_files):
        print(f"\n[{idx+1}/{len(image_files)}] Processing: {image_file}")
        
        image_path = os.path.join(input_dir, image_file)
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"  Warning: Failed to load {image_file}, skipping...")
            continue
        
        # 调整图像大小到512x512
        if image.shape[:2] != (512, 512):
            image = cv2.resize(image, (512, 512))
        
        # 提取水印
        extracted_message, accuracy, extracted_bits = wm.extract(image, message)
        
        print(f"  Extracted: '{extracted_message}'")
        print(f"  Accuracy: {accuracy*100:.2f}%")
        
        accuracy_values.append(accuracy)
    
    # 输出统计信息
    if accuracy_values:
        avg_accuracy = np.mean(accuracy_values)
        std_accuracy = np.std(accuracy_values)
        
        print(f"\n{'='*80}")
        print("BATCH DECODING STATISTICS")
        print(f"{'='*80}")
        print(f"Total images processed: {len(accuracy_values)}")
        print(f"Average Accuracy: {avg_accuracy*100:.2f}% (std: {std_accuracy*100:.2f}%)")
        print(f"Mean Accuracy: {avg_accuracy*100:.2f}%")
        print(f"Variance: {np.var(accuracy_values)*10000:.2f}%^2")
        print(f"Min Accuracy: {np.min(accuracy_values)*100:.2f}%")
        print(f"Max Accuracy: {np.max(accuracy_values)*100:.2f}%")
        print(f"{'='*80}")


def main():
    """命令行主函数"""
    parser = argparse.ArgumentParser(
        description='CWT Watermark Method - 基于DFT域的水印嵌入和提取',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 单图像嵌入水印
  python cwt_method.py encode -i input.png -o output.png -m "Hello!!"
  
  # 批量嵌入水印（文件夹）
  python cwt_method.py encode -i input_folder -o output_folder -m "Hello!!"
  
  # 单图像提取水印（需要GT消息）
  python cwt_method.py decode -i output.png -m "Hello!!"
  
  # 批量提取水印（文件夹）
  python cwt_method.py decode -i output_folder -m "Hello!!"
        """
    )

    subparsers = parser.add_subparsers(dest='mode', help='操作模式: encode 或 decode')

    # Encode模式
    encode_parser = subparsers.add_parser('encode', help='嵌入水印')
    encode_parser.add_argument('-i', '--input', required=True, help='输入图像路径或文件夹')
    encode_parser.add_argument('-o', '--output', required=True, help='输出图像路径或文件夹')
    encode_parser.add_argument('-m', '--message', required=True, help='8字符的水印消息')

    # Decode模式
    decode_parser = subparsers.add_parser('decode', help='提取水印')
    decode_parser.add_argument('-i', '--input', required=True, help='输入图像路径或文件夹（含水印）')
    decode_parser.add_argument('-m', '--message', required=True, help='原始消息（GT，用于计算准确率）')

    args = parser.parse_args()

    if args.mode is None:
        parser.print_help()
        sys.exit(1)

    # 初始化水印系统
    wm = CWTWatermark(block_size=512)

    if args.mode == 'encode':
        # 嵌入模式
        if len(args.message) != 8:
            print(f"Error: Message must be exactly 8 characters, got {len(args.message)}")
            print(f"Your message: '{args.message}'")
            sys.exit(1)

        # 判断是文件还是文件夹
        if os.path.isdir(args.input):
            # 批量处理
            batch_encode(wm, args.input, args.output, args.message)
        else:
            # 单文件处理
            if not os.path.exists(args.input):
                print(f"Error: Input image not found: {args.input}")
                sys.exit(1)

            image = cv2.imread(args.input)
            if image is None:
                print(f"Error: Failed to load image: {args.input}")
                sys.exit(1)

            # 调整图像大小到512x512
            if image.shape[:2] != (512, 512):
                print(f"Resizing image from {image.shape[:2]} to (512, 512)")
                image = cv2.resize(image, (512, 512))

            # 嵌入水印
            watermarked, psnr, ssim = wm.embed(image, args.message)

            # 保存结果
            cv2.imwrite(args.output, watermarked)
            print(f"\nWatermarked image saved to: {args.output}")
            print(f"PSNR: {psnr:.2f} dB")
            print(f"SSIM: {ssim:.4f}")

    elif args.mode == 'decode':
        # 提取模式
        if len(args.message) != 8:
            print(f"Error: GT message must be exactly 8 characters, got {len(args.message)}")
            print(f"Your message: '{args.message}'")
            sys.exit(1)

        # 判断是文件还是文件夹
        if os.path.isdir(args.input):
            # 批量处理
            batch_decode(wm, args.input, args.message)
        else:
            # 单文件处理
            if not os.path.exists(args.input):
                print(f"Error: Input image not found: {args.input}")
                sys.exit(1)

            image = cv2.imread(args.input)
            if image is None:
                print(f"Error: Failed to load image: {args.input}")
                sys.exit(1)

            # 调整图像大小到512x512
            if image.shape[:2] != (512, 512):
                print(f"Resizing image from {image.shape[:2]} to (512, 512)")
                image = cv2.resize(image, (512, 512))

            # 提取水印
            extracted_message, accuracy, extracted_bits = wm.extract(image, args.message)

            # 输出结果
            print(f"\n{'='*60}")
            print("EXTRACTION RESULTS")
            print(f"{'='*60}")
            print(f"GT Message:      '{args.message}'")
            print(f"Extracted Message: '{extracted_message}'")
            print(f"Accuracy:        {accuracy*100:.2f}%")
            print(f"Extracted Bits:  {extracted_bits}")
            print(f"{'='*60}")

            # 判断是否匹配
            if extracted_message == args.message:
                print("✓ Watermark verification: PASSED")
            else:
                print("✗ Watermark verification: FAILED")


if __name__ == "__main__":
    main()

