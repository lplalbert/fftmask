import numpy as np
from numpy.fft import ifftshift, ifft2, fftshift, fft2
import cv2

class Watermark16Sector:
    def __init__(self, L1=676, k1=120.0, r_min=5, r_max=12, n_sectors=16):
        self.L1 = L1        # 模板大小
        self.k1 = k1       # 频域幅值强度
        self.r_min = r_min # 低频内半径
        self.r_max = r_max # 低频外半径
        self.n_sectors = n_sectors# 180度分16份

    # ======================
    # 【嵌入】生成16分类低频水印模板（你的共轭对称代码）
    # ======================
    def generate_template(self, num: int):
        """
        输入：num 0~15
        输出：空域水印Tm + 频域谱M1
        """
        assert 0 <= num <= 15
        M1 = np.zeros((self.L1, self.L1), dtype=np.float32)
        cx, cy = self.L1 // 2, self.L1 // 2

        # 核心：180度(0~π) 均分16份
        angles = np.linspace(0, np.pi, self.n_sectors + 1)
        start_angle = angles[num]
        end_angle = angles[num + 1]

        # 遍历低频圆环（连续区域，纯低频）
        for r in range(self.r_min, self.r_max + 1):
            # 圆环采样点，保证平滑
            theta_list = np.linspace(start_angle, end_angle, int(r * 20))
            for theta in theta_list:
                # 计算坐标
                x = cx +int(np.round(r * np.cos(theta)))
                y = cy +int(np.round(r * np.sin(theta)))

                # 你的边界检查 + 幅值赋值
                if 0 <= x < self.L1 and 0 <= y < self.L1:
                    M1[y, x] = self.k1

                    # 【你要求的共轭对称】一模一样！
                    x_sym = self.L1 - x
                    y_sym = self.L1 - y
                    if 0 <= x_sym < self.L1 and 0 <= y_sym < self.L1:
                        M1[y_sym, x_sym] = M1[y, x]

        # 逆DFT → 空域水印
        spatial = np.real(ifft2(ifftshift(M1)))
        Tm = np.where(spatial < 0, 0, 255).astype(np.uint8)
        return Tm, M1

    # ======================
    # 【解码】抗循环平移 + 鲁棒提取0~15
    # ======================
    def decode_template(self, crop_block):
        # DFT + 幅值谱
        if crop_block.ndim == 3:
            crop_block = cv2.cvtColor(crop_block, cv2.COLOR_BGR2GRAY)
        mag = np.abs(fftshift(fft2(crop_block.astype(np.float32))))
        cx, cy = self.L1 // 2, self.L1 // 2

        # 统计16个扇区的平均幅值
        scores = np.zeros(self.n_sectors)
        angles = np.linspace(0, np.pi, self.n_sectors + 1)

        for i in range(self.n_sectors):
            s, e = angles[i], angles[i+1]
            values = []
            for r in range(self.r_min, self.r_max + 1):
                theta_list = np.linspace(s, e, int(r * 4))
                for theta in theta_list:
                    x = cx + int(np.round(r * np.cos(theta)))
                    y = cy + int(np.round(r * np.sin(theta)))
                    if 0 <= x < self.L1 and 0 <= y < self.L1:
                        values.append(mag[y, x])
            scores[i] = np.mean(values) if len(values) > 0 else 0

        # 最大值对应数字
        return int(np.argmax(scores))
if __name__ == "__main__":
    # 初始化
    wm = Watermark16Sector(L1=512, k1=10000, r_min=5, r_max=6, n_sectors=2)
    
    # 测试：嵌入数字 12
    for num in range(2):
        
        Tm, M1 = wm.generate_template(num)
        cv2.imwrite(f"img_mask/watermark_template_{num}.png", Tm)  # 保存水印模板图像以供调试
        cv2.imwrite(f"img_mask/watermark_spectrum_{num}.png", cv2.normalize(M1, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))  # 保存水印频谱图像以供调试

    # # 模拟【循环平移】（抗攻击测试）
    # Tm_shift = np.roll(Tm, shift=50, axis=0)
    # Tm_shift = np.roll(Tm_shift, shift=30, axis=1)
    
    # # 解码
    # pred_num = wm.decode_template(Tm_shift)
    # print(f"嵌入数字：{num}")
    # print(f"解码数字：{pred_num}")
    # print("✅ 成功！" if num == pred_num else "❌ 失败")