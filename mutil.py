import numpy as np
from numpy.fft import ifftshift, ifft2, fftshift, fft2
import cv2

class Watermark16Sector:
    def __init__(self, L1=676, k1=120.0, r=[5, 12], r_range=1, n_sectors=16):
        self.L1 = L1        # 模板大小
        self.k1 = k1       # 频域幅值强度
        self.r = r        # 低频半径
        self.r_range = r_range # 低频半径范围
        self.n_sectors = n_sectors# 180度分16份

    # ======================
    # 【嵌入】生成16分类低频水印模板（你的共轭对称代码）
    # ======================
    def generate_template(self, numbit=[]):
        """
        输入：num 0~15
        输出：空域水印Tm + 频域谱M1
        """
      
        M1 = np.zeros((self.L1, self.L1), dtype=np.float32)
        cx, cy = self.L1 // 2, self.L1 // 2
        if numbit == []:
            numbit = np.random.randint(0, 2, size=self.n_sectors)
            print(numbit)
        
        # 核心：180度(0~π) 均分16份
        angles = np.linspace(0, np.pi, self.n_sectors + 1)
        for num in range(self.n_sectors):
            start_angle = angles[num]
            end_angle = angles[num + 1]
            r=self.r[num%len(self.r)]

            # 遍历低频圆环（连续区域，纯低频）
            for thetar in range(r, r+self.r_range + 1):
                # 圆环采样点，保证平滑
                ri=thetar+r
                
                theta_list = np.linspace(start_angle, end_angle, int(ri * 20))
                for theta in theta_list:
                    # 计算坐标
                    x = cx + int(np.round(ri * np.cos(theta)))
                    y = cy + int(np.round(ri * np.sin(theta)))

                    # 你的边界检查 + 幅值赋值
                    if 0 <= x < self.L1 and 0 <= y < self.L1:
                        if numbit[num] == 1:
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

   
if __name__ == "__main__":
    # 初始化
    wm = Watermark16Sector(L1=512, k1=10000, r=[3,8], r_range=1, n_sectors=4)
    
   
        
    Tm, M1 = wm.generate_template(numbit=[])
    cv2.imwrite(f"img_mask/watermark_template.png", Tm)  # 保存水印模板图像以供调试
    cv2.imwrite(f"img_mask/watermark_spectrum.png", cv2.normalize(M1, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))  # 保存水印频谱图像以供调试
