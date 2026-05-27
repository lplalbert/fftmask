import os
import cv2
import numpy as np
import torch
import random
from torch.utils.data import Dataset
from encode2 import Watermark16Sector1
from noise_utils import add_pimog_noise, add_jpeg_compression_noise
class WatermarkDataset(Dataset):
    """
    水印数据集
    生成带水印的图像和对应的水印标签
    """
    def __init__(self, image_dir, block_size=512, num_bits=4, r=[3,6], bits=[5,15], alpha_embed=0.01, transform=None):
        self.image_dir = image_dir
        self.block_size = block_size
        self.transform = transform
        self.alpha_embed = alpha_embed
        self.num_bits = num_bits
        self.watermark_system = Watermark16Sector1(L1=block_size, k1=30000, r=r, bitsf=bits, r_range=1,n_sectors=num_bits) #2gaiwei1
        self.image_files = [f for f in os.listdir(image_dir) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = cv2.imread(img_path)
        
        if image.shape[:2] != (self.block_size, self.block_size):
            image = cv2.resize(image, (self.block_size, self.block_size))
        
        watermark_bits = np.random.randint(0, 2, size=self.num_bits)
        
        Tm, m1 = self.watermark_system.generate_template(numbit=watermark_bits)
        
        image = np.clip(image, 0, 255).astype(np.uint8)
        if len(image.shape) == 2:
            host_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            host_bgr = image.copy()
        if len(Tm.shape) == 2:
            Tm = cv2.cvtColor(Tm, cv2.COLOR_GRAY2BGR)
        else:
            Tm = Tm.copy()
        if len(m1.shape) == 3 and m1.shape[2] == 3:
            m1 = cv2.cvtColor(m1, cv2.COLOR_BGR2GRAY)
        elif len(m1.shape) == 2:
            m1 = m1.copy()
        else:
            raise ValueError(f'Unexpected m1 shape: {m1.shape}')

        markimg=host_bgr.astype(np.float32)*(1-self.alpha_embed)+Tm.astype(np.float32)*self.alpha_embed   
        # 添加数值稳定性检查
        if np.isnan(markimg).any() or np.isinf(markimg).any():
            print(f"NaN or Inf found in markimg! alpha_embed={self.alpha_embed}")
            watermarked_image = host_bgr.copy()
        else:
            watermarked_image=np.clip(markimg, 0, 255).astype(np.uint8)
        # host_yuv = cv2.cvtColor(host_bgr, cv2.COLOR_BGR2YUV)
        # host_y = host_yuv[:, :, 0].astype(np.float32)
        
        # Tm_float = Tm.astype(np.float32) / 255.0
        
        # JND = self.watermark_system.calculate_jnd(host_y)
        
        # alpha_embed = getattr(self.watermark_system, 'alpha', 0.1)
        # I_lum = host_y + Tm_float * JND * alpha_embed
        # I_lum = np.clip(I_lum, 0, 255).astype(np.uint8)
        
        # watermarked_yuv = host_yuv.copy()
        # watermarked_yuv[:, :, 0] = I_lum
        # watermarked_image = cv2.cvtColor(watermarked_yuv, cv2.COLOR_YUV2BGR)
        # watermarked_image = np.clip(watermarked_image, 0, 255).astype(np.uint8)
        
        # 添加噪声（在转换为灰度前）
        # 随机选择噪声类型
        noise_type = np.random.choice(['pimog', 'jpeg', 'tile_crop', 'none'])
        # if noise_type == 'pimog':
        #     watermarked_image = add_pimog_noise(watermarked_image, noise_level=0.1)
        # elif noise_type == 'jpeg':
        #     #加一个极端缩放模拟wx压缩，随机缩放比例为1.2~3.0
        #     scale = np.random.uniform(0.8, 4.0)
        #     watermarked_image = cv2.resize(watermarked_image, (int(self.block_size*scale), int(self.block_size*scale)))
        #     watermarked_image = add_jpeg_compression_noise(watermarked_image, quality=50)
        #     watermarked_image = cv2.resize(watermarked_image, (self.block_size, self.block_size))
        # elif noise_type == 'tile_crop':
        #     # 平铺随机裁剪噪声
        #     # 1. 放大到1024x1024
        #     # watermarked_1024 = cv2.resize(watermarked_image, (1024, 1024))
        #     host_1024 = cv2.resize(host_bgr, (1024, 1024))
        #     # tm_1024 = np.repeat(np.repeat(Tm, 2, axis=0), 2, axis=1)
        #     # tm_1024 = np.tile(Tm, (2, 2, 1))
        #     tiled = np.tile(Tm, (2, 2, 1)) if len(Tm.shape) == 3 else np.tile(Tm, (2, 2))
        #     # 2. 随机裁剪512x512
        #     watermarked_1024 = host_1024*(1-self.alpha_embed)+tiled*self.alpha_embed
        #     x = random.randint(0, 512)
        #     y = random.randint(0, 512)
        #     watermarked_image = watermarked_1024[y:y+512, x:x+512,:]
        # else:
        #     pass
        
        # Convert to grayscale
        watermarked_image = np.clip(watermarked_image, 0, 255).astype(np.uint8)
        watermarked_image = cv2.cvtColor(watermarked_image, cv2.COLOR_BGR2GRAY)
        # Add channel dimension
        watermarked_image = watermarked_image[..., np.newaxis]

        if len(m1.shape) == 2:
            m1 = m1[..., np.newaxis]

        if self.transform:
            watermarked_image = self.transform(watermarked_image)
            m1=self.transform(m1)

        watermark_tensor = torch.tensor(watermark_bits, dtype=torch.float32)

        return watermarked_image, watermark_tensor, m1