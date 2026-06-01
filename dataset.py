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
    def __init__(self, image_dir, block_size=512, num_bits=4, r=[3,6], bits=[5,15], alpha_embed=0.01, transform=None, noise_level='none'):
        self.image_dir = image_dir
        self.block_size = block_size
        self.transform = transform
        self.alpha_embed = alpha_embed
        self.num_bits = num_bits
        self.noise_level = noise_level.lower()
        self.watermark_system = Watermark16Sector1(L1=block_size, k1=30000, r=r, bitsf=bits, r_range=1,n_sectors=num_bits) #2gaiwei1
        self.image_files = [f for f in os.listdir(image_dir) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        # 噪声强度配置
        self.noise_config = {
            'none': {
                'pimog_level': 0.0,
                'jpeg_quality': 100,
                'tile_crop_ratio': 0.0  # 0-1的参数，表示裁剪比例的变化范围
            },
            'low': {
                'pimog_level': 0.05,
                'jpeg_quality': 70,
                'tile_crop_ratio': 0.02  # 裁剪比例范围: 0.8-1.2
            },
            'mid': {
                'pimog_level': 0.1,
                'jpeg_quality': 40,
                'tile_crop_ratio': 0.05 # 裁剪比例范围: 0.5-1.5
            },
            'high': {
                'pimog_level': 0.15,
                'jpeg_quality': 20,
                'tile_crop_ratio': 0.1  # 裁剪比例范围: 0.0-2.0
            }
        }
        
        # 验证噪声强度参数
        if self.noise_level not in self.noise_config:
            raise ValueError(f"noise_level must be one of: {list(self.noise_config.keys())}")
    
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
        if self.noise_level != 'none':
            config = self.noise_config[self.noise_level]
            
            # 随机选择噪声类型
            noise_types = ['none', 'pimog', 'jpeg', 'tile_crop']
            noise_type = np.random.choice(noise_types)
            
            if noise_type == 'pimog' and config['pimog_level'] > 0:
                watermarked_image = add_pimog_noise(watermarked_image, noise_level=config['pimog_level'])
            elif noise_type == 'jpeg' and config['jpeg_quality'] < 100:
                
                watermarked_image = add_jpeg_compression_noise(watermarked_image, quality=config['jpeg_quality'])
               
            elif noise_type == 'tile_crop' and config['tile_crop_ratio'] > 0:
                # 平铺随机裁剪噪声 - 固定平铺 2x2，创建 1024x1024 的图像
                target_size = self.block_size * 2  # 固定 1024x1024
                
                # 放大原图到 1024x1024
                host_1024 = cv2.resize(host_bgr, (target_size, target_size))
                
                # 平铺水印到 1024x1024
                tiled = np.tile(Tm, (2, 2, 1)) if len(Tm.shape) == 3 else np.tile(Tm, (2, 2))
                tiled = cv2.resize(tiled, (target_size, target_size))
                
                # 创建带水印的 1024x1024 图像
                watermarked_1024 = host_1024.astype(np.float32) * (1 - self.alpha_embed) + tiled.astype(np.float32) * self.alpha_embed
                watermarked_1024 = np.clip(watermarked_1024, 0, 255).astype(np.uint8)
                
                # 根据裁剪比例计算裁剪大小
                ratio = config['tile_crop_ratio']
                crop_ratio = np.random.uniform(max(0, 1 - ratio), 1 + ratio)
                crop_size = int(self.block_size * crop_ratio)
                crop_size = max(1, min(crop_size, self.block_size))  # 确保裁剪大小有效
                
                # 随机裁剪 1024x1024 图像
                max_offset = target_size - crop_size
                x = random.randint(0, max_offset)
                y = random.randint(0, max_offset)
                watermarked_image = watermarked_1024[y:y+crop_size, x:x+crop_size,:]
                
                # resize 回 512x512
                watermarked_image = cv2.resize(watermarked_image, (self.block_size, self.block_size))
            else:
                watermarked_image = host_bgr.copy()
        
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