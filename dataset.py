import os
import cv2
import numpy as np
import torch
import random
from torch.utils.data import Dataset
from encode2 import Watermark16Sector1
from noise_utils import add_pimog_noise, add_jpeg_compression_noise, add_tile_rotate_crop_noise, add_wechat_noise
class WatermarkDataset(Dataset):
    """
    水印数据集
    生成带水印的图像和对应的水印标签
    """
    def __init__(self, image_dir, block_size=512, num_bits=4, r=[3,6], bits=[5,15], alpha_embed=0.01, transform=None, noise_level='none', noise_pool=None, max_angle=180, crop_scale_range=None, max_images=0):
        self.image_dir = image_dir
        self.block_size = block_size
        self.transform = transform
        self.alpha_embed = alpha_embed
        self.num_bits = num_bits
        self.noise_level = noise_level.lower()
        self.noise_pool = noise_pool
        self.max_angle = max_angle
        self.crop_scale_range = crop_scale_range
        self.force_noise_pair = None  # 设为 (type1, type2) 可强制指定噪声组合
        self.watermark_system = Watermark16Sector1(L1=block_size, k1=30000, r=r, bitsf=bits, r_range=1,n_sectors=num_bits) #2gaiwei1
        self.image_files = [f for f in os.listdir(image_dir)
                           if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        if max_images > 0:
            self.image_files = self.image_files[:max_images]
        
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
            },
            'tile_rotate': {
                'pimog_level': 0.0,
                'jpeg_quality': 100,
                'tile_crop_ratio': 0.0
            }
        }
        
        # 验证噪声强度参数 (pair 模式在 __getitem__ 中特殊处理)
        valid_levels = list(self.noise_config.keys()) + ['pair']
        if self.noise_level not in valid_levels:
            raise ValueError(f"noise_level must be one of: {valid_levels}")
    
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

        # Cb通道嵌入：转换到YCrCb，在Cb通道加水印
        ycrcb = cv2.cvtColor(host_bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32)
        y_ch, cr_ch, cb_ch = cv2.split(ycrcb)

        # 将Tm转为单通道用于Cb嵌入
        if len(Tm.shape) == 3:
            Tm_gray = cv2.cvtColor(Tm, cv2.COLOR_BGR2GRAY).astype(np.float32)
        else:
            Tm_gray = Tm.astype(np.float32)

        # 在Cb通道嵌入水印
        cb_wm = cb_ch * (1 - self.alpha_embed) + Tm_gray * self.alpha_embed

        # 添加数值稳定性检查
        if np.isnan(cb_wm).any() or np.isinf(cb_wm).any():
            print(f"NaN or Inf found in cb_wm! alpha_embed={self.alpha_embed}")
            cb_wm = cb_ch.copy()

        cb_wm = np.clip(cb_wm, 0, 255).astype(np.uint8)
        ycrcb_wm = cv2.merge([y_ch.astype(np.uint8), cr_ch.astype(np.uint8), cb_wm])
        watermarked_image = cv2.cvtColor(ycrcb_wm, cv2.COLOR_YCrCb2BGR)
        watermarked_image = np.clip(watermarked_image, 0, 255).astype(np.uint8)
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
        
        # 添加噪声（在转换为Cb通道前）
        if self.noise_level == 'tile_rotate':
            # tile_rotate噪声：3x3平铺 + 旋转 + 裁剪
            from noise_utils import add_tile_rotate_crop_noise
            watermarked_image = add_tile_rotate_crop_noise(
                watermarked_image,
                angle_range=(-self.max_angle, self.max_angle),
                crop_scale_range=self.crop_scale_range
            )
        elif self.noise_level == 'pair':
            # 两两配对噪声：从 identity/wechat/tile_crop/pimog 中随机选两种不同的依次应用
            if self.force_noise_pair is not None:
                selected = self.force_noise_pair
            else:
                noise_pool = ['identity', 'wechat', 'tile_crop', 'pimog']
                first = np.random.choice(noise_pool)
                second_pool = [n for n in noise_pool if n != first]
                second = np.random.choice(second_pool)
                selected = [first, second]
            for noise_type in selected:
                if noise_type == 'identity':
                    pass  # 不加噪声
                elif noise_type == 'wechat':
                    watermarked_image = add_wechat_noise(watermarked_image)
                elif noise_type == 'tile_crop':
                    # 循环平移：对已嵌入水印的图像做随机平移（不旋转）
                    shift_x = random.randint(0, self.block_size - 1)
                    shift_y = random.randint(0, self.block_size - 1)
                    watermarked_image = np.roll(np.roll(watermarked_image, shift_x, axis=1), shift_y, axis=0)
                elif noise_type == 'pimog':
                    watermarked_image = add_pimog_noise(watermarked_image)
        elif self.noise_level != 'none':
            config = self.noise_config[self.noise_level]

            # 随机选择噪声类型
            noise_types = ['none', 'pimog', 'jpeg', 'tile_crop']
            noise_type = np.random.choice(noise_types)

            if noise_type == 'pimog' and config['pimog_level'] > 0:
                watermarked_image = add_pimog_noise(watermarked_image, noise_level=config['pimog_level'])
            elif noise_type == 'jpeg' and config['jpeg_quality'] < 100:

                watermarked_image = add_jpeg_compression_noise(watermarked_image, quality=config['jpeg_quality'])

            elif noise_type == 'tile_crop' and config['tile_crop_ratio'] > 0:
                # 循环平移：对水印模板进行随机平移
                shift_x = random.randint(0, self.block_size - 1)
                shift_y = random.randint(0, self.block_size - 1)
                # 循环平移Tm
                Tm_shifted = np.roll(np.roll(Tm, shift_x, axis=1), shift_y, axis=0)
                # 重新嵌入
                watermarked_image = host_bgr.astype(np.float32) * (1 - self.alpha_embed) + Tm_shifted.astype(np.float32) * self.alpha_embed
                watermarked_image = np.clip(watermarked_image, 0, 255).astype(np.uint8)
            else:
                watermarked_image = host_bgr.copy()
        
        # 提取Cb通道作为单通道输入
        watermarked_image = np.clip(watermarked_image, 0, 255).astype(np.uint8)
        ycrcb_out = cv2.cvtColor(watermarked_image, cv2.COLOR_BGR2YCrCb)
        cb_out = ycrcb_out[:, :, 2]  # Cb通道
        # Add channel dimension
        watermarked_image = cb_out[..., np.newaxis]

        if len(m1.shape) == 2:
            m1 = m1[..., np.newaxis]

        if self.transform:
            watermarked_image = self.transform(watermarked_image)
            m1=self.transform(m1)

        watermark_tensor = torch.tensor(watermark_bits, dtype=torch.float32)

        return watermarked_image, watermark_tensor, m1