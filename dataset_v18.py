"""
v18 水印数据集 — 镂空模板
在 v17 基础上使用镂空模板嵌入

镂空原理:
- 生成二值模板 (0/255)
- 对白色区域(环)随机挖掉 hollow_ratio 比例的像素变为0
- 黑色区域(背景)保持0不变
- 镂空模板值只有两种: 0 和 255
"""
import os
import cv2
import numpy as np
import torch
import random
from torch.utils.data import Dataset
from encode_v18 import WatermarkV18
from noise_utils import add_pimog_noise, add_jpeg_compression_noise, add_wechat_noise, add_tile_rotate_crop_noise


class WatermarkDatasetV18(Dataset):
    """
    v18 水印数据集 — 镂空模板
    """
    def __init__(self, image_dir, block_size=512, num_bits=60,
                 r_watermark=[12, 25], bitsf=[15, 45],
                 alpha_embed=0.016, transform=None,
                 noise_level='none', noise_pool=None,
                 max_angle=360, crop_scale_range=None,
                 max_rotation=5.0, max_shift=0.5,
                 max_images=0,
                 M_w=255, M_b=0, hollow_ratio=0.3):
        """
        Args:
            image_dir: 图像目录
            block_size: 图像块大小
            num_bits: 水印位数
            r_watermark: 水印环半径
            bitsf: 每个环的位数
            alpha_embed: 嵌入强度
            transform: 数据变换
            noise_level: 噪声级别
            noise_pool: 噪声池
            max_angle: 最大旋转角度
            crop_scale_range: 裁剪缩放范围
            max_rotation: 最大旋转角度 (度，用于tile_crop噪声)
            max_shift: 最大循环平移比例 (0-1)，用于tile_crop噪声
            max_images: 最大图像数
            M_w: 水印高值区域亮度
            M_b: 水印低值区域亮度
            hollow_ratio: 镂空比例 (0-1)
        """
        self.image_dir = image_dir
        self.block_size = block_size
        self.transform = transform
        self.alpha_embed = alpha_embed
        self.num_bits = num_bits
        self.noise_level = noise_level.lower() if noise_level else 'none'
        self.noise_pool = noise_pool
        self.max_angle = max_angle
        self.crop_scale_range = crop_scale_range
        self.max_rotation = max_rotation
        self.max_shift = max_shift
        self.force_noise_pair = None
        self.hollow_ratio = hollow_ratio

        # 水印生成器 (v18 镂空版)
        self.watermark_system = WatermarkV18(
            L1=block_size,
            k1=30000,
            r_watermark=r_watermark,
            bitsf=bitsf,
            r_range=1,
            n_sectors=num_bits,
            M_w=M_w,
            M_b=M_b,
            hollow_ratio=hollow_ratio
        )

        # 图像文件列表
        self.image_files = [f for f in os.listdir(image_dir)
                           if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        if max_images > 0:
            self.image_files = self.image_files[:max_images]

        # 噪声强度配置
        self.noise_config = {
            'none': {
                'pimog_level': 0.0,
                'jpeg_quality': 100,
                'tile_crop_ratio': 0.0
            },
            'low': {
                'pimog_level': 0.05,
                'jpeg_quality': 70,
                'tile_crop_ratio': 0.02
            },
            'mid': {
                'pimog_level': 0.1,
                'jpeg_quality': 40,
                'tile_crop_ratio': 0.05
            },
            'high': {
                'pimog_level': 0.15,
                'jpeg_quality': 20,
                'tile_crop_ratio': 0.1
            },
            'tile_rotate': {
                'pimog_level': 0.0,
                'jpeg_quality': 100,
                'tile_crop_ratio': 0.0
            }
        }

        # 验证噪声强度参数
        valid_levels = list(self.noise_config.keys()) + ['pair', 'fixed_triple', 'shift_only']
        if self.noise_level not in valid_levels:
            raise ValueError(f"noise_level must be one of: {valid_levels}")

    def __len__(self):
        return len(self.image_files)

    def rotate_image(self, image, angle):
        """旋转图像 (保持尺寸)"""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT)
        return rotated

    def __getitem__(self, idx):
        # 读取图像
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = cv2.imread(img_path)

        if image.shape[:2] != (self.block_size, self.block_size):
            image = cv2.resize(image, (self.block_size, self.block_size))

        # 随机生成水印bits
        watermark_bits = np.random.randint(0, 2, size=self.num_bits)

        # 生成镂空水印模板
        Tm, M1, _ = self.watermark_system.generate_template(numbit=watermark_bits, hollow=True)

        # 确保图像格式正确
        image = np.clip(image, 0, 255).astype(np.uint8)
        if len(image.shape) == 2:
            host_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            host_bgr = image.copy()

        # Cb通道嵌入
        ycrcb = cv2.cvtColor(host_bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32)
        y_ch, cr_ch, cb_ch = cv2.split(ycrcb)

        # Tm 已经是单通道 (M_w/M_b 值)
        Tm_gray = Tm.astype(np.float32)

        # 在Cb通道嵌入水印
        cb_wm = cb_ch * (1 - self.alpha_embed) + Tm_gray * self.alpha_embed

        # 数值稳定性检查
        if np.isnan(cb_wm).any() or np.isinf(cb_wm).any():
            print(f"NaN or Inf found in cb_wm! alpha_embed={self.alpha_embed}")
            cb_wm = cb_ch.copy()

        cb_wm = np.clip(cb_wm, 0, 255).astype(np.uint8)
        ycrcb_wm = cv2.merge([y_ch.astype(np.uint8), cr_ch.astype(np.uint8), cb_wm])
        watermarked_image = cv2.cvtColor(ycrcb_wm, cv2.COLOR_YCrCb2BGR)
        watermarked_image = np.clip(watermarked_image, 0, 255).astype(np.uint8)

        # 添加噪声
        if self.noise_level == 'tile_rotate':
            watermarked_image = add_tile_rotate_crop_noise(
                watermarked_image,
                angle_range=(-self.max_angle, self.max_angle),
                crop_scale_range=self.crop_scale_range
            )
        elif self.noise_level == 'shift_only':
            watermarked_image = add_tile_rotate_crop_noise(
                watermarked_image,
                angle_range=(0, 0),
                max_shift=self.max_shift
            )
        elif self.noise_level == 'pair':
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
                    pass
                elif noise_type == 'wechat':
                    watermarked_image = add_wechat_noise(watermarked_image)
                elif noise_type == 'tile_crop':
                    watermarked_image = add_tile_rotate_crop_noise(
                        watermarked_image,
                        angle_range=(-self.max_rotation, self.max_rotation),
                        max_shift=self.max_shift
                    )
                elif noise_type == 'pimog':
                    watermarked_image = add_pimog_noise(watermarked_image)
        elif self.noise_level == 'fixed_triple':
            watermarked_image = add_tile_rotate_crop_noise(
                watermarked_image,
                angle_range=(-self.max_rotation, self.max_rotation),
                max_shift=self.max_shift
            )
            watermarked_image = add_pimog_noise(watermarked_image)
            watermarked_image = add_wechat_noise(watermarked_image)
        elif self.noise_level != 'none':
            config = self.noise_config[self.noise_level]
            noise_types = ['none', 'pimog', 'jpeg', 'tile_crop']
            noise_type = np.random.choice(noise_types)

            if noise_type == 'pimog' and config['pimog_level'] > 0:
                watermarked_image = add_pimog_noise(watermarked_image, noise_level=config['pimog_level'])
            elif noise_type == 'jpeg' and config['jpeg_quality'] < 100:
                watermarked_image = add_jpeg_compression_noise(watermarked_image, quality=config['jpeg_quality'])
            elif noise_type == 'tile_crop' and config['tile_crop_ratio'] > 0:
                shift_x = random.randint(0, self.block_size - 1)
                shift_y = random.randint(0, self.block_size - 1)
                Tm_shifted = np.roll(np.roll(Tm, shift_x, axis=1), shift_y, axis=0)
                watermarked_image = host_bgr.astype(np.float32) * (1 - self.alpha_embed) + Tm_shifted.astype(np.float32) * self.alpha_embed
                watermarked_image = np.clip(watermarked_image, 0, 255).astype(np.uint8)
            else:
                watermarked_image = host_bgr.copy()

        # 提取Cb通道作为单通道输入
        watermarked_image = np.clip(watermarked_image, 0, 255).astype(np.uint8)
        ycrcb_out = cv2.cvtColor(watermarked_image, cv2.COLOR_BGR2YCrCb)
        cb_out = ycrcb_out[:, :, 2]  # Cb通道
        watermarked_image = cb_out[..., np.newaxis]  # (H, W, 1)

        if self.transform:
            watermarked_image = self.transform(watermarked_image)

        watermark_tensor = torch.tensor(watermark_bits, dtype=torch.float32)

        return watermarked_image, watermark_tensor