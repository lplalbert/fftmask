"""
v17 水印数据集
无旋转矫正环，仅包含水印环
"""
import os
import cv2
import numpy as np
import torch
import random
from torch.utils.data import Dataset
from encode_v17 import WatermarkV17
from noise_utils import add_pimog_noise, add_jpeg_compression_noise, add_wechat_noise, add_tile_rotate_crop_noise, add_physical_moire_noise


class WatermarkDatasetV17(Dataset):
    """
    v17 水印数据集
    无旋转矫正环
    """
    def __init__(self, image_dir=None, block_size=512, num_bits=60,
                 r_watermark=[8, 15], bitsf=[20, 40],
                 alpha_embed=0.016, transform=None,
                 noise_level='none', noise_pool=None,
                 max_angle=360, crop_scale_range=None,
                 max_rotation=5.0, max_shift=0.5,
                 max_images=0, wechat_downsample_factor=4,
                 datasets=None):
        """
        Args:
            image_dir: 图像目录 (单目录模式，兼容旧配置)
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
            datasets: 多目录配置列表，每项为 dict:
                {'dir': path, 'mode': 'resize'|'crop', 'max_images': N}
                mode='resize': 直接resize到block_size (适合小图如COCO/Document)
                mode='crop': 随机裁剪block_size×block_size块 (适合大图如PPT截图)
        """
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
        self.force_noise_pair = None  # 设为 (type1, type2) 可强制指定噪声组合
        self.wechat_downsample_factor = wechat_downsample_factor

        # 水印生成器
        self.watermark_system = WatermarkV17(
            L1=block_size,
            k1=30000,
            r_watermark=r_watermark,
            bitsf=bitsf,
            r_range=1,
            n_sectors=num_bits
        )

        # 加载图像列表 (支持多目录)
        bs = self.block_size
        if datasets is not None:
            self.image_items = []  # (dir, filename, mode, tile_idx)
            for ds in datasets:
                d = ds['dir']
                mode = ds.get('mode', 'resize')
                ds_max = ds.get('max_images', 0)
                files = [f for f in os.listdir(d)
                        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                if ds_max > 0:
                    files = files[:ds_max]
                if mode == 'tile':
                    # tile模式：预扫描图像尺寸，计算每张图的裁剪数
                    tiles_per_img = ds.get('tiles_per_img', 0)
                    tile_count = 0
                    for f in files:
                        img_path = os.path.join(d, f)
                        img = cv2.imread(img_path)
                        if img is None:
                            continue
                        h, w = img.shape[:2]
                        ny = h // bs
                        nx = w // bs
                        if tiles_per_img > 0:
                            n = min(tiles_per_img, ny * nx)
                        else:
                            n = ny * nx
                        for ti in range(n):
                            self.image_items.append((d, f, 'tile', ti))
                        tile_count += n
                    print(f"  [dataset] {d}: {len(files)} images → {tile_count} tiles (512×512)")
                else:
                    for f in files:
                        self.image_items.append((d, f, mode, 0))
                    print(f"  [dataset] {d}: {len(files)} images, mode={mode}")
        else:
            # 兼容旧的单目录模式
            self.image_dir = image_dir
            self.image_files = [f for f in os.listdir(image_dir)
                               if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            if max_images > 0:
                self.image_files = self.image_files[:max_images]
            self.image_items = [(image_dir, f, 'resize') for f in self.image_files]

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
        return len(self.image_items)

    def rotate_image(self, image, angle):
        """
        旋转图像 (保持尺寸)

        Args:
            image: 输入图像
            angle: 旋转角度 (度)

        Returns:
            rotated: 旋转后的图像
        """
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT)
        return rotated

    def _prepare_image(self, image, mode, tile_idx=0):
        """根据mode处理图像到block_size×block_size"""
        h, w = image.shape[:2]
        bs = self.block_size

        if mode == 'tile':
            # 平铺裁剪：将图像网格化，取第tile_idx个块
            ny = h // bs
            nx = w // bs
            row = tile_idx // nx
            col = tile_idx % nx
            image = image[row*bs:(row+1)*bs, col*bs:(col+1)*bs]
        elif mode == 'crop' and (h > bs or w > bs):
            # 随机裁剪
            if h < bs or w < bs:
                scale = max(bs / h, bs / w)
                image = cv2.resize(image, (int(w * scale), int(h * scale)))
                h, w = image.shape[:2]
            y = random.randint(0, h - bs) if h > bs else 0
            x = random.randint(0, w - bs) if w > bs else 0
            image = image[y:y+bs, x:x+bs]
        else:
            # resize模式
            if image.shape[:2] != (bs, bs):
                image = cv2.resize(image, (bs, bs))
        return image

    def __getitem__(self, idx):
        # 读取图像
        img_dir, img_file, mode, tile_idx = self.image_items[idx]
        img_path = os.path.join(img_dir, img_file)
        image = cv2.imread(img_path)

        image = self._prepare_image(image, mode, tile_idx)

        # 随机生成水印bits
        watermark_bits = np.random.randint(0, 2, size=self.num_bits)

        # 生成水印模板
        Tm, M1, _ = self.watermark_system.generate_template(numbit=watermark_bits)

        # 确保图像格式正确
        image = np.clip(image, 0, 255).astype(np.uint8)
        if len(image.shape) == 2:
            host_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            host_bgr = image.copy()

        # 确保Tm格式正确
        if len(Tm.shape) == 2:
            Tm_bgr = cv2.cvtColor(Tm, cv2.COLOR_GRAY2BGR)
        else:
            Tm_bgr = Tm.copy()

        # Cb通道嵌入
        ycrcb = cv2.cvtColor(host_bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32)
        y_ch, cr_ch, cb_ch = cv2.split(ycrcb)

        # 将Tm转为单通道用于Cb嵌入
        if len(Tm_bgr.shape) == 3:
            Tm_gray = cv2.cvtColor(Tm_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        else:
            Tm_gray = Tm_bgr.astype(np.float32)

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
            # 仅循环平移，不加旋转
            watermarked_image = add_tile_rotate_crop_noise(
                watermarked_image,
                angle_range=(0, 0),
                max_shift=self.max_shift
            )
        elif self.noise_level == 'pair':
            # 两两配对噪声：从 identity/wechat/tile_crop/pimog 中随机选两种不同的依次应用
            if self.force_noise_pair is not None:
                selected = self.force_noise_pair
            else:
                noise_pool = ['identity', 'wechat', 'tile_crop', 'physical_moire']
                first = np.random.choice(noise_pool)
                second_pool = [n for n in noise_pool if n != first]
                second = np.random.choice(second_pool)
                selected = [first, second]
            for noise_type in selected:
                if noise_type == 'identity':
                    pass  # 不加噪声
                elif noise_type == 'wechat':
                    watermarked_image = add_wechat_noise(watermarked_image, downsample_factor=self.wechat_downsample_factor)
                elif noise_type == 'tile_crop':
                    # 3x3拼接 + 旋转 + 裁剪 (循环平移+旋转 combined)
                    watermarked_image = add_tile_rotate_crop_noise(
                        watermarked_image,
                        angle_range=(-self.max_rotation, self.max_rotation),
                        max_shift=self.max_shift
                    )
                elif noise_type == 'pimog':
                    watermarked_image = add_pimog_noise(watermarked_image)
                elif noise_type == 'physical_moire':
                    watermarked_image = add_physical_moire_noise(watermarked_image)
        elif self.noise_level == 'fixed_triple':
            # 固定三重噪声: tile_crop → pimog → wechat
            watermarked_image = add_tile_rotate_crop_noise(
                watermarked_image,
                angle_range=(-self.max_rotation, self.max_rotation),
                max_shift=self.max_shift
            )
            watermarked_image = add_pimog_noise(watermarked_image)
            watermarked_image = add_wechat_noise(watermarked_image, downsample_factor=self.wechat_downsample_factor)
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
