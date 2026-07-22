import cv2
import numpy as np
import torch
from torchvision import transforms
import math
import random
import kornia
from kornia.geometry.transform import get_rotation_matrix2d, warp_affine, get_perspective_transform, warp_perspective


def _compute_translation_matrix(translation: torch.Tensor) -> torch.Tensor:
    """Computes affine matrix for translation."""
    matrix = translation.new_zeros((translation.shape[0], 3, 3))
    matrix[:, 0, 0] = 1
    matrix[:, 1, 1] = 1
    matrix[:, 2, 2] = 1
    matrix[:, 0, 2] = translation[:, 0]
    matrix[:, 1, 2] = translation[:, 1]
    return matrix


def _compute_tensor_center(tensor: torch.Tensor) -> torch.Tensor:
    """Computes the center of tensor plane for (H, W), (C, H, W) and (B, C, H, W)."""
    assert 2 <= len(tensor.shape) <= 4, f"Must be a 3D tensor as HW, CHW and BCHW. Got {tensor.shape}."
    height, width = tensor.shape[-2:]
    center_x: float = float(width - 1) / 2
    center_y: float = float(height - 1) / 2
    center: torch.Tensor = torch.tensor(
        [center_x, center_y],
        device=tensor.device, dtype=tensor.dtype)
    return center


def _compute_scaling_matrix(scale: torch.Tensor,
                            center: torch.Tensor) -> torch.Tensor:
    """Computes affine matrix for scaling."""
    angle: torch.Tensor = torch.zeros(scale.shape[0], device=scale.device, dtype=scale.dtype)
    matrix: torch.Tensor = get_rotation_matrix2d(center, angle, scale)
    return matrix


def _compute_rotation_matrix(angle: torch.Tensor,
                             center: torch.Tensor) -> torch.Tensor:
    """Computes a pure affine rotation matrix."""
    scale: torch.Tensor = torch.ones((angle.shape[0], 2), device=angle.device, dtype=angle.dtype)
    matrix: torch.Tensor = get_rotation_matrix2d(center, angle, scale)
    return matrix


def translate(image, device, d=8):
    is_unbatched: bool = image.ndimension() == 3
    if is_unbatched:
        image = torch.unsqueeze(image, dim=0)

    c = image.shape[0]
    h = image.shape[-2]
    w = image.shape[-1]
    
    trans = image.new_empty((c, 2)).uniform_(-d, d)
    
    translation_matrix: torch.Tensor = _compute_translation_matrix(trans)
    matrix = translation_matrix[..., :2, :3]
    
    data_warp = warp_affine(image, matrix, dsize=(h, w), padding_mode='border')
    
    if is_unbatched:
        data_warp = torch.squeeze(data_warp, dim=0)

    return data_warp


def rotate(image, device, d=8):
    is_unbatched: bool = image.ndimension() == 3
    if is_unbatched:
        image = torch.unsqueeze(image, dim=0)

    c = image.shape[0]
    h = image.shape[-2]
    w = image.shape[-1]
    
    angle = image.new_empty(c).uniform_(-d, d)
    center = image.new_tensor([[w / 2 - 1, h / 2 - 1]]).expand(c, -1)
    
    rotation_matrix: torch.Tensor = _compute_rotation_matrix(angle, center)
    matrix = rotation_matrix[..., :2, :3]
    
    data_warp = warp_affine(image, matrix, dsize=(h, w), padding_mode='border')
    
    if is_unbatched:
        data_warp = torch.squeeze(data_warp, dim=0)

    return data_warp


def perspective(image, device, d=8):
    c = image.shape[0]
    h = image.shape[2]
    w = image.shape[3]
    
    points_src = image.new_tensor([
        [0., 0.], [w - 1., 0.], [w - 1., h - 1.], [0., h - 1.]
    ]).expand(c, -1, -1)
    
    shifts = image.new_empty((c, 4, 2)).uniform_(-d, d)
    base_dst = image.new_tensor([
        [0., 0.], 
        [w - 1., 0.], 
        [w - 1., h - 1.], 
        [0., h - 1.]
    ]).expand(c, -1, -1)
    
    points_dst = base_dst + shifts
    
    image = image.float()
    M = get_perspective_transform(points_src.float(), points_dst.float())
    data_warp = warp_perspective(image, M, dsize=(h, w))
    
    return data_warp


def Light_Distortion(c, embed_image, device):
    device = embed_image.device
    B, C, H, W = embed_image.shape
    a = 0.7 + random.random() * 0.2
    b = 1.1 + random.random() * 0.2
    
    if c == 0:
        direction = random.randint(1, 4)
        i_vec = torch.arange(H, dtype=embed_image.dtype, device=device)
        
        val = -((b - a) / (H - 1)) * (i_vec - W) + a
        mask_2d = val.unsqueeze(1).expand(H, W)
        
        if direction == 1:
            O = mask_2d
        elif direction == 2:
            O = torch.rot90(mask_2d, 1, [0, 1])
        elif direction == 3:
            O = torch.rot90(mask_2d, 2, [0, 1])
        else:
            O = torch.rot90(mask_2d, 3, [0, 1])
            
        return O.unsqueeze(0).unsqueeze(0).expand(B, C, H, W)
    else:
        x = random.randint(0, H - 1)
        y = random.randint(0, W - 1)
        
        max_len = math.sqrt(max(
            x**2 + y**2, 
            (x - (H - 1))**2 + y**2, 
            x**2 + (y - (W - 1))**2, 
            (x - (H - 1))**2 + (y - (W - 1))**2
        ))
        
        Y, X = torch.meshgrid(
            torch.arange(H, dtype=embed_image.dtype, device=device),
            torch.arange(W, dtype=embed_image.dtype, device=device),
            indexing='ij'
        )
        
        dist = torch.sqrt((Y - x)**2 + (X - y)**2)
        mask_2d = dist / max_len * (a - b) + b
        
        return mask_2d.unsqueeze(0).unsqueeze(0).expand(B, C, H, W)


def Moire_Distortion(embed_image, device):
    device = embed_image.device
    B, C, H, W = embed_image.shape
    
    Y, X = torch.meshgrid(
        torch.arange(1, H + 1, dtype=embed_image.dtype, device=device),
        torch.arange(1, W + 1, dtype=embed_image.dtype, device=device),
        indexing='ij'
    )
    
    channels_to_gen = min(3, C)
    
    theta = embed_image.new_empty((channels_to_gen, 1, 1)).uniform_(0, math.pi)
    center_y = embed_image.new_empty((channels_to_gen, 1, 1)).uniform_(0, H)
    center_x = embed_image.new_empty((channels_to_gen, 1, 1)).uniform_(0, W)
    
    dist = torch.sqrt((Y.unsqueeze(0) - center_y)**2 + (X.unsqueeze(0) - center_x)**2)
    z1 = 0.5 + 0.5 * torch.cos(2 * math.pi * dist)
    
    phase = torch.cos(theta) * X.unsqueeze(0) + torch.sin(theta) * Y.unsqueeze(0)
    z2 = 0.5 + 0.5 * torch.cos(phase)
    
    z = torch.minimum(z1, z2)
    M = (z + 1) / 2
    
    if C > 3:
        padded = embed_image.new_zeros((C, H, W))
        padded[:3] = M
        M = padded
    
    return M.unsqueeze(0).expand(B, -1, -1, -1)


class ScreenShooting:
    def __init__(self):
        self._cache_key = None
        self._grid_y0 = torch.empty(0)
        self._grid_x0 = torch.empty(0)
        self._grid_y1 = torch.empty(0)
        self._grid_x1 = torch.empty(0)
        self._perspective_src = torch.empty(0)
        self._perspective_dst_base = torch.empty(0)

    def _prepare_cache(self, image):
        _, _, h, w = image.shape
        key = (image.device, image.dtype, h, w)
        if self._cache_key == key:
            return

        y0 = torch.arange(h, dtype=image.dtype, device=image.device)
        x0 = torch.arange(w, dtype=image.dtype, device=image.device)
        self._grid_y0, self._grid_x0 = torch.meshgrid(y0, x0, indexing='ij')

        y1 = torch.arange(1, h + 1, dtype=image.dtype, device=image.device)
        x1 = torch.arange(1, w + 1, dtype=image.dtype, device=image.device)
        self._grid_y1, self._grid_x1 = torch.meshgrid(y1, x1, indexing='ij')

        self._perspective_src = image.new_tensor([
            [0., 0.], [w - 1., 0.], [w - 1., h - 1.], [0., h - 1.]
        ]).unsqueeze(0)
        self._perspective_dst_base = self._perspective_src.clone()
        self._cache_key = key

    def _perspective(self, image, d=8):
        b, _, h, w = image.shape
        points_src = self._perspective_src.expand(b, -1, -1)
        shifts = image.new_empty((b, 4, 2)).uniform_(-d, d)
        points_dst = self._perspective_dst_base.expand(b, -1, -1) + shifts

        image = image.float()
        matrix = get_perspective_transform(points_src.float(), points_dst.float())
        return warp_perspective(image, matrix, dsize=(h, w))

    def _light_distortion(self, c, image):
        _, _, h, w = image.shape
        a = 0.7 + random.random() * 0.2
        b = 1.1 + random.random() * 0.2

        if c == 0:
            direction = random.randint(1, 4)
            val = -((b - a) / (h - 1)) * (self._grid_y0[:, 0] - w) + a
            mask_2d = val.unsqueeze(1).expand(h, w)

            if direction == 1:
                mask_2d = mask_2d
            elif direction == 2:
                mask_2d = torch.rot90(mask_2d, 1, [0, 1])
            elif direction == 3:
                mask_2d = torch.rot90(mask_2d, 2, [0, 1])
            else:
                mask_2d = torch.rot90(mask_2d, 3, [0, 1])
        else:
            x = random.randint(0, h - 1)
            y = random.randint(0, w - 1)
            max_len = math.sqrt(max(
                x**2 + y**2,
                (x - (h - 1))**2 + y**2,
                x**2 + (y - (w - 1))**2,
                (x - (h - 1))**2 + (y - (w - 1))**2
            ))

            dist = torch.sqrt((self._grid_y0 - x)**2 + (self._grid_x0 - y)**2)
            mask_2d = dist / max_len * (a - b) + b

        return mask_2d.unsqueeze(0).unsqueeze(0)

    def _moire_distortion(self, image):
        _, channels, h, w = image.shape
        channels_to_gen = min(3, channels)

        theta = image.new_empty((channels_to_gen, 1, 1)).uniform_(0, math.pi)
        center_y = image.new_empty((channels_to_gen, 1, 1)).uniform_(0, h)
        center_x = image.new_empty((channels_to_gen, 1, 1)).uniform_(0, w)

        y_grid = self._grid_y1.unsqueeze(0)
        x_grid = self._grid_x1.unsqueeze(0)
        dist = torch.sqrt((y_grid - center_y)**2 + (x_grid - center_x)**2)
        z1 = 0.5 + 0.5 * torch.cos(2 * math.pi * dist)

        phase = torch.cos(theta) * x_grid + torch.sin(theta) * y_grid
        z2 = 0.5 + 0.5 * torch.cos(phase)

        moire = (torch.minimum(z1, z2) + 1) / 2
        if channels > channels_to_gen:
            padded = image.new_zeros((channels, h, w))
            padded[:channels_to_gen] = moire
            moire = padded

        return moire.unsqueeze(0)

    def forward(self, image):
        self._prepare_cache(image)
              
        # perspective transform
        noised_image = self._perspective(image, 2)

        # Light Distortion
        c = random.randint(0, 1)
        L = self._light_distortion(c, image)

        # Moire Distortion
        Z = self._moire_distortion(image) * 2 - 1
        
        # Mingle Light and Moire
        noised_image = noised_image * L * 0.85 + Z * 0.15

        # Gaussian noise
        noised_image = noised_image + (0.001**0.5) * torch.randn_like(noised_image)

        return noised_image


def add_pimog_noise(image, noise_level=0.1):
    """
    添加PIMOG (Perceptually Important Map Guided) 噪声
    与pimog.py中的ScreenShooting实现保持一致
    
    Args:
        image: 输入图像 (H, W, C) 或 (H, W)
        noise_level: 噪声强度，范围 0-1
        
    Returns:
        加噪后的图像
    """
    # 转换为tensor
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # 转换为tensor并归一化到[-1, 1]
    image_tensor = transforms.ToTensor()(image)
    image_tensor = transforms.Normalize(mean=[0.5], std=[0.5])(image_tensor)
    image_tensor = image_tensor.unsqueeze(0)  # 添加batch维度
    
    # 应用PIMOG噪声
    device = image_tensor.device
    shooter = ScreenShooting()
    noised_tensor = shooter.forward(image_tensor)
    
    # 反归一化并转换回numpy
    noised_tensor = noised_tensor.squeeze(0)
    noised_image = (noised_tensor * 0.5 + 0.5) * 255
    noised_image = torch.clip(noised_image, 0, 255)
    noised_image = noised_image.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    
    return noised_image


def add_jpeg_compression_noise(image, quality=50):
    """
    添加JPEG压缩噪声
    
    Args:
        image: 输入图像 (H, W, C) 或 (H, W)
        quality: JPEG压缩质量，范围 0-100
        
    Returns:
        压缩后的图像
    """
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # 编码为JPEG
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    result, encimg = cv2.imencode('.jpg', image, encode_param)
    
    # 解码
    decimg = cv2.imdecode(encimg, 1)
    
    return decimg


def add_noise_to_batch(images, noise_type='pimog', noise_level=0.1):
    """
    批量添加噪声
    
    Args:
        images: 输入图像批次 (B, C, H, W)
        noise_type: 噪声类型，可选 'pimog' 或 'jpeg'
        noise_level: 噪声强度
        
    Returns:
        加噪后的图像批次
    """
    noisy_images = []
    
    for i in range(images.size(0)):
        # 转换为numpy数组并反归一化
        img = images[i].cpu().numpy()
        img = (img * 0.5 + 0.5) * 255  # 从 [-1, 1] 转换到 [0, 255]
        img = img.transpose(1, 2, 0).astype(np.uint8)
        
        # 添加噪声
        if noise_type == 'pimog':
            noisy_img = add_pimog_noise(img, noise_level)
        elif noise_type == 'jpeg':
            # 对于JPEG，使用噪声强度作为质量参数
            quality = max(1, int(100 - noise_level * 99))
            noisy_img = add_jpeg_compression_noise(img, quality)
        elif noise_type == 'tile_rotate_crop':
            noisy_img = add_tile_rotate_crop_noise(img)
        else:
            noisy_img = img
        
        # 转换回tensor并归一化
        noisy_img = transforms.ToTensor()(noisy_img)
        noisy_img = transforms.Normalize(mean=[0.5], std=[0.5])(noisy_img)
        noisy_images.append(noisy_img)
    
    return torch.stack(noisy_images).to(images.device)


def apply_noise_during_training(images, noise_prob=0.5, noise_level=0.1):
    """
    在训练过程中随机应用噪声
    
    Args:
        images: 输入图像批次 (B, C, H, W)
        noise_prob: 应用噪声的概率
        noise_level: 噪声强度
        
    Returns:
        处理后的图像批次
    """
    if np.random.random() < noise_prob:
        # 随机选择噪声类型
        noise_type = np.random.choice(['pimog', 'jpeg', 'tile_rotate_crop'])
        images = add_noise_to_batch(images, noise_type, noise_level)
    
    return images


# ── 微信压缩模拟 (向量化版本, 无 Python 循环) ──────────────────

import torch
import torch.nn.functional as F_torch

# 预计算 DCT 矩阵 (8×8, 正交归一化)
_DCT_MAT_NP = np.zeros((8, 8), dtype=np.float64)
for u in range(8):
    for x in range(8):
        alpha = math.sqrt(1.0 / 8.0) if u == 0 else math.sqrt(2.0 / 8.0)
        _DCT_MAT_NP[u, x] = alpha * math.cos((2 * x + 1) * u * math.pi / 16.0)
_DCT_MAT_T_NP = _DCT_MAT_NP.T.copy()

# 预计算高频清零掩码 (8×8, True=保留, False=置零)
_JPEG_ZIGZAG_IDX = [
    (0,0),(0,1),(1,0),(2,0),(1,1),(0,2),(0,3),(1,2),
    (2,1),(3,0),(4,0),(3,1),(2,2),(1,3),(0,4),(0,5),
    (1,4),(2,3),(3,2),(4,1),(5,0),(6,0),(5,1),(4,2),
    (3,3),(2,4),(1,5),(0,6),(0,7),(1,6),(2,5),(3,4),
    (4,3),(5,2),(6,1),(7,0),(7,1),(6,2),(5,3),(4,4),
    (3,5),(2,6),(1,7),(2,7),(3,6),(4,5),(5,4),(6,3),
    (7,2),(7,3),(6,4),(5,5),(4,6),(3,7),(4,7),(5,6),
    (6,5),(7,4),(7,5),(6,6),(5,7),(6,7),(7,6),(7,7),
]

# 预计算各 zigzag_keep 值的掩码缓存
_ZIGZAG_MASK_CACHE = {}

def _get_zigzag_mask(zigzag_keep):
    """获取高频清零掩码 (8,8) float64, 1.0=保留, 0.0=置零"""
    if zigzag_keep not in _ZIGZAG_MASK_CACHE:
        mask = np.zeros((8, 8), dtype=np.float64)
        for k in range(min(zigzag_keep, 64)):
            r, c = _JPEG_ZIGZAG_IDX[k]
            mask[r, c] = 1.0
        _ZIGZAG_MASK_CACHE[zigzag_keep] = mask
    return _ZIGZAG_MASK_CACHE[zigzag_keep]

# 标准 JPEG 亮度/色度量化表 (8×8 rowmajor, 与 wechat_jpeg_encode_once_hf_zero.py 一致)
_Q_LUMINANCE_8x8 = np.array([
    [13, 9, 8, 13, 19, 32, 41, 49],
    [10, 10, 11, 15, 21, 46, 48, 44],
    [11, 10, 13, 19, 32, 46, 55, 45],
    [11, 14, 18, 23, 41, 70, 64, 50],
    [14, 18, 30, 45, 54, 87, 82, 62],
    [19, 28, 44, 51, 65, 83, 90, 74],
    [39, 51, 62, 70, 82, 97, 96, 81],
    [58, 74, 76, 78, 90, 80, 82, 79],
], dtype=np.float64)

_Q_CHROMINANCE_8x8 = np.array([
    [14, 14, 19, 38, 79, 79, 79, 79],
    [14, 17, 21, 53, 79, 79, 79, 79],
    [19, 21, 45, 79, 79, 79, 79, 79],
    [38, 53, 79, 79, 79, 79, 79, 79],
    [79, 79, 79, 79, 79, 79, 79, 79],
    [79, 79, 79, 79, 79, 79, 79, 79],
    [79, 79, 79, 79, 79, 79, 79, 79],
    [79, 79, 79, 79, 79, 79, 79, 79],
], dtype=np.float64)


def _block_dct_hf_zero_vec(plane, zigzag_keep):
    """
    向量化 8×8 分块 DCT → 高频清零 → IDCT (无 Python 循环)

    Args:
        plane: (H, W) float64, 值域 [0, 255]
        zigzag_keep: 保留的之字形系数个数 (1-64)

    Returns:
        (H, W) float64, 值域 [0, 255]
    """
    h, w = plane.shape
    ph = int(math.ceil(h / 8.0) * 8)
    pw = int(math.ceil(w / 8.0) * 8)

    # pad 并减去 128
    padded = np.zeros((ph, pw), dtype=np.float64)
    padded[:h, :w] = plane - 128.0

    # reshape 为 (nbh, 8, nbw, 8) → transpose → (nbh*nbw, 8, 8)
    nbh, nbw = ph // 8, pw // 8
    blocks = padded.reshape(nbh, 8, nbw, 8).transpose(0, 2, 1, 3).reshape(-1, 8, 8)

    # DCT: F = C @ f @ C.T  (批量矩阵乘法)
    dct_coeff = np.einsum('ij,bjk,kl->bil', _DCT_MAT_NP, blocks, _DCT_MAT_T_NP)

    # 高频清零
    mask = _get_zigzag_mask(zigzag_keep)
    dct_coeff *= mask

    # IDCT: f = C.T @ F @ C
    recon = np.einsum('ij,bjk,kl->bil', _DCT_MAT_T_NP, dct_coeff, _DCT_MAT_NP)

    # reshape 回图像
    recon = recon.reshape(nbh, nbw, 8, 8).transpose(0, 2, 1, 3).reshape(ph, pw)

    return np.clip(recon[:h, :w] + 128.0, 0.0, 255.0)


def _chroma_420_downsample_vec(plane):
    """4:2:0 色度下采样 (向量化, 2×2 块取均值)"""
    h, w = plane.shape
    # pad 到偶数尺寸
    h_even = h + (h % 2)
    w_even = w + (w % 2)
    padded = np.zeros((h_even, w_even), dtype=plane.dtype)
    padded[:h, :w] = plane
    ch, cw = h_even // 2, w_even // 2
    return padded.reshape(ch, 2, cw, 2).mean(axis=(1, 3))


def _chroma_420_upsample_vec(plane2, h, w):
    """4:2:0 色度上采样 (np.repeat)"""
    return np.repeat(np.repeat(plane2, 2, axis=0), 2, axis=1)[:h, :w]


def _jpeg_quantize_sim(plane, qtable_8x8, quality):
    """
    模拟 JPEG 量化: DCT → 量化(四舍五入) → 反量化 → IDCT
    等效于 JPEG 编码的量化损失, 但不需要真正编解码
    """
    h, w = plane.shape
    ph = int(math.ceil(h / 8.0) * 8)
    pw = int(math.ceil(w / 8.0) * 8)
    padded = np.zeros((ph, pw), dtype=np.float64)
    padded[:h, :w] = plane - 128.0

    # 根据 quality 计算缩放因子 (标准 JPEG 缩放公式)
    if quality < 50:
        scale = 5000.0 / quality
    else:
        scale = 200.0 - 2.0 * quality
    q_scaled = np.floor((qtable_8x8 * scale + 50.0) / 100.0)
    q_scaled = np.clip(q_scaled, 1, 255)

    nbh, nbw = ph // 8, pw // 8
    blocks = padded.reshape(nbh, 8, nbw, 8).transpose(0, 2, 1, 3).reshape(-1, 8, 8)

    # DCT
    dct_coeff = np.einsum('ij,bjk,kl->bil', _DCT_MAT_NP, blocks, _DCT_MAT_T_NP)

    # 量化 + 反量化
    dct_quant = np.round(dct_coeff / q_scaled) * q_scaled

    # IDCT
    recon = np.einsum('ij,bjk,kl->bil', _DCT_MAT_T_NP, dct_quant, _DCT_MAT_NP)
    recon = recon.reshape(nbh, nbw, 8, 8).transpose(0, 2, 1, 3).reshape(ph, pw)

    return np.clip(recon[:h, :w] + 128.0, 0.0, 255.0)


def _preprocess_hf_zero_rgb(rgb, zigzag_keep, quality):
    """
    RGB uint8 → 高频清零 + JPEG 量化模拟 → RGB uint8

    流程: RGB → YCbCr → 4:2:0 下采样 → DCT 高频清零 → JPEG量化 → IDCT → 上采样 → RGB
    """
    r, g, b = rgb[..., 0].astype(np.float64), rgb[..., 1].astype(np.float64), rgb[..., 2].astype(np.float64)
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 128.0
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 128.0

    # 4:2:0 下采样
    cb2 = _chroma_420_downsample_vec(cb.astype(np.float64))
    cr2 = _chroma_420_downsample_vec(cr.astype(np.float64))

    # DCT 高频清零
    yq = _block_dct_hf_zero_vec(y, zigzag_keep)
    cbq = _block_dct_hf_zero_vec(cb2, zigzag_keep)
    crq = _block_dct_hf_zero_vec(cr2, zigzag_keep)

    # JPEG 量化模拟
    yq = _jpeg_quantize_sim(yq, _Q_LUMINANCE_8x8, quality)
    cbq = _jpeg_quantize_sim(cbq, _Q_CHROMINANCE_8x8, quality)
    crq = _jpeg_quantize_sim(crq, _Q_CHROMINANCE_8x8, quality)

    # 上采样 + 合并
    hh, ww = y.shape
    cb_up = _chroma_420_upsample_vec(cbq, hh, ww)
    cr_up = _chroma_420_upsample_vec(crq, hh, ww)

    out_r = y + 1.402 * (cr_up - 128.0)
    out_g = y - 0.344136 * (cb_up - 128.0) - 0.714136 * (cr_up - 128.0)
    out_b = y + 1.772 * (cb_up - 128.0)
    out = np.stack([out_r, out_g, out_b], axis=-1)
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def _whole_plane_dct_hf_zero(plane, keep_ratio):
    """
    整图 DCT → 高频清零 → IDCT (快速版本, ~9ms/512x512)

    Args:
        plane: (H, W) float64, 值域 [0, 255]
        keep_ratio: 保留低频比例 (0-1)

    Returns:
        (H, W) float64, 值域 [0, 255]
    """
    from scipy.fft import dctn, idctn
    h, w = plane.shape
    coeff = dctn(plane - 128.0, type=2, norm='ortho')
    kh = max(1, int(h * keep_ratio ** 0.5))
    kw = max(1, int(w * keep_ratio ** 0.5))
    coeff[kh:, :] = 0.0
    coeff[:, kw:] = 0.0
    recon = idctn(coeff, type=2, norm='ortho')
    return np.clip(recon + 128.0, 0.0, 255.0)


def add_wechat_noise(image, quality=60, zigzag_keep=21, **kwargs):
    """
    模拟微信JPEG压缩噪声

    流程：
    1. 下采样 512→256 再上采样 256→512 (模拟微信传输的分辨率损失)
    2. RGB → YCbCr (BT.601) → 4:2:0 色度下采样
    3. 整图 DCT → 高频清零 → IDCT (保留 zigzag_keep/64 比例的低频)
    4. 色度上采样 → YCbCr → RGB

    Args:
        image: 输入图像 (H, W, C) uint8
        quality: JPEG 编码质量 (未使用, 保留接口兼容)
        zigzag_keep: DCT 之字形保留系数个数 (1-64), 默认 21

    Returns:
        压缩后的图像 (H, W, C) uint8
    """
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    h, w = image.shape[:2]

    # 1. 下采样→上采样 (模拟微信分辨率损失)
    small = cv2.resize(image, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    image = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

    # 2-4. DCT 高频清零
    keep_ratio = zigzag_keep / 64.0
    img_f = image.astype(np.float64)
    r, g, b = img_f[..., 0], img_f[..., 1], img_f[..., 2]

    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 128.0
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 128.0

    cb2 = _chroma_420_downsample_vec(cb)
    cr2 = _chroma_420_downsample_vec(cr)

    yq = _whole_plane_dct_hf_zero(y, keep_ratio)
    cbq = _whole_plane_dct_hf_zero(cb2, keep_ratio)
    crq = _whole_plane_dct_hf_zero(cr2, keep_ratio)

    hh, ww = y.shape
    cb_up = _chroma_420_upsample_vec(cbq, hh, ww)
    cr_up = _chroma_420_upsample_vec(crq, hh, ww)

    out_r = yq + 1.402 * (cr_up - 128.0)
    out_g = yq - 0.344136 * (cb_up - 128.0) - 0.714136 * (cr_up - 128.0)
    out_b = yq + 1.772 * (cb_up - 128.0)
    out = np.stack([out_r, out_g, out_b], axis=-1)
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def add_perspective_noise(image, d_range=(0, 0.2)):
    """
    模拟透视变换噪声（stub，降级为轻微仿射变换）
    """
    h, w = image.shape[:2]
    d_ratio = np.random.uniform(d_range[0], d_range[1])
    d_px = d_ratio * min(h, w)

    src_pts = np.float32([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]])
    dst_pts = src_pts + np.random.uniform(-d_px, d_px, src_pts.shape).astype(np.float32)
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return cv2.warpPerspective(image, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)


def add_tile_rotate_crop_noise(image, angle_range=(-180, 180), crop_scale_range=None):
    """
    模拟循环平移+旋转噪声：将图像拼成3x3大图，随机旋转后从中间裁剪。

    原理：
    - 3x3拼接使中心区域四周都有内容，旋转后裁剪不会出现黑边
    - 等效于先随机平移（循环移位）再随机旋转，增强模型对水印位置和角度的鲁棒性

    Args:
        image: 输入图像 (H, W, C) 或 (H, W)，numpy数组
        angle_range: 旋转角度范围（度），默认 (-180, 180)
        crop_scale_range: 裁剪尺寸浮动范围，默认 None (不浮动)
                         例如 (0.9, 1.1) 表示裁剪边长为原尺寸的 0.9~1.1 倍

    Returns:
        裁剪后的图像，尺寸与输入相同
    """
    h, w = image.shape[:2]
    is_gray = len(image.shape) == 2

    # 1. 拼成3x3大图
    if is_gray:
        tiled = np.tile(image, (3, 3))
    else:
        tiled = np.tile(image, (3, 3, 1))

    # 2. 随机旋转
    angle = np.random.uniform(angle_range[0], angle_range[1])
    center = (tiled.shape[1] / 2, tiled.shape[0] / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(tiled, M, (tiled.shape[1], tiled.shape[0]),
                             borderMode=cv2.BORDER_WRAP)

    # 3. 计算裁剪尺寸
    if crop_scale_range is not None:
        # 浮动尺寸裁剪
        scale = np.random.uniform(crop_scale_range[0], crop_scale_range[1])
        crop_h = int(h * scale)
        crop_w = int(w * scale)
    else:
        # 原始尺寸裁剪
        crop_h = h
        crop_w = w

    # 4. 从中心区域随机偏移裁剪（偏移范围限制在±0.5个原始尺寸内，保证裁剪区域完全在3x3图内）
    # 确保裁剪区域不超出3x3图边界
    max_offset_x = min(w // 2, (3 * w - crop_w) // 2 - 1)
    max_offset_y = min(h // 2, (3 * h - crop_h) // 2 - 1)

    crop_cx = w + np.random.randint(-max_offset_x, max_offset_x + 1)
    crop_cy = h + np.random.randint(-max_offset_y, max_offset_y + 1)
    x1 = crop_cx - crop_w // 2
    y1 = crop_cy - crop_h // 2

    # 边界检查
    x1 = max(0, min(x1, 3 * w - crop_w))
    y1 = max(0, min(y1, 3 * h - crop_h))

    cropped = rotated[y1:y1 + crop_h, x1:x1 + crop_w]

    # 如果裁剪尺寸与原图不同，resize回原尺寸
    if crop_h != h or crop_w != w:
        cropped = cv2.resize(cropped, (w, h))

    return cropped
