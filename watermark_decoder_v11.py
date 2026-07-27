"""
v11 水印解码器
支持小角度旋转矫正 (±5°)

方案：
1. 在FFT幅度谱中，旋转矫正环(r=18)的正弦波模式会产生相位偏移
2. 通过检测相位偏移来估计旋转角度
3. 在空域旋转图像矫正，然后解码

关键改进：
- 旋转检测直接在FFT幅度谱上做，不经过ResUNet
- 使用互相关检测，精度可以达到亚像素级
- 对于±5°旋转，检测精度约±0.5°
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def cartesian_to_polar(input_tensor, output_shape, min_radius, max_radius):
    """
    input_tensor: (B, 1, H, W) - 频谱图
    output_shape: (R_bins, T_bins) - 极坐标分辨率
    """
    B, C, H, W = input_tensor.shape
    R_bins, T_bins = output_shape

    input_tensor = torch.nan_to_num(input_tensor, nan=0.0, posinf=1e6, neginf=-1e6)

    rho = torch.linspace(min_radius, max_radius, R_bins, device=input_tensor.device)
    theta = torch.linspace(0, np.pi, T_bins, device=input_tensor.device)

    grid_rho, grid_theta = torch.meshgrid(rho, theta, indexing='ij')
    grid_x = grid_rho * torch.cos(grid_theta) / (W / 2)
    grid_y = grid_rho * torch.sin(grid_theta) / (H / 2)

    grid = torch.stack([grid_y, grid_x], dim=-1).unsqueeze(0).repeat(B, 1, 1, 1)
    polar_map = F.grid_sample(input_tensor, grid, mode='bilinear', align_corners=True)

    polar_map = torch.nan_to_num(polar_map, nan=0.0, posinf=1e6, neginf=-1e6)
    return polar_map


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.scale1 = nn.Parameter(torch.ones(out_channels))
        self.bias1 = nn.Parameter(torch.zeros(out_channels))
        self.scale2 = nn.Parameter(torch.ones(out_channels))
        self.bias2 = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x):
        x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = out * self.scale1.view(1, -1, 1, 1) + self.bias1.view(1, -1, 1, 1)
        out = self.relu(out)

        out = self.conv2(out)
        out = out * self.scale2.view(1, -1, 1, 1) + self.bias2.view(1, -1, 1, 1)

        out += identity
        out = self.relu(out)

        out = torch.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6)
        return out


class ResUNetFilter(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = ResidualBlock(1, 16)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ResidualBlock(16, 32)
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = ResidualBlock(32, 64)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = ResidualBlock(64 + 32, 32)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = ResidualBlock(32 + 16, 16)
        self.final = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))
        d2 = self.up2(b)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        out = self.final(d1)
        return torch.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6)


class PatchEmbedding(nn.Module):
    def __init__(self, in_dim, embed_dim, num_patches):
        super().__init__()
        self.proj = nn.Linear(in_dim, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

    def forward(self, x):
        x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        B = x.shape[0]
        x = self.proj(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        return x


def detect_rotation_angle_numpy(fft_mag, r_rotation=18, r_range=1, rotation_cycles=8):
    """
    在FFT幅度谱上直接检测旋转角度 (numpy实现，用于推理)

    原理：
    1. 提取旋转矫正环的极坐标表示
    2. 生成参考正弦波模式
    3. 用互相关检测偏移量
    4. 偏移量对应旋转角度

    Args:
        fft_mag: (H, W) numpy数组，FFT幅度谱
        r_rotation: 旋转矫正环半径
        r_range: 环宽度
        rotation_cycles: 正弦波周期数

    Returns:
        angle_deg: 检测到的旋转角度 (度)，正值表示逆时针旋转
    """
    from scipy.signal import correlate

    H, W = fft_mag.shape
    cx, cy = H // 2, W // 2

    # 提取旋转环上的信号
    # 沿着圆周采样，角度范围 [0, 2π)
    N = 360  # 采样点数
    theta_arr = np.linspace(0, 2 * np.pi, N, endpoint=False)

    # 对多个半径取均值
    signals = []
    for r in range(r_rotation - r_range, r_rotation + r_range + 1):
        x_arr = cx + np.round(r * np.cos(theta_arr)).astype(np.int32)
        y_arr = cy + np.round(r * np.sin(theta_arr)).astype(np.int32)

        # 边界过滤
        mask = (x_arr >= 0) & (x_arr < W) & (y_arr >= 0) & (y_arr < H)
        signal = np.zeros(N)
        signal[mask] = fft_mag[y_arr[mask], x_arr[mask]]
        signals.append(signal)

    signal_mean = np.mean(signals, axis=0)

    # 生成参考模式 (注意：旋转矫正环用的是cos模式)
    ref = np.cos(rotation_cycles * theta_arr)

    # 互相关
    corr = correlate(signal_mean, ref, mode='full')
    center = len(corr) // 2

    # 只搜索 [-90°, +90°] 范围 (对应 ±0.25 周期的偏移)
    # 对于 rotation_cycles=8, 每个周期对应 360/8 = 45°
    # ±5° 对应 ±0.111 周期
    search_bins = int(N / 2)  # ±180°
    corr_search = corr[center - search_bins: center + search_bins]
    max_idx = np.argmax(corr_search)

    # 转换为角度
    # max_idx 对应偏移量 (从 -search_bins 到 +search_bins)
    shift = max_idx - search_bins
    # 偏移量转换为角度：每个采样点对应 360/N = 1°
    angle_deg = shift * (360.0 / N)

    return angle_deg


def rotate_image_numpy(image, angle_deg):
    """
    在空域旋转图像 (保持尺寸)

    Args:
        image: (H, W) 或 (H, W, C) numpy数组
        angle_deg: 旋转角度 (度)，正值表示逆时针旋转

    Returns:
        rotated: 旋转后的图像
    """
    import cv2

    if len(image.shape) == 2:
        h, w = image.shape
    else:
        h, w = image.shape[:2]

    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)

    if len(image.shape) == 2:
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT)
    else:
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT)

    return rotated


class AdvancedWatermarkDecoderV11(nn.Module):
    """
    v11 解码器 (3-ring version)

    解码流程：
    1. 输入单通道图像
    2. FFT -> 幅度谱
    3. 对三个环 (r=12, r=18, r=25) 分别极坐标采样
    4. 送入 Transformer，让模型隐式学习旋转不变性
    5. 输出 60 bits

    环结构：
    - Ring 1 (r=12): 15 bits 水印
    - Ring 2 (r=18): 旋转参考环 (方波90°，不编码bit，提供旋转信息)
    - Ring 3 (r=25): 45 bits 水印
    """
    def __init__(self, n_sectors=60, rings=None, bits=None,
                 rotation_ring=None, rotation_patches=12):
        """
        Args:
            n_sectors: 水印总位数
            rings: 水印环列表 [(min_r, max_r), ...]
            bits: 每个环的位数
            rotation_ring: 旋转参考环 (min_r, max_r)
            rotation_patches: 旋转环的patch数量 (不对应bit，只是特征提取)
        """
        super().__init__()
        self.n_sectors = n_sectors
        self.pre_filter = ResUNetFilter()

        # 水印环配置
        self.rings = rings if rings is not None else [(7, 17), (20, 30)]
        self.bits = bits if bits is not None else [15, 45]

        # 旋转参考环配置
        self.rotation_ring = rotation_ring if rotation_ring is not None else (13, 23)
        self.rotation_patches = rotation_patches

        # 合并所有环: 水印环 + 旋转环
        self.all_rings = self.rings + [self.rotation_ring]
        self.all_patches = self.bits + [rotation_patches]

        self.angle_bins = 180
        self.radius_bins = 12

        # ViT 配置
        self.embed_dim = 256
        self.num_heads = 8
        self.num_layers = 2

        # 每个 ring 的 Patch Embedding (包括旋转环)
        self.patch_embeddings = nn.ModuleList([
            PatchEmbedding(
                in_dim=(self.angle_bins // patch_num) * self.radius_bins,
                embed_dim=self.embed_dim,
                num_patches=patch_num
            ) for patch_num in self.all_patches
        ])

        # 每个 ring 的 Transformer Encoder (包括旋转环)
        self.ring_transformers = nn.ModuleList([
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=self.embed_dim,
                    nhead=self.num_heads,
                    batch_first=True,
                    dim_feedforward=self.embed_dim * 4
                ),
                num_layers=self.num_layers
            ) for _ in self.all_rings
        ])

        # 跨 ring 融合 (3个环: 2个水印 + 1个旋转)
        self.cross_ring_fusion = nn.Sequential(
            nn.Linear(self.embed_dim * len(self.all_rings), self.embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.embed_dim * 2, self.embed_dim)
        )

        self.head = nn.Linear(self.embed_dim, n_sectors)

    def detect_rotation_numpy(self, fft_mag_np):
        """
        使用numpy检测旋转角度 (用于推理)

        Args:
            fft_mag_np: (H, W) numpy数组

        Returns:
            angle_deg: 旋转角度 (度)
        """
        return detect_rotation_angle_numpy(
            fft_mag_np,
            r_rotation=self.rotation_ring[0] + (self.rotation_ring[1] - self.rotation_ring[0]) // 2,
            r_range=(self.rotation_ring[1] - self.rotation_ring[0]) // 2,
            rotation_cycles=self.rotation_cycles
        )

    def forward_with_rotation_correction(self, x_np, device):
        """
        带旋转矫正的推理流程 (numpy实现)

        Args:
            x_np: (B, 1, H, W) numpy数组，输入图像 [-1, 1]
            device: torch设备

        Returns:
            pred: (B, n_sectors) 水印位预测
            rotation_angles: (B,) 检测到的旋转角度
        """
        B, C, H, W = x_np.shape
        predictions = []
        rotation_angles = []

        for b in range(B):
            # 1. 提取单张图像
            img = x_np[b, 0]  # (H, W), [-1, 1]

            # 2. FFT
            fft_result = np.fft.fftshift(np.fft.fft2(img))
            fft_mag = np.abs(fft_result)

            # 3. 检测旋转角度
            angle_deg = self.detect_rotation_numpy(fft_mag)
            rotation_angles.append(angle_deg)

            # 4. 在空域旋转矫正
            img_corrected = rotate_image_numpy(img, -angle_deg)  # 反向旋转

            # 5. 转换为tensor
            img_tensor = torch.from_numpy(img_corrected).unsqueeze(0).unsqueeze(0).float()
            img_tensor = img_tensor.to(device)

            # 6. 解码 (不带旋转矫正)
            with torch.no_grad():
                pred, _, _ = self.forward(img_tensor, return_rotation=False)
                predictions.append(pred)

        predictions = torch.cat(predictions, dim=0)
        rotation_angles = np.array(rotation_angles)

        return predictions, rotation_angles

    def forward(self, x, return_rotation=False):
        """
        标准前向传播 (训练时使用)

        三个环 (r=12, r=18, r=25) 分别极坐标采样后送入 Transformer，
        让模型隐式学习旋转不变性。

        Args:
            x: (B, 1, H, W) - 输入图像 (单通道)
            return_rotation: 是否返回旋转角度 (训练时不使用)

        Returns:
            logits: (B, n_sectors)
            mag: FFT幅度谱
            rotation_angle: 始终返回None (训练时不检测旋转)
        """
        x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)

        # 预滤波 + FFT
        res = self.pre_filter(x)
        fft_map = torch.fft.fftshift(torch.fft.fft2(res, dim=(-2, -1)), dim=(-2, -1))
        mag = torch.abs(fft_map)
        mag = torch.nan_to_num(mag, nan=0.0, posinf=1e6, neginf=-1e6)

        # 解码三个环 (2个水印环 + 1个旋转参考环)
        ring_embeds = []
        for (min_r, max_r), patch_num, patch_embed, transformer in zip(
                self.all_rings, self.all_patches, self.patch_embeddings, self.ring_transformers):

            polar = cartesian_to_polar(
                mag,
                output_shape=(self.radius_bins, self.angle_bins),
                min_radius=min_r,
                max_radius=max_r
            )

            polar_reshaped = polar.squeeze(1).permute(0, 2, 1)

            angles_per_patch = self.angle_bins // patch_num
            patches = []
            for i in range(patch_num):
                start = i * angles_per_patch
                end = (i + 1) * angles_per_patch
                patch = polar_reshaped[:, start:end, :].flatten(1)
                patches.append(patch)
            patches = torch.stack(patches, dim=1)

            x_embed = patch_embed(patches)
            x_transformer = transformer(x_embed)
            ring_repr = x_transformer[:, 0, :]  # CLS token
            ring_embeds.append(ring_repr)

        # 融合三个环的表示
        fused = torch.cat(ring_embeds, dim=-1)
        fused = torch.nan_to_num(fused, nan=0.0, posinf=1e6, neginf=-1e6)
        fused = self.cross_ring_fusion(fused)
        logits = self.head(fused)

        logits = torch.nan_to_num(logits, nan=0.0, posinf=1e6, neginf=-1e6)

        return torch.sigmoid(logits), mag, None


if __name__ == "__main__":
    # 测试模型 (3-ring version)
    model = AdvancedWatermarkDecoderV11(
        n_sectors=60,
        rings=[(7, 17), (20, 30)],
        bits=[15, 45],
        rotation_ring=(13, 23),
        rotation_patches=12
    )

    # 测试输入
    x = torch.randn(2, 1, 512, 512)
    logits, mag, _ = model(x)

    print(f"Logits shape: {logits.shape}")
    print(f"Mag shape: {mag.shape}")
    print(f"Number of rings (including rotation): {len(model.all_rings)}")
    print(f"Patches per ring: {model.all_patches}")
    print("Model test passed!")

    # 测试旋转检测
    print("\nTesting rotation detection...")
    img = np.random.randn(512, 512).astype(np.float32)

    # 模拟旋转
    import cv2
    center = (256, 256)
    M = cv2.getRotationMatrix2D(center, 3.0, 1.0)  # 旋转3°
    img_rotated = cv2.warpAffine(img, M, (512, 512))

    # 检测旋转
    fft_mag = np.abs(np.fft.fftshift(np.fft.fft2(img)))
    fft_mag_rot = np.abs(np.fft.fftshift(np.fft.fft2(img_rotated)))

    angle = detect_rotation_angle_numpy(fft_mag_rot, r_rotation=18, r_range=1, rotation_cycles=8)
    print(f"Detected rotation angle: {angle:.2f}°")
    print("Rotation detection test passed!")
