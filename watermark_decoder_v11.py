"""
v11 水印解码器
在v10解码器基础上增加旋转矫正功能

环结构:
- Ring 1: r=12, 15 bits 水印
- Ring 2: r=18, 旋转矫正环 (固定正弦波模式)
- Ring 3: r=25, 45 bits 水印

解码流程:
1. FFT -> 极坐标
2. 从旋转矫正环检测旋转角度
3. 在极坐标中旋转矫正
4. 解码水印bits
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


class RotationDetector(nn.Module):
    """
    旋转角度检测器
    从旋转矫正环的极坐标表示中预测旋转角度
    """
    def __init__(self, radius_bins=12, angle_bins=180):
        super().__init__()
        self.radius_bins = radius_bins
        self.angle_bins = angle_bins

        # 简单的CNN + 全连接层
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * angle_bins, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()  # 输出 [0, 1]，对应 [0, 180) 度
        )

    def forward(self, rotation_polar):
        """
        Args:
            rotation_polar: (B, radius_bins, angle_bins) - 旋转环的极坐标表示

        Returns:
            angle_normalized: (B, 1) - 归一化的角度 [0, 1]
        """
        # 沿半径方向取均值 -> (B, 1, angle_bins)
        x = rotation_polar.mean(dim=1, keepdim=True)

        # CNN
        x = self.conv(x)  # (B, 32, angle_bins)

        # Flatten + FC
        x = x.flatten(1)
        angle = self.fc(x)

        return angle


class AdvancedWatermarkDecoderV11(nn.Module):
    """
    v11 解码器
    增加旋转矫正功能
    """
    def __init__(self, n_sectors=60, rings=None, bits=None,
                 rotation_ring=None, rotation_cycles=8):
        """
        Args:
            n_sectors: 水印总位数
            rings: 水印环列表 [(min_r, max_r), ...]
            bits: 每个环的位数
            rotation_ring: 旋转矫正环 (min_r, max_r)
            rotation_cycles: 旋转矫正环的周期数
        """
        super().__init__()
        self.n_sectors = n_sectors
        self.pre_filter = ResUNetFilter()

        # 水印环配置
        self.rings = rings if rings is not None else [(7, 17), (20, 30)]
        self.bits = bits if bits is not None else [15, 45]

        # 旋转矫正环配置
        self.rotation_ring = rotation_ring if rotation_ring is not None else (13, 23)
        self.rotation_cycles = rotation_cycles

        self.angle_bins = 180
        self.radius_bins = 12

        # 旋转角度检测器
        self.rotation_detector = RotationDetector(
            radius_bins=self.radius_bins,
            angle_bins=self.angle_bins
        )

        # ViT 配置
        self.embed_dim = 256
        self.num_heads = 8
        self.num_layers = 2

        # 每个 ring 的 Patch Embedding
        self.patch_embeddings = nn.ModuleList([
            PatchEmbedding(
                in_dim=(self.angle_bins // bit_num) * self.radius_bins,
                embed_dim=self.embed_dim,
                num_patches=bit_num
            ) for bit_num in self.bits
        ])

        # 每个 ring 的 Transformer Encoder
        self.ring_transformers = nn.ModuleList([
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=self.embed_dim,
                    nhead=self.num_heads,
                    batch_first=True,
                    dim_feedforward=self.embed_dim * 4
                ),
                num_layers=self.num_layers
            ) for _ in self.rings
        ])

        # 跨 ring 融合
        self.cross_ring_fusion = nn.Sequential(
            nn.Linear(self.embed_dim * len(self.rings), self.embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.embed_dim * 2, self.embed_dim)
        )

        self.head = nn.Linear(self.embed_dim, n_sectors)

    def detect_rotation(self, mag):
        """
        检测旋转角度

        Args:
            mag: (B, 1, H, W) - FFT幅度谱

        Returns:
            angle_normalized: (B, 1) - 归一化角度 [0, 1]
        """
        # 提取旋转环的极坐标表示
        rotation_polar = cartesian_to_polar(
            mag,
            output_shape=(self.radius_bins, self.angle_bins),
            min_radius=self.rotation_ring[0],
            max_radius=self.rotation_ring[1]
        )  # (B, 1, R, T)

        rotation_polar = rotation_polar.squeeze(1)  # (B, R, T)

        # 检测角度
        angle = self.rotation_detector(rotation_polar)

        return angle

    def rotate_polar(self, polar, angle_shift):
        """
        在极坐标中旋转 (循环移位)

        Args:
            polar: (B, C, R, T) - 极坐标表示
            angle_shift: (B, 1) - 角度偏移量 (归一化 [0, 1])

        Returns:
            rotated: (B, C, R, T) - 旋转后的极坐标表示
        """
        B, C, R, T = polar.shape

        # 将归一化角度转换为bin偏移量
        shift_bins = (angle_shift * T).long()  # (B, 1)

        # 循环移位
        rotated = torch.zeros_like(polar)
        for b in range(B):
            shift = shift_bins[b, 0].item()
            rotated[b] = torch.roll(polar[b], shifts=-shift, dims=-1)

        return rotated

    def forward(self, x, return_rotation=False):
        """
        Args:
            x: (B, 1, H, W) - 输入图像 (单通道)
            return_rotation: 是否返回旋转角度

        Returns:
            logits: (B, n_sectors) - 水印位logits
            mag: FFT幅度谱
            rotation_angle: 旋转角度 (如果return_rotation=True)
        """
        x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)

        # 1. 预滤波 + FFT
        res = self.pre_filter(x)
        fft_map = torch.fft.fftshift(torch.fft.fft2(res, dim=(-2, -1)), dim=(-2, -1))
        mag = torch.abs(fft_map)
        mag = torch.nan_to_num(mag, nan=0.0, posinf=1e6, neginf=-1e6)

        # 2. 检测旋转角度
        rotation_angle = self.detect_rotation(mag)  # (B, 1)

        # 3. 解码水印 (带旋转矫正)
        ring_embeds = []
        for (min_r, max_r), bit_num, patch_embed, transformer in zip(
                self.rings, self.bits, self.patch_embeddings, self.ring_transformers):

            # 提取极坐标
            polar = cartesian_to_polar(
                mag,
                output_shape=(self.radius_bins, self.angle_bins),
                min_radius=min_r,
                max_radius=max_r
            )  # (B, 1, R, T)

            # 旋转矫正
            polar = self.rotate_polar(polar, rotation_angle)  # (B, 1, R, T)

            # [B, 1, R, T] -> [B, T, R]
            polar_reshaped = polar.squeeze(1).permute(0, 2, 1)

            # 分块
            angles_per_bit = self.angle_bins // bit_num
            patches = []
            for i in range(bit_num):
                start = i * angles_per_bit
                end = (i + 1) * angles_per_bit
                bit_patch = polar_reshaped[:, start:end, :].flatten(1)
                patches.append(bit_patch)
            patches = torch.stack(patches, dim=1)

            # Patch Embedding + Transformer
            x_embed = patch_embed(patches)
            x_transformer = transformer(x_embed)
            ring_repr = x_transformer[:, 0, :]
            ring_embeds.append(ring_repr)

        # 4. 融合
        fused = torch.cat(ring_embeds, dim=-1)
        fused = torch.nan_to_num(fused, nan=0.0, posinf=1e6, neginf=-1e6)
        fused = self.cross_ring_fusion(fused)
        logits = self.head(fused)

        logits = torch.nan_to_num(logits, nan=0.0, posinf=1e6, neginf=-1e6)

        if return_rotation:
            return torch.sigmoid(logits), mag, rotation_angle
        return torch.sigmoid(logits), mag, None


if __name__ == "__main__":
    # 测试模型
    model = AdvancedWatermarkDecoderV11(
        n_sectors=60,
        rings=[(7, 17), (20, 30)],
        bits=[15, 45],
        rotation_ring=(13, 23),
        rotation_cycles=8
    )

    # 测试输入
    x = torch.randn(2, 1, 512, 512)
    logits, mag, angle = model(x, return_rotation=True)

    print(f"Logits shape: {logits.shape}")
    print(f"Mag shape: {mag.shape}")
    print(f"Angle shape: {angle.shape}")
    print(f"Angle values: {angle}")
