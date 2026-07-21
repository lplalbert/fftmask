"""
v11 Cb通道解码训练脚本
增加旋转矫正能力的训练

用法：
    python train_cb_v11.py --config config/train_cb_v11.yaml
"""
import argparse
import os
import random
import logging
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datetime import datetime
from torchvision import transforms
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from watermark_decoder_v11 import AdvancedWatermarkDecoderV11
from dataset_v11 import WatermarkDatasetV11


def setup_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_noise_schedule(cfg, epoch):
    """获取当前epoch的噪声配置"""
    schedule = cfg.get('noise_schedule', None)
    if schedule is None:
        return cfg.get('alpha', 0.01), 'none', None, 360

    for entry in schedule:
        end_epoch = entry[0]
        if epoch < end_epoch:
            alpha = entry[1]
            noise_level = entry[2]
            noise_pool = entry[3] if len(entry) > 3 else None
            max_angle = entry[4] if len(entry) > 4 else 360
            return alpha, noise_level, noise_pool, max_angle

    last = schedule[-1]
    max_angle = last[4] if len(last) > 4 else 360
    return last[1], last[2], last[3] if len(last) > 3 else None, max_angle


def get_tile_rotate_params(cfg):
    """获取tile_rotate噪声的额外参数"""
    max_angle = cfg.get('max_angle', 360)
    crop_scale_range = cfg.get('crop_scale_range', None)
    return max_angle, crop_scale_range


def build_dataset(cfg, transform, noise_level='none', alpha_embed=0.016,
                  noise_pool=None, max_angle=360, crop_scale_range=None):
    """构建训练数据集"""
    train_dir = cfg.get('train_dir')

    return WatermarkDatasetV11(
        image_dir=train_dir,
        block_size=cfg['block_size'],
        num_bits=cfg['num_bits'],
        r_watermark=cfg.get('r_watermark', [12, 25]),
        bitsf=cfg.get('bitsf', [15, 45]),
        r_rotation=cfg.get('r_rotation', 18),
        rotation_cycles=cfg.get('rotation_cycles', 8),
        alpha_embed=alpha_embed,
        transform=transform,
        noise_level=noise_level,
        noise_pool=noise_pool,
        max_angle=max_angle,
        crop_scale_range=crop_scale_range,
        rotation_aug=cfg.get('rotation_aug', True),
        max_images=cfg.get('train_length', 0),
    )


def build_val_dataset(cfg, transform, noise_level='none', alpha_embed=0.016,
                      noise_pool=None, max_angle=360, crop_scale_range=None):
    """构建验证数据集"""

    return WatermarkDatasetV11(
        image_dir=cfg['val_dir'],
        block_size=cfg['block_size'],
        num_bits=cfg['num_bits'],
        r_watermark=cfg.get('r_watermark', [12, 25]),
        bitsf=cfg.get('bitsf', [15, 45]),
        r_rotation=cfg.get('r_rotation', 18),
        rotation_cycles=cfg.get('rotation_cycles', 8),
        alpha_embed=alpha_embed,
        transform=transform,
        noise_level=noise_level,
        noise_pool=noise_pool,
        max_angle=max_angle,
        crop_scale_range=crop_scale_range,
        rotation_aug=cfg.get('rotation_aug', True),
        max_images=cfg.get('val_length', 0),
    )


def save_ckpt(path, epoch, model, optimizer, scheduler, val_loss):
    """保存checkpoint"""
    state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    torch.save({
        'model': state,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch,
        'val_loss': val_loss,
    }, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    setup_seed(cfg.get('seed', 42))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(cfg['output_dir'], timestamp)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'logs'), exist_ok=True)

    # 配置日志
    log_file = os.path.join(output_dir, 'logs', f'train_cb_v11_{timestamp}.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger()

    # Tensorboard
    writer = SummaryWriter(log_dir=os.path.join(output_dir, 'tensorboard'))

    # 记录配置
    logger.info(f"Config: {cfg}")
    logger.info(f"Tensorboard: tensorboard --logdir {os.path.join(output_dir, 'tensorboard')}")

    # 设备
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.get('device', '0')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 数据预处理transform
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    # 获取初始噪声配置
    init_alpha, init_noise, init_noise_pool, _ = get_noise_schedule(cfg, 0)
    tile_max_angle, tile_crop_scale_range = get_tile_rotate_params(cfg)

    # 构建数据集
    train_dataset = build_dataset(cfg, transform, noise_level=init_noise,
                                  alpha_embed=init_alpha, noise_pool=init_noise_pool,
                                  max_angle=tile_max_angle, crop_scale_range=tile_crop_scale_range)
    val_dataset = build_val_dataset(cfg, transform, noise_level=init_noise,
                                    alpha_embed=init_alpha, noise_pool=init_noise_pool,
                                    max_angle=tile_max_angle, crop_scale_range=tile_crop_scale_range)

    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # 构建模型
    r_watermark = cfg.get('r_watermark', [12, 25])
    bitsf = cfg.get('bitsf', [15, 45])
    ring_width = cfg.get('ring_width', 5)
    rings = [(ri - ring_width, ri + ring_width) for ri in r_watermark]

    r_rotation = cfg.get('r_rotation', 18)
    rotation_ring = (r_rotation - ring_width, r_rotation + ring_width)

    model = AdvancedWatermarkDecoderV11(
        n_sectors=cfg['num_bits'],
        rings=rings,
        bits=bitsf,
        rotation_ring=rotation_ring,
        rotation_cycles=cfg.get('rotation_cycles', 8),
    )

    # 加载预训练权重
    resume_path = cfg.get('resume', None)
    if resume_path and os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location='cpu', weights_only=False)
        if isinstance(ckpt, dict) and 'model' in ckpt:
            state_dict = ckpt['model']
        else:
            state_dict = ckpt
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        # 只加载匹配的权重 (旋转检测器的权重是新的，不会加载)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items()
                          if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        logger.info(f"Loaded checkpoint: {resume_path} ({len(pretrained_dict)} layers loaded)")

    model = model.to(device)

    # 多GPU
    if torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.get('lr', 0.001)),
        weight_decay=0.01
    )

    # 学习率调度
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg['epochs'],
        eta_min=float(cfg.get('lr', 0.001)) * 0.01
    )

    # DataLoader
    num_workers = cfg.get('num_workers', 4)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.get('batch_size', 40),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.get('batch_size', 40),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # 损失函数权重
    lambda_bit = cfg.get('lambda_bit', 15.0)
    lambda_rotation = cfg.get('lambda_rotation', 5.0)
    lambda_shape = cfg.get('lambda_shape', 0.0)

    best_val_loss = float('inf')
    epochs = cfg['epochs']

    for epoch in range(epochs):
        epoch_start = time.time()

        # 获取当前噪声配置
        alpha_embed, noise_level, noise_pool, _ = get_noise_schedule(cfg, epoch)
        tile_max_angle, tile_crop_scale_range = get_tile_rotate_params(cfg)

        train_dataset.alpha_embed = alpha_embed
        train_dataset.noise_level = noise_level if isinstance(noise_level, str) else 'none'
        train_dataset.max_angle = tile_max_angle
        train_dataset.crop_scale_range = tile_crop_scale_range
        if noise_pool is not None:
            train_dataset.noise_pool = noise_pool

        val_dataset.alpha_embed = alpha_embed
        val_dataset.noise_level = noise_level if isinstance(noise_level, str) else 'none'
        val_dataset.max_angle = tile_max_angle
        val_dataset.crop_scale_range = tile_crop_scale_range

        # 训练
        model.train()
        train_loss = 0.0
        train_bit_correct = 0
        train_bit_total = 0
        train_rotation_error = 0.0
        train_rotation_count = 0

        grad_accum_steps = cfg.get('grad_accum_steps', 1)
        optimizer.zero_grad()

        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for batch_idx, (watermarked_image, watermark_bits, rotation_angle) in enumerate(train_pbar):
            # BGR 3通道转灰度 1通道
            if watermarked_image.shape[1] == 3:
                gray_image = watermarked_image.mean(dim=1, keepdim=True).to(device)
            else:
                gray_image = watermarked_image[:, 0:1, :, :].to(device)
            watermark_bits = watermark_bits.to(device)
            rotation_angle = rotation_angle.to(device)

            # 前向传播
            pred, mag, pred_rotation = model(gray_image, return_rotation=True)

            # 水印位损失 (BCE)
            pred_safe = torch.clamp(pred, 1e-7, 1 - 1e-7)
            bce_loss = nn.functional.binary_cross_entropy(pred_safe, watermark_bits)

            # 旋转角度损失 (MSE)
            rotation_loss = nn.functional.mse_loss(pred_rotation, rotation_angle)

            # 总损失
            shape_loss = torch.tensor(0.0, device=device)
            loss = (lambda_bit * bce_loss +
                    lambda_rotation * rotation_loss +
                    lambda_shape * shape_loss)
            loss = loss / grad_accum_steps

            loss.backward()

            if (batch_idx + 1) % grad_accum_steps == 0:
                grad_clip = cfg.get('grad_clip', 1.0)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                optimizer.step()
                optimizer.zero_grad()

            train_loss += loss.item() * grad_accum_steps

            # 水印位准确率
            pred_bits = (pred > 0.5).float()
            correct = (pred_bits == watermark_bits).sum().item()
            total = watermark_bits.numel()
            train_bit_correct += correct
            train_bit_total += total

            # 旋转角度误差
            rotation_error = torch.abs(pred_rotation - rotation_angle).mean().item()
            train_rotation_error += rotation_error
            train_rotation_count += 1

            current_acc = train_bit_correct / train_bit_total
            current_rot_err = train_rotation_error / train_rotation_count * 180  # 转换为度
            train_pbar.set_postfix({
                'loss': f'{loss.item() * grad_accum_steps:.4f}',
                'bit_acc': f'{current_acc:.4f}',
                'rot_err': f'{current_rot_err:.1f}°'
            })

        train_loss /= len(train_loader)
        train_bit_acc = train_bit_correct / train_bit_total
        train_rotation_error = train_rotation_error / train_rotation_count * 180  # 度

        torch.cuda.empty_cache()

        # 验证
        model.eval()
        val_loss = 0.0
        val_bit_correct = 0
        val_bit_total = 0
        val_rotation_error = 0.0
        val_rotation_count = 0

        with torch.no_grad():
            for watermarked_image, watermark_bits, rotation_angle in tqdm(val_loader, desc='Validating'):
                if watermarked_image.shape[1] == 3:
                    gray_image = watermarked_image.mean(dim=1, keepdim=True).to(device)
                else:
                    gray_image = watermarked_image[:, 0:1, :, :].to(device)
                watermark_bits = watermark_bits.to(device)
                rotation_angle = rotation_angle.to(device)

                pred, mag, pred_rotation = model(gray_image, return_rotation=True)

                pred_safe = torch.clamp(pred, 1e-7, 1 - 1e-7)
                bce_loss = nn.functional.binary_cross_entropy(pred_safe, watermark_bits)
                rotation_loss = nn.functional.mse_loss(pred_rotation, rotation_angle)
                shape_loss = torch.tensor(0.0, device=device)
                loss = lambda_bit * bce_loss + lambda_rotation * rotation_loss + lambda_shape * shape_loss

                val_loss += loss.item()

                pred_bits = (pred > 0.5).float()
                correct = (pred_bits == watermark_bits).sum().item()
                total = watermark_bits.numel()
                val_bit_correct += correct
                val_bit_total += total

                rotation_error = torch.abs(pred_rotation - rotation_angle).mean().item()
                val_rotation_error += rotation_error
                val_rotation_count += 1

        val_loss /= len(val_loader)
        val_bit_acc = val_bit_correct / val_bit_total
        val_rotation_error = val_rotation_error / val_rotation_count * 180  # 度

        scheduler.step()

        epoch_time = time.time() - epoch_start

        # 日志
        logger.info(
            f'Epoch {epoch+1}/{epochs} | '
            f'alpha={alpha_embed:.3f} noise={noise_level} | '
            f'train_loss={train_loss:.4f} val_loss={val_loss:.4f} | '
            f'train_bit_acc={train_bit_acc:.4f} val_bit_acc={val_bit_acc:.4f} | '
            f'train_rot_err={train_rotation_error:.1f}° val_rot_err={val_rotation_error:.1f}° | '
            f'{epoch_time:.1f}s'
        )

        # Tensorboard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('BitAcc/train', train_bit_acc, epoch)
        writer.add_scalar('BitAcc/val', val_bit_acc, epoch)
        writer.add_scalar('RotError/train', train_rotation_error, epoch)
        writer.add_scalar('RotError/val', val_rotation_error, epoch)
        writer.add_scalar('LR', scheduler.get_last_lr()[0], epoch)

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_ckpt(
                os.path.join(output_dir, 'models', 'best_cb_decoder_v11.pth'),
                epoch, model, optimizer, scheduler, val_loss
            )
            logger.info(f'Best model saved, val_loss={val_loss:.4f}')

        # 定期保存
        if (epoch + 1) % 10 == 0:
            save_path = os.path.join(output_dir, 'models', f'epoch_{epoch+1}.pth')
            save_ckpt(save_path, epoch, model, optimizer, scheduler, val_loss)
            logger.info(f'Saved: {save_path}')

    writer.close()
    logger.info(f"Training complete. Best val_loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    main()
