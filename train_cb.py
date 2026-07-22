"""
Cb通道解码训练脚本

用法：
    python train_cb.py --config config/train_cb_v1_valnoise.yaml
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

from watermark_decoder3 import AdvancedWatermarkDecoder
from dataset import WatermarkDataset


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


def build_dataset(cfg, transform, noise_level='none', alpha_embed=0.01, noise_pool=None, max_angle=360, crop_scale_range=None):
    """构建训练数据集"""
    train_dir = cfg.get('train_dir')
    if train_dir is None:
        train_data_paths = cfg.get('train_data_paths', [])
        train_dir = train_data_paths[0] if train_data_paths else None

    return WatermarkDataset(
        image_dir=train_dir,
        block_size=cfg['block_size'],
        num_bits=cfg['num_bits'],
        r=cfg.get('r', [12, 25]),
        bits=cfg.get('bitsf', [15, 45]),
        alpha_embed=alpha_embed,
        transform=transform,
        noise_level=noise_level,
        noise_pool=noise_pool,
        max_angle=max_angle,
        crop_scale_range=crop_scale_range,
        max_images=cfg.get('train_length', 0),
    )


def build_val_dataset(cfg, transform, noise_level='none', alpha_embed=0.01, noise_pool=None, max_angle=360, crop_scale_range=None):
    """构建验证数据集"""

    return WatermarkDataset(
        image_dir=cfg['val_dir'],
        block_size=cfg['block_size'],
        num_bits=cfg['num_bits'],
        r=cfg.get('r', [12, 25]),
        bits=cfg.get('bitsf', [15, 45]),
        alpha_embed=alpha_embed,
        transform=transform,
        noise_level=noise_level,
        noise_pool=noise_pool,
        max_angle=max_angle,
        crop_scale_range=crop_scale_range,
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
    log_file = os.path.join(output_dir, 'logs', f'train_cb_{timestamp}.log')
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
    train_dataset = build_dataset(cfg, transform, noise_level=init_noise, alpha_embed=init_alpha, noise_pool=init_noise_pool, max_angle=tile_max_angle, crop_scale_range=tile_crop_scale_range)
    val_dataset = build_val_dataset(cfg, transform, noise_level=init_noise, alpha_embed=init_alpha, noise_pool=init_noise_pool, max_angle=tile_max_angle, crop_scale_range=tile_crop_scale_range)

    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # 构建模型
    r = cfg.get('r', [12, 25])
    bitsf = cfg.get('bitsf', [15, 45])
    ring_width = cfg.get('ring_width', 5)
    rings = [(ri-ring_width, ri+ring_width) for ri in r]

    model = AdvancedWatermarkDecoder(
        n_sectors=cfg['num_bits'],
        rings=rings,
        bits=bitsf,
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
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        logger.info(f"Loaded checkpoint: {resume_path}")

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

    # 损失函数
    lambda_bit = cfg.get('lambda_bit', 1.0)
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
        train_correct = 0
        train_total = 0

        grad_accum_steps = cfg.get('grad_accum_steps', 1)
        optimizer.zero_grad()

        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for batch_idx, (watermarked_image, watermark_bits, m1) in enumerate(train_pbar):
            # BGR 3通道转灰度 1通道
            if watermarked_image.shape[1] == 3:
                gray_image = watermarked_image.mean(dim=1, keepdim=True).to(device)
            else:
                gray_image = watermarked_image[:, 0:1, :, :].to(device)
            watermark_bits = watermark_bits.to(device)

            pred, mag, _ = model(gray_image)

            # BCE损失
            pred_safe = torch.clamp(pred, 1e-7, 1 - 1e-7)
            bce_loss = nn.functional.binary_cross_entropy(pred_safe, watermark_bits)

            shape_loss = torch.tensor(0.0, device=device)
            loss = lambda_bit * bce_loss + lambda_shape * shape_loss
            loss = loss / grad_accum_steps

            loss.backward()

            if (batch_idx + 1) % grad_accum_steps == 0:
                grad_clip = cfg.get('grad_clip', 1.0)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                optimizer.step()
                optimizer.zero_grad()

            train_loss += loss.item() * grad_accum_steps

            pred_bits = (pred > 0.5).float()
            correct = (pred_bits == watermark_bits).sum().item()
            total = watermark_bits.numel()
            train_correct += correct
            train_total += total

            current_acc = train_correct / train_total
            train_pbar.set_postfix({'loss': f'{loss.item() * grad_accum_steps:.4f}', 'acc': f'{current_acc:.4f}'})

        train_loss /= len(train_loader)
        train_acc = train_correct / train_total

        torch.cuda.empty_cache()

        # 验证：随机噪声 + 各噪声组合分别验证
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        # 12种噪声组合 (两种不同的噪声, 有序)
        noise_combos = [
            ('identity', 'wechat'),
            ('identity', 'tile_crop'),
            ('identity', 'pimog'),
            ('wechat', 'identity'),
            ('wechat', 'tile_crop'),
            ('wechat', 'pimog'),
            ('tile_crop', 'identity'),
            ('tile_crop', 'wechat'),
            ('tile_crop', 'pimog'),
            ('pimog', 'identity'),
            ('pimog', 'wechat'),
            ('pimog', 'tile_crop'),
        ]
        combo_stats = {f'{a}+{b}': {'correct': 0, 'total': 0} for a, b in noise_combos}

        with torch.no_grad():
            # 随机噪声验证 (主指标)
            val_dataset.force_noise_pair = None
            for watermarked_image, watermark_bits, m1 in tqdm(val_loader, desc='Validating'):
                if watermarked_image.shape[1] == 3:
                    gray_image = watermarked_image.mean(dim=1, keepdim=True).to(device)
                else:
                    gray_image = watermarked_image[:, 0:1, :, :].to(device)
                watermark_bits = watermark_bits.to(device)

                pred, mag, _ = model(gray_image)

                pred_safe = torch.clamp(pred, 1e-7, 1 - 1e-7)
                bce_loss = nn.functional.binary_cross_entropy(pred_safe, watermark_bits)
                shape_loss = torch.tensor(0.0, device=device)
                loss = lambda_bit * bce_loss + lambda_shape * shape_loss

                val_loss += loss.item()

                pred_bits = (pred > 0.5).float()
                correct = (pred_bits == watermark_bits).sum().item()
                total = watermark_bits.numel()
                val_correct += correct
                val_total += total

            # 各噪声组合分别验证 (用子集 + num_workers=0 确保 force_noise_pair 生效)
            combo_subset = min(128, len(val_dataset))
            combo_loader = DataLoader(val_dataset, batch_size=32, shuffle=False,
                                      num_workers=0, pin_memory=False)
            for combo_name, combo_stats_item in tqdm(combo_stats.items(), desc='Noise combos'):
                a, b = combo_name.split('+')
                val_dataset.force_noise_pair = (a, b)
                n_done = 0
                for watermarked_image, watermark_bits, m1 in combo_loader:
                    if watermarked_image.shape[1] == 3:
                        gray_image = watermarked_image.mean(dim=1, keepdim=True).to(device)
                    else:
                        gray_image = watermarked_image[:, 0:1, :, :].to(device)
                    watermark_bits = watermark_bits.to(device)

                    pred, _, _ = model(gray_image)
                    pred_bits = (pred > 0.5).float()
                    correct = (pred_bits == watermark_bits).sum().item()
                    total = watermark_bits.numel()
                    combo_stats_item['correct'] += correct
                    combo_stats_item['total'] += total

                    n_done += watermarked_image.shape[0]
                    if n_done >= combo_subset:
                        break

        val_dataset.force_noise_pair = None
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total

        # 终端输出各噪声组合 acc
        combo_acc_strs = []
        for combo_name, stats in combo_stats.items():
            if stats['total'] > 0:
                acc = stats['correct'] / stats['total']
                combo_acc_strs.append(f'{combo_name}={acc:.4f}')
        logger.info(f'  噪声组合acc: {" | ".join(combo_acc_strs)}')

        scheduler.step()

        epoch_time = time.time() - epoch_start

        # 日志
        logger.info(
            f'Epoch {epoch+1}/{epochs} | '
            f'alpha={alpha_embed:.3f} noise={noise_level} | '
            f'train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f} | '
            f'{epoch_time:.1f}s'
        )

        # Tensorboard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Acc/train', train_acc, epoch)
        writer.add_scalar('Acc/val', val_acc, epoch)
        writer.add_scalar('LR', scheduler.get_last_lr()[0], epoch)

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_ckpt(
                os.path.join(output_dir, 'models', 'best_cb_decoder.pth'),
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
