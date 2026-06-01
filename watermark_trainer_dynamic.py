import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
from torchvision import transforms
import os
from cwt_method import CWTWatermark
import torch.nn.functional as F
import torchvision.utils
from dataset import WatermarkDataset
from watermark_decoder3 import AdvancedWatermarkDecoder


def train_model(model, train_dataset, val_dataset, criterion, optimizer, num_epochs=20, output_dir='output', logger=None, writer=None, args=None):
    """
    训练模型 - 支持动态调整水印强度和噪声强度
    """
    # 检查GPU数量
    num_gpus = torch.cuda.device_count()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Found {num_gpus} GPUs. Using device: {device}')
    #创建schedule
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',  # 最小化损失
        factor=0.5,  # 每次降低一半学习率
        patience=2,  # 2个epoch损失不下降则降低学习率
        verbose=True,  # 打印学习率变化
        min_lr=1e-7  # 学习率下限
    )
    # 使用多GPU
    if num_gpus > 1:
        if logger:
            logger.info(f'Using {num_gpus} GPUs for training')
        else:
            print(f'Using {num_gpus} GPUs for training')
        model = torch.nn.DataParallel(model)
    
    model.to(device)
    saveok=0
    best_val_loss = float('inf')
    # load_model='output_001/models/best_watermark_decoder4.pth'
    # load_model="/home/ylu2024/workspace/fftmask/output_60_dynamic/models/model010.pth"
    load_model=None

    # load_model=None
    # if load_model:
    #     model.load_state_dict(torch.load(load_model))
    if load_model:
        if os.path.exists(load_model):
            # 添加 weights_only=True 解决安全警告
            state_dict = torch.load(load_model, map_location=device, weights_only=True)
            
            # 处理 DataParallel 键名问题
            # 检查当前模型是否是 DataParallel
            is_data_parallel = isinstance(model, torch.nn.DataParallel)
            
            # 检查 state_dict 中的键是否有 module. 前缀
            has_module_prefix = any(k.startswith('module.') for k in state_dict.keys())
            
            new_state_dict = {}
            if is_data_parallel and not has_module_prefix:
                # 当前模型是 DataParallel，但 state_dict 没有 module. 前缀
                for k, v in state_dict.items():
                    new_state_dict[f"module.{k}"] = v
            elif not is_data_parallel and has_module_prefix:
                # 当前模型不是 DataParallel，但 state_dict 有 module. 前缀
                for k, v in state_dict.items():
                    new_state_dict[k.replace("module.", "")] = v
            else:
                # 两者一致，直接使用
                new_state_dict = state_dict
            
            model.load_state_dict(new_state_dict)
            logger.info(f"Loaded model from {load_model}")
        else:
            logger.warning(f"Model not found at {load_model}, using untrained model")
    # 导入tqdm
    from tqdm import tqdm
    
    for epoch in range(num_epochs):
        if saveok and epoch >0:
            break
        import time
        start_time = time.time()
        
        # 动态调整水印强度和噪声强度 - 阶梯式下降 + 准确率驱动
        # 预定义配置阶梯
        config_stages = [
            {'alpha': 0.50, 'noise_levels': ['none', 'low', 'mid', 'high']},
            {'alpha': 0.40, 'noise_levels': ['none', 'low', 'mid', 'high']},
            {'alpha': 0.30, 'noise_levels': ['none', 'low', 'mid', 'high']},
            {'alpha': 0.20, 'noise_levels': ['none', 'low', 'mid', 'high']},
            {'alpha': 0.10, 'noise_levels': ['none', 'low', 'mid', 'high']},
            {'alpha': 0.08, 'noise_levels': ['none', 'low', 'mid', 'high']},
            {'alpha': 0.06, 'noise_levels': ['none', 'low', 'mid', 'high']},
            {'alpha': 0.04, 'noise_levels': ['none', 'low', 'mid', 'high']},
            {'alpha': 0.02, 'noise_levels': ['none', 'low', 'mid', 'high']},
            {'alpha': 0.01, 'noise_levels': ['none', 'low', 'mid', 'high']}
        ]
        
        # 初始化阶段和噪声索引
        if not hasattr(model, 'current_stage'):
            model.current_stage = 0
            model.current_noise_idx = 0
            model.best_accuracy = 0.0
        
        # 获取当前配置
        current_config = config_stages[model.current_stage]
        alpha_embed = current_config['alpha']
        noise_level = current_config['noise_levels'][model.current_noise_idx]
        
        # 检查是否需要切换噪声配置
        if epoch > 0 and model.best_accuracy > 0.95:
            model.current_noise_idx += 1
            if model.current_noise_idx >= len(current_config['noise_levels']):
                model.current_noise_idx = 0
                model.current_stage += 1
                if model.current_stage >= len(config_stages):
                    model.current_stage = len(config_stages) - 1  # 保持最后一个阶段
            model.best_accuracy = 0.0  # 重置准确率阈值
            logger.info(f'Switched to stage {model.current_stage}, noise level {noise_level}')
        
        # 更新数据集的参数
        train_dataset.alpha_embed = alpha_embed
        train_dataset.noise_level = noise_level
        val_dataset.alpha_embed = alpha_embed  # 验证集也使用相同的水印强度
        val_dataset.noise_level = 'none'  # 验证集不添加噪声，保持一致
        
        logger.info(f'Epoch {epoch+1}/{num_epochs}: alpha_embed={alpha_embed:.4f}, noise_level={noise_level}')
        
        # 创建新的 DataLoader（因为数据集参数变化了）
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=12,pin_memory=True,persistent_workers=True,prefetch_factor=2,)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=12,pin_memory=True,persistent_workers=True,prefetch_factor=2,)
        
        # 训练阶段
        model.train()
        train_loss = 0.0
        
        # 使用tqdm添加进度条
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch') as pbar:
            for step, (images, watermarks, m1) in enumerate(train_loader):
                if saveok and step > 0:
                    break
                images = images.to(device)
                watermarks = watermarks.to(device)
            
                #保存images[0]归一化
                if saveok:
                    if len(m1[0].shape) == 2:
                        m1_img = cv2.cvtColor(m1[0].cpu().numpy(), cv2.COLOR_GRAY2BGR)
                    else:
                        m1_img = m1[0].cpu().numpy()
                    print(watermarks[0])
                    img_normalized = (images[0].cpu().numpy().transpose(1,2,0) - images[0].cpu().numpy().transpose(1,2,0).min()) / (images[0].cpu().numpy().transpose(1,2,0).max() - images[0].cpu().numpy().transpose(1,2,0).min() + 1e-8)
                    cv2.imwrite(os.path.join('tmp', f'train_image_{epoch}_{step}.png'), img_normalized * 255)
                    cv2.imwrite(os.path.join('tmp', f'train_m1_{epoch}_{step}.png'), m1_img)
                # 前向传播
                outputs ,mag,polar_feat= model(images,save_debug=False)
                if isinstance(m1, np.ndarray):
                    m1tensor = torch.from_numpy(m1)
                else:
                    m1tensor = m1
                # if m1tensor.dim() == 4 and m1tensor.shape[-1] in (1, 3):
                #     m1tensor = m1tensor.permute(0, 3, 1, 2)
                # if m1tensor.dim() == 3:
                #     m1tensor = m1tensor.unsqueeze(1)
                m1tensor = m1tensor.float().to(device)
                # loss_m1 = F.mse_loss(mag, m1tensor)
                # B=mag.size(0)
                # loss_m1 = (1 - F.cosine_similarity(mag.view(B,-1), m1tensor.view(B,-1)).mean())
                # 数值稳定性检查
                if torch.isnan(mag).any() or torch.isinf(mag).any():
                    print(f"[WARNING] NaN/Inf found in mag at step {step}!")
                    print(f"  NaN count: {torch.isnan(mag).sum()}")
                    print(f"  Inf count: {torch.isinf(mag).sum()}")
                mag = torch.nan_to_num(mag, nan=0.0, posinf=1e6, neginf=-1e6)
                
                if torch.isnan(m1tensor).any() or torch.isinf(m1tensor).any():
                    print(f"[WARNING] NaN/Inf found in m1tensor at step {step}!")
                    print(f"  NaN count: {torch.isnan(m1tensor).sum()}")
                    print(f"  Inf count: {torch.isinf(m1tensor).sum()}")
                m1tensor = torch.nan_to_num(m1tensor, nan=0.0, posinf=1e6, neginf=-1e6)
                
                # 计算均值和标准差，添加更大的 epsilon
                mag_mean = mag.mean()
                mag_std = mag.std()
                m1_mean = m1tensor.mean()
                m1_std = m1tensor.std()
                
                # 防止除零
                mag_norm = (mag - mag_mean) / (mag_std + 1e-6)
                m1_norm = (m1tensor - m1_mean) / (m1_std + 1e-6)
                
                # 再次检查归一化后的结果
                if torch.isnan(mag_norm).any() or torch.isinf(mag_norm).any():
                    print(f"[WARNING] NaN/Inf found in mag_norm at step {step}!")
                    print(f"  NaN count: {torch.isnan(mag_norm).sum()}")
                    print(f"  Inf count: {torch.isinf(mag_norm).sum()}")
                    print(f"  mag mean: {mag_mean.item()}, std: {mag_std.item()}")
                mag_norm = torch.nan_to_num(mag_norm, nan=0.0, posinf=1e6, neginf=-1e6)
                
                if torch.isnan(m1_norm).any() or torch.isinf(m1_norm).any():
                    print(f"[WARNING] NaN/Inf found in m1_norm at step {step}!")
                    print(f"  NaN count: {torch.isnan(m1_norm).sum()}")
                    print(f"  Inf count: {torch.isinf(m1_norm).sum()}")
                    print(f"  m1 mean: {m1_mean.item()}, std: {m1_std.item()}")
                m1_norm = torch.nan_to_num(m1_norm, nan=0.0, posinf=1e6, neginf=-1e6)
                
                loss_shape = F.mse_loss(mag_norm, m1_norm)
                #给loss_shape加mask  只关注低频区域，比如只关注半径小于20的区域
                # ======================
                B, C, H, W = mag.shape
                cx = W // 2
                cy = H // 2

                # 生成坐标网格
                y_grid, x_grid = torch.meshgrid(torch.arange(H, device=mag.device) - cy,
                                                torch.arange(W, device=mag.device) - cx,
                                                indexing='ij')
                # 距离中心的平方
                dist_sq = x_grid ** 2 + y_grid ** 2
                # 低频mask：半径 <= 20 的区域全部为 1
                mask = (dist_sq <= 20 ** 2).float().unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
                # 只计算低频区域的 loss
                loss_shape = (loss_shape * mask).sum() / (mask.sum() + 1e-4)


                # 3. 引入对比度约束 (在极坐标域)
                #loss_contrast = contrastive_energy_loss(polar_feat, watermarks, margin=100)
                # loss_m1 = loss_contrast
                loss_m1=1*loss_shape
                # print(outputs[0])
                # print(watermarks[0])
                loss_bit = criterion(outputs, watermarks)
                malpha=0.02
                if loss_m1.item() < 1.3:
                    malpha=0.01
                else:
                    malpha=0.02
                loss=10*loss_bit+malpha*loss_m1  #根据实际情况调整权重 之前是1
                accuracy = (outputs > 0.5).float() == watermarks
                accuracy = accuracy.sum().item() / (watermarks.size(0) * watermarks.size(1))
                 # 更新进度条
                # pbar.set_postfix({'loss': f'{loss.item():.4f}', 'accuracy': f'{accuracy:.4f}'})
                pbar.update(1)
                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                # 添加梯度裁剪防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if writer:
                    writer.add_scalar('Loss/Train_bit', loss_bit.item(), global_step=step+epoch*len(train_loader))
                    writer.add_scalar('Loss/Train_m1', loss_m1.item(), global_step=step+epoch*len(train_loader))
                    writer.add_scalar('Loss/Train_total', loss.item(), global_step=step+epoch*len(train_loader))
                    writer.add_scalar('Accuracy/Train', accuracy, global_step=step+epoch*len(train_loader))
                    writer.add_scalar('Params/alpha_embed', alpha_embed, global_step=epoch)
                    writer.add_scalar('Params/noise_level', model.current_noise_idx, global_step=epoch)
                    writer.add_scalar('Params/current_stage', model.current_stage, global_step=epoch)
                
                # 更新最佳准确率
                if accuracy > model.best_accuracy:
                    model.best_accuracy = accuracy
                
                # train_loss += loss.item() * images.size(0)
                train_loss += loss.item()
                
        
        train_loss = train_loss / (len(train_loader.dataset)/images.size(0))
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_accuracy = 0.0
        
        with torch.no_grad():
            for step, (images, watermarks,_) in enumerate(val_loader):
                images = images.to(device)
                watermarks = watermarks.to(device)
                
                outputs, mag,_ = model(images)
                loss = criterion(outputs, watermarks)
                # val_loss += loss.item() * images.size(0)
                val_loss += loss.item()
                
                # 计算准确率
                predicted = (outputs > 0.5).float()
                correct = (predicted == watermarks).sum().item()
                accuracy = correct / (watermarks.size(0) * watermarks.size(1))
                val_accuracy += accuracy
                if writer:
                    writer.add_scalar('Accuracy/Validation', accuracy, global_step=step+epoch*len(val_loader))
                    writer.add_scalar('Loss/Validation', loss.item(), global_step=step+epoch*len(val_loader))
        val_loss = val_loss / (len(val_loader.dataset)/images.size(0))
        # val_accuracy = val_accuracy / (len(val_loader.dataset)/images.size(0))
        val_accuracy = val_accuracy / (len(val_loader))
        
        # 计算训练时间
        end_time = time.time()
        epoch_time = end_time - start_time
        
        # 更新学习率
        scheduler.step(val_loss)
        logger.info(f'Learning rate: {scheduler.get_last_lr()[0]:.6f}')
        
        # 记录日志
        if logger:
            logger.info(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, Time: {epoch_time:.2f}s')
        else:
            print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, Time: {epoch_time:.2f}s')
        
       
        #每10个epoch保存一次模型
        if epoch % 10 == 0:
            model_path = os.path.join(output_dir, 'models', f'epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), model_path)
            if logger:
                logger.info(f'Model saved to {model_path}')
            else:
                print(f'Model saved to {model_path}')
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = os.path.join(output_dir, 'models', 'best_watermark_decoder.pth')
            torch.save(model.state_dict(), model_path)
            if logger:
                logger.info(f'Best model saved to {model_path} with loss: {best_val_loss:.4f}')
            else:
                print(f'Best model saved to {model_path} with loss: {best_val_loss:.4f}')

def test_model(model, test_loader, logger=None):
    """
    测试模型
    """
    # 检查GPU数量
    num_gpus = torch.cuda.device_count()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 使用多GPU
    if num_gpus > 1:
        if logger:
            logger.info(f'Using {num_gpus} GPUs for testing')
        else:
            print(f'Using {num_gpus} GPUs for testing')
        model = torch.nn.DataParallel(model)
    
    model.to(device)
    model.eval()
    
    test_accuracy = 0.0
    
    with torch.no_grad():
        for images, watermarks in test_loader:
            images = images.to(device)
            watermarks = watermarks.to(device)
            
            outputs = model(images)
            predicted = (outputs > 0.5).float()
            correct = (predicted == watermarks).sum().item()
            test_accuracy += correct / (watermarks.size(0) * watermarks.size(1))
    
    test_accuracy = test_accuracy / len(test_loader)
    if logger:
        logger.info(f'Test Accuracy: {test_accuracy:.4f}')
    else:
        print(f'Test Accuracy: {test_accuracy:.4f}')
    return test_accuracy


def contrastive_energy_loss(polar_feat, watermarks, margin=1):
    """
    polar_feat: (B, 1, R, T) -> 极坐标特征图，例如 (B, 1, 32, 128)
    watermarks: (B, 16) -> 0/1 标签
    margin: 预期的能量差值门限
    """
    B, _, R, T = polar_feat.shape
    n_sectors = watermarks.shape[1]
    cols_per_sector = T // n_sectors
    
    # 1. 计算每个扇区的平均能量 (B, 16)
    # 将 T 维拆分为 (16, cols_per_sector)，然后在半径 R 和列维求均值
    # 得到每个 batch 中 16 个扇区的各自能量
    sector_energies = polar_feat.view(B, R, n_sectors, cols_per_sector).mean(dim=(1, 3))
    
    # 2. 根据标签分离 正样本(1) 和 负样本(0) 的能量
    pos_mask = (watermarks == 1).float()
    neg_mask = (watermarks == 0).float()
    
    # 计算每个 batch 中有水印扇区的平均能量 (B,)
    # 加 1e-6 防止某张图全是 0 或全是 1 导致的除零错误
    pos_avg = (sector_energies * pos_mask).sum(dim=1) / (pos_mask.sum(dim=1) + 1e-6)
    
    # 计算每个 batch 中无水印扇区的平均能量 (B,)
    neg_avg = (sector_energies * neg_mask).sum(dim=1) / (neg_mask.sum(dim=1) + 1e-6)
    
    # 3. 最大边际损失 (Max-Margin Loss)
    # 我们希望：pos_avg > neg_avg + margin
    # 如果满足则 Loss 为 0，如果不满足则计算差距
    # print(neg_avg-pos_avg)
    loss = torch.relu(neg_avg + margin - pos_avg).mean()
    
    return loss
def main():
    import argparse
    import logging
    import os
    
    # 参数解析
    parser = argparse.ArgumentParser(description='Watermark Trainer with Dynamic Parameters')
    parser.add_argument('--train_dir', type=str, default="/mnt/ylyu/COCO-train2017/", help='Training data directory')
    parser.add_argument('--val_dir', type=str, default="/mnt/ylyu/COCO-val2017/", help='Validation data directory')
    parser.add_argument('--test_dir', type=str, default="/mnt/ylyu/COCO-test2017/", help='Test data directory')
    parser.add_argument('--output_dir', type=str, default='/home/ylu2024/workspace/fftmask/output_60_dynamic', help='Output directory for models and logs')
    parser.add_argument('--batch_size', type=int, default=80, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--device', type=str, default='0,1', help='Device to use for training')
    parser.add_argument('--block_size', type=int, default=512, help='Block size for watermark decoding')
    parser.add_argument('--num_bits', type=int, default=60, help='Number of bits for watermark decoding')
    parser.add_argument('--r', type=list, default=[5,10,15], help='Radius for watermark decoding')
    parser.add_argument('--bitsf', type=list, default=[5,20,35], help='Bits for each radius')
    parser.add_argument('--alpha_embed', type=float, default=0.1, help='Initial embedding strength')
    parser.add_argument('--final_alpha', type=float, default=0.01, help='Final embedding strength')
    parser.add_argument('--initial_noise', type=str, default='none', help='Initial noise level')
    parser.add_argument('--final_noise', type=str, default='high', help='Final noise level')
    args = parser.parse_args()
    
    # 设置使用的GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'runs'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'models'), exist_ok=True)
    
    # 配置日志
    import datetime
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(args.output_dir, 'logs', f'training_{timestamp}.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger()
    logger.info('Starting watermark training with dynamic parameters')
    logger.info(f'Configuration: {args}')
    logger.info(f'Using GPUs: {args.device}')
    
    # 数据变换
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    
    # 创建数据集
    train_dataset = WatermarkDataset(args.train_dir, block_size=args.block_size, num_bits=args.num_bits, r=args.r, bits=args.bitsf, alpha_embed=args.alpha_embed, transform=transform, noise_level='none')
    val_dataset = WatermarkDataset(args.val_dir, block_size=args.block_size, num_bits=args.num_bits, r=args.r, bits=args.bitsf, alpha_embed=args.alpha_embed, transform=transform, noise_level='none')
    # test_dataset = WatermarkDataset(args.test_dir, transform=transform)
    
    logger.info(f'Training dataset size: {len(train_dataset)}')
    logger.info(f'Validation dataset size: {len(val_dataset)}')
    # logger.info(f'Test dataset size: {len(test_dataset)}')
    
    # 初始化模型
    # model = WatermarkDecoder(block_size=args.block_size, num_bits=args.num_bits, r=args.r)
    model=AdvancedWatermarkDecoder(n_sectors=args.num_bits,rings=[(r-1, r+1) for r in args.r],bits=args.bitsf)
    
    # 损失函数和优化器
    criterion = nn.MSELoss()
    # criterion = nn.BCELoss()
    # optimizer = optim.SGD(model.parameters(), lr=args.lr)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # 初始化SummaryWriter
    writer = SummaryWriter(os.path.join(args.output_dir, 'runs', timestamp))
    
    # 训练模型
    train_model(model, train_dataset, val_dataset, criterion, optimizer, num_epochs=args.epochs, output_dir=args.output_dir, logger=logger, writer=writer, args=args)
    
    # 关闭SummaryWriter
    writer.close()
    
    # 测试模型
    # test_accuracy = test_model(model, test_loader, logger=logger)
    # logger.info(f'Final test accuracy: {test_accuracy:.4f}')

if __name__ == '__main__':
    main()