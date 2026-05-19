import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import argparse
import logging

from encode import Watermark16Sector
from watermark_decoder import AdvancedWatermarkDecoder
from noise_utils import add_pimog_noise, add_jpeg_compression_noise


class InferenceDataset(Dataset):
    def __init__(self, image_dir, block_size=512, num_bits=4, r=[5, 12], transform=None, max_images=100, alpha_embed=0.005):
        self.image_dir = image_dir
        self.block_size = block_size
        self.transform = transform
        self.alpha_embed = alpha_embed
        self.num_bits = num_bits
        self.watermark_system = Watermark16Sector(L1=block_size, k1=30000, r=r, r_range=2, n_sectors=num_bits)
        self.image_files = [f for f in os.listdir(image_dir)
                           if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        if max_images > 0:
            self.image_files = self.image_files[:max_images]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = cv2.imread(img_path)

        if image is None:
            raise ValueError(f"Failed to load image: {img_path}")

        if image.shape[:2] != (self.block_size, self.block_size):
            image = cv2.resize(image, (self.block_size, self.block_size))

        watermark_bits = np.random.randint(0, 2, size=self.num_bits)

        return image, watermark_bits, self.image_files[idx]


def calculate_psnr(img1, img2):
    mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))


def create_tiled_watermark(watermark_512):
    """
    将512x512的水印模板平铺成1024x1024（2x2平铺）
    """
    h, w = watermark_512.shape[:2]
    tiled = np.tile(watermark_512, (2, 2, 1)) if len(watermark_512.shape) == 3 else np.tile(watermark_512, (2, 2))
    return tiled


def random_crop_512(tiled_image):
    """
    从1024x1024图像中随机裁剪512x512
    """
    h, w = tiled_image.shape[:2]
    max_y = h - 512
    max_x = w - 512
    
    y = np.random.randint(0, max_y + 1)
    x = np.random.randint(0, max_x + 1)
    
    if len(tiled_image.shape) == 3:
        cropped = tiled_image[y:y+512, x:x+512, :]
    else:
        cropped = tiled_image[y:y+512, x:x+512]
    
    return cropped


def embed_tiled_watermark(host_1024, Tm_512, alpha_embed):
    """
    将平铺的水印嵌入到1024x1024的载体图像中
    """
    Tm_tiled = create_tiled_watermark(Tm_512)
    watermarked_1024 = host_1024 * (1 - alpha_embed) + Tm_tiled * alpha_embed
    return np.clip(watermarked_1024, 0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description='Watermark Inference')
    parser.add_argument('--input_dir', type=str, default='/mnt/ylyu/COCO-val2017/', help='Input image directory')
    parser.add_argument('--output_dir', type=str, default='inference_output_001', help='Output directory for watermarked images')
    parser.add_argument('--model_path', type=str, default='output_001/models/best_watermark_decoder.pth', help='Path to trained decoder model')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--block_size', type=int, default=512, help='Block size')
    parser.add_argument('--num_bits', type=int, default=4, help='Number of watermark bits')
    parser.add_argument('--r', type=int, nargs='+', default=[5, 11], help='Radius for watermark')
    parser.add_argument('--alpha_embed', type=float, default=0.01, help='Embedding strength')
    parser.add_argument('--max_images', type=int, default=100, help='Maximum number of images to process')
    parser.add_argument('--num_crops', type=int, default=2, help='Number of random crops for noise test')
    parser.add_argument('--device', type=str, default='3', help='GPU device')
    parser.add_argument('--test_noise', type=int, default=1, help='Test with noise')
    parser.add_argument('--jpeg_quality', type=int, default=10, help='JPEG compression quality for noise test')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device

    os.makedirs(args.output_dir, exist_ok=True)

    log_file = os.path.join(args.output_dir, 'inference_noise.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    dataset = InferenceDataset(
        image_dir=args.input_dir,
        block_size=args.block_size,
        num_bits=args.num_bits,
        r=args.r,
        transform=transform,
        max_images=args.max_images,
        alpha_embed=args.alpha_embed
    )

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = AdvancedWatermarkDecoder(n_sectors=args.num_bits)

    if os.path.exists(args.model_path):
        state_dict = torch.load(args.model_path, map_location=device)
        model.load_state_dict(state_dict)
        logger.info(f"Loaded model from {args.model_path}")
    else:
        logger.warning(f"Model not found at {args.model_path}, using untrained model")

    model = model.to(device)
    model.eval()

    # 原始测试统计
    psnr_list = []
    accuracy_list = []
    correct_all_bits_count = 0  # 所有位数都正确的图片张数
    
    # 噪声测试统计
    noise_accuracy_list = []
    noise_correct_all_bits_count = 0  # 噪声测试中所有位数都正确的图片张数
    
    # 压缩噪声测试统计
    jpeg_accuracy_list = []
    jpeg_correct_all_bits_count = 0  # JPEG测试中所有位数都正确的图片张数
    
    # 屏摄噪声测试统计
    screen_accuracy_list = []
    screen_correct_all_bits_count = 0  # 屏摄测试中所有位数都正确的图片张数


    logger.info(f"Starting inference on {len(dataset)} images...")
    logger.info(f"Noise test: {args.num_crops} random crops per image from 1024x1024 tiled watermark image")
    if args.test_noise:
        logger.info(f"Testing with compression noise (JPEG quality: {args.jpeg_quality}) and screen shooting noise")

    watermark_system = Watermark16Sector(L1=args.block_size, k1=30000, r=args.r, r_range=2, n_sectors=args.num_bits)
    savewrite=0
    with torch.no_grad():
        for batch_idx, (original_images, watermark_bits_list, filenames) in enumerate(tqdm(dataloader, desc="Processing")):
            batch_size = original_images.size(0)

            for i in range(batch_size):
                original_image = original_images[i].numpy()
                if original_image.shape[0] == 3:
                    original_image = original_image.transpose(1, 2, 0)
                original_image = original_image.astype(np.uint8)

                watermark_bits = watermark_bits_list[i].numpy()

                # ========== 原始测试：直接在512x512上嵌入水印 ==========
                Tm, m1 = watermark_system.generate_template(numbit=watermark_bits)

                host_512 = original_image.copy()
                if len(host_512.shape) == 2:
                    host_512 = cv2.cvtColor(host_512, cv2.COLOR_GRAY2BGR)

                if len(Tm.shape) == 2:
                    Tm_bgr = cv2.cvtColor(Tm, cv2.COLOR_GRAY2BGR)
                else:
                    Tm_bgr = Tm.copy()

                watermarked_512 = host_512 * (1 - args.alpha_embed) + Tm_bgr * args.alpha_embed
                watermarked_512 = np.clip(watermarked_512, 0, 255).astype(np.uint8)

                psnr = calculate_psnr(original_image, watermarked_512)
                psnr_list.append(psnr)
                if savewrite:
                    save_path = os.path.join(args.output_dir, f"wm_{filenames[i]}")
                    cv2.imwrite(save_path, watermarked_512)

                # 解码512x512水印图像
                watermarked_gray = cv2.cvtColor(watermarked_512, cv2.COLOR_BGR2GRAY)
                watermarked_tensor = transform(watermarked_gray).unsqueeze(0).to(device)

                output, _, _ = model(watermarked_tensor)
                predicted_bits = (output > 0.5).float().squeeze().cpu().numpy()

                if len(predicted_bits) == len(watermark_bits):
                    accuracy = np.mean(predicted_bits == watermark_bits)
                    # 统计所有位数都正确的情况
                    if accuracy == 1.0:
                        correct_all_bits_count += 1
                else:
                    logger.warning(f"Bit length mismatch for {filenames[i]}: predicted={len(predicted_bits)}, actual={len(watermark_bits)}")
                    accuracy = 0.0

                accuracy_list.append(accuracy)

                # ========== 噪声测试：放大到1024x1024，嵌入平铺水印，随机裁剪 ==========
                # 将载体图像放大到1024x1024
                host_1024 = cv2.resize(original_image, (1024, 1024))
                if len(host_1024.shape) == 2:
                    host_1024 = cv2.cvtColor(host_1024, cv2.COLOR_GRAY2BGR)

                # 嵌入平铺的水印
                watermarked_1024 = embed_tiled_watermark(host_1024, Tm_bgr, args.alpha_embed)
                watermarked_1024 = np.clip(watermarked_1024, 0, 255).astype(np.uint8)

                # 保存1024x1024的水印图像（可选）
                if savewrite:
                    save_path_1024 = os.path.join(args.output_dir, f"wm_1024_{filenames[i]}")
                    cv2.imwrite(save_path_1024, watermarked_1024)

                # 随机裁剪多次并解码
                crop_accuracies = []
                
                for crop_idx in range(args.num_crops):
                    cropped_image = random_crop_512(watermarked_1024)
                    cropped_gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
                    cropped_tensor = transform(cropped_gray).unsqueeze(0).to(device)
                    
                    output_noise, _, _ = model(cropped_tensor)
                    predicted_bits_noise = (output_noise > 0.5).float().squeeze().cpu().numpy()
                    
                    if len(predicted_bits_noise) == len(watermark_bits):
                        crop_accuracy = np.mean(predicted_bits_noise == watermark_bits)
                    else:
                        crop_accuracy = 0.0
                    if crop_accuracy == 1.0:
                        noise_correct_all_bits_count += 1
                    
                    crop_accuracies.append(crop_accuracy)
                
                avg_noise_accuracy = np.mean(crop_accuracies)
                noise_accuracy_list.append(avg_noise_accuracy)
                # 统计噪声测试中所有位数都正确的情况
                # if avg_noise_accuracy == 1.0:
                #     noise_correct_all_bits_count += 1

                # ========== 压缩噪声测试（JPEG压缩）==========
                jpeg_accuracy = 0.0
                if args.test_noise:
                    # 应用JPEG压缩噪声
                    watermarked_jpeg = add_jpeg_compression_noise(watermarked_512, quality=args.jpeg_quality)
                    # 保存压缩后的图像（可选）
                    watermarked_jpeg = np.clip(watermarked_jpeg, 0, 255).astype(np.uint8)
                    if savewrite:
                        save_path_jpeg = os.path.join(args.output_dir, f"wm_jpeg_{filenames[i]}")
                        cv2.imwrite(save_path_jpeg, watermarked_jpeg)
                    # 解码
                    jpeg_gray = cv2.cvtColor(watermarked_jpeg, cv2.COLOR_BGR2GRAY)
                    jpeg_tensor = transform(jpeg_gray).unsqueeze(0).to(device)
                    output_jpeg, _, _ = model(jpeg_tensor)
                    predicted_bits_jpeg = (output_jpeg > 0.5).float().squeeze().cpu().numpy()
                    if len(predicted_bits_jpeg) == len(watermark_bits):
                        jpeg_accuracy = np.mean(predicted_bits_jpeg == watermark_bits)
                        # 统计JPEG测试中所有位数都正确的情况
                        if jpeg_accuracy == 1.0:
                            jpeg_correct_all_bits_count += 1
                    jpeg_accuracy_list.append(jpeg_accuracy)

                # ========== 屏摄噪声测试（PIMOG）==========
                screen_accuracy = 0.0
                if args.test_noise:
                    # 应用屏摄噪声
                    watermarked_screen = add_pimog_noise(watermarked_512)
                    watermarked_screen = np.clip(watermarked_screen, 0, 255).astype(np.uint8)
                    # 保存屏摄后的图像（可选）
                    if savewrite:
                        save_path_screen = os.path.join(args.output_dir, f"wm_screen_{filenames[i]}")
                        cv2.imwrite(save_path_screen, watermarked_screen)
                    # 解码
                    screen_gray = cv2.cvtColor(watermarked_screen, cv2.COLOR_BGR2GRAY)
                    screen_tensor = transform(screen_gray).unsqueeze(0).to(device)
                    output_screen, _, _ = model(screen_tensor)
                    predicted_bits_screen = (output_screen > 0.5).float().squeeze().cpu().numpy()
                    if len(predicted_bits_screen) == len(watermark_bits):
                        screen_accuracy = np.mean(predicted_bits_screen == watermark_bits)
                        # 统计屏摄测试中所有位数都正确的情况
                        if screen_accuracy == 1.0:
                            screen_correct_all_bits_count += 1
                    screen_accuracy_list.append(screen_accuracy)

                

                if args.test_noise:
                    logger.info(f"{filenames[i]}: PSNR={psnr:.2f}dB, Original_Acc={accuracy*100:.2f}%, Noise_Acc={avg_noise_accuracy*100:.2f}%, JPEG_Acc={jpeg_accuracy*100:.2f}%, Screen_Acc={screen_accuracy*100:.2f}%")
                else:
                    logger.info(f"{filenames[i]}: PSNR={psnr:.2f}dB, Original_Acc={accuracy*100:.2f}%, Noise_Acc={avg_noise_accuracy*100:.2f}%")

    # ========== 统计结果 ==========
    avg_psnr = np.mean(psnr_list) if psnr_list else 0.0
    avg_accuracy = np.mean(accuracy_list) if accuracy_list else 0.0
    avg_noise_accuracy = np.mean(noise_accuracy_list) if noise_accuracy_list else 0.0
    avg_jpeg_accuracy = np.mean(jpeg_accuracy_list) if jpeg_accuracy_list else 0.0
    avg_screen_accuracy = np.mean(screen_accuracy_list) if screen_accuracy_list else 0.0


    logger.info(f"Inference Summary:")
    logger.info(f"  Total images processed: {len(psnr_list)}")
    logger.info(f"  Average PSNR: {avg_psnr:.2f} dB")
    logger.info(f"  Average Original Decode Accuracy: {avg_accuracy*100:.2f}%")
    logger.info(f"  Original Test - All bits correct: {correct_all_bits_count}/{len(psnr_list)}")
    logger.info(f"  Average Noise Decode Accuracy: {avg_noise_accuracy*100:.2f}%")
    logger.info(f"  Noise Test - All bits correct: {noise_correct_all_bits_count}/{len(psnr_list)}")
    if args.test_noise:
        logger.info(f"  Average JPEG Compression Accuracy: {avg_jpeg_accuracy*100:.2f}%")
        logger.info(f"  JPEG Test - All bits correct: {jpeg_correct_all_bits_count}/{len(psnr_list)}")
        logger.info(f"  Average Screen Shooting Accuracy: {avg_screen_accuracy*100:.2f}%")
        logger.info(f"  Screen Test - All bits correct: {screen_correct_all_bits_count}/{len(psnr_list)}")
        
    logger.info(f"  Noise test: {args.num_crops} random crops per image from 1024x1024 tiled watermark image")
    logger.info(f"  Watermarked images saved to: {args.output_dir}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
