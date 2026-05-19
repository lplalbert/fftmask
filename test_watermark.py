import numpy as np
import cv2
from cwt_method import CWTWatermark
from watermark_trainer import WatermarkDecoder
import torch
from torchvision import transforms

def test_watermark_pipeline():
    """
    测试水印生成、叠加和解码的完整流程
    """
    # 1. 初始化水印系统
    block_size = 512
    watermark_system = CWTWatermark(block_size=block_size)
    
    # 2. 生成64位随机水印
    watermark_bits = np.random.randint(0, 2, size=64)
    print(f"Generated watermark: {watermark_bits}")
    
    # 3. 生成水印模板
    Tm, M1 = watermark_system.generate_message_watermark_template2(watermark_bits)
    print(f"Watermark template shape: {Tm.shape}")
    
    # 保存水印模板
    cv2.imwrite('watermark_template.png', Tm)
    print("Watermark template saved to: watermark_template.png")
    
    # 4. 读取测试图像
    test_image_path = 'test_image.png'  # 确保此路径存在测试图像
    if not os.path.exists(test_image_path):
        # 创建一个测试图像
        test_image = np.ones((block_size, block_size, 3), dtype=np.uint8) * 128
        cv2.imwrite(test_image_path, test_image)
        print(f"Created test image: {test_image_path}")
    
    image = cv2.imread(test_image_path)
    if image.shape[:2] != (block_size, block_size):
        image = cv2.resize(image, (block_size, block_size))
    
    # 5. 透明度叠加水印
    alpha = 0.3
    image_float = image.astype(np.float32) / 255.0
    Tm_float = Tm.astype(np.float32) / 255.0
    watermarked_image = (1 - alpha) * image_float + alpha * Tm_float[:,:,np.newaxis]
    watermarked_image = (watermarked_image * 255).astype(np.uint8)
    
    # 保存叠加后的图像
    output_path = 'watermarked_test.png'
    cv2.imwrite(output_path, watermarked_image)
    print(f"Watermarked image saved to: {output_path}")
    
    # 6. 使用网络解码水印
    # 加载模型
    model_path = 'best_watermark_decoder.pth'  # 确保模型已训练
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        print("Please train the model first using watermark_trainer.py")
        return
    
    model = WatermarkDecoder()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # 准备图像
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    watermarked_rgb = cv2.cvtColor(watermarked_image, cv2.COLOR_BGR2RGB)
    image_tensor = transform(watermarked_rgb).unsqueeze(0)
    
    # 解码
    with torch.no_grad():
        output = model(image_tensor)
        extracted_watermark = (output > 0.5).float().squeeze().numpy().astype(int)
    
    print(f"Extracted watermark: {extracted_watermark}")
    
    # 计算准确率
    accuracy = np.sum(watermark_bits == extracted_watermark) / 64.0
    print(f"Decoding accuracy: {accuracy:.4f}")
    
    return watermark_bits, extracted_watermark, accuracy

if __name__ == '__main__':
    import os
    # 创建必要的目录
    os.makedirs('data/train', exist_ok=True)
    os.makedirs('data/val', exist_ok=True)
    os.makedirs('data/test', exist_ok=True)
    
    # 运行测试
    test_watermark_pipeline()
