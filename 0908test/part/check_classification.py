"""检查pz目录下图片的二维码信息与目录结构是否一致"""
import cv2
import json
import os
import numpy as np
from pathlib import Path

def decode_qr(image_path):
    """多尺度多预处理检测二维码"""
    with open(str(image_path), 'rb') as f:
        raw = np.frombuffer(f.read(), dtype=np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if img is None:
        return None

    detector = cv2.QRCodeDetector()
    h, w = img.shape[:2]

    # 不同裁剪区域
    regions = [
        (0, 0, w, h),
        (0, 0, w // 2, h // 2),
        (0, 0, w // 3, h // 3),
    ]

    for x1, y1, x2, y2 in regions:
        crop = img[y1:y2, x1:x2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        # 多种预处理
        candidates = [
            crop,
            gray,
            cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1],
            cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)[1],
            cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
            cv2.GaussianBlur(gray, (5, 5), 0),
        ]

        for processed in candidates:
            d, _, _ = detector.detectAndDecode(processed)
            if d and '|' in d:
                return d

    return None


def parse_qr_data(data):
    """解析二维码数据: v17|cb|01|1001000010..."""
    parts = data.split('|')
    if len(parts) != 4:
        return None
    return {
        'version': parts[0],
        'channel': parts[1],
        'index': parts[2],
        'bit_sequence': parts[3]
    }


def main():
    base_dir = Path(r"C:\Users\24976\Desktop\code\fftmask\0908test\part\pz")
    results = []
    contradictions = []

    for version in ['v17', 'v18']:
        for channel in ['b', 'cb']:
            dir_path = base_dir / version / channel
            if not dir_path.exists():
                continue

            images = sorted([f for f in dir_path.iterdir()
                           if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp')])

            for img_path in images:
                qr_data = decode_qr(img_path)
                parsed = parse_qr_data(qr_data) if qr_data else None

                entry = {
                    'file': img_path.name,
                    'directory_version': version,
                    'directory_channel': channel,
                    'qr_raw': qr_data,
                    'qr_version': parsed['version'] if parsed else None,
                    'qr_channel': parsed['channel'] if parsed else None,
                    'qr_index': parsed['index'] if parsed else None,
                    'bit_sequence': parsed['bit_sequence'] if parsed else None,
                }
                results.append(entry)

                # 检查是否矛盾
                if parsed:
                    if parsed['version'] != version or parsed['channel'] != channel:
                        contradictions.append(entry)
                        print(f"[矛盾] {version}/{channel}/{img_path.name} -> QR: {qr_data}")
                else:
                    print(f"[未识别] {version}/{channel}/{img_path.name}")

    # 保存结果
    output = {
        'total': len(results),
        'recognized': sum(1 for r in results if r['qr_raw']),
        'contradictions': len(contradictions),
        'details': results,
        'contradiction_list': contradictions,
    }

    output_path = base_dir / 'classification_check.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n总计: {output['total']} 张图片")
    print(f"识别成功: {output['recognized']} 张")
    print(f"矛盾: {output['contradictions']} 张")
    print(f"结果已保存: {output_path}")


if __name__ == '__main__':
    main()
