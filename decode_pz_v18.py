"""
批量解码 pz/v18 的 b 和 cb 通道图片
"""

import os
import sys
import json
import subprocess
import tempfile

BITS_FILE = "/home/lpl2025/lpl/fftmask/0908test_sorted/part/pz_bits.json"
BASE_DIR = "/home/lpl2025/lpl/fftmask/0908test_sorted/part/rectified/pz/v18"
SCRIPT = "/home/lpl2025/lpl/fftmask/sweep_decode_channel_watermark_v18.py"

# 模型路径
MODELS = {
    "b": "/home/lpl2025/lpl/fftmask/output/v18_b_pair/best_model.pth",
    "cb": "/home/lpl2025/lpl/fftmask/output/v18_hollow_pair/best_model.pth",
}

def load_bits():
    with open(BITS_FILE, "r") as f:
        return json.load(f)

def write_temp_bits(bits_array, path):
    """写入 sweep_decode 脚本期望的格式"""
    with open(path, "w") as f:
        json.dump({"watermark_bits": [int(b) for b in bits_array]}, f)

def main():
    all_bits = load_bits()
    results = {}

    for channel in ["b", "cb"]:
        print(f"\n{'='*80}")
        print(f"Channel: {channel.upper()}")
        print(f"Model: {MODELS[channel]}")
        print(f"{'='*80}")

        channel_dir = os.path.join(BASE_DIR, channel)
        if not os.path.isdir(channel_dir):
            print(f"  目录不存在: {channel_dir}")
            continue

        for index in sorted(os.listdir(channel_dir)):
            index_dir = os.path.join(channel_dir, index)
            if not os.path.isdir(index_dir):
                continue

            # 检查图片数量
            images = [f for f in os.listdir(index_dir) if f.lower().endswith((".png", ".jpg"))]
            if not images:
                continue

            bits_key = f"{channel}_{index}"
            bits_str = all_bits.get(bits_key)
            if not bits_str:
                print(f"  [{channel}/{index}] 无对应 bits，跳过")
                continue

            # 写入临时 bits 文件
            tmp_bits = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
            write_temp_bits(bits_str, tmp_bits.name)
            tmp_bits.close()

            print(f"\n  [{channel}/{index}] {len(images)} 张图片, bits={bits_str[:20]}...")

            # 调用 sweep_decode 脚本
            cmd = [
                sys.executable, SCRIPT,
                "--input_dir", index_dir,
                "--bits_file", tmp_bits.name,
                "--model_path", MODELS[channel],
                "--channels", channel if channel != "b" else "y",  # b通道用y通道提取
                "--num_bits", "60",
                "--r", "12", "25",
                "--bitsf", "15", "45",
                "--crop_size", "512",
                "--batch_size", "64",
                "--show_all",
            ]

            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                # 找到 vote_acc
                for line in result.stdout.split("\n"):
                    if "vote=" in line and "%" in line:
                        print(f"    {line.strip()}")
                    elif "Best per image:" in line:
                        # 打印后面几行
                        idx = result.stdout.split("\n").index(line)
                        for l in result.stdout.split("\n")[idx+1:idx+1+len(images)+1]:
                            if l.strip():
                                print(f"    {l.strip()}")
            except Exception as e:
                print(f"    错误: {e}")
            finally:
                os.unlink(tmp_bits.name)

if __name__ == "__main__":
    main()
