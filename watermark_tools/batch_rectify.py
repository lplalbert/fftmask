"""
批量透视矫正脚本
遍历分类目录，对每个图片调用 rectify_corners.py 进行矫正
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path


def batch_rectify(input_dir: str, output_dir: str, model: str = "Qrlocted/20250729_best.pt",
                   conf: float = 0.5, imgsz: int = 1280, screen_w: int = 1920, screen_h: int = 1080,
                   size_tolerance: float = 0.3, debug: bool = False):
    input_root = Path(input_dir)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    rectify_script = Path(__file__).parent / "rectify_corners.py"

    # Collect all jpg/jpeg/png files
    image_files = []
    for root, dirs, files in os.walk(input_root):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_files.append(Path(root) / file)

    print(f"找到 {len(image_files)} 张图片")

    success_count = 0
    fail_count = 0
    fail_files = []

    for i, img_path in enumerate(image_files):
        rel_path = img_path.relative_to(input_root)
        out_path = output_root / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"[{i+1}/{len(image_files)}] {rel_path} ... ", end="", flush=True)

        cmd = [
            sys.executable, str(rectify_script),
            "--image", str(img_path),
            "--output", str(out_path),
            "--model", model,
            "--conf", str(conf),
            "--imgsz", str(imgsz),
            "--screen-w", str(screen_w),
            "--screen-h", str(screen_h),
            "--size-tolerance", str(size_tolerance),
        ]

        if debug:
            cmd.append("--debug")

        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')

        if result.returncode == 0:
            print("[OK]")
            success_count += 1
        else:
            print("[FAIL]")
            fail_count += 1
            fail_files.append((str(rel_path), result.stderr.strip()[-200:] if result.stderr else "Unknown error"))

    print(f"\n{'='*60}")
    print(f"完成: 成功 {success_count}, 失败 {fail_count}")

    if fail_files:
        print(f"\n失败文件:")
        for file, err in fail_files[:20]:
            print(f"  {file}: {err}")


def main():
    parser = argparse.ArgumentParser(description="批量透视矫正")
    parser.add_argument("--input", "-i", required=True, help="输入目录")
    parser.add_argument("--output", "-o", required=True, help="输出目录")
    parser.add_argument("--model", default="Qrlocted/20250729_best.pt", help="YOLO模型路径")
    parser.add_argument("--conf", type=float, default=0.5, help="置信度阈值")
    parser.add_argument("--imgsz", type=int, default=1280, help="推理图片尺寸")
    parser.add_argument("--screen-w", type=int, default=1920, help="输出宽度")
    parser.add_argument("--screen-h", type=int, default=1080, help="输出高度")
    parser.add_argument("--size-tolerance", type=float, default=0.3, help="大小筛选容差")
    parser.add_argument("--debug", action="store_true", help="保存调试图")
    args = parser.parse_args()

    batch_rectify(
        input_dir=args.input,
        output_dir=args.output,
        model=args.model,
        conf=args.conf,
        imgsz=args.imgsz,
        screen_w=args.screen_w,
        screen_h=args.screen_h,
        size_tolerance=args.size_tolerance,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
