"""批量矫正：668x668，使用大小筛选，递归处理子目录"""
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).parent.parent.parent / "watermark_tools" / "rectify_corners.py"

dirs = [
    Path(__file__).parent / "pz",
    Path(__file__).parent / "pzvx",
]

total_ok = 0
total_fail = 0

for src_dir in dirs:
    if not src_dir.exists():
        print(f"跳过: {src_dir}")
        continue

    # 递归查找所有图片
    images = sorted([
        f for f in src_dir.rglob("*")
        if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp')
        and f.is_file()
    ])

    print(f"\n{src_dir.name}: {len(images)} 张图片")

    for i, img in enumerate(images, 1):
        # 保持子目录结构
        rel = img.relative_to(src_dir)
        out_dir = src_dir.parent / f"{src_dir.name}_rectified" / rel.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / img.name

        if out_file.exists():
            print(f"  [{i}/{len(images)}] 跳过: {rel}")
            continue

        print(f"  [{i}/{len(images)}] {rel}")
        result = subprocess.run(
            [sys.executable, str(SCRIPT),
             "--image", str(img),
             "--output", str(out_file),
             "--screen-w", "668",
             "--screen-h", "668"],
            capture_output=True, timeout=60
        )
        if result.returncode == 0:
            total_ok += 1
        else:
            total_fail += 1
            print(f"    失败")

print(f"\n完成! 成功: {total_ok}, 失败: {total_fail}")
