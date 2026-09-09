"""批量矫正pz目录下的图片"""
import subprocess
import sys
from pathlib import Path

base = Path(__file__).parent
pz_dir = base / "pz"
output_base = base / "pz_rectified"

rectify_script = base.parent.parent / "watermark_tools" / "rectify_corners.py"

for version in ["v17", "v18"]:
    for channel in ["b", "cb"]:
        input_dir = pz_dir / version / channel
        output_dir = output_base / version / channel
        output_dir.mkdir(parents=True, exist_ok=True)

        images = sorted([
            f for f in input_dir.iterdir()
            if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")
        ])

        print(f"\n{version}/{channel}: {len(images)} images")

        for i, img in enumerate(images, 1):
            out_file = output_dir / img.name
            if out_file.exists():
                print(f"  [{i}/{len(images)}] Skip (exists): {img.name}")
                continue

            print(f"  [{i}/{len(images)}] {img.name}")
            result = subprocess.run(
                [sys.executable, str(rectify_script),
                 "--image", str(img),
                 "--output", str(out_file)],
                capture_output=True, text=True, timeout=60
            )
            if result.returncode != 0:
                print(f"    FAILED: {result.stderr.strip()[:100]}")

print("\nDone!")
