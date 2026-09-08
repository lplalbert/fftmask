"""同时运行图片播放和水印叠加。

支持两种模式：
1. 固定模板模式（默认）：一直用同一个水印模板
2. 播放列表模式（--total）：随机分配图片给10个模板，平均分配

图层顺序（从下到上）：
1. 干净图片（image_slideshow.py）
2. 二维码（贴在图片左上角）
3. 水印透明层（qt_display.py，最上面）
"""
import argparse
import json
import os
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def generate_playlist(image_dir, total, num_templates=10, seed=None):
    """
    生成播放列表：随机选取total张图片，平均分配给num_templates个模板。

    返回: (playlist_path, stats)
    """
    # 收集所有图片
    image_paths = []
    for ext in ["*.png", "*.jpg", "*.jpeg", "*.bmp"]:
        image_paths.extend(sorted(image_dir.glob(ext)))

    if not image_paths:
        raise RuntimeError(f"{image_dir} 下没有图片")

    if total > len(image_paths):
        print(f"[RunBoth] 警告: 要求{total}张但只有{len(image_paths)}张，全部使用")
        total = len(image_paths)

    # 随机采样
    rng = random.Random(seed)
    selected = rng.sample(image_paths, total)

    # 分配模板索引：平均分配，每张图一个模板
    images = []
    template_indices = []
    per_template = total // num_templates
    remainder = total % num_templates

    # 构建分配列表：前remainder个模板多分1张
    assignment = []
    for i in range(num_templates):
        count = per_template + (1 if i < remainder else 0)
        assignment.extend([i] * count)

    # 打乱分配顺序（图片和模板一起shuffle）
    combined = list(zip(selected, assignment))
    rng.shuffle(combined)

    for img_path, tpl_idx in combined:
        images.append(str(img_path))
        template_indices.append(tpl_idx)

    # 写入临时文件
    playlist = {
        "images": images,
        "template_indices": template_indices,
        "num_templates": num_templates,
    }

    fd, playlist_path = tempfile.mkstemp(suffix=".json", prefix="watermark_playlist_")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(playlist, f, ensure_ascii=False, indent=2)

    # 统计
    from collections import Counter
    stats = Counter(template_indices)

    return playlist_path, total, stats


def main():
    parser = argparse.ArgumentParser(description="图片播放 + 二维码 + 水印叠加")
    parser.add_argument("--version", choices=["v17", "v18", "both"], default="v17")
    parser.add_argument("--channel", choices=["cb", "b"], default="cb")
    parser.add_argument("--alpha", type=float, default=0.016)
    parser.add_argument("--interval", type=int, default=1000, help="图片切换间隔ms")
    parser.add_argument("--wm_interval", type=int, default=5000, help="水印切换间隔ms")
    parser.add_argument("--delay", type=float, default=1.0, help="启动水印前等待秒数")
    parser.add_argument("--index", type=int, default=0, help="固定模式: 水印模板索引")
    parser.add_argument("--corners", action="store_true", help="四角加回字形标记")
    parser.add_argument("--local", action="store_true", help="局部模式：回字形围住768x768区域")
    parser.add_argument("--region_size", type=int, default=768, help="局部模式区域大小")

    # 播放列表模式参数
    parser.add_argument("--total", type=int, default=None, help="播放列表模式: 总图片数量（平均分给10个模板）")
    parser.add_argument("--image_dir", type=Path, default=None, help="播放列表模式: 图片目录（默认clean_image/）")
    parser.add_argument("--seed", type=int, default=None, help="随机种子（可复现）")
    args = parser.parse_args()

    script_dir = Path(__file__).parent

    # v18等效alpha
    alpha = args.alpha
    if args.version == "v18" and args.alpha == 0.016:
        alpha = 0.0228

    # 二维码版本（both时用v17）
    qr_version = "v17" if args.version == "both" else args.version

    playlist_path = None

    if args.total is not None:
        # 播放列表模式
        image_dir = args.image_dir or (script_dir / "clean_image")
        if not image_dir.is_absolute():
            image_dir = script_dir / image_dir

        playlist_path, total, stats = generate_playlist(
            image_dir, args.total, num_templates=10, seed=args.seed
        )

        print(f"[RunBoth] 播放列表模式: {total}张图片, 10个模板")
        print(f"[RunBoth] 每模板分配: {dict(sorted(stats.items()))}")
        print(f"[RunBoth] 播放列表: {playlist_path}")

        # 播放列表模式下interval同步（用同一个间隔）
        args.wm_interval = args.interval

    # 1. 启动图片播放（含二维码）
    slideshow_cmd = [
        sys.executable, str(script_dir / "image_slideshow.py"),
        "--interval", str(args.interval),
        "--version", qr_version,
        "--channel", args.channel,
        "--index", str(args.index),
    ]
    if playlist_path:
        slideshow_cmd.extend(["--playlist", playlist_path])
    if args.corners:
        slideshow_cmd.append("--corners")
    if args.local:
        slideshow_cmd.append("--local")
        slideshow_cmd.extend(["--region_size", str(args.region_size)])
    print(f"[RunBoth] 启动图片播放 + 二维码")
    slideshow_proc = subprocess.Popen(slideshow_cmd)

    # 2. 等待播放窗口就绪（播放列表模式不等待，保持同步）
    if not playlist_path:
        print(f"[RunBoth] 等待 {args.delay}s...")
        time.sleep(args.delay)

    # 3. 启动水印叠加
    wm_cmd = [
        sys.executable, str(script_dir / "qt_display.py"),
        "--version", args.version,
        "--channel", args.channel,
        "--alpha", str(alpha),
        "--interval", str(args.wm_interval),
        "--index", str(args.index),
    ]
    if playlist_path:
        wm_cmd.extend(["--playlist", playlist_path])
    print(f"[RunBoth] 启动水印叠加: {args.version} {args.channel} alpha={alpha}")
    wm_proc = subprocess.Popen(wm_cmd)

    print(f"[RunBoth] 完成。ESC退出。")

    try:
        while True:
            if slideshow_proc.poll() is not None:
                wm_proc.terminate()
                break
            if wm_proc.poll() is not None:
                slideshow_proc.terminate()
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        slideshow_proc.terminate()
        wm_proc.terminate()
    finally:
        # 清理临时播放列表
        if playlist_path and Path(playlist_path).exists():
            Path(playlist_path).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
