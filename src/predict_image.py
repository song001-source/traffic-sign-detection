"""
YOLO12 交通标志识别 - 图片批量推理
======================================
对单张图片或整个文件夹的图片进行交通标志检测，结果保存到 runs/detect/。

用法:
  # 单张图片
  python predict_image.py --source data/test.jpg

  # 整个文件夹
  python predict_image.py --source data/images/

  # 指定训练好的模型和置信度阈值
  python predict_image.py --source data/test.jpg --model runs/train/weights/best.pt --conf 0.4
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent


def find_best_model():
    """自动搜索 runs/ 下最新的 best.pt 权重文件"""
    runs_dir = ROOT / "runs"
    best_candidates = sorted(runs_dir.glob("train*/weights/best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if best_candidates:
        return str(best_candidates[0])
    # 也检查 train/weights/best.pt（无后缀的情况）
    fallback = runs_dir / "train" / "weights" / "best.pt"
    if fallback.exists():
        return str(fallback)
    return "yolo12n.pt"


def parse_args():
    p = argparse.ArgumentParser(description="YOLO12 图片推理")
    # 默认模型优先用训练好的 best.pt（自动搜索最新），找不到则回退到预训练 yolo12n.pt
    p.add_argument("--model", default=find_best_model(),
                   help="模型权重（默认自动搜索最新 best.pt，回退 yolo12n.pt）")
    p.add_argument("--source", default=str(ROOT / "data" / "coco" / "images" / "val"),
                   help="图片路径或文件夹路径（默认验证集目录）")
    p.add_argument("--conf", type=float, default=0.35,
                   help="置信度阈值（默认 0.35，低于此值的检测结果会被过滤）")
    p.add_argument("--imgsz", type=int, default=640,
                   help="推理图像尺寸（默认 640）")
    p.add_argument("--device", default="0",
                   help="推理设备（默认 0=GPU）")
    return p.parse_args()


def main():
    args = parse_args()
    source = args.source
    if not Path(source).exists() and source != "yolo12n.pt":
        # source 不是文件也不是内置资源，提示用户
        print(f"警告: 找不到 {source}")
        print("提示: 请将测试图片放到 data/ 目录下，或用 --source 指定路径")

    print(f"模型: {args.model}")
    print(f"输入: {args.source}")
    print(f"置信度: {args.conf}")
    print("=" * 60)

    model = YOLO(args.model)
    results = model.predict(
        source=args.source,
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        save=True,            # 保存带检测框的结果图
        project=str(ROOT / "runs"),
        name="predict_image",
        exist_ok=True,        # 覆盖上次结果，避免生成 predict_image2/3...
    )

    print("=" * 60)
    print(f"推理完成，共处理 {len(results)} 张图片")
    print(f"结果保存于: {ROOT / 'runs' / 'predict_image'}")

    # 打印每张图的检测结果
    for i, r in enumerate(results):
        n = len(r.boxes) if r.boxes is not None else 0
        print(f"  图片 {i + 1}: 检测到 {n} 个目标")
    print("=" * 60)


if __name__ == "__main__":
    main()
