"""
YOLO12 交通标志识别 - 训练脚本
======================================
在 PyCharm 中直接右键运行，或通过命令行加参数运行：

  # 方式1：直接运行（使用默认参数，正式训练）
  python train.py

  # 方式2：小试跑（快速验证流程，只跑 3 个 epoch）
  python train.py --smoke

  # 方式3：自定义参数
  python train.py --model yolo12s.pt --epochs 50 --batch 8

使用 COCO 格式交通标志数据集（46 类），默认配置见 configs/coco.yaml。
"""
import argparse
from pathlib import Path

from ultralytics import YOLO


# 项目根目录（本脚本位于 src/ 下，根目录是其上一级）
ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    """解析命令行参数"""
    p = argparse.ArgumentParser(description="YOLO12 交通标志识别训练")
    p.add_argument("--model", default="yolo12n.pt",
                   help="预训练权重（默认 yolo12n.pt，可选 yolo12s.pt 等）")
    p.add_argument("--data", default=str(ROOT / "configs" / "coco.yaml"),
                   help="数据集配置文件路径")
    p.add_argument("--epochs", type=int, default=80,
                   help="训练轮数（默认 80）")
    p.add_argument("--batch", type=int, default=16,
                   help="批大小（默认 16，显存不足可降到 8）")
    p.add_argument("--imgsz", type=int, default=640,
                   help="输入图像尺寸（默认 640）")
    p.add_argument("--device", default="0",
                   help="训练设备（默认 0=第一块 GPU，CPU 用 'cpu'）")
    p.add_argument("--smoke", action="store_true",
                   help="小试跑模式：3 个 epoch 快速验证流程")
    p.add_argument("--resume", action="store_true",
                   help="从最近一次中断的训练继续跑（自动搜索 runs/train-* 的 last.pt）")
    p.add_argument("--mixup", type=float, default=0.0,
                   help="Mixup 数据增强强度（0.0=关闭，0.1-0.2=推荐）")
    return p.parse_args()


def main():
    args = parse_args()

    # 小试跑模式：覆盖参数，只跑 3 轮，快速验证整个流程是否通顺
    if args.smoke:
        args.epochs = 3
        args.batch = 8
        print("=" * 60)
        print("【小试跑模式】3 个 epoch，仅验证流程是否通畅")
        print("  正式训练请去掉 --smoke 参数")
        print("=" * 60)

    print(f"项目根目录: {ROOT}")
    print(f"数据集配置: {args.data}")
    print("=" * 60)

    # ===== 加载模型 =====
    if args.resume:
        # 续训模式：自动找到最近一次训练保存的 last.pt
        runs_dir = ROOT / "runs"
        # 搜索所有 train* 目录，按修改时间从新到旧排序
        train_dirs = sorted(runs_dir.glob("train*"), key=lambda p: p.stat().st_mtime, reverse=True)
        train_dirs = [d for d in train_dirs if (d / "weights" / "last.pt").exists()]

        if not train_dirs:
            print("错误: 找不到包含 last.pt 的训练记录，无法续训")
            return

        last_pt = train_dirs[0] / "weights" / "last.pt"
        print(f"【续训模式】从 {last_pt} 继续训练")
        print(f"  训练目录: {train_dirs[0]}")
        print("=" * 60)

        # 检查 last.pt 是否包含训练状态（可续训的标志）
        import torch
        try:
            ckpt = torch.load(str(last_pt), map_location="cpu", weights_only=False)
            if "epoch" not in ckpt:
                print("错误: last.pt 不含训练状态（训练已完成的权重不可续训）")
                print("请重新运行训练命令，不加 --resume 参数")
                return
        except Exception as e:
            print(f"错误: 无法读取权重文件: {e}")
            return

        model = YOLO(str(last_pt))
        results = model.train(
            resume=True,  # ultralytics 自动从上次中断处继续
            workers=0,
            verbose=True,
        )
    else:
        # 从头训练
        print(f"预训练模型: {args.model}")
        print(f"训练参数: epochs={args.epochs}, batch={args.batch}, imgsz={args.imgsz}, device={args.device}")
        print("=" * 60)

        model = YOLO(args.model)
        results = model.train(
            data=args.data,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            device=args.device,
            project=str(ROOT / "runs"),
            name="train",
            patience=20,          # 早停：20 轮无提升则停止
            workers=0,            # Windows 下避免 DataLoader 崩溃
            cache=False,
            mixup=args.mixup,
            verbose=True,
        )

    print("=" * 60)
    print("训练完成！")
    # 找到最新的 best.pt
    runs_dir = ROOT / "runs"
    best_pts = sorted(runs_dir.glob("train*/weights/best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if best_pts:
        print(f"最佳权重: {best_pts[0]}")
    print("下一步: 用 python src/predict_image.py 进行图片推理")
    print("=" * 60)
    return results


if __name__ == "__main__":
    main()
