"""
YOLO12 交通标志识别 - 视频文件推理
======================================
对视频文件逐帧进行交通标志检测，输出标注后的新视频（或实时弹窗观看）。

用法:
  # 静默处理并保存
  python predict_video.py --source data/test.mp4

  # 实时弹窗观看检测过程（按 q 提前退出）
  python predict_video.py --source my_video.mp4 --show
"""
import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent


def find_best_model():
    """自动搜索 runs/ 下最新的 best.pt 权重文件"""
    runs_dir = ROOT / "runs"
    best_candidates = sorted(runs_dir.glob("train*/weights/best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if best_candidates:
        return str(best_candidates[0])
    fallback = runs_dir / "train" / "weights" / "best.pt"
    if fallback.exists():
        return str(fallback)
    return "yolo12n.pt"


def parse_args():
    p = argparse.ArgumentParser(description="YOLO12 视频推理")
    p.add_argument("--model", default=find_best_model(), help="模型权重（默认自动搜索最新 best.pt）")
    p.add_argument("--source", default=str(ROOT / "data" / "test.mp4"), help="视频文件路径")
    p.add_argument("--conf", type=float, default=0.35, help="置信度阈值")
    p.add_argument("--imgsz", type=int, default=640, help="推理图像尺寸")
    p.add_argument("--device", default="0", help="推理设备（默认 0=GPU）")
    p.add_argument("--show", action="store_true", help="实时弹窗显示检测过程（按 q 提前退出）")
    return p.parse_args()


def main():
    args = parse_args()

    if not Path(args.source).exists():
        print(f"错误: 视频文件不存在: {args.source}")
        print("请将视频放到 data/ 目录，或用 --source 指定路径")
        return

    print(f"模型: {args.model}")
    print(f"视频: {args.source}")
    print("处理中...（逐帧检测，请耐心等待）")
    print("=" * 60)

    model = YOLO(args.model)
    # stream=True 逐帧返回，避免一次性加载全部帧占用过多内存
    results = model.predict(
        source=args.source,
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        save=True,
        stream=True,
        project=str(ROOT / "runs"),
        name="predict_video",
        exist_ok=True,
    )

    frame_count = 0
    total_objects = 0
    win_name = "YOLO12 Traffic Sign Detection (press q to quit)"
    for r in results:
        frame_count += 1
        total_objects += len(r.boxes) if r.boxes is not None else 0

        # 实时弹窗显示
        if args.show:
            annotated = r.plot()
            cv2.imshow(win_name, annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("  用户按 q 提前退出")
                break

        if frame_count % 30 == 0:  # 每 30 帧打印一次进度
            print(f"  已处理 {frame_count} 帧...")

    cv2.destroyAllWindows()

    print("=" * 60)
    print(f"完成！共处理 {frame_count} 帧，累计检测到 {total_objects} 个目标")
    print(f"结果视频保存于: {ROOT / 'runs' / 'predict_video'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
