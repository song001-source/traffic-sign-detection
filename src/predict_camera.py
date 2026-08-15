"""
YOLO12 交通标志识别 - 摄像头实时检测
======================================
打开电脑摄像头，实时检测画面中的交通标志。按 q 退出。

用法:
  # 默认使用第 0 个摄像头
  python predict_camera.py

  # 指定摄像头编号（外接 USB 摄像头可能是 1）
  python predict_camera.py --cam 1

  # 使用训练好的模型
  python predict_camera.py --model runs/train/weights/best.pt
"""
import argparse
import time
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
    p = argparse.ArgumentParser(description="YOLO12 摄像头实时检测")
    p.add_argument("--model", default=find_best_model(), help="模型权重（默认自动搜索最新 best.pt）")
    p.add_argument("--cam", type=int, default=0, help="摄像头编号（默认 0）")
    p.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    p.add_argument("--imgsz", type=int, default=640, help="推理图像尺寸")
    p.add_argument("--device", default="0", help="推理设备（默认 0=GPU）")
    return p.parse_args()


def main():
    args = parse_args()

    # ----- 打开摄像头 -----
    # 先检查 Windows 下是否有摄像头驱动问题
    cap = cv2.VideoCapture(args.cam, cv2.CAP_DSHOW)  # CAP_DSHOW 在 Windows 上更稳定
    if not cap.isOpened():
        # 第二次尝试：不加 DSHOW 后端
        cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print(f"错误: 无法打开摄像头 {args.cam}")
        print("可能的原因和解决方法:")
        print("  1. 摄像头被其他程序占用 → 关闭微信/腾讯会议/Zoom 等")
        print("  2. 编号不对 → 笔记本内置摄像头用 --cam 0，外接 USB 用 --cam 1")
        print("  3. 驱动问题 → 设备管理器中检查摄像头驱动状态")
        print("  4. 虚拟机/远程桌面 → 摄像头无法被穿透访问")
        return

    # 设置分辨率（降低到 640x480 可提高帧率）
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    model = YOLO(args.model)
    print(f"模型: {args.model}")
    print("摄像头已打开，实时检测中...")
    print("按 q 键退出，按 s 键保存当前帧")
    print("=" * 60)

    frame_count = 0
    fps_start = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("错误: 读取摄像头画面失败")
                break

            # 逐帧推理
            results = model.predict(
                source=frame,
                conf=args.conf,
                imgsz=args.imgsz,
                device=args.device,
                verbose=False,
            )
            annotated = results[0].plot()

            # 计算并显示 FPS
            frame_count += 1
            elapsed = time.time() - fps_start
            fps = frame_count / elapsed if elapsed > 0 else 0
            cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("YOLO12 Traffic Sign Detection (press q to quit)", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                snap_dir = ROOT / "runs"
                snap_dir.mkdir(parents=True, exist_ok=True)
                save_path = str(snap_dir / f"snapshot_{frame_count}.jpg")
                cv2.imwrite(save_path, annotated)
                print(f"  已保存快照: {save_path}")

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("摄像头已关闭，程序退出。")


if __name__ == "__main__":
    main()
