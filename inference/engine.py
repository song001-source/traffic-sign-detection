"""
YOLO 推理引擎
============
封装 Ultralytics YOLO 模型，提供图片/视频/实时流的统一推理接口。
纯推理层，不依赖任何 Web 框架。
"""
import time
import uuid
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from inference.config import ROOT, RUNS_DIR, InferenceConfig
from inference.label_mapping import get_chinese_name
from inference.schemas import Detection


class DetectorService:
    """检测器服务（单例模式）"""

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_path: Optional[str] = None):
        if self._initialized:
            return
        self._initialized = True
        self._model: Optional[YOLO] = None
        self._current_model_path: Optional[str] = None
        self._class_names: List[str] = []

        load_path = model_path or self._find_best_model()
        self.load_model(load_path)

    # ── 模型管理 ─────────────────────────────────────

    def load_model(self, path: str) -> None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"模型文件不存在: {path}")
        self._model = YOLO(str(p))
        self._current_model_path = str(p.resolve())
        if hasattr(self._model, "names"):
            self._class_names = list(self._model.names.values())
        print(f"[Detector] 模型已加载: {p.name}")

    def get_current_model(self) -> Tuple[str, str]:
        if not self._current_model_path:
            return "", ""
        p = Path(self._current_model_path)
        return p.name, str(p)

    def get_available_models(self) -> List[dict]:
        models = []
        for d in [ROOT, ROOT / "models", RUNS_DIR]:
            if d.exists():
                for f in sorted(d.rglob("*.pt")):
                    models.append({
                        "name": f.name,
                        "path": str(f.resolve()),
                        "size_mb": round(f.stat().st_size / (1024 * 1024), 2),
                        "is_loaded": str(f.resolve()) == self._current_model_path,
                    })
        seen = set()
        return [m for m in models if not (m["path"] in seen or seen.add(m["path"]))]

    @staticmethod
    def _find_best_model() -> str:
        candidates = sorted(
            RUNS_DIR.glob("train*/weights/best.pt"),
            key=lambda p: p.stat().st_mtime, reverse=True,
        )
        if candidates:
            return str(candidates[0])
        best = ROOT / "best.pt"
        if best.exists():
            return str(best)
        fallback = ROOT / "yolo12n.pt"
        if fallback.exists():
            return str(fallback)
        raise FileNotFoundError("未找到任何可用模型")

    # ── 核心推理 ─────────────────────────────────────

    def detect_image(
        self,
        image: np.ndarray,
        conf: float = InferenceConfig.CONF_THRESHOLD,
        iou: float = InferenceConfig.IOU_THRESHOLD,
        imgsz: int = InferenceConfig.IMG_SIZE,
        classes: Optional[List[int]] = None,
    ) -> Tuple[List[Detection], np.ndarray, float]:
        """图片/视频帧推理，返回 (检测列表, 标注图, 耗时ms)"""
        if self._model is None:
            raise RuntimeError("模型未加载")

        start = time.perf_counter()
        results = self._model.predict(
            source=image, conf=conf, iou=iou, imgsz=imgsz,
            device=InferenceConfig.DEVICE, classes=classes, verbose=False,
        )
        elapsed = (time.perf_counter() - start) * 1000

        detections = []
        result = results[0]
        if result.boxes is not None:
            for box in result.boxes:
                cls_name = result.names[int(box.cls[0])]
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append(Detection(
                    class_id=int(box.cls[0]),
                    class_name=cls_name,
                    name_cn=get_chinese_name(cls_name),
                    confidence=float(box.conf[0]),
                    bbox=[round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                ))

        annotated = result.plot()
        return detections, annotated, elapsed

    # ── 视频关键帧处理 ───────────────────────────────

    def detect_video_with_keyframes(
        self,
        video_path: Path,
        result_dir: Path,
        conf: float = InferenceConfig.CONF_THRESHOLD,
        iou: float = InferenceConfig.IOU_THRESHOLD,
        imgsz: int = InferenceConfig.IMG_SIZE,
        progress_cb=None,
    ) -> dict:
        """处理视频并提取关键帧（每类别在连续窗口内的最高置信度帧）。

        Args:
            video_path: 视频文件路径
            result_dir: 结果输出目录（必传）
            conf/iou/imgsz: 推理参数
            progress_cb: 进度回调 callback(pct: int)
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        task_id = uuid.uuid4().hex
        out_path = result_dir / f"{task_id}_annotated.mp4"
        # H.264 编码（浏览器兼容）
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        out = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
        if not out.isOpened():
            # fallback: MPEG-4
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

        frame_idx = 0
        total_objects = 0
        progress = 0

        # 关键帧状态
        in_window = False
        window_best: dict = {}     # class_id -> per-window best
        keyframes: list = []

        def _close_window():
            nonlocal in_window, window_best
            if not in_window:
                return
            in_window = False
            # 窗口内每类别只保留置信度最高的 1 帧
            for cid, info in window_best.items():
                kf_name = f"kf_{task_id}_{info['frame_idx']}_{cid}.jpg"
                kf_path = result_dir / kf_name
                _, annotated, _ = self.detect_image(
                    image=info["frame"], conf=conf, iou=iou, imgsz=imgsz,
                )
                cv2.imwrite(str(kf_path), annotated)
                keyframes.append({
                    "frame_idx": info["frame_idx"],
                    "class_name": info["class_name"],
                    "name_cn": info["name_cn"],
                    "confidence": round(info["max_conf"], 3),
                    "url": f"/api/detect/results/{kf_name}",
                })
            window_best = {}

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            detections, annotated, _ = self.detect_image(
                image=frame, conf=conf, iou=iou, imgsz=imgsz,
            )
            out.write(annotated)
            frame_idx += 1
            total_objects += len(detections)

            # 进度
            if total_frames > 0:
                new_pct = int(frame_idx / total_frames * 100)
                if new_pct != progress and new_pct % 5 == 0:
                    progress = new_pct
                    if progress_cb:
                        progress_cb(new_pct)

            # 关键帧：窗口内每类别取最高置信度
            if detections:
                if not in_window:
                    in_window = True
                    window_best = {}
                for d in detections:
                    cid = d.class_id
                    if cid not in window_best or d.confidence > window_best[cid]["max_conf"]:
                        window_best[cid] = {
                            "max_conf": d.confidence,
                            "frame_idx": frame_idx,
                            "frame": frame.copy(),
                            "class_name": d.class_name,
                            "name_cn": d.name_cn,
                        }
            else:
                _close_window()

        _close_window()
        cap.release()
        out.release()
        import time as _time
        _time.sleep(0.3)

        return {
            "success": True,
            "task_id": task_id,
            "total_frames": frame_idx,
            "total_detections": total_objects,
            "fps": round(fps, 1),
            "result_url": f"/api/detect/results/{out_path.name}",
            "key_frames": keyframes,
        }

    def get_class_names(self) -> List[str]:
        return self._class_names
