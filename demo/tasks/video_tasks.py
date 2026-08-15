"""
Celery 异步视频处理任务
========================
处理大视频文件的异步检测，支持进度查询和结果下载。
"""
import uuid
from pathlib import Path
from typing import Optional

from celery import Celery

from demo.config import ServiceConfig
from inference.config import InferenceConfig
from inference.engine import DetectorService

# ── Celery 应用 ──────────────────────────────────────
celery_app = Celery(
    "yolo12_tasks",
    broker=ServiceConfig.REDIS_URL,
    backend=ServiceConfig.REDIS_URL,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Asia/Shanghai",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
)


@celery_app.task(bind=True)
def process_video(
    self,
    video_path: str,
    conf: float = InferenceConfig.CONF_THRESHOLD,
    iou: float = InferenceConfig.IOU_THRESHOLD,
    imgsz: int = InferenceConfig.IMG_SIZE,
    result_filename: Optional[str] = None,
) -> dict:
    """
    异步处理视频检测任务。

    参数:
        video_path: 上传视频的本地路径
        conf: 置信度阈值
        iou: IOU阈值
        imgsz: 推理尺寸
        result_filename: 结果文件名（不传则自动生成）

    返回:
        dict: 包含结果文件路径、总帧数、检测数等信息
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"视频文件不存在: {video_path}")

    task_id = self.request.id or uuid.uuid4().hex

    detector = DetectorService()

    def on_progress(pct: int):
        self.update_state(
            state="PROCESSING",
            meta={"progress": pct},
        )

    result = detector.detect_video_with_keyframes(
        video_path=video_path,
        result_dir=ServiceConfig.RESULT_DIR,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        progress_cb=on_progress,
    )

    # 删除原始上传文件
    video_path.unlink(missing_ok=True)

    return {
        "task_id": task_id,
        "status": "completed",
        "result_url": result["result_url"],
        "total_frames": result["total_frames"],
        "total_detections": result["total_detections"],
        "fps": result["fps"],
    }
