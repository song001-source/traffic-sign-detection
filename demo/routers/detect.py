"""
检测路由
========
图片/视频/文件上传检测接口。
路由层只做请求解析 → 调用推理 → 返回响应，不含推理逻辑。
"""
import threading
import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, File, UploadFile, HTTPException, Depends, Query
from fastapi.responses import FileResponse

from demo.config import ServiceConfig
from inference.config import InferenceConfig
from demo.schemas import ImageDetectResponse, TaskStatus
from inference.schemas import Detection
from inference.engine import DetectorService
from demo.tasks.video_tasks import process_video

router = APIRouter(prefix="/api/detect", tags=["检测"])

# 允许的图片格式
ALLOWED_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# 允许的视频格式
ALLOWED_VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}


def get_detector() -> DetectorService:
    """依赖注入：获取全局检测器实例"""
    return DetectorService()


def _validate_image(file: UploadFile) -> str:
    """验证上传文件是否为合法图片格式"""
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_IMAGE_EXT:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的图片格式: {ext}，支持的格式: {', '.join(ALLOWED_IMAGE_EXT)}",
        )
    return ext


# ── POST /api/detect/image ──────────────────────────

@router.post("/image", response_model=ImageDetectResponse)
async def detect_image(
    file: UploadFile = File(..., description="待检测的图片文件"),
    conf: float = Query(default=InferenceConfig.CONF_THRESHOLD, ge=0.01, le=1.0),
    iou: float = Query(default=InferenceConfig.IOU_THRESHOLD, ge=0.01, le=1.0),
    imgsz: int = Query(default=InferenceConfig.IMG_SIZE, ge=320, le=1280),
    detector: DetectorService = Depends(get_detector),
):
    """
    上传单张图片进行交通标志检测。

    返回检测框列表、标注图URL，以及推理耗时。
    """
    # 1. 验证文件
    ext = _validate_image(file)
    contents = await file.read()

    if len(contents) > InferenceConfig.MAX_IMAGE_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"图片大小超过限制 ({InferenceConfig.MAX_IMAGE_SIZE_MB}MB)",
        )

    # 2. 解码图片
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="图片解码失败，请检查文件是否为有效图片")

    # 3. 执行推理
    detections, annotated, elapsed_ms = detector.detect_image(
        image=image,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
    )

    # 4. 保存标注图
    result_filename = f"{uuid.uuid4().hex}{ext}"
    result_path = ServiceConfig.RESULT_DIR / result_filename
    cv2.imwrite(str(result_path), annotated)

    # 5. 返回结果
    h, w = image.shape[:2]
    return ImageDetectResponse(
        filename=file.filename or "unknown",
        image_width=w,
        image_height=h,
        detections=detections,
        total_detections=len(detections),
        annotated_image_url=f"/api/detect/results/{result_filename}",
        inference_time_ms=round(elapsed_ms, 1),
    )


# ── POST /api/detect/video ──────────────────────────

@router.post("/video")
async def detect_video(
    file: UploadFile = File(..., description="待检测的视频文件"),
    conf: float = Query(default=InferenceConfig.CONF_THRESHOLD, ge=0.01, le=1.0),
    iou: float = Query(default=InferenceConfig.IOU_THRESHOLD, ge=0.01, le=1.0),
    imgsz: int = Query(default=InferenceConfig.IMG_SIZE, ge=320, le=1280),
    detector: DetectorService = Depends(get_detector),
):
    """
    ⚠️ 同步视频处理（小视频用，大视频请走 /video/process）

    上传视频，逐帧推理，返回标注后的视频文件和关键帧。
    """
    # 验证文件
    ext = Path(file.filename or "video.mp4").suffix.lower()
    if ext not in ALLOWED_VIDEO_EXT:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的视频格式: {ext}",
        )

    # 保存上传的视频到临时文件
    task_id = uuid.uuid4().hex
    upload_path = ServiceConfig.UPLOAD_DIR / f"{task_id}{ext}"
    with open(upload_path, "wb") as f:
        f.write(await file.read())

    # 调用推理引擎（同步处理）
    result = detector.detect_video_with_keyframes(
        video_path=upload_path,
        result_dir=ServiceConfig.RESULT_DIR,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
    )

    # 删除上传的临时文件
    upload_path.unlink(missing_ok=True)

    return {
        "success": True,
        "task_id": task_id,
        "filename": file.filename or "unknown",
        "total_frames": result["total_frames"],
        "total_detections": result["total_detections"],
        "fps": result["fps"],
        "result_url": result["result_url"],
        "key_frames": result.get("key_frames", []),
    }


# ── POST /api/detect/video/async ────────────────────

@router.post("/video/async")
async def detect_video_async(
    file: UploadFile = File(..., description="待检测的视频文件"),
    conf: float = Query(default=InferenceConfig.CONF_THRESHOLD, ge=0.01, le=1.0),
    iou: float = Query(default=InferenceConfig.IOU_THRESHOLD, ge=0.01, le=1.0),
    imgsz: int = Query(default=InferenceConfig.IMG_SIZE, ge=320, le=1280),
):
    """
    异步视频检测（Celery 版，推荐用于长视频/大文件）。

    上传视频后立即返回 task_id，客户端轮询 /api/detect/task/{task_id} 获取进度和结果。
    """
    ext = Path(file.filename or "video.mp4").suffix.lower()
    if ext not in ALLOWED_VIDEO_EXT:
        raise HTTPException(status_code=400, detail=f"不支持的视频格式: {ext}")

    # 保存上传文件
    task_id = uuid.uuid4().hex
    upload_path = ServiceConfig.UPLOAD_DIR / f"{task_id}{ext}"
    with open(upload_path, "wb") as f:
        f.write(await file.read())

    # 提交 Celery 任务
    task = process_video.delay(
        video_path=str(upload_path),
        conf=conf,
        iou=iou,
        imgsz=imgsz,
    )

    return {
        "success": True,
        "task_id": task.id,
        "status": "pending",
        "message": "视频已提交处理，请轮询 /api/detect/task/{task_id} 获取进度",
    }


# ── GET /api/detect/task/{task_id} ──────────────────

@router.get("/task/{task_id}", response_model=TaskStatus)
async def get_task_status(task_id: str):
    """查询 Celery 异步任务状态"""
    from celery.result import AsyncResult
    from demo.tasks.video_tasks import celery_app

    result = AsyncResult(task_id, app=celery_app)

    response = TaskStatus(
        task_id=task_id,
        status=result.state,
        progress=0.0,
    )

    if result.state == "PENDING":
        response.status = "pending"
    elif result.state == "PROCESSING":
        response.status = "processing"
        meta = result.info or {}
        response.progress = meta.get("progress", 0)
    elif result.state == "SUCCESS":
        response.status = "completed"
        response.progress = 100.0
        data = result.result or {}
        response.result_url = data.get("result_url")
    elif result.state == "FAILURE":
        response.status = "failed"
        response.error = str(result.info) if result.info else "未知错误"

    return response


# ── 视频处理进度存储（内存） ─────────────────────────

_video_progress: dict = {}  # task_id -> {"progress": 0-100, "status": "...", "result": {...}}


def _process_video_bg(task_id: str, video_path: Path, detector, conf, iou, imgsz):
    """后台线程处理视频，实时更新进度 + 提取关键帧"""
    def on_progress(pct):
        _video_progress[task_id] = {"progress": pct, "status": "processing", "result": None}

    try:
        result = detector.detect_video_with_keyframes(
            video_path=video_path,
            result_dir=ServiceConfig.RESULT_DIR,
            conf=conf, iou=iou, imgsz=imgsz,
            progress_cb=on_progress,
        )
        video_path.unlink(missing_ok=True)
        _video_progress[task_id] = {
            "progress": 100,
            "status": "completed",
            "result": {
                "total_frames": result["total_frames"],
                "total_detections": result["total_detections"],
                "fps": result["fps"],
                "result_url": result["result_url"],
                "key_frames": result.get("key_frames", []),
            },
        }
    except Exception as e:
        _video_progress[task_id] = {
            "progress": 0, "status": "failed", "result": None, "error": str(e)
        }


# ── POST /api/detect/video/process ──────────────────

@router.post("/video/process")
async def detect_video_process(
    file: UploadFile = File(..., description="待检测的视频文件"),
    conf: float = Query(default=InferenceConfig.CONF_THRESHOLD, ge=0.01, le=1.0),
    iou: float = Query(default=InferenceConfig.IOU_THRESHOLD, ge=0.01, le=1.0),
    imgsz: int = Query(default=InferenceConfig.IMG_SIZE, ge=320, le=1280),
    detector: DetectorService = Depends(get_detector),
):
    """
    视频检测（实时进度版，前端使用此接口）。

    1. 上传视频 → 立即返回 task_id
    2. 轮询 GET /api/detect/video/progress/{task_id} 获取进度
    3. 完成后获取结果视频和关键帧
    """
    ext = Path(file.filename or "video.mp4").suffix.lower()
    if ext not in ALLOWED_VIDEO_EXT:
        raise HTTPException(status_code=400, detail=f"不支持的视频格式: {ext}")

    task_id = uuid.uuid4().hex
    upload_path = ServiceConfig.UPLOAD_DIR / f"{task_id}{ext}"
    with open(upload_path, "wb") as f:
        f.write(await file.read())

    # 初始化进度
    _video_progress[task_id] = {"progress": 0, "status": "pending", "result": None}

    # 后台线程处理
    t = threading.Thread(
        target=_process_video_bg,
        args=(task_id, upload_path, detector, conf, iou, imgsz),
        daemon=True,
    )
    t.start()

    return {"success": True, "task_id": task_id, "message": "视频已提交，请轮询进度接口"}


# ── GET /api/detect/video/progress/{task_id} ────────

@router.get("/video/progress/{task_id}")
async def get_video_progress(task_id: str):
    """获取视频处理进度"""
    info = _video_progress.get(task_id)
    if info is None:
        raise HTTPException(status_code=404, detail="任务不存在或已过期")
    return info


# ── GET /api/results/{filename} ─────────────────────

@router.get("/results/{filename}", tags=["文件"])
async def get_result_file(filename: str):
    """获取检测结果文件（图片或视频）"""
    file_path = ServiceConfig.RESULT_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="文件不存在或已过期")
    return FileResponse(str(file_path))
