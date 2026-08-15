"""
实时流推理路由（WebSocket）
============================
浏览器通过 WebSocket 传输摄像头帧，服务端实时返回检测结果。
"""
import asyncio
import json
import time

import cv2
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from inference.config import InferenceConfig
from inference.schemas import Detection
from inference.engine import DetectorService

router = APIRouter()


@router.websocket("/api/detect/stream")
async def websocket_detect_stream(websocket: WebSocket):
    """
    WebSocket 实时检测接口。

    客户端发送:
        - JSON 配置: {"conf": 0.35, "iou": 0.45} （可选，仅在开始时发送一次）
        - 二进制帧: JPEG 编码的图片数据

    服务端返回:
        - JSON: {"detections": [...], "fps": 30.0, "inference_ms": 12.3}
    """
    await websocket.accept()

    detector = DetectorService()
    conf = InferenceConfig.CONF_THRESHOLD
    iou = InferenceConfig.IOU_THRESHOLD
    imgsz = InferenceConfig.IMG_SIZE

    frame_count = 0
    fps_start = time.perf_counter()
    fps = 0.0

    try:
        while True:
            message = await websocket.receive()

            if message.get("type") == "websocket.receive":
                # ── 文本消息 → 更新配置 ──
                if "text" in message:
                    try:
                        config = json.loads(message["text"])
                        conf = config.get("conf", conf)
                        iou = config.get("iou", iou)
                        imgsz = config.get("imgsz", imgsz)
                        await websocket.send_json({
                            "type": "config_ack",
                            "conf": conf,
                            "iou": iou,
                            "imgsz": imgsz,
                        })
                    except json.JSONDecodeError:
                        await websocket.send_json({"type": "error", "message": "无效的JSON配置"})
                    continue

                # ── 二进制消息 → 推理 ──
                if "bytes" in message:
                    img_bytes = message["bytes"]
                    nparr = np.frombuffer(img_bytes, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if frame is None:
                        await websocket.send_json({
                            "type": "error",
                            "message": "无效的图片数据",
                        })
                        continue

                    detections, annotated, elapsed_ms = detector.detect_image(
                        image=frame,
                        conf=conf,
                        iou=iou,
                        imgsz=imgsz,
                    )

                    # 计算 FPS（每 10 帧更新一次）
                    frame_count += 1
                    if frame_count % 10 == 0:
                        now = time.perf_counter()
                        fps = 10.0 / (now - fps_start)
                        fps_start = now

                    await websocket.send_json({
                        "type": "detection",
                        "detections": [d.model_dump() for d in detections],
                        "total": len(detections),
                        "fps": round(fps, 1),
                        "inference_ms": round(elapsed_ms, 1),
                    })

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
