"""
API 数据模型
============
定义 API 请求和响应的 Pydantic 数据结构。
"""
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field

# 从推理底座 re-export 基础检测类型
from inference.schemas import Detection  # noqa: F401


# ── 检测响应模型 ─────────────────────────────────────

class ImageDetectResponse(BaseModel):
    """图片检测响应"""
    success: bool = True
    filename: str = Field(..., description="原文件名")
    image_width: int
    image_height: int
    detections: List[Detection] = Field(default_factory=list, description="检测结果列表")
    total_detections: int = 0
    annotated_image_url: Optional[str] = Field(default=None, description="标注图URL")
    inference_time_ms: float = Field(default=0.0, description="推理耗时(ms)")


class ErrorResponse(BaseModel):
    """错误响应"""
    success: bool = False
    error: str = Field(..., description="错误信息")
    detail: Optional[str] = Field(default=None, description="详细错误信息")


# ── 模型管理 ─────────────────────────────────────────

class ModelInfo(BaseModel):
    """模型信息"""
    name: str = Field(..., description="模型文件名")
    path: str = Field(..., description="模型绝对路径")
    size_mb: float = Field(..., description="文件大小(MB)")
    is_loaded: bool = Field(default=False, description="是否当前加载的模型")


class ModelListResponse(BaseModel):
    """模型列表响应"""
    models: List[ModelInfo]
    current_model: str = Field(..., description="当前使用的模型路径")


# ── 异步任务 ─────────────────────────────────────────

class TaskStatus(BaseModel):
    """任务状态"""
    task_id: str = Field(..., description="任务ID")
    status: str = Field(..., description="状态: pending/processing/completed/failed")
    progress: float = Field(default=0.0, description="进度 0-100")
    result_url: Optional[str] = Field(default=None, description="结果文件URL")
    error: Optional[str] = Field(default=None, description="错误信息")
    created_at: datetime = Field(default_factory=datetime.utcnow)
