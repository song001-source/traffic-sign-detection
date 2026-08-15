"""
推理数据模型
============
检测结果的基础数据结构（Pydantic）。
"""
from typing import List

from pydantic import BaseModel, Field


class Detection(BaseModel):
    """单个检测结果"""
    class_id: int = Field(..., description="类别ID")
    class_name: str = Field(..., description="类别原始编码")
    name_cn: str = Field(default="", description="类别中文名称")
    confidence: float = Field(..., description="置信度")
    bbox: List[float] = Field(..., description="检测框 [x1, y1, x2, y2]")
