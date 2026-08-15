"""
演示系统配置
============
服务运行参数、路径、Redis 连接。
"""
import os
from pathlib import Path

from inference.config import ROOT


class ServiceConfig:
    """服务运行配置"""
    HOST: str = os.getenv("API_HOST", "0.0.0.0")
    PORT: int = int(os.getenv("API_PORT", "8000"))
    RELOAD: bool = os.getenv("API_RELOAD", "true").lower() == "true"

    # 模型路径（供外部引用，引擎自身通过 _find_best_model 自动发现）
    DEFAULT_MODEL: str = os.getenv(
        "DEFAULT_MODEL",
        str(ROOT / "yolo12n.pt"),
    )

    # Redis（用于 Celery 任务队列）
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    # 文件存储
    UPLOAD_DIR: Path = ROOT / "demo" / "uploads"
    RESULT_DIR: Path = ROOT / "demo" / "results"


# 确保目录存在
ServiceConfig.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
ServiceConfig.RESULT_DIR.mkdir(parents=True, exist_ok=True)
