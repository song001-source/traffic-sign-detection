"""
推理底座配置
============
模型路径、推理参数、设备选择。
"""
from pathlib import Path

# ── 项目根目录 ──────────────────────────────────────
# inference/config.py → 上两层到达项目根目录
ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "runs"


# ── 推理默认参数 ─────────────────────────────────────
def _auto_device() -> str:
    """自动选择可用设备：GPU 优先，无 GPU 则 CPU"""
    try:
        import torch
        if torch.cuda.is_available():
            return "0"
    except ImportError:
        pass
    return "cpu"


class InferenceConfig:
    """推理默认参数"""
    CONF_THRESHOLD: float = 0.35        # 默认置信度阈值
    IOU_THRESHOLD: float = 0.45         # 默认 IOU 阈值
    IMG_SIZE: int = 640                 # 推理图像尺寸
    DEVICE: str = _auto_device()        # 自动选择 GPU/CPU
    MAX_IMAGE_SIZE_MB: int = 20         # 上传图片最大体积
    MAX_VIDEO_SIZE_MB: int = 500        # 上传视频最大体积
