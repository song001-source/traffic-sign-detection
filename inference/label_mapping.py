"""
COCO 交通标志类别中文名称映射
============================
将 COCO 数据集编码映射为交通标志中文正式名称。
"""
import json
from pathlib import Path

# 加载映射文件
_DATA_DIR = Path(__file__).resolve().parent
_mapping_path = _DATA_DIR / "coco_names.json"

COCO_CN_NAMES: dict[str, str] = {}
if _mapping_path.exists():
    with open(_mapping_path, "r", encoding="utf-8") as f:
        COCO_CN_NAMES.update(json.load(f))


def get_chinese_name(class_name: str) -> str:
    """
    获取类别的中文名称。
    如果不在映射表中，返回原始名称。

    示例:
        >>> get_chinese_name("pne")
        '禁止驶入'
        >>> get_chinese_name("unknown_class")
        'unknown_class'
    """
    return COCO_CN_NAMES.get(class_name, class_name)
