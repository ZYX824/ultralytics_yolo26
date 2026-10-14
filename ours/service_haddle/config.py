"""服务配置：读取同目录下的 .env，系统环境变量优先。"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

SERVICE_DIR = Path(__file__).resolve().parent
load_dotenv(SERVICE_DIR / ".env", encoding="utf-8")

# 把项目根目录加入导入路径，供 ours.extraction 等模块使用。
PROJECT_ROOT = SERVICE_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _require(key: str) -> str:
    value = os.getenv(key, "").strip()
    if not value:
        raise RuntimeError(f"缺少配置项 {key}，请检查 {SERVICE_DIR / '.env'}")
    return value


def _path(key: str) -> Path:
    """相对路径按服务目录解析。"""
    path = Path(_require(key))
    return path if path.is_absolute() else (SERVICE_DIR / path).resolve()


def _parse_classes(text: str) -> dict[int, tuple[int, str]]:
    """解析 "0:2003:道路龟裂,1:2000:道路纵裂" → {0: (2003, "道路龟裂"), ...}。"""
    classes = {}
    for item in text.split(","):
        try:
            class_id, alarm_type, name = (part.strip() for part in item.split(":"))
            classes[int(class_id)] = (int(alarm_type), name)
        except ValueError:
            raise RuntimeError(f"DETECT_CLASSES 格式错误：{item!r}，应为 类别编号:警告编号:中文名称") from None
    return classes


# 图片下载与结果上报
IMAGE_BASE_URL = _require("IMAGE_BASE_URL")
CALLBACK_URL = _require("CALLBACK_URL")
# 是否向后端上报结果；false 时只在控制台打印报文。
CALLBACK_ENABLED = os.getenv("CALLBACK_ENABLED", "true").strip().lower() in ("1", "true", "yes", "on")
DOWNLOAD_DIR = _path("DOWNLOAD_DIR")

# 结果图（预测框 + 种类 + 置信度）
RESULT_DIR = _path("RESULT_DIR")
RESULT_FONT = _require("RESULT_FONT")  # 需支持中文的字体

# YOLO
YOLO_WEIGHTS = _path("YOLO_WEIGHTS")
YOLO_CONF = float(_require("YOLO_CONF"))
YOLO_IMGSZ = int(_require("YOLO_IMGSZ"))

# SAM2
SAM_ROOT = _path("SAM_ROOT")
SAM_CONFIG = _require("SAM_CONFIG")
SAM_CHECKPOINT = _path("SAM_CHECKPOINT")

# 掩码后处理
MIN_MASK_PIXELS = int(_require("MIN_MASK_PIXELS"))  # 像素数 ≤ 该值的区域丢弃
MERGE_GAP_RATIO = float(_require("MERGE_GAP_RATIO"))  # 同类区域 缝隙/外接矩形长边 ≤ 该值时合并
CONTAIN_RATIO = float(_require("CONTAIN_RATIO"))  # 线状裂缝落在龟裂内的像素比例 ≥ 该值时只报龟裂
CROSS_CLASS_IOU = float(_require("CROSS_CLASS_IOU"))  # 不同种类掩码 IoU ≥ 该值时只保留置信度高的

# 检测种类
_CLASSES = _parse_classes(_require("DETECT_CLASSES"))
DETECT_CLASSES = list(_CLASSES)  # 参与检测的模型类别编号
CLASS_TO_TYPE = {class_id: alarm_type for class_id, (alarm_type, _) in _CLASSES.items()}  # 模型类别 → 后端警告编号
CLASS_TO_NAMES = {class_id: name for class_id, (_, name) in _CLASSES.items()}  # 模型类别 → 中文名称
