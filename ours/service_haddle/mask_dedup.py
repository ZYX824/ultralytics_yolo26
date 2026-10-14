import math
import sys
from threading import Lock

import cv2
import numpy as np
import torch

from config import (
    CONTAIN_RATIO,
    CROSS_CLASS_IOU,
    MERGE_GAP_RATIO,
    MIN_MASK_PIXELS,
    SAM_CHECKPOINT,
    SAM_CONFIG,
    SAM_ROOT,
)

# 吸收规则（后端警告编号）：线状裂缝落在龟裂区域内时，只报龟裂。
# 容器类型 → 可被吸收的类型。
ABSORB_RULES = {
    2003: {2000, 2001, 2002},  # 龟裂 ← 纵裂、横裂、斜裂
}

# ========== SAM2 分割模型（模块加载时创建一次） ==========
sys.path.insert(0, str(SAM_ROOT))
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

_sam_device = "cuda:0" if torch.cuda.is_available() else "cpu"
sam_predictor = SAM2ImagePredictor(
    build_sam2(SAM_CONFIG, str(SAM_CHECKPOINT), device=_sam_device)
)
sam_lock = Lock()


def segment_and_merge(rgb: np.ndarray, detections: list[dict]) -> list[dict]:
    """
    用 SAM 分割每个预测框，按种类分别合并相连区域，再做跨种类的吸收和去重。

    返回的每个区域带有 "_mask"：与 bbox 范围对应的局部布尔掩码，供测量使用。
    """

    # 每个种类一张归属图：-1 为背景；其余为覆盖该像素、同类中置信度最高的预测框下标。
    # 不同种类互不覆盖，大片龟裂的掩码不会被小裂缝抢走像素。
    owner_maps: dict[int, np.ndarray] = {}

    # 按置信度从低到高写入，同类重叠像素最终归属置信度最高的预测框。
    order = sorted(range(len(detections)), key=lambda i: detections[i]["confidence"])

    with sam_lock:
        sam_predictor.set_image(rgb)
        try:
            for index in order:
                class_id = detections[index]["class_id"]
                try:
                    masks, _, _ = sam_predictor.predict(
                        box=np.asarray(detections[index]["bbox"], dtype=np.float32),
                        multimask_output=False,
                    )
                    if class_id not in owner_maps:
                        owner_maps[class_id] = np.full(rgb.shape[:2], -1, dtype=np.int16)
                    owner_maps[class_id][masks[0].astype(bool)] = index
                except Exception:
                    print("警告：掩码分割失败 class_id=", class_id)
        finally:
            sam_predictor.reset_predictor()

    # 1. 每个种类单独提取连通区域，不同种类即使相连也各自成框。
    regions = []
    for owner_map in owner_maps.values():
        regions.extend(merge_connected_masks(detections, owner_map))
    del owner_maps

    # 2. 过滤 SAM 产生的孤立像素点和小区域，这些区域不画框、不上报。
    kept = [det for det in regions if det["mask_pixels"] > MIN_MASK_PIXELS]

    # 3. 同类且相交或距离近的区域再合并。
    merged = merge_nearby_same_class(kept, MERGE_GAP_RATIO)

    # 4. 线状裂缝落在龟裂区域内时，只保留龟裂。
    absorbed = absorb_contained(merged, CONTAIN_RATIO)

    # 5. 不同种类掩码几乎重合，视为同一缺陷被识别成两类，只保留置信度高的。
    deduped = dedup_cross_class(absorbed, CROSS_CLASS_IOU)

    print(
        f"YOLO 预测框 {len(detections)} → 按种类连通区域 {len(regions)}"
        f" → 过滤小区域(≤{MIN_MASK_PIXELS}像素) {len(kept)}"
        f" → 同类近邻合并(间隙/长边≤{MERGE_GAP_RATIO}) {len(merged)}"
        f" → 龟裂吸收线状裂缝(包含≥{CONTAIN_RATIO}) {len(absorbed)}"
        f" → 跨种类去重(IoU≥{CROSS_CLASS_IOU}) {len(deduped)}"
    )
    return deduped


def gap_ratio(box_a: list[float], box_b: list[float]) -> float:
    """两个矩形之间的缝隙 / 两者整体外接矩形的长边；相交或相接时为 0。"""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    # 水平、竖直方向的缝隙，重叠时为 0；缝隙取两矩形间的最短距离。
    dx = max(0.0, max(ax1, bx1) - min(ax2, bx2))
    dy = max(0.0, max(ay1, by1) - min(ay2, by2))
    gap = math.hypot(dx, dy)

    long_side = max(max(ax2, bx2) - min(ax1, bx1), max(ay2, by2) - min(ay1, by1))
    return gap / long_side if long_side > 0 else 0.0


def combine_detections(a: dict, b: dict) -> dict:
    """合并两个同类区域：类别、置信度沿用置信度高的一方，框和掩码取并集。"""
    if b["confidence"] > a["confidence"]:
        a, b = b, a

    merged = a.copy()
    merged["bbox"] = [
        min(a["bbox"][0], b["bbox"][0]),
        min(a["bbox"][1], b["bbox"][1]),
        max(a["bbox"][2], b["bbox"][2]),
        max(a["bbox"][3], b["bbox"][3]),
    ]
    # 两块局部掩码贴到合并后的外接框上取并集。
    x1, y1, x2, y2 = (int(v) for v in merged["bbox"])
    mask = np.zeros((y2 - y1, x2 - x1), dtype=bool)
    for part in (a, b):
        px1, py1, px2, py2 = (int(v) for v in part["bbox"])
        mask[py1 - y1:py2 - y1, px1 - x1:px2 - x1] |= part["_mask"]

    merged["_mask"] = mask
    merged["mask_pixels"] = int(mask.sum())
    merged["contours"] = mask_contours(mask, x1, y1)
    return merged


def mask_overlap(a: dict, b: dict, b_mask: np.ndarray | None = None) -> int:
    """两个区域掩码的交集像素数；b_mask 可替换 b 的局部掩码（尺寸须与 b 的框一致）。"""
    b_mask = b["_mask"] if b_mask is None else b_mask
    ax1, ay1, ax2, ay2 = (int(v) for v in a["bbox"])
    bx1, by1, bx2, by2 = (int(v) for v in b["bbox"])
    x1, y1, x2, y2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    if x1 >= x2 or y1 >= y2:
        return 0
    part_a = a["_mask"][y1 - ay1:y2 - ay1, x1 - ax1:x2 - ax1]
    part_b = b_mask[y1 - by1:y2 - by1, x1 - bx1:x2 - bx1]
    return int(np.count_nonzero(part_a & part_b))


def filled_mask(det: dict) -> np.ndarray:
    """填充外轮廓内部的孔洞，得到区域覆盖的整片范围（龟裂掩码常呈网状）。"""
    mask = det["_mask"].astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(mask, contours, -1, 1, thickness=cv2.FILLED)
    return mask.astype(bool)


def absorb_contained(detections: list[dict], min_ratio: float) -> list[dict]:
    """线状裂缝掩码落在龟裂区域内的比例 ≥ min_ratio 时，丢弃该线状裂缝，只报龟裂。"""
    containers = [d for d in detections if d["type"] in ABSORB_RULES]
    filled = [filled_mask(d) for d in containers]

    kept = []
    for det in detections:
        absorbed = any(
            det["type"] in ABSORB_RULES[container["type"]]
            and mask_overlap(det, container, container_mask) >= min_ratio * det["mask_pixels"]
            for container, container_mask in zip(containers, filled)
        )
        if not absorbed:
            kept.append(det)
    return kept


def dedup_cross_class(detections: list[dict], min_iou: float) -> list[dict]:
    """不同种类的掩码 IoU ≥ min_iou 时视为同一缺陷，只保留置信度高的。"""
    kept = []
    for det in sorted(detections, key=lambda d: d["confidence"], reverse=True):
        duplicate = False
        for other in kept:
            if other["class_id"] == det["class_id"]:
                continue
            inter = mask_overlap(det, other)
            union = det["mask_pixels"] + other["mask_pixels"] - inter
            if union and inter / union >= min_iou:
                duplicate = True
                break
        if not duplicate:
            kept.append(det)
    return kept


def mask_contours(mask: np.ndarray, x: int, y: int) -> list[dict]:
    """提取局部掩码的轮廓，offset 将局部坐标转换为原图坐标。"""
    contours, hierarchy = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE,
        offset=(x, y),
    )
    return [
        {
            "points_xy": contour.reshape(-1, 2).tolist(),
            "parent": int(hierarchy[0, i, 3]),
        }
        for i, contour in enumerate(contours)
    ] if hierarchy is not None else []


def merge_nearby_same_class(detections: list[dict], max_ratio: float) -> list[dict]:
    """同类区域相交，或缝隙 / 外接矩形长边 ≤ max_ratio 时合并，直到没有可合并的区域。"""
    detections = list(detections)

    changed = True
    while changed:
        changed = False
        for i in range(len(detections)):
            for j in range(i + 1, len(detections)):
                a, b = detections[i], detections[j]
                if a["class_id"] != b["class_id"]:
                    continue
                if gap_ratio(a["bbox"], b["bbox"]) > max_ratio:
                    continue

                # 合并后外接框变大，可能与其他区域变近，因此重新扫描。
                detections[i] = combine_detections(a, b)
                del detections[j]
                changed = True
                break
            if changed:
                break

    return detections


def merge_connected_masks(
    detections: list[dict],
    owner_map: np.ndarray,
) -> list[dict]:
    """
    根据同一种类 SAM 掩码的连通区域，合并检测结果。

    参数：
        detections:
            原始 YOLO 检测结果列表。

        owner_map:
            某一种类的归属图，与原图尺寸相同的二维整数矩阵。
            -1：背景。
            >= 0：覆盖该像素且 YOLO 置信度最高的同类检测结果下标。

    返回：
        合并后的检测结果，每个连通区域对应一个检测框，
        "_mask" 为该区域在外接矩形内的局部掩码。
    """
    # 1. 得到所有 SAM 掩码的并集。
    union_mask = (owner_map >= 0).astype(np.uint8)

    # 2. 提取连通区域。
    # 8 邻域：水平、垂直或对角接触都算连通。
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        union_mask,
        connectivity=8,
    )

    merged_detections = []

    # 编号 0 是背景，从 1 开始处理前景区域。
    for label_id in range(1, num_labels):
        # 3. 获取当前连通区域的水平外接矩形。
        x = int(stats[label_id, cv2.CC_STAT_LEFT])
        y = int(stats[label_id, cv2.CC_STAT_TOP])
        w = int(stats[label_id, cv2.CC_STAT_WIDTH])
        h = int(stats[label_id, cv2.CC_STAT_HEIGHT])

        # 当前区域在外接矩形内的局部掩码。
        component_mask = (
            labels[y:y + h, x:x + w] == label_id
        )

        # 4. 找出当前区域中各像素所属的原始预测框。
        source_ids = np.unique(
            owner_map[y:y + h, x:x + w][component_mask]
        )

        # 5. 选择 YOLO 置信度最高的预测框。
        best_id = max(
            source_ids.tolist(),
            key=lambda index: detections[index]["confidence"],
        )

        # 类别、告警编号、置信度等沿用最高置信度预测框。
        merged = detections[best_id].copy()

        # 不沿用原框可能已有的旧测量结果。
        merged.pop("measurement", None)

        # 外接矩形改为整个连通区域的范围。
        # 格式保持为 [x1, y1, x2, y2]。
        merged["bbox"] = [
            float(x),
            float(y),
            float(x + w),
            float(y + h),
        ]

        merged["mask_pixels"] = int(
            stats[label_id, cv2.CC_STAT_AREA]
        )

        # 局部掩码，供后续合并、包含判断、IoU 和测量使用。
        merged["_mask"] = component_mask

        # 6. 提取合并后的轮廓。
        merged["contours"] = mask_contours(component_mask, x, y)

        merged_detections.append(merged)

    return merged_detections