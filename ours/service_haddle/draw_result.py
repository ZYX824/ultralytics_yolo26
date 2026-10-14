from pathlib import Path

from PIL import Image as PILImage, ImageDraw, ImageFont

from config import RESULT_DIR, RESULT_FONT

# 每种警告编号使用固定颜色，其余类型使用默认红色。
TYPE_COLORS = {
    "2000": (255, 56, 56),
    "2001": (255, 157, 151),
    "2002": (255, 112, 31),
    "2003": (72, 249, 10),
    "2004": (0, 194, 255),
}


def draw_reports(image_path: Path, reports: list[dict]) -> Path:
    """在原图上绘制预测框、种类和置信度，保存到 RESULT_DIR 并返回保存路径。"""

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    save_path = RESULT_DIR / f"info_{image_path.name}"

    with PILImage.open(image_path) as image:
        image = image.convert("RGB")

    # 线宽和字号随图片尺寸缩放，保证大图上也能看清。
    line_width = max(2, round(max(image.size) / 800))
    font = ImageFont.truetype(RESULT_FONT, max(16, round(max(image.size) / 100)))
    draw = ImageDraw.Draw(image)

    for report in reports:
        result = report["result"]
        color = TYPE_COLORS.get(result["type"], (255, 0, 0))
        x1, y1, x2, y2 = result["bbox"]
        draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width)

        # 标签放在框的上方；框贴近顶部时放到框内。
        label = f"{result['typeName']} {result['confidence']:.2f}"
        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        text_w, text_h = right - left, bottom - top + line_width * 2
        text_y = y1 - text_h if y1 - text_h >= 0 else y1
        draw.rectangle((x1, text_y, x1 + text_w + line_width * 2, text_y + text_h), fill=color)
        draw.text((x1 + line_width, text_y + line_width - top), label, fill=(255, 255, 255), font=font)

    image.save(save_path)
    print("结果图已保存：", save_path)
    return save_path
