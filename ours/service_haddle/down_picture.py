from pathlib import Path  # 用来定位文件、创建文件夹、读写文件
from urllib.parse import urljoin, urlsplit  # 处理网络地址

import requests  # 下载图片
from PIL import Image as PILImage  # 用来打开、解码图片，读取宽高等信息

from config import DOWNLOAD_DIR, IMAGE_BASE_URL


def download_picture(picture_path: str) -> Path:
    """根据后端提供的图片路径，下载原图并返回本地路径。"""

    # 1. 得到完整的下载地址。
    if picture_path.startswith(("http://", "https://")):
        url = picture_path
    else:
        url = urljoin(IMAGE_BASE_URL, picture_path.lstrip("/"))

    # 2. 创建下载目录。
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    # 3. 从下载地址中提取原始文件名，指定本地保存路径。
    filename = Path(urlsplit(url).path).name
    local_path = DOWNLOAD_DIR / filename

    try:
        # 4. 请求图片地址，图片分块写入本地文件。
        with requests.get(url, stream=True, timeout=(5, 30)) as response:
            if response.history:
                print("重定向后的下载地址：", response.url)
            response.raise_for_status()

            with local_path.open("wb") as file:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        file.write(chunk)

        # 5. 尝试解码图片，检查文件能否正常读取。
        with PILImage.open(local_path) as image:
            image.load()
            print("图片尺寸：", image.width, image.height)

    except Exception:
        # 下载或解码失败，删除残留文件，并继续抛出异常。
        local_path.unlink(missing_ok=True)
        raise

    return local_path
