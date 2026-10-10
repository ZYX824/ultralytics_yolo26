# 后端调用接口

完整流程：原始图像与几何信息 → 双 YOLO 推理 → 框合并 → SAM → 地面投影 → 尺寸结果。

## 启动

```powershell
# 本地 4060，复现样图的五个目标，监听所有网卡
& 'D:\Program_Files\Anaconda\envs\YOLOv26\python.exe' -m ours.service.serve --host 0.0.0.0 --port 8000 --device cuda:0 --imgsz 1280 --conf 0.1
```

服务器使用同样命令，将解释器换为服务器 CUDA 环境的 `python`。`serve` 默认 `imgsz=1280、conf=0.25`，本次验证使用 `conf=0.1`，检测阈值影响误检/漏检，按业务评估后调整。一个进程常驻两个 YOLO 和 SAM，限制同时处理一张图。仅运行一个 worker。

地址用实际主机 IP 替换 `127.0.0.1`：

| 方法和路径 | 功能 |
| --- | --- |
| `POST /predict` | 上传图像和信息，同步返回检测、掩码与尺寸 |
| `GET /results/{image_id}` | 获取已保存的完整结果，无需重新推理 |
| `GET /files/{image_id}/{filename}` | 下载原图、合并框、掩码、报告 |
| `GET /health` | 查询运行状态、设备及模型加载情况 |
| `GET /docs` | Swagger 调试页面 |
| `GET /openapi.json` | OpenAPI 接口定义 |

服务是同步接口，后端建议为推理请求配置 300 秒超时。

## 上传格式

`POST /predict` 使用 `multipart/form-data`：

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `image` | 文件 | 必填 | 原始图像，保留 EXIF/XMP；最大 32 MB、2500 万像素 |
| `info` | JSON 字符串 | `{}` | 此图的几何信息，可直接使用提取报告的 `images[0]`，最大 64 KB |
| `with_sam` | 布尔值 | `true` | `false` 仅运行双 YOLO，不输出尺寸 |

图像内含必要元数据时，只需上传 `image`。传入的非空几何字段优先于图像元数据；后端数据须与当前图像匹配。图像宽高、高度和 GSD 等派生字段不接受覆盖，高度始终由当前测距和俯仰角重新计算。

无元数据图像可传入：

```json
{
  "center_range_m": 177.947,
  "range_status": "Normal",
  "gimbal_pitch_deg": -45,
  "gimbal_roll_deg": 0,
  "gimbal_yaw_deg": 76.7,
  "focal_length_35mm_mm": 24,
  "uav_latitude_deg": 30.540428408,
  "uav_longitude_deg": 104.508034722,
  "center_target_latitude_deg": 30.540686,
  "center_target_longitude_deg": 104.5093054
}
```

经纬度用于地理定位和保存原点；局部尺寸计算依赖测距、姿态和内参。图像未包含等效焦距时，可提供经过标定、对应**当前原图像素网格**的内参，替代等效焦距近似：

```json
{
  "calibrated_intrinsics": {
    "fx_px": 2795.69,
    "fy_px": 2795.69,
    "cx_px": 2015.5,
    "cy_px": 1511.5
  }
}
```

这组数字只是样图近似内参的格式示例，不是实测标定。镜头畸变应在上游处理，或使用畸变可忽略的图像；传入的内参、原图尺寸和测距目标位置必须匹配。已有数字变焦时，需提供对应视场的标定内参。图片 EXIF 方向不是 1 时，不计算尺寸，先在上游统一图像和相机参数坐标网格。

后端 Python 示例：

```python
import requests

base_url = 'http://127.0.0.1:8000'
with open('image.jpeg', 'rb') as image:
    response = requests.post(
        base_url + '/predict',
        files={'image': ('image.jpeg', image, 'image/jpeg')},
        data={'info': '{}', 'with_sam': 'true'},
        timeout=300,
    )
response.raise_for_status()
result = response.json()
for target in result['detections']:
    size = target['measurement']
    if size['status'] == 'estimated_ground_plane':
        print(target['model_id'], target['class_name'],
              size['area_m2'], size['extent_long_m'], size['extent_short_m'])
    else:
        print(size['status'], size['reason'])
print(base_url + result['artifact_urls']['sam_overlay'])
```

## 返回值

`HTTP 200` 表示图像已处理，不代表每个目标都具备测量条件。关键字段示例（只展示尺寸部分）：

```json
{
  "schema_version": 2,
  "image_id": "32位十六进制标识",
  "measurement_status": "estimated_ground_plane",
  "detections": [
    {
      "instance_id": 1,
      "model_id": "visdrone2019",
      "class_name": "car",
      "confidence": 0.30,
      "measurement": {
        "status": "estimated_ground_plane",
        "reason": null,
        "area_m2": 3.8724,
        "extent_long_m": 3.5795,
        "extent_short_m": 1.5824,
        "touches_image_border": false,
        "extent_corners_en_m": [],
        "ground_contours": []
      }
    }
  ]
}
```

完整响应还包括图像宽高、几何信息、原图框坐标、实例像素轮廓、SAM 得分、掩码文件地址、计时和 GPU 峰值占用。`measurement_method` 记录方法、地理原点、相机参数来源和假设。

- `area_m2`：投影到目标水平面后的掩码面积，包含完整像素单元，孔洞不计入面积。
- `extent_long_m` / `extent_short_m`：投影掩码的最小面积旋转外接矩形长短边，表示覆盖范围，不作为裂缝真实长度或宽度。
- `extent_corners_en_m`：外接矩形四角，坐标为相对中心测距目标的东、北方向米数。
- `ground_contours`：`points_en_m` 为像素中心轮廓的地面坐标，`parent` 为孔洞/区域层级；面积以完整像素单元计算，不以这些像素中心多边形重新计算。
- `union_measurement`：所有实例掩码的并集测量，重叠部分只计一次；与逐实例面积之和可能不同。
- `artifact_urls` / `mask_url`：相对服务根地址的 URL。

## 测量状态与限制

| 状态 | 行为 |
| --- | --- |
| `estimated_ground_plane` | 返回水平目标平面上的估计面积、覆盖范围与地面轮廓 |
| `empty_mask` | 面积为 0，尺寸为空；没有检测目标时并集也是此状态 |
| `sam_disabled` | 根状态；只进行双 YOLO，`union_measurement=null` |
| `missing_valid_range` / `missing_valid_attitude` / `missing_intrinsics` | 必要参数不完整，尺寸为空 |
| `ray_at_or_above_horizon` | 掩码含不能投影到地面的视线，不返回部分面积 |
| `orientation_requires_normalization` / `digital_zoom_requires_calibration` | 坐标方向或视场参数需核对，尺寸为空 |

每张图的高度为 `R×sin(|pitch|)`，不使用相对起飞点的高度。斜拍尺度随像素位置变化，本服务直接进行透视投影，不用中心 GSD 乘整图掩码像素数。

测距点按用户约定视为相机光轴中心点。尺寸模型假定激光和相机光轴对齐、目标地面水平、镜头畸变可忽略。坡地、地形起伏和车辆等离地物体会影响结果；返回的是地面投影估计，不能由单张照片推算物体三维表面积。图片检测/分割是否正确和相机标定误差也影响测量精度。`touches_image_border=true` 表示目标可能被截断。

## HTTP 错误

| 状态码 | 含义 |
| --- | --- |
| `413` | 文件超过大小上限 |
| `422` | 输入图像、JSON、字段类型或图像尺寸不匹配 |
| `404` | 查询的结果不存在 |
| `503` | 推理队列满、缺少模型文件或 GPU OOM |
| `500` | 其他推理运行错误；详情记录在服务日志 |

队列满时带 `Retry-After: 2`。OOM 时 `detail.code=gpu_oom`，本进程停止后续推理，按已有方案转服务器运行。

结果目录为 `ours/service/runs/<image_id>`，保留原图、掩码、可视化和完整 `result.json`，重新调用 `GET /results/{image_id}` 即可读取历史结果。
