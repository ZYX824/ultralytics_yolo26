# 接口服务
配置统一写在 `.env`：YOLO/SAM2 权重路径、下载目录、检测种类、YOLO 置信度与输入尺寸、图片存储地址和上报地址。
接口允许的警告编号白名单仍在 `app.py` 的 `ALLOWED_TYPES`。

处理流程（`app.py` 的 `process_picture()` 依次调用）：

app.py:接口服务入口，编排处理流程
config.py:读取 .env 配置
down_picture.py:下载图片到 DOWNLOAD_DIR
yolo_predict.py:YOLO 推理，返回预测框集合
mask_dedup.py:SAM 分割预测框并拼接掩码，合并相连区域（类别取置信度最高者）
measurement.py:计算每个合并区域的地面面积
callback.py:组装报文并上报（CALLBACK_ENABLED=false 时只打印不上报）
draw_result.py:绘制预测框、种类、置信度，保存到 RESULT_DIR（不受回传开关影响）

## 前置要求
1.创建这个md文档
3.安装依赖：
python -m pip install fastapi uvicorn python-dotenv
fastapi写接口，uvicorn用来启动服务

# 接口参数说明

`types` 为必填整数列表，包含 1–39 个警告编号，每个编号必须在 `app.py` 的 `ALLOWED_TYPES` 白名单中。
白名单包含：10002–10008、12001–12003、12005–12042、1、2000–2004（不包含 12004）。
例如 `[1]`、`[10002, 12001, 2000]` 均有效；空列表、超过 39 个元素、非白名单编号、字符串、小数和布尔值会返回 HTTP 422。
当前允许重复编号，重复项也计入列表长度。

请求校验失败（例如缺少图片字段、列表类型错误、警告编号不在白名单或 JSON 格式错误）时，
接口返回 HTTP 422，响应统一包含 `code` 和 `msg`，不会执行 `detect()`。
`msg` 中的详细原因使用 Pydantic/FastAPI 提供的错误提示。例如 `types` 传入 `[3]`：

```json
{
  "code": 422,
  "msg": "请求参数不合法：types: Value error, 不支持的警告编号：[3]"
}
```

以上仅校验请求参数；当前示例未下载或解码图片，不检查图片文件是否损坏。

目前后端是：请求是一张图片：
{
  "pictures": [
    {
      "pictureId": "DJI_20261010103000_0001_V",
      "picturePath": "DJ-MEDIA-OSS/workspace/job/DJI_20261010103000_0001_V.jpeg"
    }
  ],
  "types": [1, 12001]
}
请求是多张图像：
{
  "pictures": [
    {
      "pictureId": "DJI_20261010103000_0001_V",
      "picturePath": "DJ-MEDIA-OSS/workspace/job/DJI_20261010103000_0001_V.jpeg"
    },
    {
      "pictureId": "DJI_20261010103000_0002_V",
      "picturePath": "DJ-MEDIA-OSS/workspace/job/DJI_20261010103000_0002_V.jpeg"
    },
    {
      "pictureId": "DJI_20261010103000_0003_V",
      "picturePath": "DJ-MEDIA-OSS/workspace/job/DJI_20261010103000_0003_V.jpeg"
    }
  ],
  "types": [1, 12001]
}
# 响应格式
{
    "code": 200,
    "msg": "图片已接收，等待后台处理",
    "data": True
}

# 推理完成请求体JSON格式：
{
  "source": "offline",
  "isReal": 0,
  "pictureId": "DJI_20261010103000_0001_V",
  "eventId": "5831927406284510",
  "timestamp": "2026-10-14 15:30:00",
  "result": {
    "type": "2000",
    "typeName": "道路纵裂",
    "bbox": [420.0, 318.0, 756.0, 602.0],
    "confidence": 0.9642,
    "area_m2": 1.82,  # 面积
    "extent_long_m": 5.2,  # 长度
    "extent_short_m": 2.1,  # 宽度
    "measurement_status": "success"  # 测量状态
  }
}

## 测量状态

`measurement_status` 根据 `ours/service/measurement.py` 中 `measure_mask()` 返回的 `status` 生成；本接口在 `build_report()` 中将成功状态 `estimated_ground_plane` 转为 `success`，其余状态保持原值。状态及含义如下（实际返回条件见下文）：

| 状态值 | 含义 |
| --- | --- |
| `success` | 测量成功：已将掩码投影到激光目标所在的水平地面，计算出面积和最小外接矩形的长短边。结果为水平地面模型下的估计值。 |
| `empty_mask` | 掩码没有前景像素，面积为 `0.0`，长短边为 `None`。 |
| `orientation_requires_normalization` | 图片的 EXIF 方向标记不是 `1`，需要先统一像素方向和对应的几何信息。 |
| `missing_valid_range` | 激光测距距离缺失、无效或不大于零，或者测距状态不是 `Normal`（忽略大小写）。 |
| `missing_valid_attitude` | 云台姿态缺失或无效：俯仰角不满足 `-90° ≤ pitch < 0°`，或者横滚角、偏航角缺失或无效。 |
| `digital_zoom_requires_calibration` | 检测到数码变焦（倍率大于 `1.001`），但没有提供该变焦状态对应的标定内参。 |
| `missing_intrinsics` | 相机内参缺失或无效：需要正数焦距像素值 `fx_px`、`fy_px`，以及有效主点坐标 `cx_px`、`cy_px`。 |
| `ray_at_or_above_horizon` | 掩码涉及的视线接近水平或朝上，无法得到有效的地面交点。 |
| `nonfinite_projection` | 地面投影结果出现 `NaN` 或无穷大，无法有效测量。 |

### 当前报文中的返回规则

`callback.py` 中的 `build_report()` 仅在测量结果存在且 `area_m2` 不为 `None` 时，附加 `area_m2`、`extent_long_m`、`extent_short_m` 和 `measurement_status` 四个字段。因此，当前报文实际只会包含以下两种测量状态：

- `success`：面积有数值，返回面积、长短边和状态。
- `empty_mask`：sam返回一个空的掩码数组[]，返回面积为 `0.0`，长短边为 `null`。

其他七种状态的 `area_m2` 为 `None`，当前报文会省略上述四个字段，不会直接返回失败状态。未执行分割或测量、SAM 调用失败，或者测量抛出异常而没有生成测量结果时，也会省略这些字段。

`extent_long_m` 和 `extent_short_m` 分别表示地面最小外接矩形的长边、短边，不代表裂缝沿曲线的长度或裂缝开口宽度。
