# 无人机图像检测服务：后端对接说明

一次提交 1–50 张 MinIO 原图地址，AI 默认仅执行目标检测，返回类别、置信度、检测框和带框图片。
调用方必须通过 `types` 选择平台算法 ID。当前仅实现“车辆违停”（算法 ID `12001`），配置回调地址后，检测到车辆就主动 POST 告警报文。

车辆范围为 VisDrone 模型的汽车 `car`、面包车 `van`、卡车 `truck` 和公交车 `bus`。
按当前联调规则，车辆通过检测置信度阈值（启动命令默认 `0.25`）就触发“车辆违停”告警，
不判断停留时间或禁停区域；行人、自行车、摩托车、三轮车及道路病害不触发此告警。

## 1. 服务地址与调用流程

```text
AI 服务地址：https://627635a1d95d4803b3ff1bcba918c739.region2.waas.aigate.cc
```

[接口调试页](https://627635a1d95d4803b3ff1bcba918c739.region2.waas.aigate.cc/docs) ·
[服务状态](https://627635a1d95d4803b3ff1bcba918c739.region2.waas.aigate.cc/health)

服务器和 AI 程序必须处于运行状态。这个唯一 HTTPS 域名已绑定服务器内部 **9000** 端口，公网调用直接使用域名加接口路径，无需再加 `:9000`。
`/docs` 是接口调试页面；提交任务时使用下面的检测接口地址。

| 用途 | 完整公网地址 |
| --- | --- |
| 接口调试页 | `https://627635a1d95d4803b3ff1bcba918c739.region2.waas.aigate.cc/docs` |
| 提交任务（POST） | `https://627635a1d95d4803b3ff1bcba918c739.region2.waas.aigate.cc/detect` |
| 文档同路径 Demo 入口（POST） | `https://627635a1d95d4803b3ff1bcba918c739.region2.waas.aigate.cc/alarm/record/detection2` |
| 查询任务（GET） | `https://627635a1d95d4803b3ff1bcba918c739.region2.waas.aigate.cc/tasks/{task_id}` |

```text
前端选择图片 → 业务后端 /alarm/record/detection2 → AI /detect
AI 下载 MinIO 原图并处理 → 逐目标 POST 告警上报接口，同时保存结果供查询和展示
```

后端将 `system-param.alarm-endpoint` 配置为：

```text
https://627635a1d95d4803b3ff1bcba918c739.region2.waas.aigate.cc/detect
```

| 接口 | 所属服务 / 用途 |
| --- | --- |
| `POST /alarm/record/detection2` | 业务后端入口，接收前端请求并转发；AI 也提供同路径 Demo 别名，可直接联调 |
| `POST /detect` | AI 批量检测入口 |
| `GET /tasks/{task_id}` | AI 任务进度及逐图结果查询 |
| `POST http://113.249.105.17:28096/api/alarm/record/report` | 业务后端告警接收接口；配置为 AI 回调地址后主动上报，无需鉴权 |

## 2. 提交检测任务

向 AI 的 `/detect` 发送 **POST** 请求，`Content-Type: application/json`：

```json
{
  "pictures": [
    {
      "pictureId": "后端媒体表中的真实图片ID",
      "picturePath": "http://minio-host:9000/bucket/path/image.jpeg"
    }
  ],
  "types": [12001]
}
```

| 字段 | 对接要求 |
| --- | --- |
| `pictures` | 1–50 个图片条目，传图片信息即可，无需上传图片文件 |
| `pictureId` | 后端登记的原图 ID；同一批次内唯一，结果原样带回 |
| `picturePath` | AI 能下载的完整 HTTP/HTTPS 原图地址，支持预签名 URL |
| `types` | 必填，非空整数列表，由调用方指定平台算法 ID，应用于本次所有图片；当前传 `[12001]` 选择车辆违停，重复 ID 不会重复上报 |
| `with_sam` | 可省略，默认 `false`，仅做目标检测；显式传 `true` 才启用 SAM2 分割和尺寸测量 |
| `callbackUrl` | 可选，覆盖启动时配置的回调地址；通常由服务统一配置，无需每次传入 |

仅做目标检测无需测距、姿态或焦距参数。若显式启用分割与测量，原图需保留 EXIF/DJI XMP，
相关参数由 AI 自动提取，后端通常无需另传。
每张图片最多 32 MB、2500 万像素，下载链接有效期需覆盖排队和处理时间。
`types` 使用平台算法库中的 ID，不是 YOLO 类别 ID，也不是自定义临时编号。
例如 `10007` 为机动车、`12001` 为车辆违停、`12002` 为交通标识线模糊，各自代表不同业务算法。
目前仅支持 `12001`，其他算法需完成实现及映射后才能使用；旧编码 `1` 不再接受。
缺少 `types`、传空列表或选择尚未支持的算法均返回 HTTP 422。
混合提交 `[12001,12002]` 会整体拒绝并列出未支持的 ID，不会偷偷改成车辆违停或忽略选项。

受理响应示例：

```json
{
  "code": 200,
  "success": true,
  "msg": "操作成功",
  "data": true,
  "task_id": "0123456789abcdef0123456789abcdef",
  "status": "queued",
  "total": 1,
  "status_url": "/tasks/0123456789abcdef0123456789abcdef"
}
```

**成功响应表示已受理，尚未处理完成。后端必须将顶层 `task_id` 透传给前端，不能只返回 `data=true`。**
重复提交会生成新任务；默认最多一批处理、一批等待，队列满时返回 HTTP 503 和 `Retry-After: 2`。

## 3. 查询进度与展示结果

向 AI 服务发送 `GET /tasks/{task_id}`，建议每 2 秒查询一次。

| 任务 `status` | 含义 |
| --- | --- |
| `queued` / `running` | 排队中 / 处理中，继续查询 |
| `completed` | 全部图片处理成功，停止查询 |
| `partial_failed` / `failed` | 部分 / 全部图片失败，停止查询并检查逐图错误 |
| `interrupted` | 服务重启或处理被中断，停止查询，按需重新提交未完成图片 |

响应中的 `total/processed/succeeded/failed` 分别为总数、已处理数、成功数和失败数。
逐图结果在 `pictures[i]`，用 `pictureId` 与原始图片关联：

| 字段（相对于 `pictures[i]`） | 用途 |
| --- | --- |
| `status` / `error` | 图片是否完成；失败原因 |
| `result.artifact_urls.combined_yolo` | 预测框图片链接 |
| `result.artifact_urls.sam_overlay` | 启用 SAM 时提供的预测框、掩码和测量叠加图链接 |
| `result.artifact_urls.union_mask` | 启用 SAM 时提供的全部目标掩码并集图片链接 |
| `result.detections[]` | 每个目标的类别、框、置信度、掩码及测量数据 |
| `result.union_measurement.area_m2` | 全部目标掩码的并集面积，重叠部分只计一次 |
| `result.result_url` | 单图完整报告查询链接，包含几何信息及轮廓 |
| `reports[].payload` | 原告警接口文档格式的逐目标 POST 报文，同时保存用于核对 |
| `reports[].callback` | 对应目标的上报状态、发送次数及错误；`delivered` 表示平台已受理，发送失败为 `failed`，report 模式不自动重试 |
| `callback.status` | 整张图上报状态：全部被平台受理为 `delivered`，有发送失败为 `failed`；无匹配目标或推理失败为 `skipped`，未配置地址为 `disabled` |

`result.detections[]` 的单目标示例（数值仅作格式说明）：

```json
{
  "instance_id": 1,
  "model_id": "visdrone2019",
  "class_name": "car",
  "confidence": 0.9,
  "box_xyxy": [420, 318, 756, 602],
  "mask_url": "https://AI服务域名/files/结果ID/masks/0001.png",
  "measurement": {
    "status": "estimated_ground_plane",
    "reason": null,
    "area_m2": 3.8,
    "extent_long_m": 3.5,
    "extent_short_m": 1.5,
    "touches_image_border": false
  }
}
```

直接使用返回的图片链接展示，无需自行拼接文件路径。
`box_xyxy` 是原图左上角为原点的像素坐标 `[x1,y1,x2,y2]`，不是 0–1 归一化坐标。

面积单位为平方米，长短边单位为米。**只有 `measurement.status=estimated_ground_plane` 时才展示估算数值。**
缺少测量条件时仍输出检测和分割，尺寸为空，使用 `reason` 展示原因。
尺寸是 SAM 分割区域在水平地面上的近似覆盖范围，不是物体三维尺寸或裂缝实际长度。

无目标时图片仍处理成功，`detections=[]`；单张失败不影响其他图片，失败图片的 `result=null`。
GPU 显存不足后当前进程停止推理，需要排查并重启。

默认 `report` 模式每辆检测到的车辆发送一个 JSON 对象，包含 `source=offline`、`isReal=0`、原始
`pictureId`、16 位随机数字字符串 `eventId`、北京时间 `timestamp` 和 `result`。
每条告警生成一次 `eventId`，保存的报文与发送的报文使用同一编号。
`result` 仅包含 `type`、`typeName`、`bbox`、`confidence` 四个字段。
告警 `result.type="12001"`、`result.typeName="车辆违停"`，与请求选择的平台算法 ID 对应；一张图有多辆车时会收到多次 POST。
`seq` 省略，由平台生成当天递增序号。实际报文示例如下（ID 和数值仅作格式说明）：

```json
{
  "source": "offline",
  "isReal": 0,
  "pictureId": "DJI_20261010103000_0001_V",
  "eventId": "5831927406284510",
  "timestamp": "2026-10-12 15:30:00",
  "result": {
    "type": "12001",
    "typeName": "车辆违停",
    "bbox": [420.0, 318.0, 756.0, 602.0],
    "confidence": 0.9642
  }
}
```

`pictureId` 必须是平台媒体文件表中存在的原始图片 ID；图片文件名不一定是媒体 ID。
平台通过 `pictureId` 获取原图，按 `bbox` 生成标注图并上传、入库。
模型及类别、掩码、测量和 AI 图片链接保留在任务查询结果中。
无车辆或推理失败时不会创建告警，仍可通过任务查询确认处理状态。
默认仅检测框即可触发告警，请求不必传 `with_sam`。此时不加载 SAM2，掩码和测量结果为空。

平台返回 `code=200`、`success=true` 仅表示请求已受理，后续标注和入库是异步处理。
因此 `reports[].callback.status=delivered` 也仅代表受理成功，需通过平台告警记录、标注图、
WebSocket 消息或日志核验最终结果。平台当前不按 `eventId` 去重，report 模式每条报文仅尝试一次，
包括超时在内的发送失败不会自动重试，避免重复告警；人工重发前需核对平台记录。
上报失败保留检测结果，不影响后续目标和图片；任务 `completed` 仅代表推理成功。
AI 保存 `eventId`、`pictureId` 和完整推送报文供核对。服务重启不自动重发已有报文。

## 4. 部署与联调

在项目根目录、已安装 CUDA PyTorch/torchvision、OpenCV 和 YOLO 依赖的环境中启动（Linux）：

```bash
python -m pip install -r ours/service/requirements.txt
export YOLO_SERVICE_PUBLIC_BASE_URL="https://627635a1d95d4803b3ff1bcba918c739.region2.waas.aigate.cc"
export YOLO_SERVICE_CALLBACK_FORMAT="report"
unset YOLO_SERVICE_TYPE_MAP  # 使用当前默认的车辆违停映射
python -m ours.service.serve --host 0.0.0.0 --port 9000 --device cuda:0 --imgsz 1280 --conf 0.25 \
  --callback-url "http://113.249.105.17:28096/api/alarm/record/report"
```

`YOLO_SERVICE_PUBLIC_BASE_URL` 填公网域名，不带 `/docs` 或 `:9000`，用于生成可访问的结果链接。
如果服务器重启后平台分配了新域名，请同步更新后端调用地址和此环境变量，再启动 AI 服务。

仅运行一个 worker。两个 YOLO 权重位于 `ours/weights/UAV_PDD2023/best.pt`、
`ours/weights/visdrone2019/best.pt`。默认目标检测无需 SAM2 源码或权重；显式启用分割时需提供 `ours/sam2/` 中的资源。
模型、数据集和运行结果不随 Git 提交，部署时需另行提供模型资源。

| 按需配置 | 说明 |
| --- | --- |
| `YOLO_SERVICE_CORS_ORIGINS` | 前端直接跨域调用 AI 时配置实际前端 origin，多个地址用逗号分隔；仅后端调用无需配置 |
| `YOLO_SERVICE_IMAGE_BASE_URL` | 完整图片 URL 无需配置；若传 `DJ-MEDIA-OSS/...` 相对路径，设置为 `http://113.249.105.17:9000/dbxq/` |
| `YOLO_SERVICE_CALLBACK_URL` | 固定告警上报地址；也可用 `--callback-url` 指定，命令行优先；都不配置时仅保存报文 |
| `YOLO_SERVICE_CALLBACK_FORMAT` | 默认 `report`，逐目标告警报文；`demo` 为逐图完整结果通知，包含处理失败通知，报文格式不同 |
| `YOLO_SERVICE_CALLBACK_TOKEN` | 可选 Bearer token，仅发送到固定配置的回调地址；当前告警接收接口无需配置 |
| `YOLO_SERVICE_TYPE_MAP` | 可覆盖已实现算法的告警映射；当前无需配置，默认 `12001` 匹配 `visdrone2019` 的类别 `[3,4,5,8]`，名称“车辆违停”；仅增加编号不能使模型具备新算法能力 |

首次联调：检查 `/health` → 提交一张真实原图 → 保存并透传 `task_id` → 查询到终态 →
确认告警接口收到报文且 `reports[].callback.status=delivered` → 打开结果图片、核对尺寸字段 → 再测试 50 张批量请求。
`/health` 的 `ready` 仅表示服务未因显存不足停止，模型在首次推理时加载。

结果保存于 `ours/service/runs/`，重启后保留已完成结果，未完成任务标记 `interrupted`，不自动重跑。
下载和推理失败在任务结果中查看；提交请求 HTTP 422 表示参数错误，503 表示队列满或服务正在退出。

单图上传与补充几何参数的详细说明见 [API.md](API.md)；日常批量对接使用本 README 即可。
