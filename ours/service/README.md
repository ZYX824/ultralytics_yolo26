# 双 YOLO + SAM + 地面尺寸服务

后端调用的完整服务已实现，详见 [API.md](API.md)。

流程：接收原图和对应几何信息 → 双 YOLO 推理 → 合并框 → SAM 分割 → 地面投影 → 面积与覆盖范围。
两套模型保持独立类别命名空间，不做跨模型 NMS。SAM 使用未绘制框的原图，每张图仅编码一次。

## 文件

- `measurement.py`：基于每张图的中心激光测距、云台姿态和相机内参，将像素投影到中心目标水平面。返回掩码面积、覆盖范围、地面轮廓及测量状态。
- `pipeline.py`：双 YOLO、SAM、测量的完整流程，保留原图和几何信息，生成实例掩码、合并掩码、可视化和 JSON。
- `app.py`：FastAPI 上传、历史结果、文件下载与健康检查接口。
- `serve.py`：后端服务启动入口，单进程使用 GPU，默认监听 `0.0.0.0:8000`。
- `__main__.py`：不启动 HTTP 的单图命令行入口。

## 启动

本机使用已有 CUDA 环境 `YOLOv26`；`tiangong` 没有 PyTorch。

```powershell
# HTTP 服务；本次样图测试配置
& 'D:\Program_Files\Anaconda\envs\YOLOv26\python.exe' -m ours.service.serve --device cuda:0 --imgsz 1280 --conf 0.1

# 不启动 HTTP，直接生成单图完整结果
& 'D:\Program_Files\Anaconda\envs\YOLOv26\python.exe' -m ours.service --source image.jpeg --device cuda:0 --imgsz 1280 --conf 0.1

# 提供第一步的单图信息报告
& 'D:\Program_Files\Anaconda\envs\YOLOv26\python.exe' -m ours.service --source image.jpeg --info ours/extraction/runs/sample/metadata.json
```

HTTP 的 `serve` 默认 `1280 / conf=0.25`；单图 CLI 保留 `640 / conf=0.25`。用户可调整阈值。

接口：`POST /predict`、`GET /results/{image_id}`、`GET /health`、`GET /files/...`。
本机调试页：[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)。

## 输出

默认输出到 `ours/service/runs/<image_id>`：

- 原始文件字节与 EXIF/XMP，`original.*`。
- `combined_yolo.jpg`：双权重合并预测框。
- `sam_overlay.jpg`：框、掩码、面积与长短边覆盖范围。
- `masks/*.png`：原图尺寸的各实例掩码。
- `union_mask.png`：实例掩码并集。
- `result.json`：完整检测、像素轮廓、地面轮廓、几何信息、测量、计时和 GPU 占用。

高度按 `中心斜距×sin(|俯仰角|)` 每张独立计算。面积累加投影后的完整像素单元，包含孔洞与分离区域，单像素区域也可以计算；并集面积避免重叠重复计数。

长度字段是掩码在目标水平面上的旋转外接矩形长、短边，不能用来表示裂缝实际长度/宽度。
默认内参由等效焦距近似，支持通过 `info.calibrated_intrinsics` 提供对应原图的标定内参。
结果属于水平目标平面上的估计，坡地、地形起伏、离地物体和检测/分割误差影响准确度。

缺少测量参数时仍提供检测和分割结果，测量字段为空且标出原因。空掩码面积为 0。测量不会凭空使用固定 GSD 或相对起飞点高度。

## 资源与迁移

现有两个 YOLO 默认路径分别为 `ours/weights/UAV_PDD2023/best.pt`、`ours/weights/visdrone2019/best.pt`。
SAM 默认使用现有 `ours/sam2/checkpoints/sam2.1_hiera_small.pt` 与相应源码。

模型加载后常驻，一张图推理、一张图排队；队列满时返回 503。锁覆盖共享 YOLO/SAM 状态，避免并发请求交叉。
如果 GPU OOM，停止当前进程后续推理并返回 `gpu_oom`，使用相同服务转服务器运行。仅运行一个 worker。

服务器需要当前仓库、两个权重、SAM 源码和检查点、CUDA PyTorch/torchvision。
在服务器环境安装当前仓库和 `ours/service/requirements.txt`；不要直接复制 Windows conda 环境。
启动：`python -m ours.service.serve --host 0.0.0.0 --port 8000 --device cuda:0`。

直接启动 `python -m uvicorn ours.service.app:app --workers 1` 时，可通过以下环境变量配置：
`YOLO_SERVICE_DEVICE`、`YOLO_SERVICE_WEIGHTS_UAV`、`YOLO_SERVICE_WEIGHTS_VISDRONE`、
`YOLO_SERVICE_SAM_CHECKPOINT`、`YOLO_SERVICE_SAM_SIZE`、`YOLO_SERVICE_OUTPUT`、
`YOLO_SERVICE_IMGSZ`、`YOLO_SERVICE_CONF`、`YOLO_SERVICE_IOU`、`YOLO_SERVICE_MAX_DET_PER_MODEL`。
`serve` 的命令行参数覆盖对应设备、分辨率和置信度环境变量。

## 验证

`python -m unittest discover -s ours/service/tests -v`。
测试包含垂直拍摄已知矩形面积、45°斜拍针孔模型、横滚/偏航、孔洞、单像素、
中心测距独立计算高度、无参数和地平线无交点、HTTP 上传和结果查询、OOM 停止。

本机 RTX 4060 Laptop 8 GB 已通过完整 HTTP 实测：样图 `1280 / conf=0.1`，
VisDrone 检出 5 个目标，UAV-PDD2023 检出 0 个道路病害，SAM 输出 5 张 `4032×3024` 掩码，
每个实例均返回地面投影测量。PyTorch 峰值分配约 663 MB、保留约 790 MB，未发生 OOM。
首次请求包含加载模型耗时约 4 s，本次几何计算约 0.145 s。
这些是本次图像和配置的资源实测，不代表地面实物尺寸已通过外部标定验证。
