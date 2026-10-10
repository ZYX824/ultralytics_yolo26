# 轮廓映射所需几何信息提取

本地读取图片 EXIF / DJI XMP，仅输出轮廓换算真实尺寸需要的数据。支持每张图独立提取和计算，不上传网站，不保存拍摄时间、曝光、ISO、设备序列号、完整 EXIF 等无关信息。

## 计算依据

按本项目约定，激光测距目标就是图像中心点，且测距轴与相机光轴对齐。对 DJI 云台俯仰角（水平为 0°，垂直向下为 -90°）：

```text
R = 每张图的 LRFTargetDistance，斜距，单位 m
θ = abs(GimbalPitchDegree)，相对水平面的向下夹角
H = R × sin(θ)，无人机相对中心目标所在水平面的垂直高度
D = R × cos(θ)，无人机到中心目标的水平距离
```

直接使用每张图片自己的测距和角度，**不使用相对起飞点的 RelativeAltitude 计算对地高度**。测距必须为正、`LRFStatus=Normal`、俯仰角在 `[-90°, 0°)` 范围内；缺失或异常时保留空值和明确状态，不填固定高度，也不固定使用 45°。

样图 `R=177.947 m`、俯仰角 `-45°`，计算得到 **H≈125.82753 m**。这是相对测距目标平面的高度，不是相对起飞点的约 110 m。图像中的无人机绝对高程和目标绝对高程也会提取，供后续定位与几何核对。

## 输出字段

每张图片一条记录，JSON / CSV 都包含以下信息：

| 信息 | 字段 |
| --- | --- |
| 原图像素坐标网格 | `width_px`, `height_px`, `orientation` |
| 无人机经纬度、高程 | `uav_latitude_deg`, `uav_longitude_deg`, `uav_absolute_altitude_m` |
| 中心目标经纬度、高程 | `center_target_latitude_deg`, `center_target_longitude_deg`, `center_target_absolute_altitude_m`，来自 `LRFTargetLat/Lon/AbsAlt` |
| 无人机到中心的测距及状态 | `center_range_m`, `range_status` |
| 云台俯仰、横滚、偏航 | `gimbal_pitch_deg`, `gimbal_roll_deg`, `gimbal_yaw_deg` |
| 焦距、等效焦距、数字变焦 | `focal_length_mm`, `focal_length_35mm_mm`, `digital_zoom_ratio` |
| 推算的对目标地面高度和水平距离 | `height_agl_m`, `height_source`, `horizontal_distance_to_center_m` |
| 近似内参和来源 | `focal_x_px`, `focal_y_px`, `principal_x_px`, `principal_y_px`, `intrinsics_source` |
| 中心点局部地面比例 | `center_gsd_cross_track_m_per_pixel`, `center_gsd_along_track_m_per_pixel` |
| 垂直拍摄的统一尺度 | `gsd_m_per_pixel`；斜拍时为空 |

缺少参数或测距状态异常时，字段仍被保留。读取成功不代表几何条件齐全，需检查 `status`。

相机内参优先根据用户提供的当前视场有效感光区域尺寸和实际焦距计算；未提供时，使用 EXIF 35mm 等效焦距近似匹配对角线视场：

```text
fx ≈ fy ≈ 35mm等效焦距 × sqrt(图宽² + 图高²) / sqrt(36² + 24²)
cx ≈ (图宽 - 1) / 2；cy ≈ (图高 - 1) / 2
```

相机标定内参比整数等效焦距更准确。默认主点是图像中心近似；输入应为未裁切、视场与参数一致的原图。

## 使用

在仓库根目录运行，`python` 可替换为本机 `D:\Program_Files\Anaconda\envs\tiangong\python.exe`。

```powershell
python ours/extraction/extract_gsd.py --source "C:\Users\ZYX\Desktop\2d1ff7278ec68e7d9e5ff53dd0632d72.jpeg"
python ours/extraction/extract_gsd.py --source "D:\images" --recursive --output ours/extraction/runs/batch

# 可选：提供当前相机视场的有效感光区域和实际焦距
python ours/extraction/extract_gsd.py --source "D:\images" --sensor-width-mm 8 --sensor-height-mm 6 --focal-length-mm 6.72
```

默认输出：`ours/extraction/runs/latest/metadata.json` 和 `summary.csv`。重复运行会覆盖同一输出目录的这两个报告；源图不会修改。CSV 使用 UTF-8 BOM，可用 Excel 打开。批次中损坏图像会记录为 `error`，其他图像继续处理，退出码为 1。

接口：

```python
from ours.extraction.extract_gsd import extract_image

geometry = extract_image(image_path)
height = geometry["height_agl_m"]  # 每张图自己的测距 × sin(自己的俯仰角)
```

## 后续轮廓映射

本脚本只提取所需几何参数并计算高度、中心局部尺度，不运行 YOLO/SAM，不实现轮廓投影。

斜拍时，把每个轮廓像素按相机内参和云台姿态生成视线，与中心目标所在水平面求交，再计算真实距离/面积。局部中心比例在零横滚时为：横向 `R/fx`，纵向 `R/(fy×sinθ)`；它们只适用于中心附近。横滚非零时地面横向/纵向方向与图像 x/y 方向会旋转。相机像素比例不同且存在横滚时，不输出这两个简化局部比例。

单个测距点提供的是目标平面基准，整幅图像的地形起伏、镜头畸变和目标是否位于地面仍会影响映射；地形起伏明显时应结合 DEM 或深度信息。斜拍的整图比例会随像素位置变化，因此 `oblique_requires_projection` 表示需要用已提取的几何参数进行投影，不能将中心局部比例当作整图统一 GSD。
