# 安装YOLO、SAM2、后端开发环境
conda create -n YOLOv26 python=3.12 -y  -y表示需要确认的地方全是yes
conda activate YOLOv26
python -m pip install -e .   默认安装 CPU 版 PyTorch，见下面表格，需要二次开发使用-m pip,不需要使用安装requirements

查看显卡和驱动 nvidia-smi

移除 CPU 版及相关包
python -m pip uninstall torch torchvision torchaudio -y

安装一组明确匹配的 GPU 版本
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128

添加.gitignore：   /ours/sam2/
官方仓库fork--git clone 到YOLO26的ours/sam2文件夹

cd C:\Users\ZYX\Desktop\ultralytics_yolo26\ours\sam2
python -c "import torchvision; print('TorchVision:', torchvision.__version__)"

$env:SAM2_BUILD_CUDA="0"
python -m pip install -e .

安装后验证
python -c "import torch, sam2; from sam2.sam2_image_predictor import SAM2ImagePredictor; print('SAM2:', sam2.__file__); print('CUDA:', torch.cuda.is_available())"

下载权重，可以官网下载，更换权重需要同步更新yaml文件
New-Item -ItemType Directory -Force checkpoints | Out-Null
curl.exe -L --fail https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt -o checkpoints\sam2.1_hiera_small.pt

pip install fastapi -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install uvicorn
pip install python-multipart
python -m pip install "hydra-core>=1.3.2"
python -m pip install "tqdm>=4.66.1" "iopath>=0.1.10"
# 服务启动命令：
python -m ours.service.serve --host 0.0.0.0 --port 8000 --device cuda:0 --imgsz 1280 --conf 0.25
# 后端调用该服务的注意事项：
