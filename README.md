# Badminton AI Analysis System

羽毛球AI分析系统，包含球轨迹追踪、运动员姿态估计、坐标统一、信息整合和球种识别等功能模块。

## 项目结构

```
badmintonAI/
├── README.md
├── environment.yml              # Conda环境配置
├── requirements.txt             # Python依赖
├── setup.sh                     # 环境设置脚本
├── src/                         # 源代码目录
│   ├── __init__.py
│   ├── main.py                  # 主程序入口
│   ├── config/                  # 配置文件
│   │   ├── __init__.py
│   │   └── config.yaml          # 系统配置
│   ├── ball_tracking/           # 球轨迹追踪模块
│   │   ├── __init__.py
│   │   ├── yolo_tracker.py      # YOLO检测+轻量化单目标追踪
│   │   └── tracknetv3.py        # TrackNetv3模型
│   ├── pose_estimation/         # 运动员姿态估计模块
│   │   ├── __init__.py
│   │   ├── rtmw.py              # RTMW模型
│   │   └── yolo_pose.py         # YOLOv11-n-Pose模型
│   ├── coordinate_unify/        # 坐标统一模块
│   │   ├── __init__.py
│   │   └── unify.py             # 坐标统一处理
│   ├── info_integration/        # 信息整合模块
│   │   ├── __init__.py
│   │   └── integration.py       # 信息整合处理
│   ├── shot_classification/     # 球种识别模块
│   │   ├── __init__.py
│   │   ├── classifier.py        # 简单分类器
│   │   └── rule_based.py        # 显式逻辑判断
│   └── utils/                   # 工具函数
│       ├── __init__.py
│       ├── video_utils.py       # 视频处理工具
│       └── visualization.py     # 可视化工具
├── models/                      # 模型文件目录
│   ├── ball_tracking/
│   ├── pose_estimation/
│   └── shot_classification/
├── data/                        # 数据目录
│   ├── videos/                  # 输入视频
│   └── results/                 # 输出结果
└── scripts/                     # 脚本目录
    ├── download_models.sh       # 下载模型脚本
    └── test.py                  # 测试脚本
```

## 环境配置

### 1. 创建Conda环境

```bash
conda env create -f environment.yml
conda activate badmintonAI
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 运行环境设置脚本

```bash
bash setup.sh
```

## 使用方法

### 基本使用

```bash
python src/main.py --video path/to/video.mp4 --output path/to/output
```

### 测试用例

```bash
python src/main.py --video TrackNetV2/Test/match1/video/1_05_02.mp4 --output data/results/match1
```

## 模块说明

### 1. 球轨迹追踪模块
- **YOLO检测+轻量化单目标追踪**: 使用YOLO进行初始检测，然后使用轻量化追踪器进行跟踪
- **TrackNetv3**: 使用TrackNetv3模型进行球轨迹追踪

### 2. 运动员姿态估计模块
- **RTMW**: 使用RTMW模型进行2D姿态估计
- **YOLOv11-n-Pose**: 使用YOLOv11-n-Pose模型进行姿态估计

### 3. 坐标统一模块
将不同坐标系下的坐标统一到同一坐标系

### 4. 信息整合模块
整合球轨迹和运动员姿态信息，计算击球方和球的物理信息

### 5. 球种识别模块
- **简单分类器**: 基于机器学习的分类器
- **显式逻辑判断**: 基于规则的球种识别

## 模型下载

运行以下脚本下载所需的模型：

```bash
bash scripts/download_models.sh
```

## 开发说明

本项目支持从GitHub仓库克隆相关模型代码，并在Conda环境中自动配置。
