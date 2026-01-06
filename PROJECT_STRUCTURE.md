# 项目结构说明

## 目录结构

```
badmintonAI/
├── README.md                    # 项目说明文档
├── QUICKSTART.md               # 快速开始指南
├── PROJECT_STRUCTURE.md        # 本文件
├── environment.yml             # Conda环境配置
├── requirements.txt            # Python依赖
├── setup.sh                    # 环境设置脚本
├── .gitignore                  # Git忽略文件
│
├── src/                        # 源代码目录
│   ├── __init__.py
│   ├── main.py                 # 主程序入口
│   │
│   ├── config/                 # 配置模块
│   │   ├── __init__.py
│   │   └── config.yaml         # 系统配置文件
│   │
│   ├── ball_tracking/          # 球轨迹追踪模块
│   │   ├── __init__.py
│   │   ├── yolo_tracker.py     # YOLO检测+卡尔曼滤波追踪
│   │   └── tracknetv3.py       # TrackNetv3模型接口
│   │
│   ├── pose_estimation/        # 运动员姿态估计模块
│   │   ├── __init__.py
│   │   ├── yolo_pose.py        # YOLOv11-n-Pose姿态估计
│   │   └── rtmw.py             # RTMW模型接口
│   │
│   ├── coordinate_unify/       # 坐标统一模块
│   │   ├── __init__.py
│   │   └── unify.py            # 像素坐标到世界坐标转换
│   │
│   ├── info_integration/       # 信息整合模块
│   │   ├── __init__.py
│   │   └── integration.py      # 整合球轨迹和姿态信息
│   │
│   ├── shot_classification/    # 球种识别模块
│   │   ├── __init__.py
│   │   ├── rule_based.py       # 基于规则的分类器
│   │   └── classifier.py       # 机器学习分类器
│   │
│   └── utils/                  # 工具函数
│       ├── __init__.py
│       ├── video_utils.py      # 视频处理工具
│       └── visualization.py   # 可视化工具
│
├── models/                     # 模型文件目录
│   ├── ball_tracking/          # 球追踪模型
│   ├── pose_estimation/        # 姿态估计模型
│   └── shot_classification/    # 球种分类模型
│
├── data/                      # 数据目录
│   ├── videos/                # 输入视频
│   └── results/               # 输出结果
│
├── scripts/                    # 脚本目录
│   ├── test.py                # 测试脚本
│   └── download_models.sh     # 模型下载脚本
│
└── logs/                      # 日志目录
```

## 模块说明

### 1. 球轨迹追踪模块 (`ball_tracking/`)

**功能**: 检测和追踪视频中的羽毛球

**实现方法**:
- `yolo_tracker.py`: YOLO检测 + 卡尔曼滤波追踪
  - 使用YOLO进行初始检测
  - 使用卡尔曼滤波器进行轨迹平滑和预测
  - 支持丢失追踪后的恢复
  
- `tracknetv3.py`: TrackNetv3模型接口
  - 提供TrackNetv3模型的封装
  - 支持模型加载和推理

**输入**: 视频帧 (numpy array)
**输出**: 球心坐标 (x, y) + 置信度

### 2. 运动员姿态估计模块 (`pose_estimation/`)

**功能**: 估计视频中运动员的2D姿态关键点

**实现方法**:
- `yolo_pose.py`: YOLOv11-n-Pose
  - 使用Ultralytics的YOLO Pose模型
  - 支持COCO格式的17个关键点
  - 自动识别前两名玩家
  
- `rtmw.py`: RTMW模型接口
  - 提供RTMW模型的封装
  - 支持实时多人姿态估计

**输入**: 视频帧 (numpy array)
**输出**: 每名球员的关键点序列 (N, 17, 3) - (x, y, confidence)

### 3. 坐标统一模块 (`coordinate_unify/`)

**功能**: 将不同坐标系下的坐标统一到世界坐标系

**实现**:
- `unify.py`: 坐标转换
  - 像素坐标 → 世界坐标
  - 支持相机标定（可选）
  - 简化版线性映射

**输入**: 
- 羽毛球中心坐标（像素）
- 球员关键点坐标（像素）
- 球场信息
- 相机信息（可选）

**输出**: 
- 统一后的羽毛球中心坐标（世界坐标）
- 统一后的球员关键点坐标（世界坐标）

### 4. 信息整合模块 (`info_integration/`)

**功能**: 整合球轨迹和运动员姿态信息，计算物理参数

**实现**:
- `integration.py`: 信息整合
  - 计算球的物理信息（速度、方向、加速度）
  - 检测击球事件
  - 识别击球方

**输入**: 
- 羽毛球中心坐标
- 球员关键点坐标

**输出**: 
- 击球方（0或1）
- 羽毛球物理信息（球速、飞行方向）

### 5. 球种识别模块 (`shot_classification/`)

**功能**: 识别羽毛球的球种类型

**实现方法**:
- `rule_based.py`: 基于规则的分类器
  - 根据速度、角度、位置等特征判断
  - 支持：发球、高远球、吊球、扣杀、平抽、网前球、挑球
  
- `classifier.py`: 机器学习分类器
  - 使用随机森林分类器
  - 需要训练数据
  - 支持模型保存和加载

**输入**: 
- 羽毛球物理信息（坐标、球速、飞行方向）
- 球员关键点坐标
- 击球方

**输出**: 球种类别 + 置信度

## 数据流

```
视频帧
  ↓
[球轨迹追踪] → 球心坐标(像素) → [坐标统一] → 球心坐标(世界)
  ↓
[姿态估计] → 关键点(像素) → [坐标统一] → 关键点(世界)
  ↓
[信息整合] → 击球信息 + 物理参数
  ↓
[球种识别] → 球种类别
  ↓
结果输出 (JSON + 可视化视频)
```

## 配置文件

`src/config/config.yaml` 包含所有模块的配置参数，可以根据需要调整：

- 模型路径
- 阈值参数
- 球场尺寸
- 视频处理参数

## 扩展说明

### 添加新的追踪方法

1. 在 `ball_tracking/` 目录创建新文件
2. 实现 `track(frame)` 方法，返回 `(x, y, confidence)` 或 `None`
3. 在 `config.yaml` 添加配置
4. 在 `main.py` 中添加对应的初始化代码

### 添加新的姿态估计方法

1. 在 `pose_estimation/` 目录创建新文件
2. 实现 `estimate(frame)` 方法，返回关键点列表
3. 在 `config.yaml` 添加配置
4. 在 `main.py` 中添加对应的初始化代码

### 训练球种分类器

1. 准备训练数据（特征和标签）
2. 使用 `MLShotClassifier` 进行训练
3. 保存模型到 `models/shot_classification/`
4. 在配置文件中指定模型路径
