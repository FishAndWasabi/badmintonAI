# 快速开始指南

## 1. 环境配置

### 创建Conda环境

```bash
# 创建环境
conda env create -f environment.yml

# 激活环境
conda activate badmintonAI

# 安装依赖
pip install -r requirements.txt
```

### 运行环境设置脚本

```bash
bash setup.sh
```

## 2. 下载模型（可选）

如果需要使用TrackNetv3或RTMW模型，运行：

```bash
bash scripts/download_models.sh
```

注意：YOLO模型会在首次使用时自动下载。

## 3. 运行测试

### 使用测试脚本

```bash
python scripts/test.py
```

### 使用主程序

```bash
# 基本使用
python src/main.py --video TrackNetV2/Test/match1/video/1_05_02.mp4 --output data/results/match1

# 不生成可视化视频（更快）
python src/main.py --video TrackNetV2/Test/match1/video/1_05_02.mp4 --output data/results/match1 --no-vis
```

## 4. 配置说明

编辑 `src/config/config.yaml` 可以修改系统配置：

- **球轨迹追踪方法**: `ball_tracking.method` - 选择 "yolo_tracker" 或 "tracknetv3"
- **姿态估计方法**: `pose_estimation.method` - 选择 "yolo_pose" 或 "rtmw"
- **球种识别方法**: `shot_classification.method` - 选择 "rule_based" 或 "classifier"

## 5. 输出结果

处理完成后，在输出目录中会生成：

- `results.json`: 包含所有分析结果的JSON文件
- `visualization.mp4`: 可视化视频（如果启用）

## 6. 常见问题

### 模型下载失败

如果TrackNetv3或RTMW模型下载失败，系统会使用占位实现。可以手动克隆仓库到对应目录。

### CUDA错误

如果遇到CUDA相关错误，系统会自动回退到CPU模式。

### 内存不足

对于长视频，可以修改代码只处理部分帧，或降低视频分辨率。
