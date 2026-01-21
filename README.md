# Badminton AI Analysis System

羽毛球AI分析系统 - 基于深度学习的羽毛球比赛视频分析工具

## 功能特性

- ✅ **球轨迹追踪** - 支持YOLO Tracker和TrackNetv3两种方法
- ✅ **姿态估计** - 支持YOLO Pose和RTMW（mmpose）两种方法
- ✅ **坐标统一** - 像素坐标转世界坐标
- ✅ **信息整合** - 击球检测和球速估计
- ✅ **球种识别** - 规则-based和ML分类器
- ✅ **性能监控** - GPU显存使用和处理时延监控
- ✅ **并行处理** - 多进程、多GPU支持

## 快速开始

### 1. 环境配置

```bash
# 创建Conda环境
conda env create -f environment.yml

# 激活环境
conda activate badmintonAI

# 安装依赖
pip install -r requirements.txt
```

### 2. 运行测试

```bash
# 快速测试
python scripts/test_video.py

# 或使用并行处理
python scripts/process_videos_parallel.py --video-dir data/videos
```

## 文档

详细文档请查看 [docs/](docs/) 目录：

- **[快速开始指南](docs/QUICKSTART.md)** - 环境配置和快速开始
- **[项目结构说明](docs/PROJECT_STRUCTURE.md)** - 项目目录结构
- **[脚本使用指南](docs/scripts_README.md)** - 所有脚本的使用说明
- **[性能探查指南](docs/BENCHMARK_GUIDE.md)** - 性能测试和优化
- **[完整文档索引](docs/README.md)** - 所有文档的索引

## 系统要求

- Python 3.8+
- CUDA 11.0+ (推荐)
- GPU: NVIDIA GPU with 8GB+ VRAM (推荐双3090)
- 操作系统: Linux

## 项目结构

```
badmintonAI/
├── docs/              # 文档目录
├── src/               # 源代码
├── scripts/           # 脚本文件
├── models/            # 模型文件
├── data/              # 数据目录
└── ckpts/             # 模型权重
```

详细结构说明请查看 [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md)

## 主要脚本

### 并行处理（推荐）

```bash
# 并行处理所有视频
python scripts/process_videos_parallel.py --video-dir data/videos --num-gpus 2
```

### 顺序处理（调试/资源受限）

```bash
# 顺序处理所有视频
python scripts/process_videos_sequential.py --video-dir data/videos
```

### 性能探查

```bash
# 测试不同并发数下的性能
python scripts/benchmark_concurrency.py --video-dir data/videos --num-gpus 2
```

## 配置

配置文件位于 `src/config/config.yaml`，可以配置：

- 球追踪方法（yolo_tracker / tracknetv3）
- 姿态估计方法（yolo_pose / rtmw）
- 模型路径和参数
- 处理阈值

## 许可证

[添加许可证信息]

## 联系方式

[添加联系方式]
