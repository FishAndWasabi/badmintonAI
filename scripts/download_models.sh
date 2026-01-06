#!/bin/bash

# 下载模型脚本

echo "Downloading models for Badminton AI system..."

# 创建模型目录
mkdir -p models/ball_tracking
mkdir -p models/pose_estimation
mkdir -p models/shot_classification

# 下载YOLO模型（ultralytics会自动下载，这里只是占位）
echo "YOLO models will be downloaded automatically when first used"

# 下载TrackNetv3模型（需要根据实际仓库调整）
if [ ! -d "models/ball_tracking/TrackNetv3" ]; then
    echo "Cloning TrackNetv3..."
    git clone https://github.com/Chang-Chia-Chi/TrackNetv3.git models/ball_tracking/TrackNetv3 || echo "TrackNetv3 clone failed"
fi

# 下载RTMW模型（需要根据实际仓库调整）
if [ ! -d "models/pose_estimation/RTMW" ]; then
    echo "Cloning RTMW..."
    git clone https://github.com/Tau-J/rtmw.git models/pose_estimation/RTMW || echo "RTMW clone failed"
fi

echo "Model download completed!"
