#!/bin/bash

# Badminton AI 环境设置脚本

echo "Setting up Badminton AI environment..."

# 创建必要的目录
mkdir -p models/ball_tracking
mkdir -p models/pose_estimation
mkdir -p models/shot_classification
mkdir -p data/videos
mkdir -p data/results
mkdir -p logs

# 克隆TrackNetv3仓库（如果不存在）
if [ ! -d "models/ball_tracking/TrackNetv3" ]; then
    echo "Cloning TrackNetv3 repository..."
    git clone https://github.com/qaz812345/TrackNetV3.git models/ball_tracking/TrackNetv3 || echo "TrackNetv3 repository clone failed or already exists"
fi

# 克隆RTMW仓库（如果不存在）
if [ ! -d "models/pose_estimation/mmpose" ]; then
    echo "Cloning RTMW repository..."
    git clone https://github.com/open-mmlab/mmpose.git models/pose_estimation/mmpose || echo "RTMW repository clone failed or already exists"
fi

echo "Environment setup completed!"
echo "Please run: conda activate badmintonAI"
echo "Then install dependencies: pip install -r requirements.txt"
