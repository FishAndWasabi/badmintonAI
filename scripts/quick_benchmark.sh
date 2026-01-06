#!/bin/bash

# TrackNetv3 快速性能测试脚本

echo "=========================================="
echo "TrackNetv3 快速性能测试"
echo "=========================================="

# 设置默认路径
TRACKNET_MODEL="ckpts/ball_tracking/TrackNet_best.pt"
INPAINTNET_MODEL="ckpts/ball_tracking/InpaintNet_best.pt"

# 检查模型文件是否存在
if [ ! -f "$TRACKNET_MODEL" ]; then
    echo "错误: TrackNet模型文件不存在: $TRACKNET_MODEL"
    echo "请先下载模型文件到指定位置"
    exit 1
fi

# 检查是否使用InpaintNet
USE_INPAINT=""
if [ -f "$INPAINTNET_MODEL" ]; then
    echo "检测到InpaintNet模型，将进行完整测试"
    USE_INPAINT="--inpaintnet_file $INPAINTNET_MODEL --test_e2e"
else
    echo "未检测到InpaintNet模型，仅测试TrackNet"
fi

# 运行测试
echo ""
echo "开始性能测试..."
echo ""

python scripts/benchmark_tracknetv3.py \
    --tracknet_file "$TRACKNET_MODEL" \
    $USE_INPAINT \
    --batch_sizes 1 4 8 16 \
    --num_warmup 10 \
    --num_iterations 100 \
    --output "benchmark_results.json"

echo ""
echo "=========================================="
echo "测试完成！结果已保存到 benchmark_results.json"
echo "=========================================="
