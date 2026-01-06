"""Test script for Badminton AI system"""

import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.main import BadmintonAISystem


def test_system():
    """测试系统基本功能"""
    print("=" * 50)
    print("Badminton AI System Test")
    print("=" * 50)
    
    # 测试视频路径
    test_video = "TrackNetV2/Test/match1/video/1_05_02.mp4"
    test_video_path = project_root / test_video
    
    if not test_video_path.exists():
        print(f"错误: 测试视频不存在: {test_video_path}")
        return
    
    # 输出目录
    output_dir = project_root / "data" / "results" / "test"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 初始化系统
    print("\n1. 初始化系统...")
    try:
        system = BadmintonAISystem()
        print("✓ 系统初始化成功")
    except Exception as e:
        print(f"✗ 系统初始化失败: {e}")
        return
    
    # 处理视频（只处理前100帧用于快速测试）
    print("\n2. 处理测试视频（前100帧）...")
    try:
        # 修改主程序以支持帧数限制（这里简化处理）
        results = system.process_video(
            str(test_video_path),
            str(output_dir),
            visualize=True
        )
        print("✓ 视频处理完成")
        print(f"  - 球轨迹点: {len(results['ball_trajectory'])}")
        print(f"  - 击球事件: {len(results['hit_events'])}")
        print(f"  - 球种分类: {len(results['shot_classifications'])}")
    except Exception as e:
        print(f"✗ 视频处理失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "=" * 50)
    print("测试完成！")
    print(f"结果保存在: {output_dir}")
    print("=" * 50)


if __name__ == '__main__':
    test_system()
