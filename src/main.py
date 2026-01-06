"""Main entry point for Badminton AI Analysis System"""

import argparse
import yaml
import cv2
import numpy as np
from pathlib import Path
from typing import Optional
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.video_utils import load_video, frame_generator
from src.utils.visualization import draw_ball, draw_pose, draw_trajectory, draw_text
from src.ball_tracking.yolo_tracker import YOLOBallTracker
from src.ball_tracking.tracknetv3 import TrackNetv3Tracker
from src.pose_estimation.yolo_pose import YOLOPoseEstimator
from src.pose_estimation.rtmw import RTMWPoseEstimator
from src.coordinate_unify.unify import CoordinateUnifier
from src.info_integration.integration import InfoIntegrator
from src.shot_classification.rule_based import RuleBasedClassifier
from src.shot_classification.classifier import MLShotClassifier


class BadmintonAISystem:
    """羽毛球AI分析系统主类"""
    
    def __init__(self, config_path: str = "src/config/config.yaml"):
        """
        初始化系统
        
        Args:
            config_path: 配置文件路径
        """
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 初始化球轨迹追踪模块
        ball_tracking_method = self.config['ball_tracking']['method']
        if ball_tracking_method == 'yolo_tracker':
            yolo_config = self.config['ball_tracking']['yolo']
            self.ball_tracker = YOLOBallTracker(
                model_path=yolo_config.get('model_path'),
                conf_threshold=yolo_config.get('conf_threshold', 0.5),
                iou_threshold=yolo_config.get('iou_threshold', 0.45)
            )
        elif ball_tracking_method == 'tracknetv3':
            tracknet_config = self.config['ball_tracking']['tracknetv3']
            self.ball_tracker = TrackNetv3Tracker(
                model_path=tracknet_config.get('model_path'),
                input_size=tuple(tracknet_config.get('input_size', [288, 512])),
                confidence_threshold=tracknet_config.get('confidence_threshold', 0.5)
            )
        else:
            raise ValueError(f"未知的球轨迹追踪方法: {ball_tracking_method}")
        
        # 初始化姿态估计模块
        pose_method = self.config['pose_estimation']['method']
        if pose_method == 'yolo_pose':
            yolo_pose_config = self.config['pose_estimation']['yolo_pose']
            self.pose_estimator = YOLOPoseEstimator(
                model=yolo_pose_config.get('model', 'yolo11n-pose.pt'),
                conf_threshold=yolo_pose_config.get('conf_threshold', 0.5),
                keypoint_threshold=yolo_pose_config.get('keypoint_threshold', 0.5)
            )
        elif pose_method == 'rtmw':
            rtmw_config = self.config['pose_estimation']['rtmw']
            self.pose_estimator = RTMWPoseEstimator(
                model_path=rtmw_config.get('model_path'),
                input_size=tuple(rtmw_config.get('input_size', [256, 192]))
            )
        else:
            raise ValueError(f"未知的姿态估计方法: {pose_method}")
        
        # 初始化坐标统一模块
        coord_config = self.config['coordinate_unify']
        self.coord_unifier = CoordinateUnifier(
            court_width=coord_config.get('court_width', 6.1),
            court_length=coord_config.get('court_length', 13.4),
            camera_matrix=None,  # 可以后续添加相机标定
            dist_coeffs=None
        )
        
        # 初始化信息整合模块
        info_config = self.config['info_integration']
        video_config = self.config.get('video', {})
        self.info_integrator = InfoIntegrator(
            ball_speed_threshold=info_config.get('ball_speed_threshold', 5.0),
            hit_detection_radius=info_config.get('hit_detection_radius', 0.5),
            fps=video_config.get('fps', 30.0)
        )
        
        # 初始化球种识别模块
        shot_method = self.config['shot_classification']['method']
        if shot_method == 'rule_based':
            rule_config = self.config['shot_classification']['rule_based']
            self.shot_classifier = RuleBasedClassifier(
                speed_thresholds=rule_config.get('speed_thresholds'),
                angle_thresholds=rule_config.get('angle_thresholds')
            )
        elif shot_method == 'classifier':
            classifier_config = self.config['shot_classification']['classifier']
            self.shot_classifier = MLShotClassifier(
                model_path=classifier_config.get('model_path')
            )
        else:
            raise ValueError(f"未知的球种识别方法: {shot_method}")
    
    def process_video(self, video_path: str, output_dir: str, 
                     visualize: bool = True) -> dict:
        """
        处理视频
        
        Args:
            video_path: 视频路径
            output_dir: 输出目录
            visualize: 是否生成可视化视频
            
        Returns:
            处理结果字典
        """
        # 创建输出目录
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 加载视频
        cap, video_info = load_video(video_path)
        fps = video_info['fps']
        width = video_info['width']
        height = video_info['height']
        
        print(f"视频信息: {width}x{height}, {fps} FPS, {video_info['frame_count']} 帧")
        
        # 初始化结果存储
        results = {
            'ball_trajectory': [],
            'player1_poses': [],
            'player2_poses': [],
            'hit_events': [],
            'shot_classifications': []
        }
        
        # 可视化帧列表
        vis_frames = []
        
        # 处理每一帧
        frame_idx = 0
        for frame_idx, frame in frame_generator(video_path):
            if frame_idx % 30 == 0:
                print(f"处理帧 {frame_idx}/{video_info['frame_count']}")
            
            # 1. 球轨迹追踪
            ball_result = self.ball_tracker.track(frame)
            ball_pixel = None
            ball_world = None
            if ball_result:
                x, y, conf = ball_result
                ball_pixel = (x, y)
                ball_world = self.coord_unifier.unify_ball_coordinate(
                    ball_pixel, (width, height)
                )
                results['ball_trajectory'].append({
                    'frame': frame_idx,
                    'pixel': ball_pixel,
                    'world': ball_world,
                    'confidence': conf
                })
            
            # 2. 姿态估计
            if hasattr(self.pose_estimator, 'estimate_top2'):
                player1_kpts, player2_kpts = self.pose_estimator.estimate_top2(frame)
            else:
                poses = self.pose_estimator.estimate(frame)
                player1_kpts = poses[0] if len(poses) > 0 else None
                player2_kpts = poses[1] if len(poses) > 1 else None
            
            # 统一坐标
            player1_kpts_unified = None
            player2_kpts_unified = None
            if player1_kpts is not None:
                player1_kpts_unified = self.coord_unifier.unify_pose_coordinates(
                    player1_kpts, (width, height)
                )
                results['player1_poses'].append({
                    'frame': frame_idx,
                    'keypoints': player1_kpts_unified.tolist()
                })
            
            if player2_kpts is not None:
                player2_kpts_unified = self.coord_unifier.unify_pose_coordinates(
                    player2_kpts, (width, height)
                )
                results['player2_poses'].append({
                    'frame': frame_idx,
                    'keypoints': player2_kpts_unified.tolist()
                })
            
            # 3. 信息整合
            hit_info = self.info_integrator.integrate(
                frame_idx, ball_world, player1_kpts_unified, player2_kpts_unified
            )
            
            if hit_info:
                results['hit_events'].append({
                    'frame': hit_info.frame_idx,
                    'hitter': hit_info.hitter,
                    'ball_position': hit_info.ball_position,
                    'ball_velocity': hit_info.ball_velocity,
                    'ball_direction': hit_info.ball_direction
                })
                
                # 4. 球种识别
                shot_info = self.shot_classifier.classify(
                    hit_info.ball_velocity,
                    hit_info.ball_direction,
                    hit_info.ball_position,
                    hit_info.hitter,
                    player1_kpts_unified if hit_info.hitter == 0 else player2_kpts_unified
                )
                
                if hasattr(shot_info, 'shot_type'):
                    results['shot_classifications'].append({
                        'frame': hit_info.frame_idx,
                        'shot_type': shot_info.shot_type,
                        'confidence': shot_info.confidence
                    })
                else:
                    shot_type, confidence = shot_info
                    results['shot_classifications'].append({
                        'frame': hit_info.frame_idx,
                        'shot_type': shot_type,
                        'confidence': confidence
                    })
            
            # 可视化
            if visualize:
                vis_frame = frame.copy()
                
                # 绘制球
                if ball_pixel:
                    vis_frame = draw_ball(vis_frame, ball_pixel, 
                                        results['ball_trajectory'][-1]['confidence'])
                
                # 绘制轨迹
                if len(results['ball_trajectory']) > 1:
                    trajectory = [r['pixel'] for r in results['ball_trajectory'][-30:]]
                    vis_frame = draw_trajectory(vis_frame, trajectory)
                
                # 绘制姿态
                if player1_kpts is not None:
                    vis_frame = draw_pose(vis_frame, player1_kpts)
                if player2_kpts is not None:
                    vis_frame = draw_pose(vis_frame, player2_kpts)
                
                # 绘制击球信息
                if hit_info:
                    text = f"Hit by Player {hit_info.hitter + 1}, Speed: {hit_info.ball_velocity:.1f} m/s"
                    vis_frame = draw_text(vis_frame, text, (10, 30))
                    
                    if results['shot_classifications']:
                        shot = results['shot_classifications'][-1]
                        shot_text = f"Shot: {shot['shot_type']} ({shot['confidence']:.2f})"
                        vis_frame = draw_text(vis_frame, shot_text, (10, 60))
                
                vis_frames.append(vis_frame)
        
        cap.release()
        
        # 保存可视化视频
        if visualize and vis_frames:
            vis_output_path = output_path / "visualization.mp4"
            from src.utils.video_utils import save_video
            save_video(vis_frames, str(vis_output_path), fps, (width, height))
            print(f"可视化视频已保存: {vis_output_path}")
        
        # 保存结果
        import json
        results_path = output_path / "results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"结果已保存: {results_path}")
        
        return results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Badminton AI Analysis System')
    parser.add_argument('--video', type=str, required=True,
                       help='输入视频路径')
    parser.add_argument('--output', type=str, default='data/results',
                       help='输出目录')
    parser.add_argument('--config', type=str, default='src/config/config.yaml',
                       help='配置文件路径')
    parser.add_argument('--no-vis', action='store_true',
                       help='不生成可视化视频')
    
    args = parser.parse_args()
    
    # 初始化系统
    print("初始化Badminton AI系统...")
    system = BadmintonAISystem(config_path=args.config)
    
    # 处理视频
    print(f"开始处理视频: {args.video}")
    results = system.process_video(
        args.video,
        args.output,
        visualize=not args.no_vis
    )
    
    # 打印统计信息
    print("\n处理完成！")
    print(f"总帧数: {len(results['ball_trajectory'])}")
    print(f"检测到击球事件: {len(results['hit_events'])}")
    print(f"球种分类: {len(results['shot_classifications'])}")


if __name__ == '__main__':
    main()
