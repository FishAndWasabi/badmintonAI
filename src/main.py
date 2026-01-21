"""Main entry point for Badminton AI Analysis System"""

import argparse
import cv2
import json
import numpy as np
import time
import yaml
from pathlib import Path
from typing import Optional, Callable, Union, Tuple, Dict

from src.utils.video_utils import load_video_stream, frame_generator
from src.utils.visualization import draw_ball, draw_pose, draw_trajectory, draw_text
from src.utils.performance import PerformanceMonitor
from src.utils.registry import registry
from src.coordinate_unify.unify import CoordinateUnifier
from src.info_integration.integration import InfoIntegrator

from src import *


class BadmintonAISystem:
    """Badminton AI Analysis System main class"""
    
    def __init__(self, config_path: str = "configs/config.yaml",
                 ball_tracking_method: Optional[str] = None,
                 pose_estimation_method: Optional[str] = None):
        """
        Initialize the system
        
        Args:
            config_path: Path to configuration file
            ball_tracking_method: Ball tracking method ("yolo_tracker" or "tracknetv3"). 
                                  If None, will use default from config or "tracknetv3"
            pose_estimation_method: Pose estimation method ("rtmw" or "yolo_pose").
                                    If None, will use default from config or "yolo_pose"
        """
        # Load configuration
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_file, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # Store method overrides
        self.ball_tracking_method = ball_tracking_method
        self.pose_estimation_method = pose_estimation_method
        
        # Initialize ball tracking module
        self.ball_tracker = self._init_ball_tracker()
        
        # Initialize pose estimation module
        self.pose_estimator = self._init_pose_estimator()
        
        # Initialize coordinate unification module
        coord_config = self.config['coordinate_unify']
        self.coord_unifier = CoordinateUnifier(
            court_width=coord_config.get('court_width', 6.1),
            court_length=coord_config.get('court_length', 13.4),
            camera_matrix=None,  # Can be added later with camera calibration
            dist_coeffs=None
        )
        
        # Initialize information integration module
        info_config = self.config['info_integration']
        video_config = self.config.get('video', {})
        self.info_integrator = InfoIntegrator(
            ball_speed_threshold=info_config.get('ball_speed_threshold', 5.0),
            hit_detection_radius=info_config.get('hit_detection_radius', 0.5),
            fps=video_config.get('fps', 30.0)
        )
        
        # Initialize shot classification module
        self.shot_classifier = self._init_shot_classifier()
    
    def _init_ball_tracker(self):
        """Initialize ball tracking module based on config (MMDetection/MMPose style)"""
        # Use method from parameter, or from config, or default to "tracknetv3"
        method = self.ball_tracking_method
        if method is None:
            method = self.config['ball_tracking'].get('method', 'tracknetv3')
        
        tracker_config = self.config['ball_tracking'].get(method, {}).copy()
        tracker_config['type'] = method
        
        # Handle special cases for input_size conversion
        if method == 'tracknetv3' and 'input_size' in tracker_config:
            tracker_config['input_size'] = tuple(tracker_config['input_size'])
        
        return registry.build_ball_tracker(tracker_config)
    
    def _init_pose_estimator(self):
        """Initialize pose estimation module based on config (MMDetection/MMPose style)"""
        # Use method from parameter, or from config, or default to "yolo_pose"
        method = self.pose_estimation_method
        if method is None:
            method = self.config['pose_estimation'].get('method', 'yolo_pose')
        
        estimator_config = self.config['pose_estimation'].get(method, {}).copy()
        estimator_config['type'] = method
        
        return registry.build_pose_estimator(estimator_config)
    
    def _init_shot_classifier(self):
        """Initialize shot classification module based on config (MMDetection/MMPose style)"""
        method = self.config['shot_classification']['method']
        classifier_config = self.config['shot_classification'].get(method, {}).copy()
        classifier_config['type'] = method
        
        return registry.build_shot_classifier(classifier_config)
    
    def _setup_video_stream(self, stream_url: Optional[str], test_mode: bool,
                           test_frames: int, verbose: bool) -> Tuple[Dict, bool]:
        """Setup video stream loading"""
        use_random_frames = False
        
        if test_mode and (test_frames == 0 or (test_frames > 0 and stream_url is None)):
            # Test mode: use random tensors
            width, height = 1280, 720
            fps = 30.0
            frame_count = test_frames if test_frames > 0 else 2
            video_info = {
                'fps': fps,
                'width': width,
                'height': height,
                'frame_count': frame_count
            }
            use_random_frames = True
            if verbose:
                print(f"Test mode: Using random tensors (size: {width}x{height}, frames: {frame_count})")
        elif test_mode and test_frames > 0 and stream_url:
            # Test mode: load limited frames from stream
            cap, video_info = load_video_stream(stream_url)
            video_info['frame_count'] = test_frames
            cap.release()
            if verbose:
                print(f"Test mode: Processing {test_frames} frames from stream")
                print(f"Stream info: {video_info['width']}x{video_info['height']}, {video_info['fps']} FPS")
        else:
            # Normal mode: load live stream
            if stream_url is None:
                raise ValueError("Stream URL required in normal mode")
            cap, video_info = load_video_stream(stream_url)
            cap.release()  # Release immediately, will reopen in frame_generator
            if verbose:
                frame_count_str = f"{video_info['frame_count']} frames" if video_info['frame_count'] else "live stream"
                print(f"Stream info: {video_info['width']}x{video_info['height']}, "
                      f"{video_info['fps']} FPS, {frame_count_str}")
                print("Starting real-time stream processing...")
        
        return video_info, use_random_frames
    
    def _init_visualization_writer(self, output_path: Path, visualize: bool,
                                  test_mode: bool, test_frames: int,
                                  fps: float, width: int, height: int):
        """Initialize visualization video writer"""
        if visualize and not (test_mode and test_frames == 0):
            vis_output_path = output_path / "visualization.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            return cv2.VideoWriter(str(vis_output_path), fourcc, fps, (width, height))
        return None
    
    def process_video(self, stream_url: Union[str, None], output_dir: str, 
                     visualize: bool = True, 
                     monitor_performance: bool = True,
                     num_gpus: int = 2,
                     verbose: bool = True,
                     progress_callback: Optional[Callable[[int], None]] = None,
                     test_mode: bool = False,
                     test_frames: int = 2) -> dict:
        """
        Process online video stream (RTSP, RTMP, HTTP, etc.)
        
        Args:
            stream_url: Video stream URL (e.g., rtsp://, rtmp://, http://, or camera index)
            output_dir: Output directory
            visualize: Whether to generate visualization video
            monitor_performance: Whether to monitor performance
            num_gpus: Number of GPUs
            verbose: Whether to output detailed information
            progress_callback: Progress callback function
            test_mode: Test mode, if True, process limited frames or use random tensors
            test_frames: Number of frames to process in test mode (0=use random tensor, >0=load specified frames)
            
        Returns:
            Processing results dictionary
        """
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize performance monitor
        perf_monitor = PerformanceMonitor(num_gpus=num_gpus) if monitor_performance else None
        
        # Load video stream or setup test mode
        video_info, use_random_frames = self._setup_video_stream(
            stream_url, test_mode, test_frames, verbose
        )
        width, height = video_info['width'], video_info['height']
        fps = video_info['fps']
        
        # Initialize results storage
        results = {
            'ball_trajectory': [],
            'player1_poses': [],  # For backward compatibility
            'player2_poses': [],  # For backward compatibility
            'all_poses': [],  # All detected people poses
            'hit_events': [],
            'shot_classifications': []
        }
        
        # Initialize visualization writer
        vis_writer = self._init_visualization_writer(
            output_path, visualize, test_mode, test_frames, fps, width, height
        )
        
        # Trajectory history for visualization
        trajectory_history = []
        
        # Process frames in real-time streaming
        if test_mode and (test_frames == 0 or (test_frames > 0 and stream_url is None)):
            # Use random tensors for testing (test_frames=0 or test_frames>0 and stream_url=None)
            for frame_idx in range(video_info['frame_count']):
                # Generate random RGB image (BGR format)
                frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
                
                if perf_monitor:
                    perf_monitor.start_frame()
                
                # Call progress callback (if provided)
                if progress_callback:
                    progress_callback(frame_idx)
                
                # Process frame (use random results, don't call actual model)
                if perf_monitor:
                    start_time = time.time()
                
                # 1. Ball tracking (using random results)
                ball_result = None
                if np.random.random() > 0.5:  # 50% probability of detecting ball
                    x = np.random.uniform(0, width)
                    y = np.random.uniform(0, height)
                    conf = np.random.uniform(0.7, 1.0)
                    ball_result = (x, y, conf)
                
                if perf_monitor:
                    elapsed = time.time() - start_time
                    perf_monitor.record_ball_tracking(elapsed)
                
                ball_pixel = None
                ball_world = None
                if ball_result:
                    x, y, conf = ball_result
                    ball_pixel = (x, y)
                    ball_world = self.coord_unifier.unify_ball_coordinate(
                        ball_pixel, (width, height)
                    )
                    current_velocity = self.info_integrator.get_current_ball_velocity()
                    
                    results['ball_trajectory'].append({
                        'frame': frame_idx,
                        'pixel': ball_pixel,
                        'world': ball_world,
                        'confidence': conf,
                        'velocity': current_velocity
                    })
                    trajectory_history.append(ball_pixel)
                    if len(trajectory_history) > 30:
                        trajectory_history.pop(0)
                
                # 2. Pose estimation (using random results)
                if perf_monitor:
                    start_time = time.time()
                
                # Generate random keypoints
                player1_kpts = np.random.uniform(0, 1, (17, 3))
                player1_kpts[:, :2] *= np.array([width, height])
                player1_kpts[:, 2] = np.random.uniform(0.7, 1.0, 17)
                
                player2_kpts = np.random.uniform(0, 1, (17, 3))
                player2_kpts[:, :2] *= np.array([width, height])
                player2_kpts[:, 2] = np.random.uniform(0.7, 1.0, 17)
                
                if perf_monitor:
                    elapsed = time.time() - start_time
                    perf_monitor.record_pose_estimation(elapsed)
                
                # Unify coordinates for all detected people
                all_poses_list = [player1_kpts, player2_kpts]
                all_poses_unified = []
                for pose in all_poses_list:
                    if pose is not None:
                        pose_unified = self.coord_unifier.unify_pose_coordinates(
                            pose, (width, height)
                        )
                        all_poses_unified.append(pose_unified)
                
                # Store all poses
                if len(all_poses_unified) > 0:
                    frame_poses = []
                    for pose_unified in all_poses_unified:
                        frame_poses.append({
                            'keypoints': pose_unified.tolist()
                        })
                    results['all_poses'] = results.get('all_poses', [])
                    results['all_poses'].append({
                        'frame': frame_idx,
                        'people': frame_poses
                    })
                    
                    # For backward compatibility: store first two as player1/player2
                    player1_kpts_unified = all_poses_unified[0] if len(all_poses_unified) > 0 else None
                    player2_kpts_unified = all_poses_unified[1] if len(all_poses_unified) > 1 else None
                    
                    if player1_kpts_unified is not None:
                        results['player1_poses'].append({
                            'frame': frame_idx,
                            'keypoints': player1_kpts_unified.tolist()
                        })
                    
                    if player2_kpts_unified is not None:
                        results['player2_poses'].append({
                            'frame': frame_idx,
                            'keypoints': player2_kpts_unified.tolist()
                        })
                else:
                    player1_kpts_unified = None
                    player2_kpts_unified = None
                
                # 3. Information integration
                if perf_monitor:
                    start_time = time.time()
                
                if ball_world is not None:
                    hit_info = self.info_integrator.integrate(
                        frame_idx,
                        ball_world,
                        player1_kpts_unified,
                        player2_kpts_unified
                    )
                    
                    if hit_info is not None:
                        results['hit_events'].append({
                            'frame': frame_idx,
                            'hitter': hit_info.hitter,
                            'position': hit_info.ball_position,
                            'ball_speed': hit_info.ball_velocity
                        })
                
                if perf_monitor:
                    elapsed = time.time() - start_time
                    perf_monitor.record_integration(elapsed)
                
                # End inference timing (before visualization)
                if perf_monitor:
                    perf_monitor.end_inference()
                
                # 4. Visualization (skip in test mode)
                if visualize and vis_writer is not None:
                    vis_frame = frame.copy()
                    if ball_pixel:
                        vis_frame = draw_ball(vis_frame, ball_pixel)
                    # Draw poses for all detected people (use original keypoints before unification)
                    if player1_kpts is not None:
                        vis_frame = draw_pose(vis_frame, player1_kpts)
                    if player2_kpts is not None:
                        vis_frame = draw_pose(vis_frame, player2_kpts)
                    vis_writer.write(vis_frame)
                
                if perf_monitor:
                    perf_monitor.end_frame()
                    if frame_idx % 10 == 0:
                        perf_monitor.update_gpu_memory()
        else:
            # Normal mode or test mode (load limited frames from stream)
            frame_count = 0
            for frame_idx, frame in frame_generator(stream_url):
                if test_mode and test_frames > 0 and frame_count >= test_frames:
                    break
                frame_count += 1
                
                if perf_monitor:
                    perf_monitor.start_frame()
                
                # Call progress callback (if provided)
                if progress_callback:
                    progress_callback(frame_idx)
                
                # 1. Ball tracking
                if perf_monitor:
                    start_time = time.time()
                
                ball_result = self.ball_tracker.track(frame)
                
                if perf_monitor:
                    elapsed = time.time() - start_time
                    perf_monitor.record_ball_tracking(elapsed)
                
                ball_pixel = None
                ball_world = None
                if ball_result:
                    x, y, conf = ball_result
                    ball_pixel = (x, y)
                    ball_world = self.coord_unifier.unify_ball_coordinate(
                        ball_pixel, (width, height)
                    )
                    # Get current ball velocity
                    current_velocity = self.info_integrator.get_current_ball_velocity()
                    
                    results['ball_trajectory'].append({
                        'frame': frame_idx,
                        'pixel': ball_pixel,
                        'world': ball_world,
                        'confidence': conf,
                        'velocity': current_velocity  # Add ball velocity information
                    })
                    trajectory_history.append(ball_pixel)
                    # Keep only last 30 frames of trajectory
                    if len(trajectory_history) > 30:
                        trajectory_history.pop(0)
                
                # 2. Pose estimation (detect all people)
                if perf_monitor:
                    start_time = time.time()
                
                # Get all detected poses
                all_poses = self.pose_estimator.estimate(frame)
                
                if perf_monitor:
                    elapsed = time.time() - start_time
                    perf_monitor.record_pose_estimation(elapsed)
                
                # Unify coordinates for all detected people
                all_poses_unified = []
                for pose in all_poses:
                    if pose is not None:
                        pose_unified = self.coord_unifier.unify_pose_coordinates(
                            pose, (width, height)
                        )
                        all_poses_unified.append(pose_unified)
                
                # Store all poses (for backward compatibility, also store first two as player1/player2)
                if len(all_poses_unified) > 0:
                    # Store all poses
                    frame_poses = []
                    for pose_unified in all_poses_unified:
                        frame_poses.append({
                            'keypoints': pose_unified.tolist()
                        })
                    results['all_poses'] = results.get('all_poses', [])
                    results['all_poses'].append({
                        'frame': frame_idx,
                        'people': frame_poses
                    })
                    
                    # For backward compatibility: store first two as player1/player2
                    player1_kpts_unified = all_poses_unified[0] if len(all_poses_unified) > 0 else None
                    player2_kpts_unified = all_poses_unified[1] if len(all_poses_unified) > 1 else None
                    
                    if player1_kpts_unified is not None:
                        results['player1_poses'].append({
                            'frame': frame_idx,
                            'keypoints': player1_kpts_unified.tolist()
                        })
                    
                    if player2_kpts_unified is not None:
                        results['player2_poses'].append({
                            'frame': frame_idx,
                            'keypoints': player2_kpts_unified.tolist()
                        })
                else:
                    player1_kpts_unified = None
                    player2_kpts_unified = None
                
                # 3. Information integration
                if perf_monitor:
                    start_time = time.time()
                
                hit_info = self.info_integrator.integrate(
                    frame_idx, ball_world, player1_kpts_unified, player2_kpts_unified
                )
                
                if perf_monitor:
                    elapsed = time.time() - start_time
                    perf_monitor.record_integration(elapsed)
                
                if hit_info:
                    results['hit_events'].append({
                        'frame': hit_info.frame_idx,
                        'hitter': hit_info.hitter,
                        'ball_position': hit_info.ball_position,
                        'ball_velocity': hit_info.ball_velocity,
                        'ball_direction': hit_info.ball_direction
                    })
                    
                    # 4. Shot classification
                    # Select corresponding keypoints based on hitter
                    player_kpts = None
                    if hit_info.hitter is not None:
                        player_kpts = player1_kpts_unified if hit_info.hitter == 0 else player2_kpts_unified
                    
                    shot_info = self.shot_classifier.classify(
                        hit_info.ball_velocity,
                        hit_info.ball_direction,
                        hit_info.ball_position,
                        hit_info.hitter,
                        player_kpts
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
                
                # End inference timing (before visualization)
                if perf_monitor:
                    perf_monitor.end_inference()
                
                # Visualization
                if visualize and vis_writer is not None:
                    vis_frame = frame.copy()
                    
                    # Draw ball
                    if ball_pixel:
                        conf = results['ball_trajectory'][-1]['confidence'] if results['ball_trajectory'] else 0.5
                        vis_frame = draw_ball(vis_frame, ball_pixel, conf)
                    
                    # Draw trajectory
                    if len(trajectory_history) > 1:
                        vis_frame = draw_trajectory(vis_frame, trajectory_history)
                    
                    # Draw poses for all detected people
                    # Use original keypoints (before coordinate unification) for visualization
                    if len(all_poses) > 0:
                        for pose in all_poses:
                            if pose is not None:
                                vis_frame = draw_pose(vis_frame, pose)
                    
                    # Get current ball velocity (continuous display)
                    current_velocity = self.info_integrator.get_average_ball_velocity(window_size=3)
                    if current_velocity > 0.1:  # Only show meaningful ball speed
                        speed_text = f"Ball Speed: {current_velocity:.1f} m/s"
                        vis_frame = draw_text(vis_frame, speed_text, (10, 30), color=(0, 255, 255))
                    
                    # Draw hit information
                    if hit_info:
                        y_offset = 60 if current_velocity > 0.1 else 30
                        if hit_info.hitter is not None:
                            text = f"Hit by Player {hit_info.hitter + 1}, Speed: {hit_info.ball_velocity:.1f} m/s"
                        else:
                            text = f"Hit detected, Speed: {hit_info.ball_velocity:.1f} m/s"
                        vis_frame = draw_text(vis_frame, text, (10, y_offset), color=(255, 255, 0))
                        
                        if results['shot_classifications']:
                            shot = results['shot_classifications'][-1]
                            shot_text = f"Shot: {shot['shot_type']} ({shot['confidence']:.2f})"
                            vis_frame = draw_text(vis_frame, shot_text, (10, y_offset + 30))
                    
                    # Write frame (real-time write, no caching)
                    vis_writer.write(vis_frame)
                
                # Update performance monitoring
                if perf_monitor:
                    perf_monitor.end_frame()
                    if frame_idx % 30 == 0:  # Update GPU memory every 30 frames
                        perf_monitor.update_gpu_memory()
        
        if vis_writer:
            vis_writer.release()
        
        # Test mode (test_frames=0): only inference, no results saved
        if test_mode and test_frames == 0:
            if verbose:
                print("Test mode: Inference only, no results saved")
            # Only return basic performance stats (if monitoring performance)
            if perf_monitor:
                return {
                    'performance_stats': perf_monitor.stats.get_stats(),
                    'gpu_memory_info': perf_monitor.get_gpu_memory(),
                    'test_mode': True
                }
            return {'test_mode': True}
        
        # Normal mode: save results and visualization
        if visualize:
            print(f"Visualization video saved: {output_path / 'visualization.mp4'}")
        
        # Save results
        results_path = output_path / "results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        if verbose:
            print(f"Results saved: {results_path}")
        
        # Print performance statistics
        if perf_monitor:
            if verbose:
                perf_monitor.print_stats()
            perf_stats_path = output_path / "performance_stats.json"
            perf_monitor.save_stats(str(perf_stats_path))
            # Add performance statistics to results
            results['performance_stats'] = perf_monitor.stats.get_stats()
            results['gpu_memory_info'] = perf_monitor.get_gpu_memory()
        
        return results


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Badminton AI Analysis System - Online Stream Processing')
    parser.add_argument('--stream', type=str, required=True,
                       help='Video stream URL (e.g., rtsp://, rtmp://, http://, or camera index like 0)')
    parser.add_argument('--output', type=str, default='data/results',
                       help='Output directory')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Configuration file path')
    parser.add_argument('--ball-tracking-method', type=str, choices=['yolo_tracker', 'tracknetv3'],
                       default=None, help='Ball tracking method (yolo_tracker or tracknetv3)')
    parser.add_argument('--pose-estimation-method', type=str, choices=['rtmw', 'yolo_pose'],
                       default=None, help='Pose estimation method (rtmw or yolo_pose)')
    parser.add_argument('--no-vis', action='store_true',
                       help='Do not generate visualization video')
    parser.add_argument('--no-perf', action='store_true',
                       help='Do not monitor performance')
    parser.add_argument('--num-gpus', type=int, default=2,
                       help='Number of GPUs (for performance monitoring)')
    parser.add_argument('--test-mode', action='store_true',
                       help='Test mode: use random tensors or process limited frames')
    parser.add_argument('--test-frames', type=int, default=2,
                       help='Number of frames to process in test mode (0=use random tensor, >0=load specified frames)')
    
    args = parser.parse_args()
    
    # Initialize system
    print("Initializing Badminton AI system...")
    system = BadmintonAISystem(
        config_path=args.config,
        ball_tracking_method=args.ball_tracking_method,
        pose_estimation_method=args.pose_estimation_method
    )
    
    # Process video stream
    print(f"Starting stream processing: {args.stream}")
    results = system.process_video(
        args.stream,
        args.output,
        visualize=not args.no_vis,
        monitor_performance=not args.no_perf,
        num_gpus=args.num_gpus,
        test_mode=args.test_mode,
        test_frames=args.test_frames
    )
    
    # Print statistics
    print("\nProcessing completed!")
    print(f"Total frames: {len(results.get('ball_trajectory', []))}")
    print(f"Hit events detected: {len(results.get('hit_events', []))}")
    print(f"Shot classifications: {len(results.get('shot_classifications', []))}")


if __name__ == '__main__':
    main()
