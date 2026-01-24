#!/usr/bin/env python3
"""
Flask API for Badminton AI Analysis System
Multi-process architecture: Reader -> Processor -> Streamer
"""

import sys
import json
import time
import subprocess
import multiprocessing
import threading
import queue
import cv2
from pathlib import Path
from typing import List, Dict
from datetime import datetime

from flask import Flask, request, jsonify
from flask_cors import CORS

from src.main import BadmintonAISystem
from src.utils.video_utils import load_video_stream, frame_generator
from src.utils.visualization import draw_ball, draw_pose, draw_trajectory, draw_text
from src.utils.performance import PerformanceMonitor


app = Flask(__name__)
CORS(app)

# Store active process information (thread-safe, Flask is multi-threaded)
active_processes: Dict[str, Dict] = {}
process_lock = threading.Lock()


def frame_reader_process(stream_url: str, frame_queue: multiprocessing.Queue, 
                        stop_event: multiprocessing.Event, task_id: str):
    """
    Process 1: Read frames from stream and put into queue
    
    Args:
        stream_url: Input stream URL
        frame_queue: Queue to put raw frames
        stop_event: Event to signal stop
        task_id: Task ID for logging
    """
    print(f"[Task {task_id}] Reader: Starting frame reading from {stream_url}")
    
    cap, video_info = load_video_stream(stream_url)
    fps = video_info.get('fps', 30.0)
    width = video_info.get('width', 1280)
    height = video_info.get('height', 720)
    
    try:
        # Put video info into queue as first item
        frame_queue.put(('info', {'fps': fps, 'width': width, 'height': height}))
        
        frame_idx = 0
        for frame_idx, frame in frame_generator(stream_url):
            if stop_event.is_set():
                break
            
            # Put frame into queue (non-blocking, drop old frames if full to prevent lag)
            # Optimized: try to put directly first, only drop frames if necessary
            # IMPORTANT: Never drop 'info' or 'end' messages, only drop 'frame' messages
            try:
                frame_queue.put(('frame', (frame_idx, frame)), block=False)
            except queue.Full:
                # Queue full, drop oldest FRAME items only (never drop info/end)
                dropped = 0
                non_frame_items = []  # Store non-frame items to put back
                while frame_queue.full() and dropped < 5:  # Limit drops to prevent infinite loop
                    try:
                        item = frame_queue.get_nowait()
                        if item[0] == 'frame':
                            dropped += 1  # Only drop frame items
                        else:
                            # Keep non-frame items (info, end) to put back
                            non_frame_items.append(item)
                    except queue.Empty:
                        break
                
                # Put back non-frame items first (they have priority)
                for item in non_frame_items:
                    try:
                        frame_queue.put(item, block=False)
                    except queue.Full:
                        # If still full after putting back non-frame items, 
                        # we need to drop more frames
                        break
                
                # Try to put the new frame again
                try:
                    frame_queue.put(('frame', (frame_idx, frame)), block=False)
                except queue.Full:
                    pass  # Skip this frame if still can't put
        
        frame_queue.put(('end', None))
        print(f"[Task {task_id}] Reader: Finished reading {frame_idx} frames")
    finally:
        # Cleanup: release video capture
        if cap is not None:
            cap.release()


def frame_processor_process(stream_url: str, frame_queue: multiprocessing.Queue,
                           processed_queue: multiprocessing.Queue,
                           stop_event: multiprocessing.Event, task_id: str,
                           config_path: str, temp_dir: str,
                           ball_tracking_method: str = None,
                           pose_estimation_method: str = None):
    """
    Process 2: Process frames (ball tracking, pose estimation, visualization)
    
    Args:
        stream_url: Input stream URL (for reference)
        frame_queue: Queue to get raw frames
        processed_queue: Queue to put processed frames
        stop_event: Event to signal stop
        task_id: Task ID for logging
        config_path: Config file path
        temp_dir: Temporary directory
        ball_tracking_method: Ball tracking method override
        pose_estimation_method: Pose estimation method override
    """
    print(f"[Task {task_id}] Processor: Starting frame processing")
    
    # Initialize system
    system = BadmintonAISystem(
        config_path=config_path,
        ball_tracking_method=ball_tracking_method,
        pose_estimation_method=pose_estimation_method
    )
    
    # Initialize performance monitor
    perf_monitor = PerformanceMonitor(num_gpus=1)
    
    # Get video info from queue (wait for info message)
    # The info message should be the first message, but handle edge cases
    max_retries = 100  # Prevent infinite loop
    retry_count = 0
    item = None
    while retry_count < max_retries:
        item = frame_queue.get()
        if item[0] == 'info':
            break
        # If we got a frame or other message before info, this is unexpected
        # Put it back and wait briefly
        print(f"[Task {task_id}] Processor: Warning - got {item[0]} before info (retry {retry_count}), waiting for info...")
        try:
            frame_queue.put(item, block=False)  # Put it back
        except queue.Full:
            # If queue is full, we can't put it back, so we'll process it later
            pass
        time.sleep(0.01)
        retry_count += 1
    
    if item is None or item[0] != 'info':
        print(f"[Task {task_id}] Processor: Error - failed to get info message after {max_retries} retries")
        processed_queue.put(('end', None))
        return
    
    video_info = item[1]
    fps = video_info['fps']
    width = video_info['width']
    height = video_info['height']
    
    # Initialize visualization writer
    task_temp_dir = Path(temp_dir) / task_id
    task_temp_dir.mkdir(parents=True, exist_ok=True)
    output_video_path = task_temp_dir / "visualization.mp4"
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vis_writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
    
    if not vis_writer.isOpened():
        print(f"[Task {task_id}] Processor: Warning - Failed to open video writer")
        vis_writer = None
    
    # Put video info to processed queue
    processed_queue.put(('info', video_info))
    
    # Initialize tracking data
    trajectory_history = []
    results = {
        'ball_trajectory': [],
        'player1_poses': [],  # For backward compatibility
        'player2_poses': [],  # For backward compatibility
        'all_poses': [],  # All detected people poses
        'hit_events': [],
        'shot_classifications': []
    }
    
    # Process frames
    frame_count = 0
    while not stop_event.is_set():
        # Get frame from queue (non-blocking, wait if empty)
        if frame_queue.empty():
            time.sleep(0.01)
            continue
        item = frame_queue.get()
        if item[0] == 'end':
            break
        elif item[0] != 'frame':
            continue  
        frame_idx, frame = item[1]
        frame_count += 1
        
        # Start performance monitoring
        perf_monitor.start_frame()
        inference_start_time = time.time()
        
        # 1. Ball tracking
        ball_track_start = time.time()
        ball_result = system.ball_tracker.track(frame)
        ball_track_time = time.time() - ball_track_start
        perf_monitor.record_ball_tracking(ball_track_time)
        
        ball_pixel = None
        ball_world = None
        if ball_result:
            x, y, conf = ball_result
            ball_pixel = (x, y)
            ball_world = system.coord_unifier.unify_ball_coordinate(
                ball_pixel, (width, height)
            )
            current_velocity = system.info_integrator.get_current_ball_velocity()
            
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
        
        # 2. Pose estimation (detect all people)
        pose_start = time.time()
        all_poses = system.pose_estimator.estimate(frame)
        pose_time = time.time() - pose_start
        perf_monitor.record_pose_estimation(pose_time)
        
        # Unify coordinates for all detected people
        all_poses_unified = []
        for pose in all_poses:
            if pose is not None:
                pose_unified = system.coord_unifier.unify_pose_coordinates(
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
            results['all_poses'].append({
                'frame': frame_idx,
                'people': frame_poses
            })
            
            # For backward compatibility: store first two as player1/player2
            player1_kpts_unified = all_poses_unified[0]
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
        integration_start = time.time()
        hit_info = system.info_integrator.integrate(
            frame_idx, ball_world, player1_kpts_unified, player2_kpts_unified
        )
        integration_time = time.time() - integration_start
        perf_monitor.record_integration(integration_time)
        
        # End inference timing (before visualization)
        inference_time = time.time() - inference_start_time
        perf_monitor.end_inference()
        
        if hit_info:
            results['hit_events'].append({
                'frame': hit_info.frame_idx,
                'hitter': hit_info.hitter,
                'ball_position': hit_info.ball_position,
                'ball_velocity': hit_info.ball_velocity,
                'ball_direction': hit_info.ball_direction
            })
            
            # 4. Shot classification
            player_kpts = None
            if hit_info.hitter is not None:
                player_kpts = player1_kpts_unified if hit_info.hitter == 0 else player2_kpts_unified
            
            shot_info = system.shot_classifier.classify(
                hit_info.ball_velocity,
                hit_info.ball_direction,
                hit_info.ball_position,
                hit_info.hitter,
                player_kpts
            )
            
            # Handle both object and tuple return types
            if hasattr(shot_info, 'shot_type'):
                shot_type, confidence = shot_info.shot_type, shot_info.confidence
            else:
                shot_type, confidence = shot_info
            
            results['shot_classifications'].append({
                'frame': hit_info.frame_idx,
                'shot_type': shot_type,
                'confidence': confidence
            })
            
            # Output prediction results to terminal
            hitter_str = f"Player {hit_info.hitter + 1}" if hit_info.hitter is not None else "Unknown"
            ball_pos_str = f"({hit_info.ball_position[0]:.2f}, {hit_info.ball_position[1]:.2f}, {hit_info.ball_position[2]:.2f})"
            ball_dir_str = f"({hit_info.ball_direction[0]:.2f}, {hit_info.ball_direction[1]:.2f}, {hit_info.ball_direction[2]:.2f})"
            
            print(f"\n[Task {task_id}] ========== PREDICTION RESULT ==========")
            print(f"[Task {task_id}] Frame: {hit_info.frame_idx}")
            print(f"[Task {task_id}] Hit Event Detected:")
            print(f"[Task {task_id}]   - Hitter: {hitter_str}")
            print(f"[Task {task_id}]   - Ball Position (world): {ball_pos_str} m")
            print(f"[Task {task_id}]   - Ball Velocity: {hit_info.ball_velocity:.2f} m/s")
            print(f"[Task {task_id}]   - Ball Direction: {ball_dir_str}")
            print(f"[Task {task_id}] Shot Classification:")
            print(f"[Task {task_id}]   - Shot Type: {shot_type}")
            print(f"[Task {task_id}]   - Confidence: {confidence:.2%}")
            print(f"[Task {task_id}] =========================================\n")
        
        # 5. Visualization
        # Use in-place operations where possible to reduce memory copies
        vis_frame = frame.copy()
        
        # Draw ball
        if ball_pixel:
            conf = ball_result[2] if ball_result else 0.5
            vis_frame = draw_ball(vis_frame, ball_pixel, conf)
            
            # Output ball tracking info periodically (every 30 frames)
            if frame_idx % 30 == 0 and ball_world is not None:
                print(f"[Task {task_id}] Frame {frame_idx}: Ball detected at pixel ({ball_pixel[0]:.1f}, {ball_pixel[1]:.1f}), "
                      f"world ({ball_world[0]:.2f}, {ball_world[1]:.2f}, {ball_world[2]:.2f}) m, "
                      f"confidence: {conf:.2f}, velocity: {current_velocity:.2f} m/s")
        
        # Draw trajectory
        if len(trajectory_history) > 1:
            vis_frame = draw_trajectory(vis_frame, trajectory_history)
        
        # Draw poses for all detected people
        if len(all_poses) > 0:
            for pose in all_poses:
                if pose is not None:
                    vis_frame = draw_pose(vis_frame, pose)
        
        # Draw text information
        y_offset = 30
        current_velocity = system.info_integrator.get_average_ball_velocity(window_size=3)
        if current_velocity > 0.1:
            speed_text = f"Ball Speed: {current_velocity:.1f} m/s"
            vis_frame = draw_text(vis_frame, speed_text, (10, y_offset), color=(0, 255, 255))
            y_offset += 30
        
        # Draw inference latency
        inference_ms = inference_time * 1000
        fps_actual = 1.0 / inference_time if inference_time > 0 else 0
        latency_text = f"Inference: {inference_ms:.1f}ms ({fps_actual:.1f} FPS)"
        vis_frame = draw_text(vis_frame, latency_text, (10, y_offset), color=(0, 255, 0))
        y_offset += 30
        
        if hit_info:
            if hit_info.hitter is not None:
                text = f"Hit by Player {hit_info.hitter + 1}, Speed: {hit_info.ball_velocity:.1f} m/s"
            else:
                text = f"Hit detected, Speed: {hit_info.ball_velocity:.1f} m/s"
            vis_frame = draw_text(vis_frame, text, (10, y_offset), color=(255, 255, 0))
            y_offset += 30
            
            if results['shot_classifications']:
                shot = results['shot_classifications'][-1]
                shot_text = f"Shot: {shot['shot_type']} ({shot['confidence']:.2f})"
                vis_frame = draw_text(vis_frame, shot_text, (10, y_offset), color=(255, 255, 255))
        
        # End frame timing (after visualization)
        perf_monitor.end_frame()
        
        # Write to video file
        if vis_writer is not None:
            vis_writer.write(vis_frame)
        
        # Put processed frame to queue (non-blocking, drop old frames if full to prevent lag)
        # Optimized: try to put directly first, only drop frames if necessary
        # IMPORTANT: Never drop 'info' or 'end' messages, only drop 'frame' messages
        try:
            processed_queue.put(('frame', vis_frame), block=False)
        except queue.Full:
            # Queue full, drop oldest FRAME items only (never drop info/end)
            dropped = 0
            non_frame_items = []  # Store non-frame items to put back
            while processed_queue.full() and dropped < 5:  # Limit drops to prevent infinite loop
                try:
                    item = processed_queue.get_nowait()
                    if item[0] == 'frame':
                        dropped += 1  # Only drop frame items
                    else:
                        # Keep non-frame items (info, end) to put back
                        non_frame_items.append(item)
                except queue.Empty:
                    break
            
            # Put back non-frame items first (they have priority)
            for item in non_frame_items:
                try:
                    processed_queue.put(item, block=False)
                except queue.Full:
                    # If still full after putting back non-frame items,
                    # we need to drop more frames
                    break
            
            # Try to put the new frame again
            try:
                processed_queue.put(('frame', vis_frame), block=False)
            except queue.Full:
                pass  # Skip this frame if still can't put
    
    # Cleanup resources
    if vis_writer is not None:
        vis_writer.release()
    
    # Save results
    results_path = task_temp_dir / "results.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Save performance statistics
    perf_stats_path = task_temp_dir / "performance_stats.json"
    perf_monitor.save_stats(str(perf_stats_path))
    
    # Print final summary to terminal
    print(f"\n[Task {task_id}] ========== PROCESSING SUMMARY ==========")
    print(f"[Task {task_id}] Total frames processed: {frame_count}")
    print(f"[Task {task_id}] Ball trajectory points: {len(results.get('ball_trajectory', []))}")
    print(f"[Task {task_id}] Hit events detected: {len(results.get('hit_events', []))}")
    print(f"[Task {task_id}] Shot classifications: {len(results.get('shot_classifications', []))}")
    print(f"[Task {task_id}] Player poses detected: {len(results.get('all_poses', []))}")
    
    # Print shot classification summary
    if results.get('shot_classifications'):
        print(f"\n[Task {task_id}] Shot Classification Summary:")
        shot_types = {}
        for shot in results['shot_classifications']:
            shot_type = shot['shot_type']
            shot_types[shot_type] = shot_types.get(shot_type, 0) + 1
        for shot_type, count in sorted(shot_types.items(), key=lambda x: x[1], reverse=True):
            print(f"[Task {task_id}]   - {shot_type}: {count} times")
    
    print(f"[Task {task_id}] Results saved to: {results_path}")
    print(f"[Task {task_id}] Performance stats saved to: {perf_stats_path}")
    print(f"[Task {task_id}] =========================================\n")
    
    # Print performance summary
    print(f"[Task {task_id}] Processor: Finished processing {frame_count} frames")
    perf_monitor.print_stats()
    
    processed_queue.put(('end', None))


def stream_writer_process(processed_queue: multiprocessing.Queue,
                         output_stream_url: str, stop_event: multiprocessing.Event,
                         task_id: str, temp_dir: str):
    """
    Process 3: Stream processed frames to RTMP server via ffmpeg pipe
    
    Args:
        processed_queue: Queue to get processed frames
        output_stream_url: RTMP output stream URL
        stop_event: Event to signal stop
        task_id: Task ID for logging
        temp_dir: Temporary directory (for reference)
    """
    print(f"[Task {task_id}] Streamer: Starting stream to {output_stream_url}")
    
    # Get video info from queue (wait for info message)
    # The info message should be the first message, but handle edge cases
    max_retries = 100  # Prevent infinite loop
    retry_count = 0
    item = None
    while retry_count < max_retries:
        item = processed_queue.get()
        if item[0] == 'info':
            break
        # If we got a frame or other message before info, this is unexpected
        # Put it back and wait briefly
        print(f"[Task {task_id}] Streamer: Warning - got {item[0]} before info (retry {retry_count}), waiting for info...")
        try:
            processed_queue.put(item, block=False)  # Put it back
        except queue.Full:
            # If queue is full, we can't put it back, so we'll process it later
            pass
        time.sleep(0.01)
        retry_count += 1
    
    if item is None or item[0] != 'info':
        print(f"[Task {task_id}] Streamer: Error - failed to get info message after {max_retries} retries")
        return
    
    video_info = item[1]
    fps = video_info['fps']
    width = video_info['width']
    height = video_info['height']
    
    # Build ffmpeg command for streaming from pipe
    # Important: Set both input and output frame rate to match original video
    cmd = [
        'ffmpeg',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', f'{width}x{height}',
        '-pix_fmt', 'bgr24',
        '-r', str(fps),  # Input frame rate
        '-i', '-',  # Read from stdin
        '-c:v', 'libx264',
        '-preset', 'veryfast',
        '-r', str(fps),  # Output frame rate (must match input)
        '-b:v', '2000k',
        '-maxrate', '2000k',
        '-bufsize', '4000k',
        '-pix_fmt', 'yuv420p',
        '-f', 'flv',
        '-flvflags', 'no_duration_filesize',
        output_stream_url
    ]
    
    print(f"[Task {task_id}] Streamer: Starting ffmpeg process (pipe input)")
    print(f"[Task {task_id}] Streamer: Output FPS: {fps}")
    # Use larger buffer size to reduce blocking on writes
    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=1024 * 1024  # 1MB buffer for smoother writes
    )
    
    # Frame buffer for 0.5s delay to reduce stuttering
    # Let ffmpeg handle FPS control, we just buffer frames and write them
    CACHE_DELAY = 0.5  # 0.5 second delay
    frame_buffer = []  # List to store frames before output
    frame_count = 0
    buffer_start_time = None  # Time when first frame was received
    streaming_started = False
    
    print(f"[Task {task_id}] Streamer: Using {CACHE_DELAY}s delay buffer, ffmpeg controls FPS")
    
    # Stream frames from queue - ffmpeg handles FPS timing
    while not stop_event.is_set():
        # Get frame from queue (non-blocking, wait if empty)
        try:
            item = processed_queue.get_nowait()
        except queue.Empty:
            # If streaming started, output buffered frames to ffmpeg
            if streaming_started and len(frame_buffer) > 0:
                frame = frame_buffer.pop(0)
                if process.stdin:
                    try:
                        frame_bytes = frame.tobytes()
                        process.stdin.write(frame_bytes)
                        frame_count += 1
                        
                        # Debug output every 30 frames
                        if frame_count % 30 == 0:
                            print(f"[Task {task_id}] Streamer: Output {frame_count} frames, buffer: {len(frame_buffer)} frames")
                    except (BrokenPipeError, Exception) as e:
                        print(f"[Task {task_id}] Streamer: Error writing frame: {e}")
                        break
            else:
                time.sleep(0.001)  # Brief wait if not streaming yet
            continue
        
        if item[0] == 'end':
            # Output all remaining frames in buffer
            print(f"[Task {task_id}] Streamer: End signal received, outputting {len(frame_buffer)} remaining frames")
            while len(frame_buffer) > 0:
                frame = frame_buffer.pop(0)
                if process.stdin:
                    try:
                        frame_bytes = frame.tobytes()
                        process.stdin.write(frame_bytes)
                        frame_count += 1
                    except Exception as e:
                        print(f"[Task {task_id}] Streamer: Error writing frame: {e}")
            break
        elif item[0] != 'frame':
            continue
        
        frame = item[1]
        
        # Add frame to buffer
        frame_buffer.append(frame)
        
        # Set buffer start time when first frame arrives
        if buffer_start_time is None:
            buffer_start_time = time.time()
            print(f"[Task {task_id}] Streamer: First frame received, starting {CACHE_DELAY}s delay buffer")
        
        # Check if we should start streaming (after CACHE_DELAY seconds)
        if not streaming_started:
            current_time = time.time()
            elapsed_time = current_time - buffer_start_time
            
            if elapsed_time >= CACHE_DELAY:
                streaming_started = True
                print(f"[Task {task_id}] Streamer: Starting output after {elapsed_time:.2f}s delay (buffer: {len(frame_buffer)} frames)")
        
        # If streaming started, output frames from buffer (ffmpeg controls timing)
        if streaming_started and len(frame_buffer) > 0:
            frame = frame_buffer.pop(0)
            if process.stdin:
                try:
                    frame_bytes = frame.tobytes()
                    process.stdin.write(frame_bytes)
                    frame_count += 1
                    
                    # Debug output every 30 frames
                    if frame_count % 30 == 0:
                        print(f"[Task {task_id}] Streamer: Output {frame_count} frames, buffer: {len(frame_buffer)} frames")
                except (BrokenPipeError, Exception) as e:
                    print(f"[Task {task_id}] Streamer: Error writing frame: {e}")
                    break
    
    # Flush and close stdin to signal end of stream
    if process.stdin:
        try:
            process.stdin.flush()
            process.stdin.close()
        except:
            pass
    
    # Wait for ffmpeg to finish
    process.wait(timeout=10)
    
    if stop_event.is_set():
        process.terminate()
        process.wait(timeout=5)
        if process.poll() is None:
            process.kill()
            process.wait()
    
    if process.returncode == 0:
        print(f"[Task {task_id}] Streamer: Streamed {frame_count} frames successfully")
    else:
        stderr_output = process.stderr.read().decode('utf-8', errors='ignore') if process.stderr else ""
        print(f"[Task {task_id}] Streamer: Stream ended with return code {process.returncode}")
        if stderr_output:
            print(f"[Task {task_id}] Streamer stderr: {stderr_output[:500]}")


def process_video_streams(stream_urls: List[str], output_stream_url: str,
                         task_id: str, config_path: str = "configs/config.yaml",
                         temp_dir: str = "data/temp",
                         ball_tracking_method: str = None,
                         pose_estimation_method: str = None):
    """
    Main process coordinator: Start and manage three worker processes
    
    Args:
        stream_urls: Input stream URLs (1-2 streams)
        output_stream_url: Output RTMP stream URL
        task_id: Task ID
        config_path: Config file path
        temp_dir: Temporary directory
        ball_tracking_method: Ball tracking method override
        pose_estimation_method: Pose estimation method override
    """
    print(f"[Task {task_id}] Starting multi-process video processing")
    print(f"[Task {task_id}] Input streams: {stream_urls}")
    print(f"[Task {task_id}] Output stream: {output_stream_url}")
    
    # Use first stream for now
    main_stream_url = stream_urls[0]
    
    # Create shared queues and events
    # Increased queue size to reduce blocking and improve smoothness
    frame_queue = multiprocessing.Queue(maxsize=30)  # Raw frames
    processed_queue = multiprocessing.Queue(maxsize=30)  # Processed frames
    stop_event = multiprocessing.Event()
    
    # Create three processes
    reader_process = multiprocessing.Process(
        target=frame_reader_process,
        args=(main_stream_url, frame_queue, stop_event, task_id),
        name=f"Reader-{task_id}"
    )
    
    processor_process = multiprocessing.Process(
        target=frame_processor_process,
        args=(main_stream_url, frame_queue, processed_queue, stop_event, task_id,
              config_path, temp_dir, ball_tracking_method, pose_estimation_method),
        name=f"Processor-{task_id}"
    )
    
    streamer_process = multiprocessing.Process(
        target=stream_writer_process,
        args=(processed_queue, output_stream_url, stop_event, task_id, temp_dir),
        name=f"Streamer-{task_id}"
    )
    
    # Start all processes
    reader_process.start()
    processor_process.start()
    streamer_process.start()
    
    # Wait for all processes to complete
    reader_process.join()
    processor_process.join()
    streamer_process.join()
    
    print(f"[Task {task_id}] All processes completed")


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'active_tasks': len(active_processes)
    })


@app.route('/process', methods=['POST'])
def process_streams():
    """
    Process video streams endpoint
    
    Request body:
    {
        "stream_urls": ["rtsp://...", "rtsp://..."],  // 1-2 stream URLs
        "output_stream_url": "rtmp://..."  // Output RTMP stream URL
    }
    """
    data = request.get_json()
    
    # Validate parameters
    if not data:
        return jsonify({'error': 'Request body is required'}), 400
    
    stream_urls = data.get('stream_urls', [])
    output_stream_url = data.get('output_stream_url')
    
    if not stream_urls:
        return jsonify({'error': 'stream_urls is required'}), 400
    
    if not isinstance(stream_urls, list):
        return jsonify({'error': 'stream_urls must be a list'}), 400
    
    if len(stream_urls) < 1 or len(stream_urls) > 2:
        return jsonify({'error': 'stream_urls must contain 1 or 2 URLs'}), 400
    
    if not output_stream_url:
        return jsonify({'error': 'output_stream_url is required'}), 400
    
    # Get method overrides from request
    ball_tracking_method = data.get('ball_tracking_method')
    pose_estimation_method = data.get('pose_estimation_method')
    
    # Generate task ID
    task_id = f"task_{int(time.time() * 1000)}"
    
    # Create new process to coordinate worker processes
    coordinator_process = multiprocessing.Process(
        target=process_video_streams,
        args=(
            stream_urls,
            output_stream_url,
            task_id,
            "configs/config.yaml",
            "data/temp",
            ball_tracking_method,
            pose_estimation_method
        ),
        name=f"Coordinator-{task_id}"
    )
    
    coordinator_process.start()
    
    # Record process information
    with process_lock:
        active_processes[task_id] = {
            'process': coordinator_process,
            'stream_urls': stream_urls,
            'output_stream_url': output_stream_url,
            'start_time': datetime.now().isoformat(),
            'status': 'running'
        }
    
    return jsonify({
        'status': 'success',
        'task_id': task_id,
        'message': 'Video processing started with multi-process architecture'
    }), 200


@app.route('/tasks', methods=['GET'])
def list_tasks():
    """List all tasks"""
    with process_lock:
        tasks = []
        for task_id, info in active_processes.items():
            process = info['process']
            status = 'running' if process.is_alive() else 'completed'
            
            tasks.append({
                'task_id': task_id,
                'status': status,
                'stream_urls': info['stream_urls'],
                'output_stream_url': info['output_stream_url'],
                'start_time': info['start_time']
            })
            
            # Clean up completed processes
            if not process.is_alive():
                process.join()
                info['status'] = 'completed'
        
        return jsonify({
            'tasks': tasks,
            'total': len(tasks)
        }), 200


@app.route('/tasks/<task_id>', methods=['GET'])
def get_task(task_id):
    """Get specific task information"""
    with process_lock:
        if task_id not in active_processes:
            return jsonify({'error': 'Task not found'}), 404
        
        info = active_processes[task_id]
        process = info['process']
        status = 'running' if process.is_alive() else 'completed'
        
        return jsonify({
            'task_id': task_id,
            'status': status,
            'stream_urls': info['stream_urls'],
            'output_stream_url': info['output_stream_url'],
            'start_time': info['start_time']
        }), 200


@app.route('/tasks/<task_id>', methods=['DELETE'])
def stop_task(task_id):
    """Stop specific task"""
    with process_lock:
        if task_id not in active_processes:
            return jsonify({'error': 'Task not found'}), 404
        
        info = active_processes[task_id]
        process = info['process']
        
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join()
        
        del active_processes[task_id]
        
        return jsonify({
            'status': 'success',
            'message': f'Task {task_id} stopped'
        }), 200


if __name__ == '__main__':
    # Ensure spawn method for process start (better compatibility)
    if sys.platform != 'win32':
        multiprocessing.set_start_method('spawn', force=True)
    
    # Create necessary directories
    Path("data/temp").mkdir(parents=True, exist_ok=True)
    
    print("Starting Badminton AI API Server...")
    print("Multi-process architecture: Reader -> Processor -> Streamer")
    print("API endpoints:")
    print("  POST   /process  - Process video streams")
    print("  GET    /tasks    - List all tasks")
    print("  GET    /tasks/<task_id> - Get task info")
    print("  DELETE /tasks/<task_id> - Stop task")
    print("  GET    /health   - Health check")
    
    app.run(host='0.0.0.0', port=2070, debug=False, threaded=True)
