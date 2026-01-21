"""Parallel processing script: Process multiple video streams concurrently"""

import sys
import os
import argparse
import time
import json
from pathlib import Path
from typing import List, Dict, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from tqdm.auto import tqdm
import threading

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.main import BadmintonAISystem
from src.utils.performance import PerformanceMonitor


def get_gpu_memory_usage(gpu_id: int) -> Dict[str, float]:
    """获取指定GPU的显存使用情况"""
    import torch
    if not torch.cuda.is_available() or gpu_id >= torch.cuda.device_count():
        return {'available_mb': 0, 'used_mb': 0, 'total_mb': 0}
    
    torch.cuda.set_device(gpu_id)
    allocated = torch.cuda.memory_allocated(gpu_id) / (1024 ** 2)  # MB
    reserved = torch.cuda.memory_reserved(gpu_id) / (1024 ** 2)  # MB
    total = torch.cuda.get_device_properties(gpu_id).total_memory / (1024 ** 2)  # MB
    available = total - reserved
    
    return {
        'available_mb': available,
        'used_mb': reserved,
        'total_mb': total,
        'used_percent': (reserved / total) * 100 if total > 0 else 0
    }


def find_available_gpu(num_gpus: int, min_free_memory_mb: float = 2000.0) -> Optional[int]:
    """查找可用的GPU（显存充足）"""
    for gpu_id in range(num_gpus):
        mem_info = get_gpu_memory_usage(gpu_id)
        if mem_info['available_mb'] >= min_free_memory_mb:
            return gpu_id
    return None


def process_single_video_stream(args_tuple):
    """
    Process single video stream (independent process, real-time stream processing)
    
    Args:
        args_tuple: (stream_url, output_dir, config_path, num_gpus, stream_idx, total_streams, gpu_id, progress_dict, test_mode, test_frames)
        
    Returns:
        Processing result dictionary
    """
    stream_url, output_dir, config_path, num_gpus, stream_idx, total_streams, assigned_gpu_id, progress_dict, test_mode, test_frames = args_tuple
    
    # 抑制详细输出
    import os
    import sys
    from io import StringIO
    
    # 重定向stdout和stderr以抑制详细输出
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = StringIO()
    sys.stderr = StringIO()
    
    # 设置环境变量抑制YOLO详细输出
    os.environ['YOLO_VERBOSE'] = 'False'
    
    # 确定使用的GPU
    gpu_id = assigned_gpu_id
    
    # 如果未指定GPU或指定GPU显存不足，自动查找可用GPU
    if gpu_id is None or gpu_id >= num_gpus:
        gpu_id = find_available_gpu(num_gpus, min_free_memory_mb=2000.0)
        if gpu_id is None:
            # 如果所有GPU显存都不足，使用显存使用率最低的GPU
            min_usage = float('inf')
            best_gpu = 0
            for gid in range(num_gpus):
                mem_info = get_gpu_memory_usage(gid)
                if mem_info['used_percent'] < min_usage:
                    min_usage = mem_info['used_percent']
                    best_gpu = gid
            gpu_id = best_gpu
    
    # 设置GPU设备
    import torch
    if torch.cuda.is_available() and gpu_id < torch.cuda.device_count():
        torch.cuda.set_device(gpu_id)
    else:
        gpu_id = None
    
    # 初始化性能监控
    perf_monitor = PerformanceMonitor(num_gpus=1 if gpu_id is not None else 0)
    
    # 初始化系统（每个进程独立初始化）
    init_start = time.time()
    system = BadmintonAISystem(config_path=config_path)
    init_time = time.time() - init_start
    
    # Create output directory
    # Use stream URL as identifier (sanitize for filesystem)
    stream_name = stream_url.replace('://', '_').replace('/', '_').replace(':', '_')
    stream_output_dir = Path(output_dir) / stream_name
    stream_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get stream info for progress bar
    # For streams, frame_count is typically unknown (None)
    if test_mode and test_frames == 0:
        # Test mode: use random tensor, fixed frame count
        total_frames = 2
    elif test_mode and test_frames > 0:
        # Test mode: limit frames
        total_frames = test_frames
    else:
        # Live stream: unknown frame count, use None or large number for progress
        total_frames = None  # Will be updated as frames are processed
    
    # Initialize progress
    if progress_dict is not None:
        progress_dict[stream_idx] = {'current': 0, 'total': total_frames or 0, 'status': 'processing'}
    
    # Record start time
    start_time = time.time()
    
    # Process stream (real-time stream processing, suppress output)
    # Define progress callback function
    def update_progress_callback(frame_idx):
        if progress_dict is not None:
            current_total = progress_dict[stream_idx].get('total', 0) if progress_dict[stream_idx] else 0
            # For live streams, update total as we go
            if current_total == 0 or frame_idx + 1 > current_total:
                current_total = frame_idx + 1
            progress_dict[stream_idx] = {'current': frame_idx + 1, 'total': current_total, 'status': 'processing'}
    
    # Test mode (test_frames=0): pass None to avoid loading stream
    stream_url_for_processing = None if (test_mode and test_frames == 0) else str(stream_url)
    
    results = system.process_video(
        stream_url_for_processing,
        str(stream_output_dir),
        visualize=True,
        monitor_performance=True,
        num_gpus=1 if gpu_id is not None else 0,
        verbose=False,  # Suppress detailed output
        progress_callback=update_progress_callback,
        test_mode=test_mode,
        test_frames=test_frames
    )
    
    # Update progress to completed
    if progress_dict is not None:
        final_total = progress_dict[stream_idx].get('total', 0) if progress_dict[stream_idx] else 0
        progress_dict[stream_idx] = {'current': final_total, 'total': final_total, 'status': 'completed'}
    
    # 恢复输出（但保持抑制状态，避免输出到主进程）
    sys.stdout = old_stdout
    sys.stderr = old_stderr
    
    # 记录结束时间
    end_time = time.time()
    total_time = end_time - start_time
    
    # 从process_video返回的结果中获取性能统计（如果存在）
    perf_stats = results.get('performance_stats', {})
    memory_info = results.get('gpu_memory_info', {})
    
    # 如果没有从process_video获取到性能统计，尝试从本地perf_monitor获取
    if not perf_stats:
        perf_stats = perf_monitor.stats.get_stats()
    if not memory_info:
        memory_info = perf_monitor.get_gpu_memory()
    
    # Build result
    result = {
        'stream_url': str(stream_url),
        'stream_name': stream_name,
        'output_dir': str(stream_output_dir),
        'status': 'success',
        'gpu_id': gpu_id,
        'init_time_seconds': init_time,
        'total_time_seconds': total_time,
        'total_frames': len(results.get('ball_trajectory', [])),
        'hit_events': len(results.get('hit_events', [])),
        'shot_classifications': len(results.get('shot_classifications', [])),
        'performance': {
            'frame_processing': perf_stats.get('frame_processing', {}),
            'ball_tracking': perf_stats.get('ball_tracking', {}),
            'pose_estimation': perf_stats.get('pose_estimation', {}),
            'integration': perf_stats.get('integration', {}),
            'gpu_memory': perf_stats.get('gpu_memory', {}),
        },
        'gpu_memory_info': memory_info,
    }
    
    return result


def print_summary(results: List[Dict], output_file: Optional[str] = None):
    """Print processing results summary"""
    print("\n" + "=" * 80)
    print("Parallel Processing Results Summary")
    print("=" * 80)
    
    successful = [r for r in results if r.get('status') == 'success']
    failed = [r for r in results if r.get('status') == 'error']
    
    print(f"\nTotal streams: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if successful:
        print("\nSuccessfully processed streams:")
        print("-" * 80)
        total_time = 0
        total_frames = 0
        total_init_time = 0
        
        # Group statistics by GPU
        gpu_stats = {}
        
        for result in successful:
            stream_name = result.get('stream_name', 'Unknown')
            total_time_stream = result.get('total_time_seconds', 0)
            init_time_stream = result.get('init_time_seconds', 0)
            total_frames_stream = result.get('total_frames', 0)
            gpu_id = result.get('gpu_id', 'N/A')
            
            total_time += total_time_stream
            total_init_time += init_time_stream
            total_frames += total_frames_stream
            
            # GPU statistics
            if gpu_id not in gpu_stats:
                gpu_stats[gpu_id] = {'count': 0, 'time': 0, 'frames': 0}
            gpu_stats[gpu_id]['count'] += 1
            gpu_stats[gpu_id]['time'] += total_time_stream
            gpu_stats[gpu_id]['frames'] += total_frames_stream
            
            # Performance information
            perf = result.get('performance', {})
            fp = perf.get('frame_processing', {})
            fps = fp.get('fps', 0)
            avg_time = fp.get('mean_ms', 0)
            
            # GPU memory
            gpu_mem = perf.get('gpu_memory', {})
            avg_mem = gpu_mem.get('mean_mb', 0)
            max_mem = gpu_mem.get('max_mb', 0)
            
            print(f"  {stream_name} (GPU {gpu_id}):")
            print(f"    Init time: {init_time_stream:.2f}s")
            print(f"    Process time: {total_time_stream:.2f}s")
            print(f"    Total frames: {total_frames_stream}")
            print(f"    Processing speed: {fps:.2f} FPS")
            print(f"    Avg frame time: {avg_time:.2f} ms")
            print(f"    Avg memory: {avg_mem:.2f} MB")
            print(f"    Max memory: {max_mem:.2f} MB")
            print(f"    Hit events: {result.get('hit_events', 0)}")
        
        print("\nOverall Statistics:")
        print("-" * 80)
        print(f"  Total init time: {total_init_time:.2f}s ({total_init_time/60:.2f}min)")
        print(f"  Total process time: {total_time:.2f}s ({total_time/60:.2f}min)")
        print(f"  Total frames: {total_frames}")
        if total_time > 0:
            avg_fps = total_frames / total_time
            print(f"  Avg processing speed: {avg_fps:.2f} FPS")
        
        # GPU usage statistics
        if gpu_stats:
            print("\nGPU Usage Statistics:")
            print("-" * 80)
            for gpu_id, stats in sorted(gpu_stats.items()):
                print(f"  GPU {gpu_id}:")
                print(f"    Processed streams: {stats['count']}")
                print(f"    Total process time: {stats['time']:.2f}s")
                print(f"    Total frames: {stats['frames']}")
                if stats['time'] > 0:
                    print(f"    Avg FPS: {stats['frames']/stats['time']:.2f}")
        
        # GPU memory statistics
        all_avg_mem = [r.get('performance', {}).get('gpu_memory', {}).get('mean_mb', 0) 
                       for r in successful if r.get('performance', {}).get('gpu_memory', {}).get('mean_mb', 0) > 0]
        all_max_mem = [r.get('performance', {}).get('gpu_memory', {}).get('max_mb', 0) 
                      for r in successful if r.get('performance', {}).get('gpu_memory', {}).get('max_mb', 0) > 0]
        
        if all_avg_mem:
            print(f"\nMemory Usage Statistics:")
            print(f"  Avg memory usage: {sum(all_avg_mem)/len(all_avg_mem):.2f} MB")
        if all_max_mem:
            print(f"  Max memory usage: {max(all_max_mem):.2f} MB")
    
    if failed:
        print("\nFailed streams:")
        print("-" * 80)
        for result in failed:
            stream_name = result.get('stream_name', 'Unknown')
            error = result.get('error', 'Unknown error')
            print(f"  {stream_name}: {error}")
    
    print("=" * 80)
    
    # Save results to file
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nDetailed results saved to: {output_file}")


def update_progress_bars(progress_dict, progress_bars):
    """更新所有进度条"""
    for idx, pbar in progress_bars.items():
        if idx in progress_dict:
            progress = progress_dict[idx]
            # 检查progress是否为None或不是字典
            if progress is None or not isinstance(progress, dict):
                continue
            if progress.get('total', 0) > 0:
                pbar.n = progress.get('current', 0)
                pbar.total = progress.get('total', 0)
                status = progress.get('status', 'processing')
                if status == 'completed':
                    pbar.set_postfix_str("✓ 完成")
                elif status == 'error':
                    pbar.set_postfix_str("✗ 错误")
                else:
                    pbar.set_postfix_str(f"处理中...")
                pbar.refresh()


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Parallel processing script: Process multiple video streams concurrently')
    parser.add_argument('--streams', type=str, nargs='+', required=True,
                       help='Video stream URLs (e.g., rtsp://, rtmp://, http://, or camera indices like 0)')
    parser.add_argument('--output', type=str, default='data/results/parallel',
                       help='Output directory')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Configuration file path')
    parser.add_argument('--num-gpus', type=int, default=2,
                       help='Number of GPUs')
    parser.add_argument('--max-workers', type=int, default=None,
                       help='Maximum parallel processes (default: number of streams, all streams in parallel)')
    parser.add_argument('--output-json', type=str, default=None,
                       help='Results JSON file path (default: output_dir/results.json)')
    parser.add_argument('--test-mode', action='store_true',
                       help='Test mode: use random tensors or process limited frames')
    parser.add_argument('--test-frames', type=int, default=2,
                       help='Number of frames to process in test mode (0=use random tensor, >0=load specified frames)')
    
    args = parser.parse_args()
    
    # Collect stream URLs
    stream_urls = list(set(args.streams))  # Remove duplicates
    
    if not stream_urls:
        print("Error: No stream URLs provided")
        print("Please use --streams to specify video stream URLs")
        return 1
    
    # Determine parallel process count
    if args.max_workers is not None:
        max_workers = args.max_workers
    else:
        # Default: each GPU can handle multiple streams (adjust based on memory)
        max_workers = min(len(stream_urls), args.num_gpus * 10) if args.num_gpus > 0 else len(stream_urls)
    
    # Ensure not exceeding stream count
    max_workers = min(max_workers, len(stream_urls))
    
    print("=" * 80)
    print("Parallel Stream Processing (Real-time)")
    print("=" * 80)
    print(f"Found {len(stream_urls)} stream(s)")
    print(f"Output directory: {args.output}")
    print(f"Number of GPUs: {args.num_gpus}")
    print(f"Parallel processes: {max_workers}")
    print(f"Processing mode: Each stream processed as independent real-time stream")
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create shared progress dictionary (for multiprocess communication)
    manager = mp.Manager()
    progress_dict = manager.dict()
    
    # Get stream info for progress bars
    # For streams, frame_count is typically unknown
    stream_info = {}
    for idx, stream_url in enumerate(stream_urls, 1):
        if args.test_mode and args.test_frames == 0:
            # Test mode: use random tensor, fixed frame count
            total_frames = 2
        elif args.test_mode and args.test_frames > 0:
            # Test mode: limit frames
            total_frames = args.test_frames
        else:
            # Live stream: unknown frame count
            total_frames = 0  # Will be updated as frames are processed
        stream_name = stream_url.replace('://', '_').replace('/', '_').replace(':', '_')
        stream_info[idx] = {
            'name': stream_name[:50],  # Truncate long URLs
            'total_frames': total_frames
        }
        progress_dict[idx] = {'current': 0, 'total': total_frames, 'status': 'waiting'}
    
    # Prepare arguments, initial GPU assignment (round-robin)
    args_list = []
    for idx, stream_url in enumerate(stream_urls):
        # Initial GPU assignment (round-robin), may adjust based on memory usage
        initial_gpu_id = idx % args.num_gpus if args.num_gpus > 0 else None
        args_list.append((
            stream_url, 
            str(output_dir), 
            args.config, 
            args.num_gpus, 
            idx + 1, 
            len(stream_urls),
            initial_gpu_id,
            progress_dict,
            args.test_mode,
            args.test_frames
        ))
    
    # Process streams (parallel processing)
    start_time = time.time()
    results = []
    
    print(f"\nStarting processing...\n")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_stream = {
            executor.submit(process_single_video_stream, args): (args[0], args[4])
            for args in args_list
        }
        
        # Create independent progress bars for each stream
        progress_bars = {}
        for idx in range(1, len(stream_urls) + 1):
            stream_name = stream_info[idx]['name']
            total_frames = stream_info[idx]['total_frames']
            # Use position parameter to display each progress bar on different line
            pbar = tqdm(
                total=total_frames if total_frames > 0 else None,
                desc=f"Stream{idx:2d}: {stream_name[:35]:<35}",
                unit="frames",
                position=idx-1,
                leave=True,
                ncols=100
            )
            progress_bars[idx] = pbar
        
        # Progress update thread function
        stop_progress_thread = threading.Event()
        active_tasks = set(range(1, len(stream_urls) + 1))  # Track active tasks
        
        def update_progress():
            while not stop_progress_thread.is_set() or active_tasks:
                # Update all progress bars
                for idx, pbar in progress_bars.items():
                    if idx in progress_dict:
                        progress = progress_dict[idx]
                        # Check if progress is None or not a dict
                        if progress is None or not isinstance(progress, dict):
                            continue
                        current = progress.get('current', 0)
                        total = progress.get('total', 0)
                        status = progress.get('status', 'processing')
                        
                        # Update progress bar
                        pbar.n = current
                        if total > 0:
                            pbar.total = total
                        else:
                            # For live streams, update total dynamically
                            if current > pbar.total:
                                pbar.total = current
                        
                        if status == 'completed':
                            pbar.set_postfix_str("✓ Completed")
                            if idx in active_tasks:
                                active_tasks.remove(idx)
                        elif status == 'error':
                            pbar.set_postfix_str("✗ Error")
                            if idx in active_tasks:
                                active_tasks.remove(idx)
                        else:
                            pbar.set_postfix_str("Processing...")
                        
                        pbar.refresh()
                
                # Exit loop if all tasks completed
                if not active_tasks:
                    break
                    
                time.sleep(0.1)  # Update every 0.1 seconds
        
        # Start progress update thread
        progress_thread = threading.Thread(target=update_progress, daemon=True)
        progress_thread.start()
        
        # Collect results
        for future in as_completed(future_to_stream):
            stream_url, stream_idx = future_to_stream[future]
            result = future.result(timeout=None)
            results.append(result)
            
            # Update progress bar status
            if stream_idx in progress_dict:
                progress = progress_dict.get(stream_idx)
                if progress is not None and isinstance(progress, dict):
                    final_total = progress.get('total', 0)
                    progress_dict[stream_idx] = {
                        'current': final_total,
                        'total': final_total,
                        'status': 'completed' if result.get('status') == 'success' else 'error'
                    }
                else:
                    # If progress is invalid, create new one
                    progress_dict[stream_idx] = {
                        'current': 0,
                        'total': 0,
                        'status': 'completed' if result.get('status') == 'success' else 'error'
                    }
            # Remove from active tasks
            if stream_idx in active_tasks:
                active_tasks.remove(stream_idx)
        
        # Stop progress update thread
        stop_progress_thread.set()
        time.sleep(0.3)  # Wait for thread to complete last update
        
        # Close all progress bars
        for pbar in progress_bars.values():
            pbar.close()
    
    total_time = time.time() - start_time
    
    # Sort by original order
    results.sort(key=lambda x: stream_urls.index(x['stream_url']) if x['stream_url'] in stream_urls else 999)
    
    # Output results file path
    if args.output_json:
        output_json = args.output_json
    else:
        output_json = str(output_dir / "results.json")
    
    # Print summary
    print_summary(results, output_json)
    
    print(f"\nTotal time: {total_time:.2f}s ({total_time/60:.2f}min)")
    
    return 0


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    exit(main())
