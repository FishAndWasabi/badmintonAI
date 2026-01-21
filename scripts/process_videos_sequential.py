"""Sequential processing script: Process video streams sequentially with real-time results display"""

import sys
import os
import argparse
import time
import json
from pathlib import Path
from typing import List, Dict, Optional, Callable
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.main import BadmintonAISystem
from src.utils.performance import PerformanceMonitor


def print_realtime_result(result: Dict, stream_idx: int, total_streams: int):
    """Print real-time processing result for a single stream"""
    stream_name = result.get('stream_name', 'Unknown')
    status = result.get('status', 'unknown')
    total_time = result.get('total_time_seconds', 0)
    total_frames = result.get('total_frames', 0)
    
    print(f"\n{'='*80}")
    print(f"Stream {stream_idx}/{total_streams}: {stream_name}")
    print(f"{'='*80}")
    
    if status == 'success':
        perf = result.get('performance', {})
        fp = perf.get('frame_processing', {})
        fps = fp.get('fps', 0)
        avg_time = fp.get('mean_ms', 0)
        
        gpu_mem = perf.get('gpu_memory', {})
        avg_mem = gpu_mem.get('mean_mb', 0)
        max_mem = gpu_mem.get('max_mb', 0)
        
        print(f"Status: ✓ Success")
        print(f"Processing time: {total_time:.2f}s")
        print(f"Total frames processed: {total_frames}")
        print(f"Processing speed: {fps:.2f} FPS")
        print(f"Average frame time: {avg_time:.2f} ms")
        print(f"GPU memory - Avg: {avg_mem:.2f} MB, Max: {max_mem:.2f} MB")
        print(f"Hit events detected: {result.get('hit_events', 0)}")
        print(f"Shot classifications: {result.get('shot_classifications', 0)}")
        print(f"Output directory: {result.get('output_dir', 'N/A')}")
    else:
        error = result.get('error', 'Unknown error')
        print(f"Status: ✗ Failed")
        print(f"Error: {error}")
    
    print(f"{'='*80}")


def print_summary(results: List[Dict], output_file: Optional[str] = None):
    """Print processing results summary"""
    print("\n" + "=" * 80)
    print("Sequential Processing Results Summary")
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
        
        for result in successful:
            stream_name = result.get('stream_name', 'Unknown')
            total_time_stream = result.get('total_time_seconds', 0)
            total_frames_stream = result.get('total_frames', 0)
            
            total_time += total_time_stream
            total_frames += total_frames_stream
            
            # Performance information
            perf = result.get('performance', {})
            fp = perf.get('frame_processing', {})
            fps = fp.get('fps', 0)
            avg_time = fp.get('mean_ms', 0)
            
            # GPU memory
            gpu_mem = perf.get('gpu_memory', {})
            avg_mem = gpu_mem.get('mean_mb', 0)
            max_mem = gpu_mem.get('max_mb', 0)
            
            print(f"  {stream_name}:")
            print(f"    Process time: {total_time_stream:.2f}s")
            print(f"    Total frames: {total_frames_stream}")
            print(f"    Processing speed: {fps:.2f} FPS")
            print(f"    Avg frame time: {avg_time:.2f} ms")
            print(f"    Avg memory: {avg_mem:.2f} MB")
            print(f"    Max memory: {max_mem:.2f} MB")
            print(f"    Hit events: {result.get('hit_events', 0)}")
        
        print("\nOverall Statistics:")
        print("-" * 80)
        print(f"  Total process time: {total_time:.2f}s ({total_time/60:.2f}min)")
        print(f"  Total frames: {total_frames}")
        if total_time > 0:
            avg_fps = total_frames / total_time
            print(f"  Avg processing speed: {avg_fps:.2f} FPS")
        
        # GPU memory statistics
        all_avg_mem = [r.get('performance', {}).get('gpu_memory', {}).get('mean_mb', 0) 
                       for r in successful if r.get('performance', {}).get('gpu_memory', {}).get('mean_mb', 0) > 0]
        all_max_mem = [r.get('performance', {}).get('gpu_memory', {}).get('max_mb', 0) 
                      for r in successful if r.get('performance', {}).get('gpu_memory', {}).get('max_mb', 0) > 0]
        
        if all_avg_mem:
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


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Sequential processing script: Process video streams sequentially')
    parser.add_argument('--streams', type=str, nargs='+', required=True,
                       help='Video stream URLs (e.g., rtsp://, rtmp://, http://, or camera indices like 0)')
    parser.add_argument('--output', type=str, default='outputs/sequential',
                       help='Output directory')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Configuration file path')
    parser.add_argument('--num-gpus', type=int, default=2,
                       help='Number of GPUs (for performance monitoring)')
    parser.add_argument('--output-json', type=str, default=None,
                       help='Results JSON file path (default: output_dir/results.json)')
    
    args = parser.parse_args()
    
    # Collect stream URLs
    stream_urls = list(set(args.streams))  # Remove duplicates
    
    if not stream_urls:
        print("Error: No stream URLs provided")
        print("Please use --streams to specify video stream URLs")
        return 1
    
    print("=" * 80)
    print("Sequential Stream Processing")
    print("=" * 80)
    print(f"Found {len(stream_urls)} stream(s)")
    print(f"Output directory: {args.output}")
    print(f"Number of GPUs: {args.num_gpus}")
    print(f"Processing mode: Single process sequential processing")
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize system (only once)
    print(f"\nInitializing system...")
    init_start = time.time()
    system = BadmintonAISystem(config_path=args.config)
    init_time = time.time() - init_start
    print(f"System initialization complete (took: {init_time:.2f}s)")
    
    # Initialize performance monitor
    perf_monitor = PerformanceMonitor(num_gpus=args.num_gpus)
    
    # Process streams sequentially
    start_time = time.time()
    results = []
    
    print(f"\nStarting processing...")
    print(f"Processing {len(stream_urls)} stream(s) sequentially\n")
    
    for idx, stream_url in enumerate(stream_urls, 1):
        # Use stream URL as identifier (sanitize for filesystem)
        stream_name = stream_url.replace('://', '_').replace('/', '_').replace(':', '_')
        stream_output_dir = output_dir / stream_name
        stream_output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting stream {idx}/{len(stream_urls)}: {stream_url[:60]}")
        
        # Process stream
        stream_start = time.time()
        try:
            # Define real-time progress callback
            last_update_time = [time.time()]
            last_frame_count = [0]
            
            def progress_callback(frame_idx: int):
                """Real-time progress callback"""
                current_time = time.time()
                # Update every 1 second or every 30 frames
                if current_time - last_update_time[0] >= 1.0 or frame_idx - last_frame_count[0] >= 30:
                    elapsed = current_time - stream_start
                    fps = (frame_idx - last_frame_count[0]) / (current_time - last_update_time[0]) if current_time > last_update_time[0] else 0
                    print(f"  [{datetime.now().strftime('%H:%M:%S')}] Frame {frame_idx:6d} | "
                          f"Elapsed: {elapsed:6.1f}s | "
                          f"Speed: {fps:5.1f} FPS", end='\r', flush=True)
                    last_update_time[0] = current_time
                    last_frame_count[0] = frame_idx
            
            stream_results = system.process_video(
                str(stream_url),
                str(stream_output_dir),
                visualize=True,
                monitor_performance=True,
                num_gpus=args.num_gpus,
                verbose=False,  # Suppress verbose output, we'll show our own
                progress_callback=progress_callback
            )
            stream_time = time.time() - stream_start
            
            # Clear progress line
            print()  # New line after progress updates
            
            # Get performance statistics
            perf_stats = perf_monitor.stats.get_stats()
            memory_info = perf_monitor.get_gpu_memory()
            
            # Build result
            result = {
                'stream_url': str(stream_url),
                'stream_name': stream_name,
                'output_dir': str(stream_output_dir),
                'status': 'success',
                'total_time_seconds': stream_time,
                'total_frames': len(stream_results.get('ball_trajectory', [])),
                'hit_events': len(stream_results.get('hit_events', [])),
                'shot_classifications': len(stream_results.get('shot_classifications', [])),
                'performance': {
                    'frame_processing': perf_stats.get('frame_processing', {}),
                    'ball_tracking': perf_stats.get('ball_tracking', {}),
                    'pose_estimation': perf_stats.get('pose_estimation', {}),
                    'integration': perf_stats.get('integration', {}),
                    'gpu_memory': perf_stats.get('gpu_memory', {}),
                },
                'gpu_memory_info': memory_info,
            }
            
        except Exception as e:
            stream_time = time.time() - stream_start
            print(f"\n  ✗ Error processing stream: {str(e)}")
            result = {
                'stream_url': str(stream_url),
                'stream_name': stream_name,
                'output_dir': str(stream_output_dir),
                'status': 'error',
                'error': str(e),
                'total_time_seconds': stream_time,
            }
        
        # Print real-time result
        print_realtime_result(result, idx, len(stream_urls))
        results.append(result)
    
    total_time = time.time() - start_time
    
    # Output results file path
    if args.output_json:
        output_json = args.output_json
    else:
        output_json = str(output_dir / "results.json")
    
    # Print summary
    print_summary(results, output_json)
    
    print(f"\nTotal time: {total_time:.2f}s ({total_time/60:.2f}min)")
    print(f"System initialization time: {init_time:.2f}s")
    
    return 0


if __name__ == '__main__':
    exit(main())
