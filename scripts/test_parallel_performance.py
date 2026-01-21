"""并行多路视频流性能测试脚本：使用随机数据测试显存占用和时延"""

import sys
import os
import argparse
import time
import json
from pathlib import Path
from typing import List, Dict
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from tqdm.auto import tqdm
import threading

# 添加项目根目录到路径
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


def find_available_gpu(num_gpus: int, min_free_memory_mb: float = 2000.0) -> int:
    """查找可用的GPU（显存充足）"""
    for gpu_id in range(num_gpus):
        mem_info = get_gpu_memory_usage(gpu_id)
        if mem_info['available_mb'] >= min_free_memory_mb:
            return gpu_id
    # 如果所有GPU显存都不足，返回显存使用率最低的GPU
    min_usage = float('inf')
    best_gpu = 0
    for gid in range(num_gpus):
        mem_info = get_gpu_memory_usage(gid)
        if mem_info['used_percent'] < min_usage:
            min_usage = mem_info['used_percent']
            best_gpu = gid
    return best_gpu


def test_single_stream(args_tuple):
    """
    测试单个视频流（独立进程，使用随机数据）
    
    Args:
        args_tuple: (stream_id, output_dir, config_path, num_gpus, assigned_gpu_id, progress_dict, test_frames)
        
    Returns:
        测试结果字典
    """
    stream_id, output_dir, config_path, num_gpus, assigned_gpu_id, progress_dict, test_frames = args_tuple
    
    # 抑制详细输出
    import os
    import sys
    from io import StringIO
    
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = StringIO()
    sys.stderr = StringIO()
    
    os.environ['YOLO_VERBOSE'] = 'False'
    
    # 确定使用的GPU
    gpu_id = assigned_gpu_id
    if gpu_id is None or gpu_id >= num_gpus:
        gpu_id = find_available_gpu(num_gpus, min_free_memory_mb=2000.0)
    
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
    
    # 创建输出目录（测试模式下实际上不会保存文件）
    stream_output_dir = Path(output_dir) / f"stream_{stream_id}"
    stream_output_dir.mkdir(parents=True, exist_ok=True)
    
    # 测试模式：使用随机数据，不加载视频
    # 可配置的帧数用于测试
    total_frames = test_frames
    
    # 初始化进度
    if progress_dict is not None:
        progress_dict[stream_id] = {'current': 0, 'total': total_frames, 'status': 'processing'}
    
    # 记录开始时间
    start_time = time.time()
    
    # 定义进度回调函数
    def update_progress_callback(frame_idx):
        if progress_dict is not None:
            progress_dict[stream_id] = {'current': frame_idx + 1, 'total': total_frames, 'status': 'processing'}
    
    # 处理视频（测试模式：使用随机数据，不保存结果）
    results = system.process_video(
        None,  # 不提供视频路径，使用随机数据
        str(stream_output_dir),
        visualize=False,  # 测试模式下不生成可视化
        monitor_performance=True,
        num_gpus=1 if gpu_id is not None else 0,
        verbose=False,
        progress_callback=update_progress_callback,
        test_mode=True,
        test_frames=test_frames  # 使用随机tensor，处理指定帧数
    )
    
    # 更新进度为完成
    if progress_dict is not None:
        progress_dict[stream_id] = {'current': total_frames, 'total': total_frames, 'status': 'completed'}
    
    # 恢复输出
    sys.stdout = old_stdout
    sys.stderr = old_stderr
    
    # 记录结束时间
    end_time = time.time()
    total_time = end_time - start_time
    
    # 获取性能统计
    perf_stats = results.get('performance_stats', {})
    memory_info = results.get('gpu_memory_info', {})
    
    # 构建结果
    result = {
        'stream_id': stream_id,
        'gpu_id': gpu_id,
        'init_time_seconds': init_time,
        'total_time_seconds': total_time,
        'total_frames': total_frames,
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


def print_performance_summary(results: List[Dict], output_file: str = None):
    """打印性能测试摘要"""
    print("\n" + "=" * 80)
    print("并行多路视频流性能测试结果")
    print("=" * 80)
    
    if not results:
        print("没有测试结果")
        return
    
    print(f"\n总流数: {len(results)}")
    
    # 统计总体性能
    total_init_time = sum(r.get('init_time_seconds', 0) for r in results)
    total_processing_time = sum(r.get('total_time_seconds', 0) for r in results)
    total_frames = sum(r.get('total_frames', 0) for r in results)
    
    print(f"\n总体统计:")
    print("-" * 80)
    print(f"  总初始化时间: {total_init_time:.2f}秒 ({total_init_time/60:.2f}分钟)")
    print(f"  总处理时间: {total_processing_time:.2f}秒 ({total_processing_time/60:.2f}分钟)")
    print(f"  总帧数: {total_frames}")
    if total_processing_time > 0:
        avg_fps = total_frames / total_processing_time
        print(f"  平均处理速度: {avg_fps:.2f} FPS")
    
    # 按GPU分组统计
    gpu_stats = {}
    for result in results:
        gpu_id = result.get('gpu_id', 'N/A')
        if gpu_id not in gpu_stats:
            gpu_stats[gpu_id] = {
                'count': 0,
                'time': 0,
                'frames': 0,
                'init_time': 0
            }
        gpu_stats[gpu_id]['count'] += 1
        gpu_stats[gpu_id]['time'] += result.get('total_time_seconds', 0)
        gpu_stats[gpu_id]['frames'] += result.get('total_frames', 0)
        gpu_stats[gpu_id]['init_time'] += result.get('init_time_seconds', 0)
    
    if gpu_stats:
        print("\nGPU使用统计:")
        print("-" * 80)
        for gpu_id, stats in sorted(gpu_stats.items()):
            print(f"  GPU {gpu_id}:")
            print(f"    流数: {stats['count']}")
            print(f"    总初始化时间: {stats['init_time']:.2f}秒")
            print(f"    总处理时间: {stats['time']:.2f}秒")
            print(f"    总帧数: {stats['frames']}")
            if stats['time'] > 0:
                print(f"    平均FPS: {stats['frames']/stats['time']:.2f}")
    
    # 性能指标统计
    all_fps = []
    all_frame_times = []
    all_ball_tracking_times = []
    all_pose_times = []
    all_mem_usage = []
    all_max_mem = []
    
    for result in results:
        perf = result.get('performance', {})
        
        # 帧处理性能
        fp = perf.get('frame_processing', {})
        if fp.get('fps', 0) > 0:
            all_fps.append(fp.get('fps', 0))
        if fp.get('mean_ms', 0) > 0:
            all_frame_times.append(fp.get('mean_ms', 0))
        
        # 球追踪性能
        bt = perf.get('ball_tracking', {})
        if bt.get('mean_ms', 0) > 0:
            all_ball_tracking_times.append(bt.get('mean_ms', 0))
        
        # 姿态估计性能
        pe = perf.get('pose_estimation', {})
        if pe.get('mean_ms', 0) > 0:
            all_pose_times.append(pe.get('mean_ms', 0))
        
        # GPU显存
        gpu_mem = perf.get('gpu_memory', {})
        if gpu_mem.get('mean_mb', 0) > 0:
            all_mem_usage.append(gpu_mem.get('mean_mb', 0))
        if gpu_mem.get('max_mb', 0) > 0:
            all_max_mem.append(gpu_mem.get('max_mb', 0))
    
    print("\n性能指标统计:")
    print("-" * 80)
    if all_fps:
        print(f"  平均FPS: {sum(all_fps)/len(all_fps):.2f} (范围: {min(all_fps):.2f} - {max(all_fps):.2f})")
    if all_frame_times:
        print(f"  平均帧处理时间: {sum(all_frame_times)/len(all_frame_times):.2f} ms (范围: {min(all_frame_times):.2f} - {max(all_frame_times):.2f} ms)")
    if all_ball_tracking_times:
        print(f"  平均球追踪时间: {sum(all_ball_tracking_times)/len(all_ball_tracking_times):.2f} ms (范围: {min(all_ball_tracking_times):.2f} - {max(all_ball_tracking_times):.2f} ms)")
    if all_pose_times:
        print(f"  平均姿态估计时间: {sum(all_pose_times)/len(all_pose_times):.2f} ms (范围: {min(all_pose_times):.2f} - {max(all_pose_times):.2f} ms)")
    if all_mem_usage:
        print(f"  平均显存使用: {sum(all_mem_usage)/len(all_mem_usage):.2f} MB (范围: {min(all_mem_usage):.2f} - {max(all_mem_usage):.2f} MB)")
    if all_max_mem:
        print(f"  最大显存使用: {max(all_max_mem):.2f} MB")
    
    print("=" * 80)
    
    # 保存结果到文件
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n详细结果已保存到: {output_file}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='并行多路视频流性能测试脚本（使用随机数据）')
    parser.add_argument('--num-streams', type=int, default=20,
                       help='测试的并行流数量（默认：20）')
    parser.add_argument('--output', type=str, default='data/results/performance_test',
                       help='输出目录（测试模式下不会保存实际文件）')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='配置文件路径')
    parser.add_argument('--num-gpus', type=int, default=2,
                       help='GPU数量')
    parser.add_argument('--max-workers', type=int, default=None,
                       help='最大并行进程数（默认：num_streams，即所有流并行）')
    parser.add_argument('--output-json', type=str, default=None,
                       help='结果JSON文件路径（默认：output_dir/performance_test_results.json）')
    parser.add_argument('--test-frames', type=int, default=2,
                       help='测试模式下处理的帧数（默认：2）')
    
    args = parser.parse_args()
    
    # 确定并行进程数
    if args.max_workers is not None:
        max_workers = args.max_workers
    else:
        max_workers = min(args.num_streams, args.num_gpus * 10) if args.num_gpus > 0 else args.num_streams
    
    max_workers = min(max_workers, args.num_streams)
    
    print("=" * 80)
    print("并行多路视频流性能测试（使用随机数据）")
    print("=" * 80)
    print(f"测试流数: {args.num_streams}")
    print(f"输出目录: {args.output}")
    print(f"GPU数量: {args.num_gpus}")
    print(f"并行进程数: {max_workers}")
    print(f"测试帧数: {args.test_frames}")
    print(f"测试模式: 使用随机生成的数据，不加载视频文件")
    
    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建共享的进度字典（用于多进程通信）
    manager = mp.Manager()
    progress_dict = manager.dict()
    
    # 初始化进度字典
    for stream_id in range(1, args.num_streams + 1):
        progress_dict[stream_id] = {'current': 0, 'total': args.test_frames, 'status': 'waiting'}
    
    # 准备参数
    args_list = []
    for stream_id in range(1, args.num_streams + 1):
        # 初始GPU分配（轮询）
        initial_gpu_id = (stream_id - 1) % args.num_gpus if args.num_gpus > 0 else None
        args_list.append((
            stream_id,
            str(output_dir),
            args.config,
            args.num_gpus,
            initial_gpu_id,
            progress_dict,
            args.test_frames
        ))
    
    # 开始测试
    start_time = time.time()
    results = []
    
    print(f"\n开始测试...\n")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_stream = {
            executor.submit(test_single_stream, args): args[0]
            for args in args_list
        }
        
        # 为每个流创建独立的进度条
        progress_bars = {}
        for stream_id in range(1, args.num_streams + 1):
            pbar = tqdm(
                total=args.test_frames,
                desc=f"流{stream_id:2d}: 测试中",
                unit="帧",
                position=stream_id-1,
                leave=True,
                ncols=100
            )
            progress_bars[stream_id] = pbar
        
        # 更新进度条的线程函数
        stop_progress_thread = threading.Event()
        active_tasks = set(range(1, args.num_streams + 1))
        
        def update_progress():
            while not stop_progress_thread.is_set() or active_tasks:
                for idx, pbar in progress_bars.items():
                    if idx in progress_dict:
                        progress = progress_dict[idx]
                        if progress is None or not isinstance(progress, dict):
                            continue
                        if progress.get('total', 0) > 0:
                            current = progress.get('current', 0)
                            total = progress.get('total', 0)
                            status = progress.get('status', 'processing')
                            
                            pbar.n = current
                            pbar.total = total
                            
                            if status == 'completed':
                                pbar.set_postfix_str("✓ 完成")
                                if idx in active_tasks:
                                    active_tasks.remove(idx)
                            elif status == 'error':
                                pbar.set_postfix_str("✗ 错误")
                                if idx in active_tasks:
                                    active_tasks.remove(idx)
                            else:
                                pbar.set_postfix_str("处理中...")
                            
                            pbar.refresh()
                
                if not active_tasks:
                    break
                    
                time.sleep(0.1)
        
        # 启动进度更新线程
        progress_thread = threading.Thread(target=update_progress, daemon=True)
        progress_thread.start()
        
        # 收集结果
        for future in as_completed(future_to_stream):
            stream_id = future_to_stream[future]
            result = future.result(timeout=None)
            results.append(result)
            
            # 更新进度条状态
            if stream_id in progress_dict:
                progress = progress_dict.get(stream_id)
                if progress is not None and isinstance(progress, dict):
                    progress_dict[stream_id] = {
                        'current': progress.get('total', 0),
                        'total': progress.get('total', 0),
                        'status': 'completed'
                    }
            if stream_id in active_tasks:
                active_tasks.remove(stream_id)
        
        # 停止进度更新线程
        stop_progress_thread.set()
        time.sleep(0.2)
        
        # 关闭所有进度条
        for pbar in progress_bars.values():
            pbar.close()
    
    total_time = time.time() - start_time
    
    # 按流ID排序
    results.sort(key=lambda x: x.get('stream_id', 0))
    
    # 输出结果文件路径
    if args.output_json:
        output_json = args.output_json
    else:
        output_json = str(output_dir / "performance_test_results.json")
    
    # 打印摘要
    print_performance_summary(results, output_json)
    
    print(f"\n总测试时间: {total_time:.2f}秒 ({total_time/60:.2f}分钟)")
    
    return 0


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    exit(main())
