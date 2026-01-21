"""Performance monitoring utilities for GPU memory and latency"""

import time
import torch
import numpy as np
from typing import Dict, List, Optional
from collections import deque
from dataclasses import dataclass, field

import psutil


@dataclass
class PerformanceStats:
    """Performance statistics"""
    frame_times: deque = field(default_factory=lambda: deque(maxlen=100))  # Total frame time (including visualization)
    inference_times: deque = field(default_factory=lambda: deque(maxlen=100))  # Inference time only (no visualization)
    ball_tracking_times: deque = field(default_factory=lambda: deque(maxlen=100))
    pose_estimation_times: deque = field(default_factory=lambda: deque(maxlen=100))
    integration_times: deque = field(default_factory=lambda: deque(maxlen=100))
    
    gpu_memory_used: List[float] = field(default_factory=list)
    gpu_memory_allocated: List[float] = field(default_factory=list)
    
    def get_stats(self) -> Dict:
        """Get statistics"""
        stats = {}
        
        # FPS based on inference time only (excluding visualization)
        if len(self.inference_times) > 0:
            stats['frame_processing'] = {
                'mean_ms': np.mean(self.inference_times) * 1000,
                'median_ms': np.median(self.inference_times) * 1000,
                'min_ms': np.min(self.inference_times) * 1000,
                'max_ms': np.max(self.inference_times) * 1000,
                'std_ms': np.std(self.inference_times) * 1000,
                'fps': 1.0 / np.mean(self.inference_times) if np.mean(self.inference_times) > 0 else 0
            }
        elif len(self.frame_times) > 0:
            # Fallback to frame_times if inference_times not available
            stats['frame_processing'] = {
                'mean_ms': np.mean(self.frame_times) * 1000,
                'median_ms': np.median(self.frame_times) * 1000,
                'min_ms': np.min(self.frame_times) * 1000,
                'max_ms': np.max(self.frame_times) * 1000,
                'std_ms': np.std(self.frame_times) * 1000,
                'fps': 1.0 / np.mean(self.frame_times) if np.mean(self.frame_times) > 0 else 0
            }
        
        # Also include total frame time (with visualization) for reference
        if len(self.frame_times) > 0:
            stats['total_frame_processing'] = {
                'mean_ms': np.mean(self.frame_times) * 1000,
                'median_ms': np.median(self.frame_times) * 1000,
                'min_ms': np.min(self.frame_times) * 1000,
                'max_ms': np.max(self.frame_times) * 1000,
                'std_ms': np.std(self.frame_times) * 1000,
            }
        
        if len(self.ball_tracking_times) > 0:
            stats['ball_tracking'] = {
                'mean_ms': np.mean(self.ball_tracking_times) * 1000,
                'median_ms': np.median(self.ball_tracking_times) * 1000,
            }
        
        if len(self.pose_estimation_times) > 0:
            stats['pose_estimation'] = {
                'mean_ms': np.mean(self.pose_estimation_times) * 1000,
                'median_ms': np.median(self.pose_estimation_times) * 1000,
            }
        
        if len(self.integration_times) > 0:
            stats['integration'] = {
                'mean_ms': np.mean(self.integration_times) * 1000,
                'median_ms': np.median(self.integration_times) * 1000,
            }
        
        if len(self.gpu_memory_used) > 0:
            stats['gpu_memory'] = {
                'mean_mb': np.mean(self.gpu_memory_used),
                'max_mb': np.max(self.gpu_memory_used),
                'min_mb': np.min(self.gpu_memory_used),
            }
        
        return stats


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self, num_gpus: int = 2):
        """
        初始化性能监控器
        
        Args:
            num_gpus: GPU数量
        """
        self.num_gpus = num_gpus
        self.stats = PerformanceStats()
        self.start_time = None
        self.inference_end_time = None
        self.end_time = None
        
    def get_gpu_memory(self) -> Dict[int, Dict[str, float]]:
        """
        Get GPU memory usage
        
        Returns:
            Dictionary with GPU ID as key and memory info (MB) as value
        """
        if not torch.cuda.is_available():
            return {}
        
        memory_info = {}
        for gpu_id in range(self.num_gpus):
            if gpu_id < torch.cuda.device_count():
                allocated = torch.cuda.memory_allocated(gpu_id) / (1024 ** 2)  # MB
                reserved = torch.cuda.memory_reserved(gpu_id) / (1024 ** 2)  # MB
                total = torch.cuda.get_device_properties(gpu_id).total_memory / (1024 ** 2)  # MB
                
                memory_info[gpu_id] = {
                    'allocated_mb': allocated,
                    'reserved_mb': reserved,
                    'total_mb': total,
                    'used_percent': (allocated / total) * 100 if total > 0 else 0
                }
        
        return memory_info
    
    def start_frame(self):
        """Start frame processing timer (for inference time measurement)"""
        self.start_time = time.time()
    
    def end_inference(self):
        """End inference timer (before visualization)"""
        if self.start_time is not None:
            elapsed = time.time() - self.start_time
            self.stats.inference_times.append(elapsed)
            self.inference_end_time = time.time()
    
    def end_frame(self):
        """End frame processing timer (after visualization)"""
        if self.start_time is not None:
            elapsed = time.time() - self.start_time
            self.stats.frame_times.append(elapsed)
            self.start_time = None
            self.inference_end_time = None
    
    def record_ball_tracking(self, elapsed_time: float):
        """Record ball tracking time"""
        self.stats.ball_tracking_times.append(elapsed_time)
    
    def record_pose_estimation(self, elapsed_time: float):
        """Record pose estimation time"""
        self.stats.pose_estimation_times.append(elapsed_time)
    
    def record_integration(self, elapsed_time: float):
        """Record information integration time"""
        self.stats.integration_times.append(elapsed_time)
    
    def update_gpu_memory(self):
        """Update GPU memory usage"""
        memory_info = self.get_gpu_memory()
        if memory_info:
            # Record average usage across all GPUs
            total_allocated = sum(info['allocated_mb'] for info in memory_info.values())
            self.stats.gpu_memory_allocated.append(total_allocated)
            
            total_used = sum(info['reserved_mb'] for info in memory_info.values())
            self.stats.gpu_memory_used.append(total_used)
    
    def print_stats(self):
        """Print statistics"""
        stats = self.stats.get_stats()
        
        print("\n" + "=" * 80)
        print("Performance Statistics")
        print("=" * 80)
        
        if 'frame_processing' in stats:
            fp = stats['frame_processing']
            print(f"\nFrame Processing Performance:")
            print(f"  Mean time: {fp['mean_ms']:.2f} ms")
            print(f"  Median time: {fp['median_ms']:.2f} ms")
            print(f"  Min/Max: {fp['min_ms']:.2f} / {fp['max_ms']:.2f} ms")
            print(f"  Std dev: {fp['std_ms']:.2f} ms")
            print(f"  Processing speed: {fp['fps']:.2f} FPS")
        
        if 'ball_tracking' in stats:
            bt = stats['ball_tracking']
            print(f"\nBall Tracking Performance:")
            print(f"  Mean time: {bt['mean_ms']:.2f} ms")
            print(f"  Median time: {bt['median_ms']:.2f} ms")
        
        if 'pose_estimation' in stats:
            pe = stats['pose_estimation']
            print(f"\nPose Estimation Performance:")
            print(f"  Mean time: {pe['mean_ms']:.2f} ms")
            print(f"  Median time: {pe['median_ms']:.2f} ms")
        
        if 'integration' in stats:
            it = stats['integration']
            print(f"\nIntegration Performance:")
            print(f"  Mean time: {it['mean_ms']:.2f} ms")
            print(f"  Median time: {it['median_ms']:.2f} ms")
        
        if 'gpu_memory' in stats:
            gm = stats['gpu_memory']
            print(f"\nGPU Memory Usage:")
            print(f"  Mean usage: {gm['mean_mb']:.2f} MB")
            print(f"  Max usage: {gm['max_mb']:.2f} MB")
            print(f"  Min usage: {gm['min_mb']:.2f} MB")
        
        # Print current GPU memory
        memory_info = self.get_gpu_memory()
        if memory_info:
            print(f"\nCurrent GPU Memory Status:")
            for gpu_id, info in memory_info.items():
                print(f"  GPU {gpu_id}:")
                print(f"    Allocated: {info['allocated_mb']:.2f} MB")
                print(f"    Reserved: {info['reserved_mb']:.2f} MB")
                print(f"    Total: {info['total_mb']:.2f} MB")
                print(f"    Usage: {info['used_percent']:.2f}%")
        
        print("=" * 80)
    
    def save_stats(self, output_path: str):
        """Save statistics to file"""
        import json
        
        stats = self.stats.get_stats()
        memory_info = self.get_gpu_memory()
        
        output = {
            'performance_stats': stats,
            'current_gpu_memory': memory_info,
            'num_gpus': self.num_gpus
        }
        
        # Convert numpy types
        def convert_to_native(obj):
            if isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(item) for item in obj]
            return obj
        
        output = convert_to_native(output)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"\nPerformance statistics saved to: {output_path}")
