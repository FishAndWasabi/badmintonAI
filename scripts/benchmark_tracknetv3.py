"""Benchmark script for TrackNetv3 latency testing"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from statistics import mean, median, stdev
from typing import Dict, List, Tuple

# 添加TrackNetv3路径
tracknetv3_path = Path(__file__).parent.parent / "models" / "ball_tracking" / "TrackNetv3"
sys.path.insert(0, str(tracknetv3_path))

from utils.general import get_model, HEIGHT, WIDTH
from model import TrackNet, InpaintNet


class TrackNetv3Benchmark:
    """TrackNetv3性能测试类"""
    
    def __init__(self, device: str = 'cuda'):
        """
        初始化测试环境
        
        Args:
            device: 计算设备 ('cuda' 或 'cpu')
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA版本: {torch.version.cuda}")
        
        self.results = {}
    
    def load_model(self, model_path: str, model_type: str = 'TrackNet') -> nn.Module:
        """
        加载模型
        
        Args:
            model_path: 模型文件路径
            model_type: 模型类型 ('TrackNet' 或 'InpaintNet')
            
        Returns:
            加载的模型
        """
        print(f"\n加载{model_type}模型: {model_path}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        start_time = time.time()
        
        if model_type == 'TrackNet':
            ckpt = torch.load(model_path, map_location=self.device)
            seq_len = ckpt['param_dict']['seq_len']
            bg_mode = ckpt['param_dict']['bg_mode']
            model = get_model('TrackNet', seq_len, bg_mode).to(self.device)
            model.load_state_dict(ckpt['model'])
            model.eval()
            self.seq_len = seq_len
            self.bg_mode = bg_mode
        elif model_type == 'InpaintNet':
            ckpt = torch.load(model_path, map_location=self.device)
            seq_len = ckpt['param_dict']['seq_len']
            model = get_model('InpaintNet').to(self.device)
            model.load_state_dict(ckpt['model'])
            model.eval()
            self.inpaint_seq_len = seq_len
        else:
            raise ValueError(f"未知的模型类型: {model_type}")
        
        load_time = time.time() - start_time
        print(f"模型加载时间: {load_time:.3f} 秒")
        
        # 计算模型大小
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        model_size_mb = (param_size + buffer_size) / (1024 ** 2)
        print(f"模型大小: {model_size_mb:.2f} MB")
        
        return model
    
    def generate_test_data(self, batch_size: int, seq_len: int, bg_mode: str = 'concat') -> torch.Tensor:
        """
        生成测试数据
        
        Args:
            batch_size: 批次大小
            seq_len: 序列长度
            bg_mode: 背景模式
            
        Returns:
            测试数据张量
        """
        if bg_mode == 'subtract':
            in_dim = seq_len
        elif bg_mode == 'subtract_concat':
            in_dim = seq_len * 4
        elif bg_mode == 'concat':
            in_dim = (seq_len + 1) * 3
        else:
            in_dim = seq_len * 3
        
        # 生成随机输入数据 (归一化到[0, 1])
        data = torch.rand(batch_size, in_dim, HEIGHT, WIDTH, dtype=torch.float32)
        return data.to(self.device)
    
    def benchmark_inference(self, model: nn.Module, 
                           batch_sizes: List[int],
                           num_warmup: int = 10,
                           num_iterations: int = 100) -> Dict:
        """
        测试推理性能
        
        Args:
            model: 模型
            batch_sizes: 要测试的批次大小列表
            num_warmup: 预热迭代次数
            num_iterations: 测试迭代次数
            
        Returns:
            性能测试结果字典
        """
        results = {}
        
        # 获取模型参数
        if hasattr(self, 'seq_len'):
            seq_len = self.seq_len
            bg_mode = self.bg_mode
        else:
            seq_len = 8  # 默认值
            bg_mode = 'concat'
        
        print(f"\n开始性能测试 (序列长度: {seq_len}, 背景模式: {bg_mode})")
        print(f"预热迭代: {num_warmup}, 测试迭代: {num_iterations}")
        
        for batch_size in batch_sizes:
            print(f"\n测试批次大小: {batch_size}")
            
            # 生成测试数据
            test_data = self.generate_test_data(batch_size, seq_len, bg_mode)
            
            # 预热
            print("  预热中...", end="", flush=True)
            with torch.no_grad():
                for _ in range(num_warmup):
                    _ = model(test_data)
            
            # 同步GPU（如果使用CUDA）
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            
            print(" 完成")
            
            # 测试推理时间
            print("  测试推理时间...", end="", flush=True)
            inference_times = []
            
            with torch.no_grad():
                for _ in range(num_iterations):
                    if self.device.type == 'cuda':
                        torch.cuda.synchronize()
                    
                    start_time = time.time()
                    _ = model(test_data)
                    
                    if self.device.type == 'cuda':
                        torch.cuda.synchronize()
                    
                    end_time = time.time()
                    inference_times.append((end_time - start_time) * 1000)  # 转换为毫秒
            
            print(" 完成")
            
            # 统计结果
            avg_time = mean(inference_times)
            median_time = median(inference_times)
            min_time = min(inference_times)
            max_time = max(inference_times)
            std_time = stdev(inference_times) if len(inference_times) > 1 else 0
            
            # 计算FPS（每秒处理的帧数）
            # 每个batch处理seq_len帧，所以总帧数 = batch_size * seq_len
            total_frames = batch_size * seq_len
            fps = total_frames / (avg_time / 1000)
            
            results[batch_size] = {
                'avg_time_ms': avg_time,
                'median_time_ms': median_time,
                'min_time_ms': min_time,
                'max_time_ms': max_time,
                'std_time_ms': std_time,
                'fps': fps,
                'frames_per_batch': total_frames
            }
            
            print(f"  平均推理时间: {avg_time:.2f} ms")
            print(f"  中位数推理时间: {median_time:.2f} ms")
            print(f"  最小/最大: {min_time:.2f} / {max_time:.2f} ms")
            print(f"  标准差: {std_time:.2f} ms")
            print(f"  FPS: {fps:.2f}")
        
        return results
    
    def benchmark_end_to_end(self, tracknet_model: nn.Module,
                            inpaintnet_model: nn.Module = None,
                            batch_size: int = 1,
                            num_iterations: int = 50) -> Dict:
        """
        端到端性能测试（TrackNet + InpaintNet）
        
        Args:
            tracknet_model: TrackNet模型
            inpaintnet_model: InpaintNet模型（可选）
            batch_size: 批次大小
            num_iterations: 迭代次数
            
        Returns:
            端到端测试结果
        """
        print(f"\n端到端性能测试 (批次大小: {batch_size})")
        
        seq_len = self.seq_len
        bg_mode = self.bg_mode
        
        # 生成测试数据
        test_data = self.generate_test_data(batch_size, seq_len, bg_mode)
        
        # 预热
        print("预热中...", end="", flush=True)
        with torch.no_grad():
            for _ in range(10):
                y_pred = tracknet_model(test_data)
                if inpaintnet_model is not None:
                    # 模拟InpaintNet输入（需要坐标预测）
                    coor_pred = torch.rand(batch_size, self.inpaint_seq_len, 2).to(self.device)
                    inpaint_mask = torch.rand(batch_size, self.inpaint_seq_len, 1).to(self.device)
                    _ = inpaintnet_model(coor_pred, inpaint_mask)
        
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        print(" 完成")
        
        # 测试
        tracknet_times = []
        inpaintnet_times = []
        total_times = []
        
        print("测试中...", end="", flush=True)
        with torch.no_grad():
            for _ in range(num_iterations):
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                
                start_total = time.time()
                
                # TrackNet推理
                start_tracknet = time.time()
                y_pred = tracknet_model(test_data)
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                tracknet_time = (time.time() - start_tracknet) * 1000
                tracknet_times.append(tracknet_time)
                
                # InpaintNet推理（如果提供）
                if inpaintnet_model is not None:
                    coor_pred = torch.rand(batch_size, self.inpaint_seq_len, 2).to(self.device)
                    inpaint_mask = torch.rand(batch_size, self.inpaint_seq_len, 1).to(self.device)
                    
                    start_inpaint = time.time()
                    _ = inpaintnet_model(coor_pred, inpaint_mask)
                    if self.device.type == 'cuda':
                        torch.cuda.synchronize()
                    inpaint_time = (time.time() - start_inpaint) * 1000
                    inpaintnet_times.append(inpaint_time)
                
                total_time = (time.time() - start_total) * 1000
                total_times.append(total_time)
        
        print(" 完成")
        
        results = {
            'tracknet': {
                'avg_ms': mean(tracknet_times),
                'median_ms': median(tracknet_times),
                'min_ms': min(tracknet_times),
                'max_ms': max(tracknet_times)
            },
            'total': {
                'avg_ms': mean(total_times),
                'median_ms': median(total_times),
                'min_ms': min(total_times),
                'max_ms': max(total_times)
            }
        }
        
        if inpaintnet_model is not None:
            results['inpaintnet'] = {
                'avg_ms': mean(inpaintnet_times),
                'median_ms': median(inpaintnet_times),
                'min_ms': min(inpaintnet_times),
                'max_ms': max(inpaintnet_times)
            }
        
        return results
    
    def print_results(self, results: Dict):
        """打印测试结果"""
        print("\n" + "=" * 80)
        print("性能测试结果")
        print("=" * 80)
        
        if 'batch_sizes' in results:
            print("\n批次大小测试结果:")
            print(f"{'Batch Size':<12} {'Avg (ms)':<12} {'Median (ms)':<12} {'FPS':<12} {'Frames/Batch':<15}")
            print("-" * 80)
            for batch_size, stats in results['batch_sizes'].items():
                print(f"{batch_size:<12} {stats['avg_time_ms']:<12.2f} {stats['median_time_ms']:<12.2f} "
                      f"{stats['fps']:<12.2f} {stats['frames_per_batch']:<15}")
        
        if 'end_to_end' in results:
            print("\n端到端测试结果:")
            e2e = results['end_to_end']
            print(f"TrackNet平均时间: {e2e['tracknet']['avg_ms']:.2f} ms")
            if 'inpaintnet' in e2e:
                print(f"InpaintNet平均时间: {e2e['inpaintnet']['avg_ms']:.2f} ms")
            print(f"总平均时间: {e2e['total']['avg_ms']:.2f} ms")
        
        print("=" * 80)
    
    def save_results(self, results: Dict, output_file: str):
        """保存测试结果到文件"""
        import json
        
        # 转换numpy类型为Python原生类型
        def convert_to_native(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(item) for item in obj]
            return obj
        
        results_native = convert_to_native(results)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results_native, f, indent=2, ensure_ascii=False)
        
        print(f"\n结果已保存到: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='TrackNetv3性能测试脚本')
    parser.add_argument('--tracknet_file', type=str, required=True,
                       help='TrackNet模型文件路径')
    parser.add_argument('--inpaintnet_file', type=str, default='',
                       help='InpaintNet模型文件路径（可选）')
    parser.add_argument('--batch_sizes', type=int, nargs='+', default=[1, 4, 8, 16],
                       help='要测试的批次大小列表')
    parser.add_argument('--num_warmup', type=int, default=10,
                       help='预热迭代次数')
    parser.add_argument('--num_iterations', type=int, default=100,
                       help='测试迭代次数')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='计算设备')
    parser.add_argument('--output', type=str, default='benchmark_results.json',
                       help='结果输出文件路径')
    parser.add_argument('--test_e2e', action='store_true',
                       help='是否进行端到端测试')
    
    args = parser.parse_args()
    
    # 创建测试实例
    benchmark = TrackNetv3Benchmark(device=args.device)
    
    # 加载模型
    tracknet_model = benchmark.load_model(args.tracknet_file, 'TrackNet')
    
    inpaintnet_model = None
    if args.inpaintnet_file and os.path.exists(args.inpaintnet_file):
        inpaintnet_model = benchmark.load_model(args.inpaintnet_file, 'InpaintNet')
    
    # 批次大小测试
    print("\n" + "=" * 80)
    print("批次大小性能测试")
    print("=" * 80)
    batch_results = benchmark.benchmark_inference(
        tracknet_model,
        args.batch_sizes,
        num_warmup=args.num_warmup,
        num_iterations=args.num_iterations
    )
    
    results = {
        'device': str(benchmark.device),
        'tracknet_model': args.tracknet_file,
        'batch_sizes': batch_results
    }
    
    # 端到端测试
    if args.test_e2e:
        print("\n" + "=" * 80)
        print("端到端性能测试")
        print("=" * 80)
        e2e_results = benchmark.benchmark_end_to_end(
            tracknet_model,
            inpaintnet_model,
            batch_size=args.batch_sizes[0],
            num_iterations=args.num_iterations
        )
        results['end_to_end'] = e2e_results
    
    # 打印结果
    benchmark.print_results(results)
    
    # 保存结果
    benchmark.save_results(results, args.output)


if __name__ == '__main__':
    main()
