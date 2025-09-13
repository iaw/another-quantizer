# metrics.py
"""Metrics collection and reporting for quantization pipeline"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path
import json
import time
import numpy as np
from collections import defaultdict
import logging
from datetime import datetime


@dataclass
class LayerMetrics:
    """Metrics for a single layer"""
    # ALL fields without defaults MUST come first
    layer_name: str
    layer_type: str
    quantization_time: float
    original_size_mb: float
    quantized_size_mb: float
    compression_ratio: float
    mse_error: float
    cosine_similarity: float
    max_abs_error: float
    relative_error: float
    weight_mean: float
    weight_std: float
    weight_min: float
    weight_max: float
    weight_sparsity: float
    scale_mean: float
    scale_std: float
    scale_min: float
    scale_max: float
    peak_gpu_memory_mb: float  # NO DEFAULT VALUE
    peak_cpu_memory_mb: float   # NO DEFAULT VALUE
    
    # ALL fields with defaults MUST come last
    activation_mean: Optional[float] = None
    activation_std: Optional[float] = None
    activation_range: Optional[Tuple[float, float]] = None


@dataclass
class ModelMetrics:
    """Overall model quantization metrics"""
    total_layers: int
    quantized_layers: int
    failed_layers: List[str]
    
    # Size metrics
    original_size_gb: float
    quantized_size_gb: float
    overall_compression_ratio: float
    
    # Time metrics
    total_quantization_time: float
    average_layer_time: float
    
    # Quality metrics
    average_mse_error: float
    average_cosine_similarity: float
    max_layer_error: float
    
    
    # Memory metrics
    peak_gpu_memory_gb: float
    peak_cpu_memory_gb: float
    average_gpu_memory_gb: float
    
    # Last quality metric
    perplexity: Optional[float] = None
    # Method info
    quantization_method: str = "awq"
    bits: int = 4
    group_size: int = 128
    

class MetricsCollector:
    """Collect and aggregate metrics during quantization"""
    
    def __init__(self):
        """Initialize metrics collector"""
        self.layer_metrics: List[LayerMetrics] = []
        self.model_metrics: Optional[ModelMetrics] = None
        self.start_time = time.time()
        self.logger = logging.getLogger(__name__)
        
        # Track running statistics
        self.total_original_size = 0.0
        self.total_quantized_size = 0.0
        self.total_layers = 0
        self.failed_layers = []
        
        # Memory tracking
        self.peak_gpu_memory = 0.0
        self.peak_cpu_memory = 0.0
        self.memory_samples = []
        
        # Error tracking
        self.error_samples = []
        self.similarity_samples = []
        
    def collect_layer_metrics(self,
                             layer_name: str,
                             layer_type: str,
                             original_layer: nn.Module,
                             quantized_layer: nn.Module,
                             quantization_result: Any,
                             calibration_data: torch.Tensor,
                             time_taken: float,
                             memory_stats: Dict[str, float]) -> LayerMetrics:
        """Collect metrics for a single layer
        
        Args:
            layer_name: Name of the layer
            layer_type: Type of layer (transformer, moe, etc.)
            original_layer: Original layer module
            quantized_layer: Quantized layer module
            quantization_result: Result from quantization
            calibration_data: Calibration data used
            time_taken: Time taken for quantization
            memory_stats: Memory usage statistics
            
        Returns:
            LayerMetrics object
        """
        # Calculate sizes
        original_size_mb = self._calculate_module_size(original_layer) / 1e6
        quantized_size_mb = self._calculate_module_size(quantized_layer) / 1e6
        compression_ratio = original_size_mb / max(quantized_size_mb, 0.001)
        
        # Calculate errors
        error_metrics = self._calculate_error_metrics(
            original_layer, quantized_layer, calibration_data
        )
        
        # Get weight statistics
        weight_stats = self._get_weight_statistics(quantized_layer)
        
        # Get scale statistics if available
        scale_stats = self._get_scale_statistics(quantization_result)
        
        # Get activation statistics if available
        activation_stats = self._get_activation_statistics(calibration_data)
        
        # Create layer metrics
        metrics = LayerMetrics(
            layer_name=layer_name,
            layer_type=layer_type,
            quantization_time=time_taken,
            original_size_mb=original_size_mb,
            quantized_size_mb=quantized_size_mb,
            compression_ratio=compression_ratio,
            mse_error=error_metrics['mse'],
            cosine_similarity=error_metrics['cosine_similarity'],
            max_abs_error=error_metrics['max_abs_error'],
            relative_error=error_metrics['relative_error'],
            weight_mean=weight_stats['mean'],
            weight_std=weight_stats['std'],
            weight_min=weight_stats['min'],
            weight_max=weight_stats['max'],
            weight_sparsity=weight_stats['sparsity'],
            scale_mean=scale_stats['mean'],
            scale_std=scale_stats['std'],
            scale_min=scale_stats['min'],
            scale_max=scale_stats['max'],
            peak_gpu_memory_mb=memory_stats.get('gpu_peak_mb', 0),
            peak_cpu_memory_mb=memory_stats.get('cpu_peak_mb', 0),
            activation_mean=activation_stats.get('mean'),
            activation_std=activation_stats.get('std'),
            activation_range=activation_stats.get('range')
        )
        
        # Add to collection
        self.layer_metrics.append(metrics)
        
        # Update running statistics
        self.total_original_size += original_size_mb
        self.total_quantized_size += quantized_size_mb
        self.total_layers += 1
        self.error_samples.append(error_metrics['mse'])
        self.similarity_samples.append(error_metrics['cosine_similarity'])
        
        # Update memory peaks
        self.peak_gpu_memory = max(self.peak_gpu_memory, memory_stats.get('gpu_peak_mb', 0))
        self.peak_cpu_memory = max(self.peak_cpu_memory, memory_stats.get('cpu_peak_mb', 0))
        self.memory_samples.append(memory_stats.get('gpu_peak_mb', 0))
        
        self.logger.debug(f"Collected metrics for {layer_name}: "
                         f"compression={compression_ratio:.2f}x, "
                         f"error={error_metrics['mse']:.6f}")
        
        return metrics
    
    def add_failed_layer(self, layer_name: str) -> None:
        """Record a failed layer"""
        self.failed_layers.append(layer_name)
        self.logger.warning(f"Layer {layer_name} failed quantization")
    
    def compute_model_metrics(self,
                            perplexity: Optional[float] = None,
                            quantization_config: Dict[str, Any] = None) -> ModelMetrics:
        """Compute overall model metrics
        
        Args:
            perplexity: Optional perplexity value
            quantization_config: Quantization configuration
            
        Returns:
            ModelMetrics object
        """
        # Calculate averages
        avg_mse = np.mean(self.error_samples) if self.error_samples else 0.0
        avg_cosine = np.mean(self.similarity_samples) if self.similarity_samples else 1.0
        max_error = max(self.error_samples) if self.error_samples else 0.0
        
        # Calculate time metrics
        total_time = time.time() - self.start_time
        avg_layer_time = total_time / max(self.total_layers, 1)
        
        # Create model metrics
        self.model_metrics = ModelMetrics(
            total_layers=self.total_layers + len(self.failed_layers),
            quantized_layers=self.total_layers,
            failed_layers=self.failed_layers,
            original_size_gb=self.total_original_size / 1e3,
            quantized_size_gb=self.total_quantized_size / 1e3,
            overall_compression_ratio=self.total_original_size / max(self.total_quantized_size, 0.001),
            total_quantization_time=total_time,
            average_layer_time=avg_layer_time,
            average_mse_error=avg_mse,
            average_cosine_similarity=avg_cosine,
            max_layer_error=max_error,
            perplexity=perplexity,
            peak_gpu_memory_gb=self.peak_gpu_memory / 1e3,
            peak_cpu_memory_gb=self.peak_cpu_memory / 1e3,
            average_gpu_memory_gb=np.mean(self.memory_samples) / 1e3 if self.memory_samples else 0.0,
            quantization_method=quantization_config.get('method', 'awq') if quantization_config else 'awq',
            bits=quantization_config.get('bits', 4) if quantization_config else 4,
            group_size=quantization_config.get('group_size', 128) if quantization_config else 128
        )
        
        return self.model_metrics
    
    def save_metrics(self, output_path: str) -> None:
        """Save all metrics to file
        
        Args:
            output_path: Path to save metrics
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare data for JSON serialization
        metrics_data = {
            'timestamp': datetime.now().isoformat(),
            'model_metrics': asdict(self.model_metrics) if self.model_metrics else None,
            'layer_metrics': [asdict(m) for m in self.layer_metrics],
            'summary': self.get_summary()
        }
        
        # Save to JSON
        with open(output_path, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        self.logger.info(f"Metrics saved to {output_path}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of metrics
        
        Returns:
            Summary dictionary
        """
        if not self.layer_metrics:
            return {'status': 'No metrics collected'}
        
        # Calculate percentiles for errors
        errors = [m.mse_error for m in self.layer_metrics]
        
        summary = {
            'total_layers_processed': len(self.layer_metrics),
            'failed_layers': len(self.failed_layers),
            'compression': {
                'original_size_gb': self.total_original_size / 1e3,
                'quantized_size_gb': self.total_quantized_size / 1e3,
                'compression_ratio': self.total_original_size / max(self.total_quantized_size, 0.001)
            },
            'error_statistics': {
                'mean_mse': np.mean(errors),
                'median_mse': np.median(errors),
                'std_mse': np.std(errors),
                'min_mse': np.min(errors),
                'max_mse': np.max(errors),
                'p95_mse': np.percentile(errors, 95),
                'p99_mse': np.percentile(errors, 99)
            },
            'time': {
                'total_seconds': time.time() - self.start_time,
                'average_per_layer': (time.time() - self.start_time) / max(len(self.layer_metrics), 1)
            },
            'memory': {
                'peak_gpu_gb': self.peak_gpu_memory / 1e3,
                'peak_cpu_gb': self.peak_cpu_memory / 1e3,
                'average_gpu_gb': np.mean(self.memory_samples) / 1e3 if self.memory_samples else 0.0
            }
        }
        
        return summary
    
    def print_summary(self) -> None:
        """Print metrics summary to console"""
        summary = self.get_summary()
        
        print("\n" + "="*80)
        print("QUANTIZATION METRICS SUMMARY")
        print("="*80)
        
        print(f"\nLayers: {summary['total_layers_processed']} processed, "
              f"{summary['failed_layers']} failed")
        
        print(f"\nCompression:")
        print(f"  Original: {summary['compression']['original_size_gb']:.2f} GB")
        print(f"  Quantized: {summary['compression']['quantized_size_gb']:.2f} GB")
        print(f"  Ratio: {summary['compression']['compression_ratio']:.2f}x")
        
        print(f"\nError Statistics (MSE):")
        print(f"  Mean: {summary['error_statistics']['mean_mse']:.6f}")
        print(f"  Median: {summary['error_statistics']['median_mse']:.6f}")
        print(f"  Max: {summary['error_statistics']['max_mse']:.6f}")
        print(f"  95th percentile: {summary['error_statistics']['p95_mse']:.6f}")
        
        print(f"\nTime:")
        print(f"  Total: {summary['time']['total_seconds']:.1f} seconds")
        print(f"  Per layer: {summary['time']['average_per_layer']:.1f} seconds")
        
        print(f"\nMemory:")
        print(f"  Peak GPU: {summary['memory']['peak_gpu_gb']:.2f} GB")
        print(f"  Average GPU: {summary['memory']['average_gpu_gb']:.2f} GB")
        
        print("="*80 + "\n")
    
    # Helper methods
    
    def _calculate_module_size(self, module: nn.Module) -> float:
        """Calculate size of module in bytes"""
        total_size = 0
        for param in module.parameters():
            total_size += param.numel() * param.element_size()
        for buffer in module.buffers():
            total_size += buffer.numel() * buffer.element_size()
        return total_size
    
    def _calculate_error_metrics(self,
                                original: nn.Module,
                                quantized: nn.Module,
                                test_input: torch.Tensor) -> Dict[str, float]:
        """Calculate error metrics between original and quantized"""
        metrics = {
            'mse': 0.0,
            'cosine_similarity': 1.0,
            'max_abs_error': 0.0,
            'relative_error': 0.0
        }
        
        try:
            with torch.no_grad():
                # Get outputs
                device = next(original.parameters()).device
                test_input = test_input.to(device)
                
                # Handle different input shapes
                if test_input.dim() == 2:
                    test_input = test_input.unsqueeze(0)
                
                orig_output = original(test_input)
                quant_output = quantized(test_input)
                
                # Handle tuple outputs
                if isinstance(orig_output, tuple):
                    orig_output = orig_output[0]
                if isinstance(quant_output, tuple):
                    quant_output = quant_output[0]
                
                # Calculate metrics
                diff = orig_output - quant_output
                metrics['mse'] = torch.mean(diff ** 2).item()
                metrics['max_abs_error'] = torch.max(torch.abs(diff)).item()
                
                # Cosine similarity
                orig_flat = orig_output.reshape(-1)
                quant_flat = quant_output.reshape(-1)
                metrics['cosine_similarity'] = F.cosine_similarity(
                    orig_flat.unsqueeze(0),
                    quant_flat.unsqueeze(0)
                ).item()
                
                # Relative error
                orig_norm = torch.norm(orig_output).item()
                if orig_norm > 0:
                    metrics['relative_error'] = torch.norm(diff).item() / orig_norm
                    
        except Exception as e:
            self.logger.warning(f"Error calculating metrics: {e}")
        
        return metrics
    
    def _get_weight_statistics(self, module: nn.Module) -> Dict[str, float]:
        """Get weight statistics from module"""
        stats = {
            'mean': 0.0,
            'std': 0.0,
            'min': 0.0,
            'max': 0.0,
            'sparsity': 0.0
        }
        
        all_weights = []
        for param in module.parameters():
            all_weights.append(param.data.cpu().flatten())
        
        if all_weights:
            weights = torch.cat(all_weights)
            stats['mean'] = weights.mean().item()
            stats['std'] = weights.std().item()
            stats['min'] = weights.min().item()
            stats['max'] = weights.max().item()
            stats['sparsity'] = (weights == 0).float().mean().item()
        
        return stats
    
    def _get_scale_statistics(self, quantization_result: Any) -> Dict[str, float]:
        """Get scale statistics from quantization result"""
        stats = {
            'mean': 0.0,
            'std': 0.0,
            'min': 0.0,
            'max': 0.0
        }
        
        if hasattr(quantization_result, 'scales') and quantization_result.scales:
            all_scales = []
            for scale_tensor in quantization_result.scales.values():
                if torch.is_tensor(scale_tensor):
                    all_scales.append(scale_tensor.cpu().flatten())
            
            if all_scales:
                scales = torch.cat(all_scales)
                stats['mean'] = scales.mean().item()
                stats['std'] = scales.std().item()
                stats['min'] = scales.min().item()
                stats['max'] = scales.max().item()
        
        return stats
    
    def _get_activation_statistics(self, activation: torch.Tensor) -> Dict[str, Any]:
        """Get activation statistics"""
        stats = {}
        
        if activation is not None and torch.is_tensor(activation):
            stats['mean'] = activation.mean().item()
            stats['std'] = activation.std().item()
            stats['range'] = (activation.min().item(), activation.max().item())
        
        return stats