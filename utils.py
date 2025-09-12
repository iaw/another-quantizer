# utils.py
"""Utility functions for GLM quantization"""

import torch
import json
import yaml
import psutil
import hashlib
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np
from datetime import datetime, timedelta
import logging
import re
from tqdm import tqdm
import safetensors.torch
import shutil


@dataclass
class ModelInfo:
    """Information about a model"""
    name: str
    type: str  # glm4, chatglm3, etc.
    size_gb: float
    num_params: int
    num_layers: int
    hidden_size: int
    vocab_size: int
    max_length: int
    

@dataclass
class SystemInfo:
    """System hardware information"""
    gpu_count: int
    gpu_names: List[str]
    gpu_memory_gb: List[float]
    cpu_cores: int
    cpu_memory_gb: float
    disk_free_gb: float
    cuda_version: str
    pytorch_version: str


def get_memory_info() -> Dict[str, float]:
    """Get current memory usage info
    
    Returns comprehensive memory stats:
    - GPU memory (used/total/free)
    - CPU memory (used/total/available)
    - Swap usage
    """
    memory_info = {}
    
    # GPU memory from torch.cuda
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1e9
            reserved = torch.cuda.memory_reserved(i) / 1e9
            total = torch.cuda.get_device_properties(i).total_memory / 1e9
            
            memory_info[f'gpu_{i}_allocated_gb'] = allocated
            memory_info[f'gpu_{i}_reserved_gb'] = reserved
            memory_info[f'gpu_{i}_total_gb'] = total
            memory_info[f'gpu_{i}_free_gb'] = total - allocated
    
    # CPU memory from psutil
    vm = psutil.virtual_memory()
    memory_info['cpu_used_gb'] = vm.used / 1e9
    memory_info['cpu_total_gb'] = vm.total / 1e9
    memory_info['cpu_available_gb'] = vm.available / 1e9
    memory_info['cpu_percent'] = vm.percent
    
    # Swap memory
    swap = psutil.swap_memory()
    memory_info['swap_used_gb'] = swap.used / 1e9
    memory_info['swap_total_gb'] = swap.total / 1e9
    memory_info['swap_percent'] = swap.percent
    
    return memory_info


def format_bytes(bytes_value: int) -> str:
    """Format bytes to human readable string
    
    Examples:
    - 1024 -> "1.0 KB"
    - 1048576 -> "1.0 MB"
    - 1073741824 -> "1.0 GB"
    """
    units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
    unit_index = 0
    value = float(bytes_value)
    
    while value >= 1024 and unit_index < len(units) - 1:
        value /= 1024
        unit_index += 1
    
    if unit_index == 0:
        return f"{int(value)} {units[unit_index]}"
    else:
        return f"{value:.1f} {units[unit_index]}"


def estimate_model_size(num_params: int, bits: int = 4) -> float:
    """Estimate model size in GB after quantization
    
    Size calculation:
    - FP16: params * 2 bytes
    - INT8: params * 1 byte
    - INT4: params * 0.5 bytes
    - Add overhead for scales/metadata
    """
    # Base size calculation
    bytes_per_param = bits / 8
    base_size_gb = (num_params * bytes_per_param) / 1e9
    
    # Add overhead for scales and metadata
    # For group_size=128, scales add about 3% overhead
    # For group_size=32, scales add about 12% overhead
    # Using average of 5% for estimation
    scale_overhead = base_size_gb * 0.05
    
    # Add fixed metadata overhead (approximately 100MB)
    metadata_overhead = 0.1
    
    total_size_gb = base_size_gb + scale_overhead + metadata_overhead
    
    return total_size_gb


def format_time(seconds: float) -> str:
    """Format seconds to human readable time
    
    Examples:
    - 65 -> "1m 5s"
    - 3665 -> "1h 1m 5s"
    - 90000 -> "1d 1h"
    """
    if seconds < 60:
        return f"{int(seconds)}s"
    
    # Calculate components
    days = int(seconds // 86400)
    hours = int((seconds % 86400) // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    # Format based on magnitude
    if days > 0:
        if hours > 0:
            return f"{days}d {hours}h"
        else:
            return f"{days}d"
    elif hours > 0:
        if minutes > 0:
            return f"{hours}h {minutes}m"
        else:
            return f"{hours}h"
    else:
        if secs > 0:
            return f"{minutes}m {secs}s"
        else:
            return f"{minutes}m"


def get_system_info() -> SystemInfo:
    """Get comprehensive system information
    
    System info includes:
    - Hardware specs
    - Software versions
    - Available resources
    """
    # GPU information
    gpu_count = 0
    gpu_names = []
    gpu_memory_gb = []
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        for i in range(gpu_count):
            gpu_names.append(torch.cuda.get_device_name(i))
            gpu_memory_gb.append(torch.cuda.get_device_properties(i).total_memory / 1e9)
    
    # CPU information
    cpu_cores = psutil.cpu_count(logical=False) or psutil.cpu_count()
    cpu_memory_gb = psutil.virtual_memory().total / 1e9
    
    # Disk information
    disk_stats = shutil.disk_usage('/')
    disk_free_gb = disk_stats.free / 1e9
    
    # Software versions
    cuda_version = torch.version.cuda if torch.cuda.is_available() else "N/A"
    pytorch_version = torch.__version__
    
    return SystemInfo(
        gpu_count=gpu_count,
        gpu_names=gpu_names,
        gpu_memory_gb=gpu_memory_gb,
        cpu_cores=cpu_cores,
        cpu_memory_gb=cpu_memory_gb,
        disk_free_gb=disk_free_gb,
        cuda_version=cuda_version,
        pytorch_version=pytorch_version
    )


# Stub remaining functions to prevent import errors
def load_glm_config(model_path: str) -> Dict[str, Any]:
    """Load GLM model configuration - STUB"""
    pass


def get_layer_names(model_config: Dict[str, Any]) -> List[str]:
    """Extract layer names from model config - STUB"""
    pass


def create_vllm_config(
    quantized_path: str,
    num_gpus: int = 4,
    max_length: int = 32768
) -> Dict[str, Any]:
    """Create vLLM serving configuration - STUB"""
    pass


def calculate_perplexity(
    model: Any,
    eval_dataloader: Any,
    device: str = "cuda"
) -> float:
    """Calculate model perplexity - STUB"""
    pass


def merge_quantized_layers(
    layer_dir: Path,
    output_path: Path
) -> None:
    """Merge separately quantized layers into single model - STUB"""
    pass


def verify_quantized_weights(weights_path: Path) -> bool:
    """Verify integrity of quantized weights - STUB"""
    pass


def create_model_card(
    base_model: str,
    quantization_config: Dict[str, Any],
    metrics: Dict[str, float]
) -> str:
    """Create model card for quantized model - STUB"""
    pass


def safe_torch_load(path: Path, 
                   device: str = "cpu",
                   dtype: Optional[torch.dtype] = None) -> Any:
    """Safely load torch checkpoint with error handling - STUB"""
    pass


def get_optimal_batch_size(available_memory: float,
                          model_config: Dict[str, Any]) -> int:
    """Calculate optimal batch size based on available memory - STUB"""
    pass


def detect_model_type(model_path: str) -> str:
    """Detect GLM model variant from config - STUB"""
    pass


def create_progress_bar(total: int, 
                       desc: str = "Processing",
                       unit: str = "layers") -> tqdm:
    """Create formatted progress bar - STUB"""
    pass


def compute_compression_ratio(original_size: float,
                             quantized_size: float) -> float:
    """Compute compression ratio - STUB"""
    pass


def validate_cuda_version() -> bool:
    """Validate CUDA version compatibility - STUB"""
    pass


def get_weight_statistics(weights: torch.Tensor) -> Dict[str, float]:
    """Get statistics for weight tensor - STUB"""
    pass


def create_sharded_index(weight_files: List[Path],
                        output_path: Path) -> None:
    """Create index for sharded model weights - STUB"""
    pass


def monitor_gpu_temperature() -> List[float]:
    """Monitor GPU temperatures - STUB"""
    pass


def estimate_memory_requirements(model_config: Dict[str, Any],
                                batch_size: int = 1) -> Dict[str, float]:
    """Estimate memory requirements for quantization - STUB"""
    pass


def create_benchmark_report(metrics: Dict[str, Any],
                           output_file: Path) -> None:
    """Create detailed benchmark report - STUB"""
    pass


def hash_model_weights(model_path: str) -> str:
    """Compute hash of model weights for verification - STUB"""
    pass


def cleanup_temp_files(temp_dir: Path) -> None:
    """Clean up temporary files - STUB"""
    pass


def convert_safetensors_to_pytorch(safetensors_path: Path,
                                  output_path: Path) -> None:
    """Convert safetensors format to PyTorch - STUB"""
    pass


def profile_memory_usage(func, *args, **kwargs) -> Tuple[Any, Dict[str, float]]:
    """Profile memory usage of a function - STUB"""
    pass