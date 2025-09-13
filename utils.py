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


def validate_model_path(model_path: str) -> Tuple[bool, str]:
    """Validate model path and check for required files
    
    Args:
        model_path: Path to model directory
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    model_path = Path(model_path)
    
    if not model_path.exists():
        return False, f"Model path does not exist: {model_path}"
    
    if not model_path.is_dir():
        return False, f"Model path is not a directory: {model_path}"
    
    # Check for required files
    required_files = ["config.json"]
    missing_files = []
    
    for file_name in required_files:
        if not (model_path / file_name).exists():
            missing_files.append(file_name)
    
    if missing_files:
        return False, f"Missing required files: {', '.join(missing_files)}"
    
    # Check for weight files
    weight_files = list(model_path.glob("*.safetensors"))
    weight_files.extend(model_path.glob("*.bin"))
    weight_files.extend(model_path.glob("*.pt"))
    
    if not weight_files:
        return False, "No model weight files found (.safetensors, .bin, or .pt)"
    
    # Check config is valid JSON
    try:
        with open(model_path / "config.json", 'r') as f:
            config = json.load(f)
        
        # Check for required config fields
        required_fields = ["hidden_size", "num_hidden_layers"]
        missing_fields = [field for field in required_fields if field not in config and f"num_{field}" not in config]
        
        if missing_fields:
            return False, f"Config missing required fields: {', '.join(missing_fields)}"
            
    except json.JSONDecodeError as e:
        return False, f"Invalid config.json: {e}"
    except Exception as e:
        return False, f"Error reading config.json: {e}"
    
    return True, "Model path is valid"


def validate_output_path(output_path: str) -> Tuple[bool, str]:
    """Validate output path is writable
    
    Args:
        output_path: Path for output
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    output_path = Path(output_path)
    
    # Check parent directory exists and is writable
    parent_dir = output_path.parent
    
    if not parent_dir.exists():
        try:
            parent_dir.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            return False, f"Cannot create output directory (permission denied): {parent_dir}"
        except Exception as e:
            return False, f"Cannot create output directory: {e}"
    
    # Check if we can write to the directory
    try:
        test_file = parent_dir / ".write_test"
        test_file.touch()
        test_file.unlink()
    except PermissionError:
        return False, f"Output directory is not writable: {parent_dir}"
    except Exception as e:
        return False, f"Cannot write to output directory: {e}"
    
    # Check disk space
    free_space = shutil.disk_usage(str(parent_dir)).free / 1e9
    if free_space < 10:  # Less than 10GB free
        return False, f"Insufficient disk space: {free_space:.1f}GB free (need at least 10GB)"
    
    return True, "Output path is valid"


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


def load_glm_config(model_path: str) -> Dict[str, Any]:
    """Load GLM model configuration
    
    Args:
        model_path: Path to model directory
        
    Returns:
        Model configuration dictionary
    """
    model_path = Path(model_path)
    config_file = model_path / "config.json"
    
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Extract key information
    model_info = {
        'model_type': config.get('model_type', 'glm'),
        'num_layers': config.get('num_hidden_layers', config.get('num_layers')),
        'hidden_size': config.get('hidden_size'),
        'num_attention_heads': config.get('num_attention_heads'),
        'intermediate_size': config.get('intermediate_size', config.get('ffn_hidden_size')),
        'vocab_size': config.get('vocab_size', config.get('padded_vocab_size')),
        'max_position_embeddings': config.get('max_position_embeddings', 2048),
        'num_experts': config.get('num_experts'),
        'num_shared_experts': config.get('num_shared_experts'),
        'torch_dtype': config.get('torch_dtype', 'float16'),
        'original_config': config  # Keep full config for reference
    }
    
    return model_info


def get_layer_names(model_config: Dict[str, Any]) -> List[str]:
    """Extract layer names from model config
    
    Args:
        model_config: Model configuration dictionary
        
    Returns:
        List of layer names in order
    """
    layer_names = []
    
    # Get number of layers
    num_layers = model_config.get('num_layers', model_config.get('num_hidden_layers', 0))
    
    # Embedding layer
    layer_names.append("transformer.embedding")
    
    # Transformer layers
    for i in range(num_layers):
        layer_names.append(f"transformer.layers.{i}")
    
    # Final layer norm
    layer_names.append("transformer.final_layernorm")
    
    # Output head
    layer_names.append("lm_head")
    
    # Alternative naming for some GLM variants
    if 'output_layer' in str(model_config.get('original_config', {})):
        layer_names.append("output_layer")
    
    return layer_names


def create_vllm_config(
    quantized_path: str,
    num_gpus: int = 4,
    max_length: int = 32768
) -> Dict[str, Any]:
    """Create vLLM serving configuration
    
    Args:
        quantized_path: Path to quantized model
        num_gpus: Number of GPUs for tensor parallel
        max_length: Maximum sequence length
        
    Returns:
        vLLM configuration dictionary
    """
    config = {
        "model": str(quantized_path),
        "tokenizer": str(quantized_path),
        "tokenizer_mode": "auto",
        "trust_remote_code": True,
        "dtype": "auto",  # Will auto-detect from model
        "quantization": "awq",
        
        # Parallelism settings
        "tensor_parallel_size": num_gpus,
        "pipeline_parallel_size": 1,
        
        # Memory settings
        "gpu_memory_utilization": 0.95,
        "max_model_len": max_length,
        "enforce_eager": False,  # Use CUDA graphs for better performance
        
        # AWQ specific settings
        "quantization_param_path": str(Path(quantized_path) / "quantize_config.json"),
        
        # Performance settings
        "max_num_batched_tokens": max_length,
        "max_num_seqs": 256,
        "max_paddings": 256,
        
        # Disable some features for AWQ
        "disable_custom_all_reduce": False,
        "enable_lora": False,
        "enable_prefix_caching": False,
    }
    
    return config


def calculate_perplexity(
    model: Any,
    eval_dataloader: Any,
    device: str = "cuda"
) -> float:
    """Calculate model perplexity
    
    Args:
        model: Language model with forward method
        eval_dataloader: DataLoader with evaluation data
        device: Device to run on
        
    Returns:
        Perplexity value
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Calculating perplexity"):
            # Move batch to device
            if isinstance(batch, dict):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch.get('attention_mask', torch.ones_like(input_ids)).to(device)
            else:
                input_ids = batch.input_ids.to(device)
                attention_mask = batch.attention_mask.to(device) if hasattr(batch, 'attention_mask') else torch.ones_like(input_ids)
            
            # Shift for language modeling
            labels = input_ids[:, 1:].contiguous()
            input_ids = input_ids[:, :-1].contiguous()
            attention_mask = attention_mask[:, :-1].contiguous()
            
            # Forward pass
            outputs = model(input_ids, attention_mask=attention_mask)
            
            # Get logits
            if isinstance(outputs, tuple):
                logits = outputs[0]
            elif hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs
            
            # Calculate loss
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            
            # Mask padding tokens
            mask = attention_mask.view(-1).bool()
            loss = loss * mask
            
            total_loss += loss.sum().item()
            total_tokens += mask.sum().item()
    
    # Calculate perplexity
    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = np.exp(avg_loss)
    
    return perplexity


def merge_quantized_layers(
    layer_dir: Path,
    output_path: Path
) -> None:
    """Merge separately quantized layers into single model
    
    Args:
        layer_dir: Directory containing individual layer files
        output_path: Output path for merged model
    """
    layer_dir = Path(layer_dir)
    output_path = Path(output_path)
    
    if not layer_dir.exists():
        raise FileNotFoundError(f"Layer directory not found: {layer_dir}")
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all layer files
    layer_files = list(layer_dir.glob("*.safetensors"))
    if not layer_files:
        # Try .pt files as fallback
        layer_files = list(layer_dir.glob("*.pt"))
    
    if not layer_files:
        raise ValueError(f"No layer files found in {layer_dir}")
    
    logging.info(f"Found {len(layer_files)} layer files to merge")
    
    # Load and merge all layers
    merged_state_dict = {}
    metadata = {}
    
    for layer_file in tqdm(layer_files, desc="Merging layers"):
        if layer_file.suffix == '.safetensors':
            # Load safetensors file
            state_dict = safetensors.torch.load_file(str(layer_file))
            # Extract metadata if available
            with safetensors.safe_open(str(layer_file), framework="pt") as f:
                file_metadata = f.metadata() if hasattr(f, 'metadata') else {}
                metadata.update(file_metadata)
        else:
            # Load PyTorch file
            state_dict = torch.load(layer_file, map_location='cpu')
        
        # Add to merged dict
        merged_state_dict.update(state_dict)
    
    # Save merged model
    output_file = output_path / "model.safetensors"
    safetensors.torch.save_file(merged_state_dict, str(output_file), metadata=metadata)
    
    logging.info(f"Merged model saved to {output_file}")
    
    # Create index file if model is large
    total_size = sum(p.numel() * p.element_size() for p in merged_state_dict.values())
    if total_size > 5e9:  # If larger than 5GB, create sharded index
        create_sharded_index([output_file], output_path)


def verify_quantized_weights(weights_path: Path) -> bool:
    """Verify integrity of quantized weights
    
    Args:
        weights_path: Path to weights file or directory
        
    Returns:
        True if weights are valid, False otherwise
    """
    weights_path = Path(weights_path)
    
    if not weights_path.exists():
        logging.error(f"Weights path does not exist: {weights_path}")
        return False
    
    try:
        if weights_path.is_file():
            # Single file verification
            if weights_path.suffix == '.safetensors':
                # Verify safetensors file
                with safetensors.safe_open(str(weights_path), framework="pt") as f:
                    # Check if we can read tensor names
                    tensor_names = f.keys()
                    if not tensor_names:
                        logging.error("No tensors found in file")
                        return False
                    
                    # Sample check: load first tensor
                    first_tensor_name = list(tensor_names)[0]
                    tensor = f.get_tensor(first_tensor_name)
                    
                    # Check for NaN or Inf
                    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                        logging.error(f"NaN or Inf found in tensor {first_tensor_name}")
                        return False
                    
                    # Check quantization metadata if present
                    metadata = f.metadata() if hasattr(f, 'metadata') else {}
                    if 'quantization_config' in metadata:
                        quant_config = json.loads(metadata['quantization_config'])
                        logging.info(f"Quantization method: {quant_config.get('quant_method', 'unknown')}")
            else:
                # Verify PyTorch file
                state_dict = torch.load(weights_path, map_location='cpu')
                if not state_dict:
                    logging.error("Empty state dict")
                    return False
                
                # Check first tensor
                for name, tensor in state_dict.items():
                    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                        logging.error(f"NaN or Inf found in tensor {name}")
                        return False
                    break
        
        else:
            # Directory verification - check all weight files
            weight_files = list(weights_path.glob("*.safetensors"))
            weight_files.extend(weights_path.glob("*.pt"))
            weight_files.extend(weights_path.glob("*.bin"))
            
            if not weight_files:
                logging.error("No weight files found in directory")
                return False
            
            for weight_file in weight_files:
                if not verify_quantized_weights(weight_file):
                    return False
        
        logging.info("Weight verification passed")
        return True
        
    except Exception as e:
        logging.error(f"Error verifying weights: {e}")
        return False

def verify_exported_model_completeness(model_path: str) -> Tuple[bool, List[str], List[str]]:
    """Verify exported model has all required components
    
    Args:
        model_path: Path to exported model
        
    Returns:
        Tuple of (is_complete, missing_files, warnings)
    """
    model_path = Path(model_path)
    missing_files = []
    warnings = []
    
    # Required files for a complete model
    required_files = {
        'config.json': 'Model configuration',
        'model.safetensors': 'Model weights (or sharded model-*.safetensors)',
    }
    
    # Optional but recommended files
    optional_files = {
        'tokenizer.json': 'Fast tokenizer',
        'tokenizer_config.json': 'Tokenizer configuration',
        'tokenizer.model': 'Sentencepiece tokenizer',
        'special_tokens_map.json': 'Special tokens mapping',
        'quantization_metadata.json': 'Quantization details',
        'validation_report.json': 'Validation results',
    }
    
    # Check required files
    for file_name, description in required_files.items():
        file_path = model_path / file_name
        if not file_path.exists():
            # Special case for sharded weights
            if file_name == 'model.safetensors':
                sharded_files = list(model_path.glob("model-*.safetensors"))
                if not sharded_files:
                    missing_files.append(f"{file_name} ({description})")
                elif len(sharded_files) > 0:
                    # Check for index file
                    index_file = model_path / "model.safetensors.index.json"
                    if not index_file.exists():
                        warnings.append("Sharded model lacks index file")
            else:
                missing_files.append(f"{file_name} ({description})")
    
    # Check optional files
    for file_name, description in optional_files.items():
        file_path = model_path / file_name
        if not file_path.exists():
            warnings.append(f"Missing optional: {file_name} ({description})")
    
    # Verify config has quantization info
    config_file = model_path / "config.json"
    if config_file.exists():
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            if 'quantization_config' not in config:
                warnings.append("Config lacks quantization_config section")
            else:
                quant_config = config['quantization_config']
                if 'quant_method' not in quant_config:
                    warnings.append("Quantization config missing method")
                if 'bits' not in quant_config:
                    warnings.append("Quantization config missing bits")
                    
        except Exception as e:
            warnings.append(f"Cannot parse config.json: {e}")
    
    is_complete = len(missing_files) == 0
    
    return is_complete, missing_files, warnings

def create_model_card(
    base_model: str,
    quantization_config: Dict[str, Any],
    metrics: Dict[str, float]
) -> str:
    """Create model card for quantized model - STUB"""
    return ''


def safe_torch_load(path: Path, 
                   device: str = "cpu",
                   dtype: Optional[torch.dtype] = None) -> Any:
    """Safely load torch checkpoint with error handling
    
    Args:
        path: Path to checkpoint file
        device: Device to load to
        dtype: Optional dtype conversion
        
    Returns:
        Loaded checkpoint or state dict
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    
    try:
        # Determine file type
        if path.suffix == '.safetensors':
            # Load safetensors
            state_dict = safetensors.torch.load_file(str(path), device=device)
            
            # Convert dtype if specified
            if dtype is not None:
                state_dict = {k: v.to(dtype) for k, v in state_dict.items()}
            
            return state_dict
        else:
            # Load PyTorch checkpoint
            checkpoint = torch.load(path, map_location=device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif isinstance(checkpoint, dict):
                state_dict = checkpoint
            else:
                # Assume it's already a state dict
                state_dict = checkpoint
            
            # Convert dtype if specified
            if dtype is not None and isinstance(state_dict, dict):
                state_dict = {k: v.to(dtype) if torch.is_tensor(v) else v 
                             for k, v in state_dict.items()}
            
            return state_dict
            
    except Exception as e:
        logging.error(f"Error loading checkpoint from {path}: {e}")
        raise


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
    """Create index for sharded model weights
    
    Args:
        weight_files: List of weight file paths
        output_path: Output directory for index file
    """
    output_path = Path(output_path)
    
    # Build weight map
    weight_map = {}
    metadata = {
        "total_size": 0,
        "format": "safetensors" if weight_files[0].suffix == '.safetensors' else "pt"
    }
    
    for weight_file in weight_files:
        if weight_file.suffix == '.safetensors':
            # Read safetensors metadata
            with safetensors.safe_open(str(weight_file), framework="pt") as f:
                for tensor_name in f.keys():
                    weight_map[tensor_name] = weight_file.name
                    # Get tensor size
                    tensor = f.get_tensor(tensor_name)
                    metadata["total_size"] += tensor.numel() * tensor.element_size()
        else:
            # Load PyTorch file metadata
            state_dict = torch.load(weight_file, map_location='cpu', map_only=True)
            for tensor_name in state_dict.keys():
                weight_map[tensor_name] = weight_file.name
    
    # Create index structure
    index = {
        "metadata": metadata,
        "weight_map": weight_map
    }
    
    # Save index file
    index_file = output_path / "model.safetensors.index.json"
    with open(index_file, 'w') as f:
        json.dump(index, f, indent=2)
    
    logging.info(f"Created index file: {index_file}")


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