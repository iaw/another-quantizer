# memory_manager.py
"""Memory management for sequential quantization"""

import torch
import gc
import psutil
import logging
from typing import Dict, Optional, Any, List, Tuple
from dataclasses import dataclass, field
from collections import deque
import warnings
import time
import threading
from queue import Queue


@dataclass
class MemoryStats:
    """Track memory usage statistics"""
    gpu_used: float  # GB
    gpu_total: float  # GB
    gpu_reserved: float  # GB (PyTorch reserved)
    cpu_used: float  # GB
    cpu_total: float  # GB
    cpu_available: float  # GB
    timestamp: float = 0.0
    
    def gpu_free(self) -> float:
        """Calculate free GPU memory"""
        return self.gpu_total - self.gpu_used
    
    def cpu_free(self) -> float:
        """Calculate free CPU memory"""
        return self.cpu_available


@dataclass 
class TensorTracker:
    """Track tensor allocations for memory management"""
    name: str
    size_gb: float
    device: str
    dtype: torch.dtype
    shape: Tuple[int, ...]
    
    
class MemoryManager:
    """Manage GPU and CPU memory during quantization"""
    
    def __init__(self, max_gpu_memory: int, max_cpu_memory: int):
        """Initialize memory manager with limits"""
        self.max_gpu_memory = max_gpu_memory  # User-specified limit in GB
        self.max_cpu_memory = max_cpu_memory
        self.device_map = {}
        self.tensor_registry = {}  # Track allocated tensors
        self.memory_history = deque(maxlen=100)  # Track memory over time
        self.logger = logging.getLogger(__name__)
        self.gpu_device = torch.device("cuda:0") if torch.cuda.is_available() else None
        self.enable_aggressive_cleanup = True
        self.min_free_memory = 2.0  # Always keep 2GB free
        
        # Prefetch queue for next layer
        self.prefetch_queue = Queue(maxsize=1)
        self.prefetch_thread = None
        
        # Initialize CUDA if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
    def get_current_memory(self) -> MemoryStats:
        """Get current memory usage"""
        stats = MemoryStats(
            gpu_used=0.0,
            gpu_total=0.0,
            gpu_reserved=0.0,
            cpu_used=0.0,
            cpu_total=0.0,
            cpu_available=0.0,
            timestamp=time.time()
        )
        
        # GPU memory from torch.cuda
        if torch.cuda.is_available():
            stats.gpu_used = torch.cuda.memory_allocated(0) / 1e9
            stats.gpu_reserved = torch.cuda.memory_reserved(0) / 1e9
            stats.gpu_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        # CPU memory from psutil
        vm = psutil.virtual_memory()
        stats.cpu_used = vm.used / 1e9
        stats.cpu_total = vm.total / 1e9
        stats.cpu_available = vm.available / 1e9
        
        # Add to history for tracking
        self.memory_history.append(stats)
        
        return stats
    
    def clear_gpu_cache(self) -> None:
        """Clear GPU cache and run garbage collection"""
        if not torch.cuda.is_available():
            return
        
        before_stats = self.get_current_memory()
        
        # Synchronize CUDA operations
        torch.cuda.synchronize()
        
        # Clear cache
        torch.cuda.empty_cache()
        
        # Garbage collection
        gc.collect()
        
        if self.enable_aggressive_cleanup:
            # Reset peak memory statistics
            torch.cuda.reset_peak_memory_stats()
            
            # Additional aggressive cleanup
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            
            # Multiple GC passes
            for _ in range(3):
                gc.collect()
        
        after_stats = self.get_current_memory()
        freed = before_stats.gpu_used - after_stats.gpu_used
        
        if freed > 0.1:  # Log if freed more than 100MB
            self.logger.info(f"Freed {freed:.2f}GB GPU memory")
    
    def offload_to_cpu(self, tensor: torch.Tensor, name: Optional[str] = None) -> torch.Tensor:
        """Move tensor to CPU memory with tracking"""
        if tensor.device.type == 'cpu':
            return tensor
        
        # Check if CPU has enough space
        tensor_size = self.estimate_tensor_size(tensor)
        current_stats = self.get_current_memory()
        
        if current_stats.cpu_available < tensor_size + self.min_free_memory:
            self.logger.warning(f"Low CPU memory: {current_stats.cpu_available:.2f}GB available, "
                              f"need {tensor_size:.2f}GB")
            # Try to free some CPU memory
            gc.collect()
            current_stats = self.get_current_memory()
            
            if current_stats.cpu_available < tensor_size:
                raise MemoryError(f"Insufficient CPU memory for offloading: "
                                f"need {tensor_size:.2f}GB, have {current_stats.cpu_available:.2f}GB")
        
        # Move to CPU with pinned memory for faster transfers
        cpu_tensor = tensor.to("cpu", non_blocking=True)
        
        # Synchronize to ensure transfer completes
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Update tensor registry
        if name:
            self.unregister_tensor(name)  # Remove old entry if exists
            self.register_tensor(cpu_tensor, name)
        
        # Clear source GPU memory
        del tensor
        self.clear_gpu_cache()
        
        return cpu_tensor
    
    def load_to_gpu(self, tensor: torch.Tensor, device: int = 0, name: Optional[str] = None) -> torch.Tensor:
        """Load tensor to specified GPU with safety checks"""
        if not torch.cuda.is_available():
            self.logger.warning("GPU not available, keeping tensor on CPU")
            return tensor
        
        # Calculate tensor size
        tensor_size = self.estimate_tensor_size(tensor)
        
        # Check if fits within max_gpu_memory limit
        current_stats = self.get_current_memory()
        projected_usage = current_stats.gpu_used + tensor_size
        
        if projected_usage > self.max_gpu_memory:
            self.logger.warning(f"Loading tensor would exceed GPU memory limit: "
                              f"{projected_usage:.2f}GB > {self.max_gpu_memory:.2f}GB")
            
            # Try clearing cache first
            self.clear_gpu_cache()
            current_stats = self.get_current_memory()
            projected_usage = current_stats.gpu_used + tensor_size
            
            if projected_usage > self.max_gpu_memory:
                # Try emergency cleanup
                self.emergency_cleanup()
                current_stats = self.get_current_memory()
                projected_usage = current_stats.gpu_used + tensor_size
                
                if projected_usage > self.max_gpu_memory:
                    raise MemoryError(f"Cannot load tensor to GPU: would exceed limit "
                                    f"({projected_usage:.2f}GB > {self.max_gpu_memory:.2f}GB)")
        
        # Check actual available GPU memory
        if current_stats.gpu_free() < tensor_size + self.min_free_memory:
            self.logger.warning(f"Low GPU memory: {current_stats.gpu_free():.2f}GB free, "
                              f"need {tensor_size:.2f}GB")
            self.clear_gpu_cache()
        
        # Move to GPU
        device_str = f"cuda:{device}"
        gpu_tensor = tensor.to(device_str, non_blocking=False)  # Blocking for safety
        
        # Update tensor registry
        if name:
            self.unregister_tensor(name)  # Remove old entry if exists
            self.register_tensor(gpu_tensor, name)
        
        return gpu_tensor
    
    def can_fit_on_gpu(self, size_gb: float, safety_margin: float = 0.1) -> bool:
        """Check if tensor of given size can fit on GPU"""
        if not torch.cuda.is_available():
            return False
        
        current_stats = self.get_current_memory()
        
        # Add safety margin (10% by default)
        required_size = size_gb * (1 + safety_margin)
        
        # Check against user-specified max_gpu_memory
        if current_stats.gpu_used + required_size > self.max_gpu_memory:
            return False
        
        # Check against actual available GPU memory
        if current_stats.gpu_free() < required_size + self.min_free_memory:
            return False
        
        return True
    
    def setup_device_map(self, model_config: Any) -> Dict[str, str]:
        """Create device map for model layers"""
        device_map = {}
        
        if not torch.cuda.is_available():
            # All layers on CPU if no GPU
            for layer_name in model_config.layer_names:
                device_map[layer_name] = "cpu"
            return device_map
        
        current_gpu_usage = 0.0
        
        for i, layer_name in enumerate(model_config.layer_names):
            layer_size = model_config.get_layer_memory_estimate(i)
            
            # Special handling for certain layers
            if "embedding" in layer_name.lower():
                # Keep embeddings on GPU if possible (frequently accessed)
                if current_gpu_usage + layer_size <= self.max_gpu_memory * 0.5:
                    device_map[layer_name] = "cuda:0"
                    current_gpu_usage += layer_size
                else:
                    device_map[layer_name] = "cpu"
            elif "lm_head" in layer_name.lower() or "output" in layer_name.lower():
                # Output layers can stay on CPU (moved when needed)
                device_map[layer_name] = "cpu"
            else:
                # Regular transformer layers
                if current_gpu_usage + layer_size <= self.max_gpu_memory * 0.7:
                    device_map[layer_name] = "cuda:0"
                    current_gpu_usage += layer_size
                else:
                    device_map[layer_name] = "cpu"
        
        self.device_map = device_map
        return device_map
    
    def monitor_memory(self, tag: str = "") -> None:
        """Log current memory usage with optional tag"""
        stats = self.get_current_memory()
        
        message = f"Memory Usage"
        if tag:
            message += f" [{tag}]"
        
        message += f": GPU: {stats.gpu_used:.1f}/{stats.gpu_total:.1f}GB "
        message += f"(reserved: {stats.gpu_reserved:.1f}GB), "
        message += f"CPU: {stats.cpu_used:.1f}/{stats.cpu_total:.1f}GB "
        message += f"(available: {stats.cpu_available:.1f}GB)"
        
        self.logger.info(message)
        
        # Warn if approaching limits
        gpu_usage_percent = (stats.gpu_used / self.max_gpu_memory) * 100 if self.max_gpu_memory > 0 else 0
        cpu_usage_percent = (stats.cpu_used / self.max_cpu_memory) * 100 if self.max_cpu_memory > 0 else 0
        
        if gpu_usage_percent > 90:
            self.logger.warning(f"GPU memory usage critical: {gpu_usage_percent:.1f}%")
        elif gpu_usage_percent > 80:
            self.logger.warning(f"GPU memory usage high: {gpu_usage_percent:.1f}%")
        
        if cpu_usage_percent > 90:
            self.logger.warning(f"CPU memory usage critical: {cpu_usage_percent:.1f}%")
        elif cpu_usage_percent > 80:
            self.logger.warning(f"CPU memory usage high: {cpu_usage_percent:.1f}%")
    
    def emergency_cleanup(self) -> None:
        """Emergency memory cleanup if OOM is imminent"""
        self.logger.warning("Performing emergency memory cleanup")
        
        # Force synchronize CUDA
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Clear all caches
        self.clear_gpu_cache()
        
        # Offload non-essential tensors to CPU
        tensors_to_offload = []
        for name, tracker in self.tensor_registry.items():
            if tracker.device != "cpu" and "essential" not in name:
                tensors_to_offload.append(name)
        
        for name in tensors_to_offload:
            self.logger.info(f"Emergency offloading tensor: {name}")
            # Note: Actual tensor offloading would need tensor reference
        
        # Multiple aggressive GC passes
        for _ in range(5):
            gc.collect()
        
        # Clear Python's internal caches
        torch.cuda.empty_cache()
        
        # Log final state
        self.monitor_memory("after_emergency_cleanup")
    
    def estimate_tensor_size(self, tensor: torch.Tensor) -> float:
        """Estimate tensor size in GB"""
        # Get number of elements
        numel = tensor.numel()
        
        # Get bytes per element based on dtype
        dtype_sizes = {
            torch.float32: 4,
            torch.float16: 2,
            torch.bfloat16: 2,
            torch.int8: 1,
            torch.uint8: 1,
            torch.int16: 2,
            torch.int32: 4,
            torch.int64: 8,
            torch.bool: 1,
        }
        
        bytes_per_element = dtype_sizes.get(tensor.dtype, 4)
        
        # For INT4, we pack 2 values per byte
        if hasattr(tensor, 'is_quantized') and tensor.is_quantized:
            bytes_per_element = 0.5
        
        # Calculate size in GB
        size_gb = (numel * bytes_per_element) / 1e9
        
        # Add 10% overhead for alignment and metadata
        return size_gb * 1.1
    
    def register_tensor(self, tensor: torch.Tensor, name: str) -> None:
        """Register tensor in tracking system"""
        tracker = TensorTracker(
            name=name,
            size_gb=self.estimate_tensor_size(tensor),
            device=str(tensor.device),
            dtype=tensor.dtype,
            shape=tuple(tensor.shape)
        )
        self.tensor_registry[name] = tracker
    
    def unregister_tensor(self, name: str) -> None:
        """Remove tensor from tracking"""
        if name in self.tensor_registry:
            del self.tensor_registry[name]
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get summary of memory usage"""
        stats = self.get_current_memory()
        
        # Aggregate stats from tensor_registry
        gpu_tensors = []
        cpu_tensors = []
        total_gpu_size = 0.0
        total_cpu_size = 0.0
        
        for name, tracker in self.tensor_registry.items():
            if "cuda" in tracker.device:
                gpu_tensors.append((name, tracker.size_gb))
                total_gpu_size += tracker.size_gb
            else:
                cpu_tensors.append((name, tracker.size_gb))
                total_cpu_size += tracker.size_gb
        
        # Sort by size
        gpu_tensors.sort(key=lambda x: x[1], reverse=True)
        cpu_tensors.sort(key=lambda x: x[1], reverse=True)
        
        return {
            "current_stats": {
                "gpu_used_gb": stats.gpu_used,
                "gpu_total_gb": stats.gpu_total,
                "gpu_free_gb": stats.gpu_free(),
                "cpu_used_gb": stats.cpu_used,
                "cpu_total_gb": stats.cpu_total,
                "cpu_free_gb": stats.cpu_free(),
            },
            "tracked_tensors": {
                "gpu_count": len(gpu_tensors),
                "gpu_total_gb": total_gpu_size,
                "gpu_largest": gpu_tensors[:5] if gpu_tensors else [],
                "cpu_count": len(cpu_tensors),
                "cpu_total_gb": total_cpu_size,
                "cpu_largest": cpu_tensors[:5] if cpu_tensors else [],
            },
            "limits": {
                "max_gpu_gb": self.max_gpu_memory,
                "max_cpu_gb": self.max_cpu_memory,
            }
        }
    
    def optimize_memory_layout(self, required_size: float) -> bool:
        """Try to optimize memory to fit required size"""
        self.logger.info(f"Optimizing memory to fit {required_size:.2f}GB")
        
        # Clear caches
        self.clear_gpu_cache()
        
        # Check if we have enough space now
        if self.can_fit_on_gpu(required_size):
            return True
        
        # Identify least recently used tensors (simplified - would need LRU tracking)
        # For now, just try to free up space
        current_stats = self.get_current_memory()
        
        if current_stats.gpu_free() < required_size:
            # Need to offload some tensors
            self.logger.info("Need to offload tensors to make space")
            # In practice, would offload specific tensors here
        
        # Consolidate fragmented memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Final check
        return self.can_fit_on_gpu(required_size)
    
    def set_memory_fraction(self, fraction: float) -> None:
        """Set PyTorch GPU memory fraction"""
        if not torch.cuda.is_available():
            return
        
        if fraction <= 0 or fraction > 1:
            raise ValueError(f"Memory fraction must be between 0 and 1, got {fraction}")
        
        torch.cuda.set_per_process_memory_fraction(fraction)
        
        # Update max_gpu_memory accordingly
        total_gpu = torch.cuda.get_device_properties(0).total_memory / 1e9
        self.max_gpu_memory = total_gpu * fraction
        
        self.logger.info(f"Set GPU memory fraction to {fraction:.2f} "
                        f"(max {self.max_gpu_memory:.1f}GB)")
    
    def checkpoint_memory_state(self) -> Dict[str, Any]:
        """Save current memory state for recovery"""
        return {
            "tensor_registry": dict(self.tensor_registry),
            "device_map": dict(self.device_map),
            "memory_stats": self.get_current_memory(),
            "max_gpu_memory": self.max_gpu_memory,
            "max_cpu_memory": self.max_cpu_memory,
        }
    
    def restore_memory_state(self, state: Dict[str, Any]) -> None:
        """Restore memory state from checkpoint"""
        # Clear current state
        self.tensor_registry.clear()
        self.device_map.clear()
        
        # Restore from checkpoint
        for name, tracker_dict in state["tensor_registry"].items():
            # Reconstruct TensorTracker from dict
            self.tensor_registry[name] = TensorTracker(**tracker_dict)
        
        self.device_map = state["device_map"]
        self.max_gpu_memory = state["max_gpu_memory"]
        self.max_cpu_memory = state["max_cpu_memory"]
        
        # Verify memory availability matches
        current_stats = self.get_current_memory()
        saved_stats = state["memory_stats"]
        
        if abs(current_stats.gpu_used - saved_stats.gpu_used) > 1.0:
            self.logger.warning(f"GPU memory mismatch: current {current_stats.gpu_used:.1f}GB, "
                              f"saved {saved_stats.gpu_used:.1f}GB")
    
    def profile_memory_usage(self, func, *args, **kwargs):
        """Profile memory usage of a function"""
        # Record memory before
        before_stats = self.get_current_memory()
        before_peak = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        
        # Run function
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        # Record memory after
        after_stats = self.get_current_memory()
        after_peak = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        
        # Calculate memory usage
        memory_profile = {
            "duration_seconds": end_time - start_time,
            "gpu_used_gb": after_stats.gpu_used - before_stats.gpu_used,
            "gpu_peak_gb": (after_peak - before_peak) / 1e9 if torch.cuda.is_available() else 0,
            "cpu_used_gb": after_stats.cpu_used - before_stats.cpu_used,
        }
        
        return result, memory_profile
    
    def start_prefetch(self, load_func, *args, **kwargs) -> None:
        """Start prefetching next layer in background"""
        def prefetch_worker():
            try:
                result = load_func(*args, **kwargs)
                self.prefetch_queue.put(("success", result))
            except Exception as e:
                self.prefetch_queue.put(("error", e))
        
        self.prefetch_thread = threading.Thread(target=prefetch_worker)
        self.prefetch_thread.start()
    
    def get_prefetched(self, timeout: float = None):
        """Get prefetched result"""
        if self.prefetch_thread is None:
            return None
        
        # Wait for prefetch to complete
        self.prefetch_thread.join(timeout=timeout)
        
        if not self.prefetch_queue.empty():
            status, result = self.prefetch_queue.get()
            if status == "error":
                raise result
            return result
        
        return None
    
    @property
    def is_gpu_available(self) -> bool:
        """Check if GPU is available and accessible"""
        return torch.cuda.is_available()
    
    @property
    def gpu_name(self) -> str:
        """Get GPU name for logging"""
        if self.is_gpu_available:
            return torch.cuda.get_device_name(0)
        return "No GPU"
    
    def __str__(self) -> str:
        """String representation of current memory state"""
        stats = self.get_current_memory()
        return (f"GPU: {stats.gpu_used:.1f}/{stats.gpu_total:.1f}GB, "
                f"CPU: {stats.cpu_used:.1f}/{stats.cpu_total:.1f}GB")