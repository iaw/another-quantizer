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
class SlidingWindowManager:
    """Manage sliding window for activation collection to limit memory usage"""
    window_size: int = 3  # Number of layers to keep in memory
    current_layers: deque = field(default_factory=lambda: deque(maxlen=3))
    activation_cache: Dict[str, torch.Tensor] = field(default_factory=dict)
    weight_cache: Dict[str, Dict[str, torch.Tensor]] = field(default_factory=dict)
    cache_size_gb: float = 0.0
    max_cache_gb: float = 4.0  # Maximum cache size in GB
    
    def add_layer(self, layer_name: str, activations: Optional[torch.Tensor] = None, 
                  weights: Optional[Dict[str, torch.Tensor]] = None) -> None:
        """Add layer to sliding window, evicting old layers if needed"""
        # Check if we need to evict
        if len(self.current_layers) >= self.window_size:
            evicted = self.current_layers[0]  # Will be auto-removed by deque
            # Clean up evicted layer's data
            if evicted in self.activation_cache:
                del self.activation_cache[evicted]
            if evicted in self.weight_cache:
                del self.weight_cache[evicted]
        
        # Add new layer
        self.current_layers.append(layer_name)
        
        if activations is not None:
            self.activation_cache[layer_name] = activations
        if weights is not None:
            self.weight_cache[layer_name] = weights
        
        # Update cache size
        self._update_cache_size()
    
    def _update_cache_size(self) -> None:
        """Calculate current cache size"""
        total_size = 0
        
        # Activation cache size
        for tensor in self.activation_cache.values():
            if tensor is not None:
                total_size += tensor.numel() * tensor.element_size()
        
        # Weight cache size
        for layer_weights in self.weight_cache.values():
            for tensor in layer_weights.values():
                if tensor is not None:
                    total_size += tensor.numel() * tensor.element_size()
        
        self.cache_size_gb = total_size / 1e9
    
    def should_offload(self) -> bool:
        """Check if cache is too large and needs offloading"""
        return self.cache_size_gb > self.max_cache_gb
    
    def get_activation(self, layer_name: str) -> Optional[torch.Tensor]:
        """Get activation for layer if in cache"""
        return self.activation_cache.get(layer_name)
    
    def clear_old_activations(self, keep_last_n: int = 1) -> None:
        """Clear old activations keeping only last n"""
        if len(self.activation_cache) > keep_last_n:
            layers_to_remove = list(self.activation_cache.keys())[:-keep_last_n]
            for layer in layers_to_remove:
                del self.activation_cache[layer]
        self._update_cache_size()


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
        
        # Initialize sliding window for activation management
        self.sliding_window = SlidingWindowManager(
            window_size=3,
            max_cache_gb=min(4.0, self.max_gpu_memory * 0.2)  # Use 20% of GPU memory for cache
        )
        
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
        
        # Single garbage collection
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
        cpu_tensor = tensor.to(device="cpu", dtype=tensor.dtype, non_blocking=True)
        
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
        gpu_tensor = tensor.to(device=device_str, dtype=tensor.dtype, non_blocking=False)
        
        # Update tensor registry
        if name:
            self.unregister_tensor(name)  # Remove old entry if exists
            self.register_tensor(gpu_tensor, name)
        
        return gpu_tensor
    
    def allocate_memory_for_layer(self, layer_size_gb: float, 
                                 layer_type: str = "transformer") -> Dict[str, Any]:
        """Adaptively allocate memory for layer based on size and type
        
        Args:
            layer_size_gb: Estimated layer size
            layer_type: Type of layer (transformer, moe, embedding, etc.)
            
        Returns:
            Allocation strategy dictionary
        """
        current_stats = self.get_current_memory()
        
        strategy = {
            'device': 'cuda:0',
            'offload_activations': False,
            'offload_weights': False,
            'use_checkpointing': False,
            'batch_size': self.max_gpu_memory // 4,  # Default
        }
        
        # Adjust based on layer type
        if layer_type == "moe":
            # MoE layers need more memory for expert routing
            required_memory = layer_size_gb * 2.5
        elif layer_type == "embedding":
            # Embeddings are large but accessed infrequently
            required_memory = layer_size_gb * 1.2
            strategy['offload_weights'] = True  # Can offload after initial use
        else:
            # Standard transformer layer
            required_memory = layer_size_gb * 2.0
        
        # Check GPU availability
        gpu_available = current_stats.gpu_free() - self.min_free_memory
        
        if required_memory > gpu_available:
            # Need to use offloading strategy
            if required_memory > gpu_available * 2:
                # Extreme case: use CPU with gradient checkpointing
                strategy['device'] = 'cpu'
                strategy['use_checkpointing'] = True
                strategy['batch_size'] = 1
                self.logger.warning(f"Layer {layer_type} requires {required_memory:.2f}GB, "
                                  f"using CPU with checkpointing")
            else:
                # Moderate case: offload activations after use
                strategy['offload_activations'] = True
                strategy['batch_size'] = max(1, int(gpu_available / layer_size_gb))
                self.logger.info(f"Layer {layer_type} requires {required_memory:.2f}GB, "
                               f"will offload activations")
        
        # Check if sliding window needs adjustment
        if self.sliding_window.should_offload():
            self.sliding_window.clear_old_activations(keep_last_n=1)
            self.clear_gpu_cache()
        
        return strategy
    
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
    
    def emergency_cleanup(self, required_memory_gb: float = 0) -> bool:
        """Emergency memory cleanup if OOM is imminent
        
        Args:
            required_memory_gb: Memory needed to free up
            
        Returns:
            True if enough memory was freed
        """
        self.logger.warning(f"Performing emergency memory cleanup (need {required_memory_gb:.2f}GB)")
        
        initial_stats = self.get_current_memory()
        
        # Step 1: Clear sliding window cache
        if self.sliding_window.cache_size_gb > 0:
            self.sliding_window.clear_old_activations(keep_last_n=0)
            self.logger.info(f"Cleared sliding window cache: {self.sliding_window.cache_size_gb:.2f}GB")
        
        # Step 2: Force synchronize and clear CUDA
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        
        # Step 3: Clear tracked tensors that are on GPU
        gpu_tensors_to_clear = []
        for name, tracker in self.tensor_registry.items():
            if 'cuda' in tracker.device and tracker.size_gb > 0.1:  # Clear tensors > 100MB
                gpu_tensors_to_clear.append(name)
        
        for name in gpu_tensors_to_clear:
            self.unregister_tensor(name)
            self.logger.debug(f"Unregistered tensor: {name}")
        
        # Step 4: Aggressive garbage collection
        for _ in range(3):
            gc.collect()
        
        # Step 5: Final CUDA cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Check if we freed enough memory
        final_stats = self.get_current_memory()
        freed_memory = initial_stats.gpu_used - final_stats.gpu_used
        
        self.logger.info(f"Emergency cleanup freed {freed_memory:.2f}GB GPU memory")
        self.monitor_memory("after_emergency_cleanup")
        
        return freed_memory >= required_memory_gb
    
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
    
    def schedule_layer_processing(self, layer_sizes: List[Tuple[str, float]]) -> List[Dict[str, Any]]:
        """Schedule layer processing to optimize memory usage
        
        Args:
            layer_sizes: List of (layer_name, size_gb) tuples
            
        Returns:
            List of processing strategies for each layer
        """
        schedule = []
        
        # Sort layers by size to process smaller ones first when possible
        # But maintain some order for layer dependencies
        grouped_layers = []
        current_group = []
        current_group_size = 0
        max_group_size = self.max_gpu_memory * 0.7  # Use 70% of GPU memory per group
        
        for layer_name, size_gb in layer_sizes:
            if current_group_size + size_gb > max_group_size and current_group:
                # Start new group
                grouped_layers.append(current_group)
                current_group = [(layer_name, size_gb)]
                current_group_size = size_gb
            else:
                current_group.append((layer_name, size_gb))
                current_group_size += size_gb
        
        if current_group:
            grouped_layers.append(current_group)
        
        # Create schedule for each group
        for group_idx, group in enumerate(grouped_layers):
            group_size = sum(size for _, size in group)
            
            # Determine strategy for this group
            if group_size < self.max_gpu_memory * 0.5:
                # Can process entirely on GPU
                strategy = 'gpu_batch'
                device = 'cuda:0'
            elif group_size < self.max_gpu_memory:
                # Process on GPU with memory management
                strategy = 'gpu_sequential'
                device = 'cuda:0'
            else:
                # Need CPU offloading
                strategy = 'cpu_offload'
                device = 'mixed'
            
            for layer_name, size_gb in group:
                schedule.append({
                    'layer_name': layer_name,
                    'size_gb': size_gb,
                    'group_idx': group_idx,
                    'strategy': strategy,
                    'device': device,
                    'clear_cache_after': True if strategy == 'cpu_offload' else False,
                })
        
        self.logger.info(f"Created processing schedule for {len(layer_sizes)} layers in {len(grouped_layers)} groups")
        
        return schedule
    
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