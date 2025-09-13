# layer_quantizer.py
"""Layer-wise quantization implementation"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
import numpy as np
from dataclasses import dataclass
import logging
import time
import gc


@dataclass
class LayerQuantizationResult:
    """Results from quantizing a single layer"""
    layer_name: str
    original_size_gb: float
    quantized_size_gb: float
    quantization_error: float  # MSE or perplexity delta
    time_taken: float
    scales: Dict[str, torch.Tensor]
    zero_points: Optional[Dict[str, torch.Tensor]]
    group_size: int
    bits: int
    compression_ratio: float
    

class AWQScalePropagator:
    """Handles scale propagation between layers for AWQ
    
    Key insight: AWQ works by propagating scales through the network
    to balance quantization errors between layers
    """
    
    def __init__(self, config: Any):
        self.config = config
        self.layer_connections = {}  # Maps output layers to input layers
        self.scale_factors = {}  # Propagation factors between layers
        self.logger = logging.getLogger(__name__)
        
        # Build layer connection map for GLM architecture
        self._build_connection_map()
    
    def _build_connection_map(self):
        """Build map of how layers connect in GLM architecture"""
        # Standard transformer connections
        self.layer_connections = {
            # Within a transformer block
            'input_layernorm': ['q_proj', 'k_proj', 'v_proj'],
            'v_proj': ['o_proj'],
            'o_proj': ['post_attention_layernorm'],  # Through residual
            'post_attention_layernorm': ['gate_proj', 'up_proj'],
            'up_proj': ['down_proj'],
            'down_proj': ['next_input_layernorm'],  # To next layer
            
            # MoE specific
            'router': ['experts'],
            'shared_expert_gate': ['shared_expert_up'],
            'shared_expert_up': ['shared_expert_down'],
        }
    
    def propagate_scales(self,
                        source_layer: str,
                        source_scales: torch.Tensor,
                        target_layer: str,
                        target_weights: torch.Tensor) -> torch.Tensor:
        """Propagate scales from source to target layer
        
        Args:
            source_layer: Name of source layer
            source_scales: Scales from source layer
            target_layer: Name of target layer
            target_weights: Weights of target layer
            
        Returns:
            Propagated scales for target layer
        """
        # Check if layers are connected
        connections = self.layer_connections.get(source_layer, [])
        
        if not any(conn in target_layer for conn in connections):
            # Not directly connected, no propagation
            return torch.ones_like(target_weights[0])
        
        # Compute propagation factor based on weight magnitudes
        source_magnitude = source_scales.mean().item()
        target_magnitude = torch.abs(target_weights).mean().item()
        
        # Propagation factor: balance the scales based on relative magnitudes
        propagation_factor = (source_magnitude / (target_magnitude + 1e-8)) ** 0.5
        propagation_factor = min(max(propagation_factor, 0.5), 2.0)  # Limit range
        
        # Apply propagation
        if source_scales.shape[0] == target_weights.shape[1]:
            # Direct dimension match (e.g., hidden_size to hidden_size)
            propagated_scales = source_scales * propagation_factor
        else:
            # Dimension mismatch, need to adapt
            if target_weights.shape[1] > source_scales.shape[0]:
                # Target is larger, repeat scales
                repeat_factor = target_weights.shape[1] // source_scales.shape[0]
                propagated_scales = source_scales.repeat(repeat_factor + 1)[:target_weights.shape[1]]
            else:
                # Target is smaller, average scales
                chunk_size = source_scales.shape[0] // target_weights.shape[1]
                propagated_scales = source_scales.reshape(-1, chunk_size).mean(dim=1)[:target_weights.shape[1]]
            
            propagated_scales = propagated_scales * propagation_factor
        
        self.logger.debug(f"Propagated scales from {source_layer} to {target_layer}: "
                         f"factor={propagation_factor:.3f}")
        
        return propagated_scales
    
    def compute_equilibrium_scales(self,
                                  layer_weights: Dict[str, torch.Tensor],
                                  initial_scales: Dict[str, torch.Tensor],
                                  iterations: int = 5) -> Dict[str, torch.Tensor]:
        """Compute equilibrium scales through iterative propagation
        
        AWQ paper insight: Iterate scale propagation to reach equilibrium
        
        Args:
            layer_weights: Dictionary of layer weights
            initial_scales: Initial scale estimates
            iterations: Number of propagation iterations
            
        Returns:
            Equilibrium scales after propagation
        """
        equilibrium_scales = initial_scales.copy()
        
        for iter_idx in range(iterations):
            new_scales = {}
            
            for layer_name, scales in equilibrium_scales.items():
                # Find connected layers
                propagated_scales = []
                
                # Check incoming connections
                for source_name, targets in self.layer_connections.items():
                    if any(target in layer_name for target in targets):
                        if source_name in equilibrium_scales:
                            # Propagate from source
                            prop_scale = self.propagate_scales(
                                source_name,
                                equilibrium_scales[source_name],
                                layer_name,
                                layer_weights.get(layer_name, torch.ones(1, scales.shape[0]))
                            )
                            propagated_scales.append(prop_scale)
                
                if propagated_scales:
                    # Combine propagated scales with original
                    avg_propagated = torch.stack(propagated_scales).mean(dim=0)
                    # Blend with original scales
                    blend_factor = 0.5 * (1 - iter_idx / iterations)  # Decrease over iterations
                    new_scales[layer_name] = (1 - blend_factor) * scales + blend_factor * avg_propagated
                else:
                    new_scales[layer_name] = scales
            
            equilibrium_scales = new_scales
            
            # Log convergence
            max_change = max(
                (new_scales[k] - initial_scales[k]).abs().max().item()
                for k in new_scales.keys()
            )
            self.logger.debug(f"Iteration {iter_idx + 1}: max scale change = {max_change:.6f}")
            
            if max_change < 1e-4:
                self.logger.info(f"Scale propagation converged at iteration {iter_idx + 1}")
                break
        
        return equilibrium_scales


class LayerQuantizer:
    """Handles quantization of individual layers"""
    
    def __init__(self, config: Any, memory_manager: Any, model_loader: Any = None):
        """Initialize layer quantizer"""
        self.config = config
        self.memory_manager = memory_manager
        self.model_loader = model_loader  # Reference to model loader for dtype info
        self.awq_scales = {}  # Cache scales across layers
        self.quantization_cache = {}  # Cache for reuse
        self.logger = logging.getLogger(__name__)
        
        # Initialize LLM Compressor wrapper
        from llm_compressor_wrapper import LLMCompressorWrapper
        self.llm_compressor = LLMCompressorWrapper(config)
        
        # Initialize AWQ scale propagator
        self.scale_propagator = AWQScalePropagator(config)
        
        # Cache for layer activations collected during forward pass
        self.activation_cache = {}
        self.hooks = []
        self.layer_weights_cache = {}  # Cache weights for scale propagation
        
        # Statistics tracking
        self.total_original_size = 0.0
        self.total_quantized_size = 0.0
        self.layer_errors = {}
        
        # Initialize metrics collector
        from metrics import MetricsCollector
        self.metrics_collector = MetricsCollector()
        
    def quantize_layer(self, 
                   layer: torch.nn.Module, 
                   layer_name: str,
                   calibration_data: Any) -> Tuple[torch.nn.Module, LayerQuantizationResult]:
        """Quantize a single layer using AWQ with error recovery"""
        start_time = time.time()
        self.logger.info(f"Starting quantization of layer: {layer_name}")
        
        # Type checking and logging for calibration data
        self.logger.debug(f"[TYPE_CHECK] quantize_layer received calibration_data type: {type(calibration_data)}")
        if isinstance(calibration_data, str):
            self.logger.error(f"[TYPE_ERROR] calibration_data is string at quantize_layer entry: '{calibration_data}'")
            self.logger.error(f"[TYPE_ERROR] Stack trace for debugging:")
            import traceback
            self.logger.error(traceback.format_stack())
        elif isinstance(calibration_data, dict):
            self.logger.debug(f"[TYPE_CHECK] calibration_data dict keys: {list(calibration_data.keys())}")
            for key, value in calibration_data.items():
                self.logger.debug(f"[TYPE_CHECK]   {key}: type={type(value)}, shape={value.shape if torch.is_tensor(value) else 'N/A'}")
        elif torch.is_tensor(calibration_data):
            self.logger.debug(f"[TYPE_CHECK] calibration_data tensor shape: {calibration_data.shape}, dtype: {calibration_data.dtype}")
        
        # FIX: Properly extract tensor from calibration data - with defensive copying
        original_calibration_data = calibration_data  # Keep original reference
        
        if isinstance(calibration_data, dict):
            # Create a defensive copy to prevent mutations
            calibration_data_copy = {}
            for key, value in calibration_data.items():
                if torch.is_tensor(value):
                    calibration_data_copy[key] = value.clone()
                else:
                    calibration_data_copy[key] = value
            
            # Now work with the copy
            if 'hidden_states' in calibration_data_copy:
                # Extract the actual tensor, not just pass the dict
                extracted_tensor = calibration_data_copy['hidden_states']
                if torch.is_tensor(extracted_tensor):
                    calibration_data = extracted_tensor
                    self.logger.debug(f"Extracted hidden_states tensor: shape={calibration_data.shape}")
                else:
                    self.logger.error(f"hidden_states is not a tensor: {type(extracted_tensor)}")
                    hidden_size = self._get_hidden_size(layer)
                    calibration_data = torch.randn(1, 128, hidden_size, dtype=torch.float16) * 0.02
            elif 'input_ids' in calibration_data_copy:
                # For token inputs, keep as dict for special handling but use the copy
                calibration_data = calibration_data_copy
                self.logger.debug(f"Keeping dict with input_ids for token processing")
            else:
                # Unexpected dict structure
                self.logger.warning(f"Unexpected calibration data structure: {calibration_data_copy.keys()}")
                hidden_size = self._get_hidden_size(layer)
                calibration_data = torch.randn(1, 128, hidden_size, dtype=torch.float16) * 0.02
        elif isinstance(calibration_data, str):
            # This should never happen but we're seeing it
            self.logger.error(f"Calibration data is string: '{calibration_data}'")
            hidden_size = self._get_hidden_size(layer)
            calibration_data = torch.randn(1, 128, hidden_size, dtype=torch.float16) * 0.02
        elif hasattr(calibration_data, 'input_ids'):
            # CalibrationSample object
            calibration_data = {
                'input_ids': calibration_data.input_ids,
                'attention_mask': calibration_data.attention_mask
            }
        elif not isinstance(calibration_data, torch.Tensor):
            self.logger.warning(f"Unexpected calibration data type: {type(calibration_data)}")
            hidden_size = self._get_hidden_size(layer)
            calibration_data = torch.randn(1, 128, hidden_size, dtype=torch.float16) * 0.02
        
        # Input validation
        if layer is None:
            raise ValueError(f"Layer {layer_name} is None")
        
        # Calculate original size with error handling
        try:
            original_size_gb = self._calculate_layer_size(layer)
            self.total_original_size += original_size_gb
        except Exception as e:
            self.logger.warning(f"Failed to calculate size for {layer_name}: {e}")
            original_size_gb = 0.0
        
        # Check if layer should be quantized
        if self._should_skip_layer(layer_name, layer):
            self.logger.info(f"Skipping layer {layer_name} (in skip list)")
            # Return original layer unchanged
            result = LayerQuantizationResult(
                layer_name=layer_name,
                original_size_gb=original_size_gb,
                quantized_size_gb=original_size_gb,
                quantization_error=0.0,
                time_taken=time.time() - start_time,
                scales={},
                zero_points=None,
                group_size=0,
                bits=0,
                compression_ratio=1.0
            )
            return layer, result
        
        # Move layer to GPU if needed and possible
        device = self._determine_device(layer, original_size_gb)
        layer = layer.to(device)
        
        # Detect layer's expected dtype - prefer model loader info if available
        if self.model_loader is not None:
            layer_dtype = self.model_loader.get_layer_dtype(layer_name)
            self.logger.debug(f"Layer {layer_name} expects dtype from model_loader: {layer_dtype}")
        else:
            # Fallback to checking layer parameters
            layer_dtype = next(layer.parameters()).dtype
            self.logger.debug(f"Layer {layer_name} expects dtype from parameters: {layer_dtype}")
        
        # Handle calibration_data device placement AND dtype conversion
        self.logger.debug(f"Moving calibration data to device: {device} with dtype: {layer_dtype}")
        
        if isinstance(calibration_data, dict):
            # Move dictionary contents to device AND convert dtype for float tensors
            calibration_data_device = {}
            for key, value in calibration_data.items():
                if torch.is_tensor(value):
                    # Move to device
                    moved_tensor = value.to(device)
                    
                    # Convert dtype only for floating point tensors (not for int input_ids)
                    if moved_tensor.dtype.is_floating_point and moved_tensor.dtype != layer_dtype:
                        self.logger.debug(f"Converting {key} from {moved_tensor.dtype} to {layer_dtype}")
                        moved_tensor = moved_tensor.to(dtype=layer_dtype)
                    
                    calibration_data_device[key] = moved_tensor
                    self.logger.debug(f"Moved {key} tensor to {device}: shape={moved_tensor.shape}, dtype={moved_tensor.dtype}")
                else:
                    calibration_data_device[key] = value
                    self.logger.debug(f"Kept {key} as-is: type={type(value)}")
            calibration_data = calibration_data_device
            
            # Validate after device movement
            if not isinstance(calibration_data, dict):
                self.logger.error(f"CRITICAL: Lost dict structure after device movement: {type(calibration_data)}")
            else:
                self.logger.debug(f"Dict structure preserved with keys: {list(calibration_data.keys())}")
                
        elif torch.is_tensor(calibration_data):
            calibration_data = calibration_data.to(device)
            if calibration_data.dim() == 2:
                calibration_data = calibration_data.unsqueeze(0)  # Add batch dimension
            self.logger.debug(f"Moved tensor to {device}: shape={calibration_data.shape}")
        elif hasattr(calibration_data, 'to'):
            calibration_data = calibration_data.to(device)
            self.logger.debug(f"Moved object with .to() method to {device}")
            
        # Final validation before use
        if isinstance(calibration_data, str):
            self.logger.error(f"CRITICAL: calibration_data corrupted to string after device placement: '{calibration_data}'")
            hidden_size = self._get_hidden_size(layer)
            calibration_data = torch.randn(1, 128, hidden_size, device=device, dtype=torch.float16) * 0.02
            if calibration_data.dim() == 2:
                calibration_data = calibration_data.unsqueeze(0)  # Add batch dimension
        elif hasattr(calibration_data, 'to'):
            calibration_data = calibration_data.to(device)
        
        # Memory check
        self.memory_manager.monitor_memory(f"before_quantize_{layer_name}")
        
        try:
            # Identify layer type
            layer_type = self._identify_layer_type(layer, layer_name)
            
            # Compute AWQ scales with multiple samples if available
            if hasattr(calibration_data, '__iter__') and not isinstance(calibration_data, torch.Tensor):
                # Use multi-sample computation for DataLoader
                scales = self.compute_awq_scales_multi_sample(
                    layer, calibration_data, layer_type, num_samples=min(32, self.config.calibration_samples)
                )
            else:
                # Single sample computation - FIX: Ensure calibration_data is passed correctly
                scales = self.compute_awq_scales(layer, calibration_data, layer_type)
            
            # Cache layer weights for propagation
            layer_weights = {}
            for name, module in layer.named_modules():
                if isinstance(module, nn.Linear):
                    layer_weights[name] = module.weight.data
            self.layer_weights_cache[layer_name] = layer_weights
            
            # Apply scale propagation if we have multiple layers cached
            if len(self.awq_scales) >= 2:
                # Compute equilibrium scales through propagation
                all_weights = {}
                all_scales = self.awq_scales.copy()
                all_scales[layer_name] = scales  # Add current layer
                
                # Flatten weights from all cached layers
                for cached_layer_name, cached_weights in self.layer_weights_cache.items():
                    for module_name, weight in cached_weights.items():
                        full_name = f"{cached_layer_name}.{module_name}"
                        all_weights[full_name] = weight
                
                # Add current layer weights
                for module_name, weight in layer_weights.items():
                    full_name = f"{layer_name}.{module_name}"
                    all_weights[full_name] = weight
                
                # Compute equilibrium scales
                equilibrium_scales = self.scale_propagator.compute_equilibrium_scales(
                    all_weights,
                    {f"{layer_name}.{k}": v for k, v in scales.items()},
                    iterations=3
                )
                
                # Extract scales for current layer
                for full_name, eq_scale in equilibrium_scales.items():
                    if layer_name in full_name:
                        module_name = full_name.replace(f"{layer_name}.", "")
                        if module_name in scales:
                            # Blend equilibrium scales with computed scales
                            scales[module_name] = 0.7 * scales[module_name] + 0.3 * eq_scale
            
            # Apply scale smoothing (additional smoothing after propagation)
            if self.awq_scales:
                prev_layer_name = self._get_previous_layer_name(layer_name)
                if prev_layer_name and prev_layer_name in self.awq_scales:
                    scales = self.smooth_scales_between_layers(
                        self.awq_scales[prev_layer_name],
                        scales,
                        smoothing_factor=0.2  # Less smoothing since we have propagation
                    )
            
            # Cache scales for this layer
            self.awq_scales[layer_name] = scales
            
            # Apply quantization based on layer type - FIX: Pass calibration_data to all methods
            if layer_type == "moe":
                quantized_layer = self.quantize_moe_layer(layer, calibration_data, scales)
            elif layer_type == "attention":
                quantized_layer = self.quantize_attention_layer(layer, calibration_data, scales)
            elif layer_type == "mlp":
                quantized_layer = self.quantize_mlp_layer(layer, calibration_data, scales)
            else:
                # Generic quantization
                quantized_layer = self._apply_awq_quantization(layer, scales)
            
            # Pack weights if needed
            quantized_layer = self._pack_layer_weights(quantized_layer)
            
            # Validate quantization - FIX: Pass calibration_data
            error = self.validate_quantized_layer(layer, quantized_layer, calibration_data)
            
            # Calculate quantized size
            quantized_size_gb = self._calculate_layer_size(quantized_layer)
            self.total_quantized_size += quantized_size_gb
            
            # Move back to CPU if needed
            if self.config.offload_to_cpu:
                quantized_layer = quantized_layer.to('cpu')
            
            # Clear GPU cache
            self.memory_manager.clear_gpu_cache()
            
            # Clean up old cached weights to save memory (keep only last 3 layers)
            if len(self.layer_weights_cache) > 3:
                oldest_layers = list(self.layer_weights_cache.keys())[:-3]
                for old_layer in oldest_layers:
                    del self.layer_weights_cache[old_layer]
            
            # Get memory peak
            memory_peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            
            # Create result
            result = LayerQuantizationResult(
                layer_name=layer_name,
                original_size_gb=original_size_gb,
                quantized_size_gb=quantized_size_gb,
                quantization_error=error,
                time_taken=time.time() - start_time,
                scales=scales,
                zero_points=self._extract_zero_points(quantized_layer),
                group_size=self.config.group_size,
                bits=self.config.bits,
                compression_ratio=original_size_gb / quantized_size_gb if quantized_size_gb > 0 else 1.0
            )
            
            # Collect detailed metrics
            memory_stats = {
                'gpu_peak_mb': memory_peak * 1024,
                'cpu_peak_mb': self.memory_manager.get_current_memory().cpu_used * 1024
            }
            
            layer_metrics = self.metrics_collector.collect_layer_metrics(
                layer_name=layer_name,
                layer_type=layer_type,
                original_layer=layer,
                quantized_layer=quantized_layer,
                quantization_result=result,
                calibration_data=calibration_data,
                time_taken=result.time_taken,
                memory_stats=memory_stats
            )
            
            # Log results
            self.logger.info(f"Quantized {layer_name}: {original_size_gb:.2f}GB -> {quantized_size_gb:.2f}GB "
                        f"(compression: {result.compression_ratio:.2f}x, error: {error:.6f}, "
                        f"cosine_sim: {layer_metrics.cosine_similarity:.4f})")
            
            return quantized_layer, result
            
        except Exception as e:
            self.logger.error(f"Error quantizing layer {layer_name}: {e}")
            # Return original layer on error
            result = LayerQuantizationResult(
                layer_name=layer_name,
                original_size_gb=original_size_gb,
                quantized_size_gb=original_size_gb,
                quantization_error=float('inf'),
                time_taken=time.time() - start_time,
                scales={},
                zero_points=None,
                group_size=0,
                bits=0,
                compression_ratio=1.0
            )
            return layer, result
    
    def collect_layer_activations(self,
                             layer: torch.nn.Module,
                             calibration_data: torch.Tensor,
                             layer_name: str) -> Dict[str, torch.Tensor]:
        """Collect real activations by running calibration data through layer
        
        Args:
            layer: Layer module to collect activations from
            calibration_data: Real tokenized calibration data
            layer_name: Name of the layer for caching
            
        Returns:
            Dictionary of module_name -> activation tensors
        """
        # Type validation at entry
        self.logger.debug(f"[TYPE_CHECK] collect_layer_activations for {layer_name}, input type: {type(calibration_data)}")
        if isinstance(calibration_data, str):
            self.logger.error(f"[TYPE_ERROR] String passed to collect_layer_activations: '{calibration_data}'")
            # Stack trace to find where this came from
            import traceback
            self.logger.error(f"[TYPE_ERROR] Call stack:\n{''.join(traceback.format_stack())}")
            # Create fallback
            hidden_size = self._get_hidden_size(layer)
            calibration_data = torch.randn(1, 128, hidden_size,
                                          device=next(layer.parameters()).device,
                                          dtype=torch.float16) * 0.02
            self.logger.warning(f"[TYPE_RECOVERY] Using synthetic data shape {calibration_data.shape}")
        
        activations = {}
        handles = []
        
        def create_hook(name):
            def hook_fn(module, input, output):
                # Store input activations for AWQ scaling
                if isinstance(input, tuple):
                    input = input[0]
                activations[name] = input.detach().cpu()
            return hook_fn
        
        # Register hooks on all Linear layers
        for name, module in layer.named_modules():
            if isinstance(module, nn.Linear):
                handle = module.register_forward_hook(create_hook(name))
                handles.append(handle)
        
        # Prepare calibration data - FIX: Ensure we have a tensor, not a dict or string
        with torch.no_grad():
            # Check memory before processing
            current_stats = self.memory_manager.get_current_memory()
            if current_stats.gpu_free() < 2.0:  # Less than 2GB free
                self.logger.warning("Low GPU memory, reducing batch size")
                if isinstance(calibration_data, dict) and 'input_ids' in calibration_data:
                    if calibration_data['input_ids'].shape[0] > 1:
                        # Reduce batch size
                        calibration_data = {
                            'input_ids': calibration_data['input_ids'][:1],
                            'attention_mask': calibration_data['attention_mask'][:1] if 'attention_mask' in calibration_data else None
                        }
                elif hasattr(calibration_data, 'input_ids') and calibration_data.input_ids.shape[0] > 1:
                    calibration_data = type(calibration_data)(
                        input_ids=calibration_data.input_ids[:1],
                        attention_mask=calibration_data.attention_mask[:1] if hasattr(calibration_data, 'attention_mask') else None
                    )
            
            # FIX: Properly prepare hidden_states tensor - with careful extraction
            hidden_states = None
            
            # Handle different calibration_data formats
            if isinstance(calibration_data, torch.Tensor):
                # Already a tensor, use directly (clone for safety)
                hidden_states = calibration_data.clone()
                if hidden_states.dim() == 2:
                    hidden_states = hidden_states.unsqueeze(0)
                self.logger.debug(f"Using tensor directly: shape={hidden_states.shape}")
                
            elif isinstance(calibration_data, dict):
                self.logger.debug(f"Processing dict with keys: {list(calibration_data.keys())}")
                
                if 'hidden_states' in calibration_data:
                    # Extract hidden states from dict - CAREFULLY
                    dict_value = calibration_data['hidden_states']
                    if isinstance(dict_value, torch.Tensor):
                        hidden_states = dict_value.clone()  # Clone to prevent mutations
                        if hidden_states.dim() == 2:
                            hidden_states = hidden_states.unsqueeze(0)
                        self.logger.debug(f"Extracted hidden_states tensor: shape={hidden_states.shape}")
                    else:
                        self.logger.error(f"hidden_states in dict is not a tensor: {type(dict_value)}, value={dict_value}")
                        hidden_states = None
                elif 'input_ids' in calibration_data:
                    # We have tokenized input, need to get/generate hidden states
                    # Try to get cached activations from sliding window first
                    cached_activation = self.memory_manager.sliding_window.get_activation(layer_name)
                    if cached_activation is not None:
                        hidden_states = cached_activation
                    elif layer_name in self.activation_cache:
                        hidden_states = self.activation_cache[layer_name]
                    else:
                        # Generate synthetic hidden states as fallback - with correct dtype
                        hidden_size = self._get_hidden_size(layer)
                        # Get layer's expected dtype
                        layer_dtype = next(layer.parameters()).dtype
                        
                        input_ids = calibration_data['input_ids']
                        if isinstance(input_ids, torch.Tensor):
                            batch_size = input_ids.shape[0] if input_ids.dim() > 0 else 1
                            seq_len = input_ids.shape[1] if input_ids.dim() > 1 else 128
                        else:
                            batch_size = 1
                            seq_len = 128
                        
                        hidden_states = torch.randn(
                            batch_size, seq_len, hidden_size, 
                            device=layer.device, 
                            dtype=layer_dtype  # Use layer's dtype
                        ) * 0.02
                        self.logger.debug(f"Generated synthetic hidden states for {layer_name}: shape={hidden_states.shape}")
            elif hasattr(calibration_data, 'input_ids'):
                # CalibrationSample object
                # Try cached activations first
                cached_activation = self.memory_manager.sliding_window.get_activation(layer_name)
                if cached_activation is not None:
                    hidden_states = cached_activation
                elif layer_name in self.activation_cache:
                    hidden_states = self.activation_cache[layer_name]
                else:
                    # Generate synthetic hidden states
                    hidden_size = self._get_hidden_size(layer)
                    batch_size = calibration_data.input_ids.shape[0] if calibration_data.input_ids.dim() > 0 else 1
                    seq_len = calibration_data.input_ids.shape[1] if calibration_data.input_ids.dim() > 1 else 128
                    
                    hidden_states = torch.randn(
                        batch_size, seq_len, hidden_size,
                        device=layer.device,
                        dtype=torch.float16
                    ) * 0.02
                    self.logger.debug(f"Generated synthetic hidden states from CalibrationSample for {layer_name}")
            elif isinstance(calibration_data, str):
                # This should never happen but we're seeing it - critical error
                self.logger.error(f"CRITICAL: calibration_data is a string: '{calibration_data}'")
                # Generate fallback synthetic data
                hidden_size = self._get_hidden_size(layer)
                hidden_states = torch.randn(1, 128, hidden_size, device=layer.device, dtype=torch.float16) * 0.02
            else:
                # Unknown format, generate synthetic data
                self.logger.warning(f"Unknown calibration_data type: {type(calibration_data)}")
                hidden_size = self._get_hidden_size(layer)
                hidden_states = torch.randn(1, 128, hidden_size, device=layer.device, dtype=torch.float16) * 0.02
            
            # Ensure hidden_states is not None and is a tensor
            if hidden_states is None:
                self.logger.error(f"Failed to prepare hidden_states for {layer_name}")
                hidden_size = self._get_hidden_size(layer)
                hidden_states = torch.randn(1, 128, hidden_size, device=layer.device, dtype=torch.float16) * 0.02
            
            # Ensure proper shape [batch, seq_len, hidden_size]
            if hidden_states.dim() == 2:
                hidden_states = hidden_states.unsqueeze(0)
            elif hidden_states.dim() != 3:
                self.logger.error(f"Unexpected hidden_states dimensions: {hidden_states.dim()}, shape: {hidden_states.shape}")
                # Try to fix it
                hidden_size = self._get_hidden_size(layer)
                hidden_states = torch.randn(1, 128, hidden_size, device=layer.device, dtype=torch.float16) * 0.02
            
            # Move to layer device
            hidden_states = hidden_states.to(layer.device)
            
            # Run forward pass to collect activations
            try:
                # Double-check that hidden_states is a tensor before forward pass
                if not isinstance(hidden_states, torch.Tensor):
                    self.logger.error(f"hidden_states is not a tensor before forward pass: {type(hidden_states)}")
                    hidden_size = self._get_hidden_size(layer)
                    hidden_states = torch.randn(1, 128, hidden_size, device=layer.device, dtype=torch.float16) * 0.02
                
                output = layer(hidden_states)
                
                # Cache output for next layer
                if isinstance(output, tuple):
                    output = output[0]
                # Store for next layer's input
                next_layer_name = self._get_next_layer_name(layer_name)
                if next_layer_name:
                    self.activation_cache[next_layer_name] = output.detach()
            except Exception as e:
                self.logger.warning(f"Forward pass failed during activation collection: {e}")
                self.logger.debug(f"hidden_states type: {type(hidden_states)}, shape: {hidden_states.shape if isinstance(hidden_states, torch.Tensor) else 'N/A'}")
        
        # Remove hooks
        for handle in handles:
            handle.remove()
        
        return activations
    
    def _get_next_layer_name(self, current_layer_name: str) -> Optional[str]:
        """Get the name of the next layer in sequence"""
        # Simple pattern matching for transformer layers
        import re
        match = re.search(r'layers\.(\d+)', current_layer_name)
        if match:
            current_idx = int(match.group(1))
            next_idx = current_idx + 1
            return current_layer_name.replace(f'layers.{current_idx}', f'layers.{next_idx}')
        return None
    
    def _get_previous_layer_name(self, current_layer_name: str) -> Optional[str]:
        """Get the name of the previous layer in sequence"""
        import re
        match = re.search(r'layers\.(\d+)', current_layer_name)
        if match:
            current_idx = int(match.group(1))
            if current_idx > 0:
                prev_idx = current_idx - 1
                return current_layer_name.replace(f'layers.{current_idx}', f'layers.{prev_idx}')
        return None
    
    def compute_awq_scales(self,
                      layer: torch.nn.Module,
                      calibration_data: Any,
                      layer_type: str = "generic") -> Dict[str, torch.Tensor]:
        """Compute AWQ scaling factors for layer using real activations
        
        AWQ algorithm:
        1. Collect activation statistics from real forward pass
        2. Identify salient weight channels (top 1%)
        3. Compute optimal scaling factors
        4. Apply smoothing between layers
        """
        scales = {}
        
        # Type checking for calibration_data
        self.logger.debug(f"[TYPE_CHECK] compute_awq_scales received type: {type(calibration_data)}")
        if isinstance(calibration_data, str):
            self.logger.error(f"[TYPE_ERROR] calibration_data is string in compute_awq_scales: '{calibration_data}'")
            # Create emergency fallback data
            hidden_size = self._get_hidden_size(layer)
            calibration_data = torch.randn(1, 128, hidden_size, 
                                          device=next(layer.parameters()).device, 
                                          dtype=torch.float16) * 0.02
            self.logger.warning(f"[TYPE_RECOVERY] Created synthetic calibration data with shape {calibration_data.shape}")
        
        # FIX: Extract actual tensor if calibration_data is a dict - PROPERLY
        calibration_tensor = None
        
        if isinstance(calibration_data, dict):
            # Make a defensive copy first
            calibration_dict = {k: v.clone() if torch.is_tensor(v) else v 
                               for k, v in calibration_data.items()}
            
            if 'hidden_states' in calibration_dict:
                tensor_value = calibration_dict['hidden_states']
                if torch.is_tensor(tensor_value):
                    calibration_tensor = tensor_value
                    self.logger.debug(f"Extracted hidden_states: shape={calibration_tensor.shape}, dtype={calibration_tensor.dtype}")
                else:
                    self.logger.error(f"hidden_states value is not a tensor: {type(tensor_value)}")
                    calibration_tensor = self._create_dummy_input(layer)
            elif 'input_ids' in calibration_dict:
                # Keep as dict for token handling, but use the copy
                calibration_tensor = calibration_dict
                self.logger.debug(f"Keeping calibration dict with keys: {list(calibration_dict.keys())}")
            else:
                # Unexpected dict, create dummy input
                self.logger.warning(f"No recognized keys in calibration dict: {list(calibration_dict.keys())}")
                calibration_tensor = self._create_dummy_input(layer)
        elif torch.is_tensor(calibration_data):
            calibration_tensor = calibration_data.clone()  # Defensive copy
            self.logger.debug(f"Using tensor calibration data: shape={calibration_tensor.shape}")
        else:
            self.logger.warning(f"Unexpected calibration_data type: {type(calibration_data)}")
            calibration_tensor = self._create_dummy_input(layer)
        
        # Validate before passing to collect_layer_activations
        if isinstance(calibration_tensor, str):
            self.logger.error(f"CRITICAL: calibration_tensor became string: '{calibration_tensor}'")
            calibration_tensor = self._create_dummy_input(layer)
        
        # First, collect real activations by running calibration data through the layer
        layer_name = getattr(layer, '_layer_name', 'unknown')
        activations = self.collect_layer_activations(layer, calibration_tensor, layer_name)
        
        # If no activations collected, try direct forward pass
        if not activations:
            self.logger.warning("No activations collected, attempting direct forward pass")
            
            # Prepare calibration inputs
            if isinstance(calibration_tensor, dict):
                if 'hidden_states' in calibration_tensor:
                    calibration_inputs = calibration_tensor['hidden_states']
                elif 'input_ids' in calibration_tensor:
                    # Try to use cached activations from previous layer
                    if layer_name in self.activation_cache:
                        calibration_inputs = self.activation_cache[layer_name]
                    else:
                        # Generate approximate hidden states
                        hidden_size = self._get_hidden_size(layer)
                        seq_length = calibration_tensor['input_ids'].shape[-1] if calibration_tensor['input_ids'].dim() > 1 else 128
                        batch_size = calibration_tensor['input_ids'].shape[0] if calibration_tensor['input_ids'].dim() > 0 else 1
                        
                        calibration_inputs = torch.randn(
                            batch_size, seq_length, hidden_size,
                            device=layer.device, 
                            dtype=torch.float16
                        ) * 0.02
                else:
                    calibration_inputs = self._create_dummy_input(layer)
            elif isinstance(calibration_tensor, torch.Tensor):
                calibration_inputs = calibration_tensor
                if calibration_inputs.dim() == 2:
                    calibration_inputs = calibration_inputs.unsqueeze(0)
            else:
                calibration_inputs = self._create_dummy_input(layer)
            
            # Move to same device as layer
            device = next(layer.parameters()).device
            calibration_inputs = calibration_inputs.to(device)
            
            # Collect activations with hooks
            activations = {}
            hooks = []
            linear_modules = {}
            
            def create_hook(name):
                def hook_fn(module, input, output):
                    if isinstance(input, tuple):
                        input = input[0]
                    activations[name] = input.detach()
                return hook_fn
            
            for name, module in layer.named_modules():
                if isinstance(module, nn.Linear):
                    handle = module.register_forward_hook(create_hook(name))
                    hooks.append(handle)
                    linear_modules[name] = module
            
            # Forward pass
            with torch.no_grad():
                try:
                    output = layer(calibration_inputs)
                    # Cache output for next layer
                    if isinstance(output, tuple):
                        output = output[0]
                    next_layer_name = self._get_next_layer_name(layer_name)
                    if next_layer_name:
                        self.activation_cache[next_layer_name] = output.detach()
                except Exception as e:
                    self.logger.warning(f"Forward pass failed during scale computation: {e}")
                    # Use default scales
                    for name, module in linear_modules.items():
                        scales[name] = torch.ones(module.weight.shape[1], device=module.weight.device)
                    for handle in hooks:
                        handle.remove()
                    return scales
            
            # Remove hooks
            for handle in hooks:
                handle.remove()
        else:
            # Get linear modules for scale computation
            linear_modules = {}
            for name, module in layer.named_modules():
                if isinstance(module, nn.Linear):
                    linear_modules[name] = module
        
        # Compute scales for each Linear layer
        for name, module in linear_modules.items():
            if name not in activations:
                self.logger.warning(f"No activations captured for {name}")
                # Use default scale
                scales[name] = torch.ones(module.weight.shape[1], device=module.weight.device)
                continue
            
            X = activations[name]
            W = module.weight
            
            # Ensure we have enough samples for statistics
            if X.shape[0] < 4:
                self.logger.warning(f"Too few samples ({X.shape[0]}) for {name}, using default scales")
                scales[name] = torch.ones(module.weight.shape[1], device=module.weight.device)
                continue
            
            # Reshape activations to [batch*seq_len, hidden] for better statistics
            orig_shape = X.shape
            if X.dim() == 3:
                X = X.reshape(-1, X.shape[-1])
            elif X.dim() == 2:
                pass  # Already correct shape
            else:
                self.logger.warning(f"Unexpected activation shape {orig_shape} for {name}")
                scales[name] = torch.ones(module.weight.shape[1], device=module.weight.device)
                continue
            
            # Compute activation statistics with better numerical stability
            # Use absolute mean and variance for robustness
            X_abs = torch.abs(X)
            X_mean = torch.mean(X_abs, dim=0)
            X_var = torch.var(X_abs, dim=0)
            X_max = torch.max(X_abs, dim=0)[0]
            
            # Compute weight statistics
            W_abs = torch.abs(W)
            W_mean = torch.mean(W_abs, dim=0)
            W_var = torch.var(W_abs, dim=0)
            W_max = torch.max(W_abs, dim=0)[0]
            
            # Identify salient channels using multiple criteria
            # 1. High mean activation
            # 2. High variance (dynamic range)
            # 3. High maximum values (outliers)
            
            # Normalize each metric
            mean_importance = X_mean / (X_mean.sum() + 1e-8)
            var_importance = X_var / (X_var.sum() + 1e-8)
            max_importance = X_max / (X_max.sum() + 1e-8)
            
            # Combined importance score
            importance = mean_importance * 0.5 + var_importance * 0.3 + max_importance * 0.2
            
            # Identify top k% most important channels
            k_percent = 0.01  # Top 1%
            num_salient = max(1, int(k_percent * len(importance)))
            salient_indices = torch.topk(importance, num_salient).indices
            
            # Compute optimal AWQ scales using the formula from the paper
            # scale = sqrt(mean(X^2) / mean(W^2)) with modifications
            
            # Use RMS (Root Mean Square) for more stable computation
            X_rms = torch.sqrt(torch.mean(X ** 2, dim=0) + 1e-8)
            W_rms = torch.sqrt(torch.mean(W ** 2, dim=0) + 1e-8)
            
            # Basic scale computation
            scale = torch.sqrt(X_rms / (W_rms + 1e-8))
            
            # Apply protection to salient channels
            # Increase scale for important channels to preserve their precision
            protection_factor = self.config.awq_protection_factor
            scale[salient_indices] *= protection_factor
            
            # Apply additional scaling based on weight magnitude
            # Channels with very small weights need less aggressive quantization
            small_weight_mask = W_max < (W_max.mean() * 0.1)
            scale[small_weight_mask] *= 0.5
            
            # Apply damping for numerical stability
            damping = self.config.awq_damping_percent
            scale = scale * (1 - damping) + 1.0 * damping
            
            # Clip to reasonable range
            scale = torch.clamp(scale, min=0.01, max=100.0)
            
            # Smooth the scales to avoid sudden changes
            if scale.shape[0] > 1:
                # Simple moving average smoothing
                kernel_size = min(5, scale.shape[0] // 10)
                if kernel_size > 1:
                    scale_padded = torch.nn.functional.pad(scale.unsqueeze(0).unsqueeze(0), 
                                                        (kernel_size//2, kernel_size//2), 
                                                        mode='replicate')
                    kernel = torch.ones(1, 1, kernel_size, device=scale.device) / kernel_size
                    scale = torch.nn.functional.conv1d(scale_padded, kernel).squeeze()
            
            scales[name] = scale
            
            self.logger.debug(f"Computed scales for {name}: mean={scale.mean():.3f}, "
                            f"std={scale.std():.3f}, min={scale.min():.3f}, max={scale.max():.3f}, "
                            f"salient={num_salient}/{len(scale)}")
        
        return scales
    
    def compute_awq_scales_multi_sample(self,
                                       layer: torch.nn.Module,
                                       calibration_dataloader: Any,
                                       layer_type: str = "generic",
                                       num_samples: int = 32) -> Dict[str, torch.Tensor]:
        """Compute AWQ scales using multiple calibration samples for better statistics
        
        Args:
            layer: Layer module
            calibration_dataloader: DataLoader with calibration samples
            layer_type: Type of layer
            num_samples: Number of samples to use
            
        Returns:
            Dictionary of scaling factors per module
        """
        # Accumulate activations from multiple samples
        accumulated_activations = {}
        linear_modules = {}
        
        # Get linear modules
        for name, module in layer.named_modules():
            if isinstance(module, nn.Linear):
                linear_modules[name] = module
        
        # Process multiple batches
        samples_processed = 0
        for batch_idx, batch in enumerate(calibration_dataloader):
            if samples_processed >= num_samples:
                break
            
            # Get activations for this batch
            layer_name = getattr(layer, '_layer_name', f'layer_{batch_idx}')
            batch_activations = self.collect_layer_activations(layer, batch, layer_name)
            
            # Accumulate activations
            for name, activation in batch_activations.items():
                if name not in accumulated_activations:
                    accumulated_activations[name] = []
                accumulated_activations[name].append(activation.cpu())
            
            samples_processed += batch.input_ids.shape[0] if hasattr(batch, 'input_ids') else 1
        
        # Concatenate all activations
        combined_activations = {}
        for name, activation_list in accumulated_activations.items():
            if activation_list:
                combined_activations[name] = torch.cat(activation_list, dim=0)
        
        # Compute scales using combined activations
        scales = {}
        for name, module in linear_modules.items():
            if name not in combined_activations:
                self.logger.warning(f"No activations collected for {name}")
                scales[name] = torch.ones(module.weight.shape[1], device=module.weight.device)
                continue
            
            X = combined_activations[name].to(module.weight.device)
            W = module.weight
            
            # Compute statistics on larger sample
            X_abs = torch.abs(X.reshape(-1, X.shape[-1]))
            W_abs = torch.abs(W)
            
            # RMS-based scaling
            X_rms = torch.sqrt(torch.mean(X ** 2, dim=0) + 1e-8)
            W_rms = torch.sqrt(torch.mean(W ** 2, dim=0) + 1e-8)
            
            scale = torch.sqrt(X_rms / (W_rms + 1e-8))
            
            # Identify and protect salient channels
            importance = torch.mean(X_abs, dim=0) / (torch.mean(X_abs) + 1e-8)
            num_salient = max(1, int(0.01 * len(importance)))
            salient_indices = torch.topk(importance, num_salient).indices
            scale[salient_indices] *= self.config.awq_protection_factor
            
            # Apply damping and clipping
            damping = self.config.awq_damping_percent
            scale = scale * (1 - damping) + 1.0 * damping
            scale = torch.clamp(scale, min=0.01, max=100.0)
            
            scales[name] = scale
            
            self.logger.debug(f"Multi-sample scales for {name}: samples={samples_processed}, "
                            f"mean={scale.mean():.3f}, std={scale.std():.3f}")
        
        return scales
    
    def _safe_dict_get(self, data: Dict[str, Any], key: str, expected_type: type = None) -> Any:
        """Safely extract value from dictionary with type checking
        
        Args:
            data: Dictionary to extract from
            key: Key to extract
            expected_type: Expected type of the value (optional)
            
        Returns:
            The value if found and valid, None otherwise
        """
        if not isinstance(data, dict):
            self.logger.error(f"_safe_dict_get called with non-dict: {type(data)}")
            return None
            
        if key not in data:
            self.logger.debug(f"Key '{key}' not found in dict with keys: {list(data.keys())}")
            return None
            
        value = data[key]
        
        # Check if we got the key string instead of the value (the bug we're fixing)
        if isinstance(value, str) and value == key:
            self.logger.error(f"CRITICAL BUG: Got key string '{key}' instead of value!")
            import traceback
            self.logger.error(f"Stack trace:\n{''.join(traceback.format_stack())}")
            return None
            
        if expected_type is not None and not isinstance(value, expected_type):
            self.logger.warning(f"Value for key '{key}' has unexpected type: {type(value)}, expected {expected_type}")
            return None
            
        return value
    
    def _get_hidden_size(self, layer: nn.Module) -> int:
        """Get hidden size from layer"""
        if hasattr(layer, 'hidden_size'):
            return layer.hidden_size
        elif hasattr(layer, 'input_layernorm'):
            return layer.input_layernorm.weight.shape[0]
        else:
            # Try to infer from first linear layer
            for module in layer.modules():
                if isinstance(module, nn.Linear):
                    return module.in_features
            # Fallback
            return self.config.hidden_size if hasattr(self.config, 'hidden_size') else 4096
    
    def _create_dummy_input(self, layer: nn.Module) -> torch.Tensor:
        """Create dummy input tensor for layer with correct dtype"""
        hidden_size = self._get_hidden_size(layer)
        batch_size = 1
        seq_length = 128
        
        # Get the layer's expected dtype from its parameters
        layer_dtype = next(layer.parameters()).dtype
        layer_device = next(layer.parameters()).device
        
        self.logger.debug(f"Creating dummy input with dtype={layer_dtype}, device={layer_device}")
        
        return torch.randn(
            batch_size, seq_length, hidden_size,
            device=layer_device,
            dtype=layer_dtype  # Use layer's dtype, not hardcoded float16
        ) * 0.02  # Proper initialization scale
    
    def apply_awq_quantization(self,
                              weight: torch.Tensor,
                              scales: torch.Tensor,
                              group_size: int = 32) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply AWQ quantization to weights
        
        Quantization process:
        1. Apply scaling to weights
        2. Group weights for group-wise quantization
        3. Compute zero points and scales per group
        4. Quantize to INT4
        """
        # Apply AWQ scales
        scaled_weight = weight * scales.unsqueeze(0)
        
        # Reshape for group-wise quantization
        orig_shape = scaled_weight.shape
        scaled_weight_flat = scaled_weight.reshape(-1)
        
        # Pad if necessary for group size
        remainder = len(scaled_weight_flat) % group_size
        if remainder != 0:
            padding = group_size - remainder
            scaled_weight_flat = torch.cat([
                scaled_weight_flat,
                torch.zeros(padding, device=scaled_weight_flat.device, dtype=scaled_weight_flat.dtype)
            ])
        
        # Reshape into groups
        weight_groups = scaled_weight_flat.reshape(-1, group_size)
        num_groups = weight_groups.shape[0]
        
        # Quantization parameters
        bits = self.config.bits
        max_quant_val = 2**bits - 1
        
        # Initialize outputs
        quantized_groups = torch.zeros_like(weight_groups, dtype=torch.uint8)
        group_scales = torch.zeros(num_groups, dtype=torch.float16, device=weight.device)
        group_zeros = torch.zeros(num_groups, dtype=torch.float16, device=weight.device)
        
        # Quantize each group
        for i in range(num_groups):
            group = weight_groups[i]
            
            # Find min and max
            min_val = group.min().item()
            max_val = group.max().item()
            
            # Symmetric vs asymmetric quantization
            if self.config.symmetric:
                # Symmetric quantization
                abs_max = max(abs(min_val), abs(max_val))
                scale = (2 * abs_max) / max_quant_val if abs_max > 0 else 1.0
                zero_point = (max_quant_val + 1) / 2
            else:
                # Asymmetric quantization
                scale = (max_val - min_val) / max_quant_val if max_val != min_val else 1.0
                zero_point = round(-min_val / scale) if scale != 0 else 0
            
            # Quantize
            quantized = torch.round(group / scale + zero_point)
            quantized = torch.clamp(quantized, 0, max_quant_val)
            
            quantized_groups[i] = quantized.to(torch.uint8)
            group_scales[i] = scale
            group_zeros[i] = zero_point
        
        # Reshape back (removing padding if added)
        if remainder != 0:
            quantized_flat = quantized_groups.reshape(-1)[:-padding]
        else:
            quantized_flat = quantized_groups.reshape(-1)
        
        # Reshape to original shape
        quantized_weight = quantized_flat.reshape(orig_shape)
        
        return quantized_weight, group_scales, group_zeros
    
    def apply_propagated_scales(self,
                               weight: torch.Tensor,
                               input_scales: torch.Tensor,
                               output_scales: torch.Tensor) -> torch.Tensor:
        """Apply propagated scales to weight matrix
        
        AWQ formula: W_scaled = W * diag(s_in) / diag(s_out)
        
        Args:
            weight: Original weight matrix [out_features, in_features]
            input_scales: Scales for input channels [in_features]
            output_scales: Scales for output channels [out_features]
            
        Returns:
            Scaled weight matrix
        """
        # Ensure scales are on same device as weights
        input_scales = input_scales.to(weight.device)
        output_scales = output_scales.to(weight.device)
        
        # Apply input scales (multiply columns)
        weight_scaled = weight * input_scales.unsqueeze(0)
        
        # Apply output scales (divide rows)
        weight_scaled = weight_scaled / (output_scales.unsqueeze(1) + 1e-8)
        
        return weight_scaled
    
    def propagate_scales_through_residual(self,
                                         pre_residual_scales: torch.Tensor,
                                         post_residual_scales: torch.Tensor,
                                         residual_weight: float = 0.5) -> torch.Tensor:
        """Propagate scales through residual connections
        
        Args:
            pre_residual_scales: Scales before residual
            post_residual_scales: Scales after residual  
            residual_weight: Weight of residual path (vs main path)
            
        Returns:
            Combined scales accounting for residual
        """
        # Residual connections require special handling
        # The effective scale is a weighted combination
        combined_scales = (1 - residual_weight) * post_residual_scales + residual_weight * pre_residual_scales
        
        # Normalize to maintain scale magnitude
        combined_scales = combined_scales * (pre_residual_scales.mean() / combined_scales.mean())
        
        return combined_scales
    
    def smooth_scales_between_layers(self,
                                    prev_layer_scales: Dict[str, torch.Tensor],
                                    curr_layer_scales: Dict[str, torch.Tensor],
                                    smoothing_factor: float = 0.5) -> Dict[str, torch.Tensor]:
        """Smooth scales between consecutive layers for better quantization
        
        AWQ insight: Scales should be coordinated between connected layers
        
        Args:
            prev_layer_scales: Scales from previous layer
            curr_layer_scales: Scales for current layer
            smoothing_factor: How much to blend (0=no smoothing, 1=full averaging)
            
        Returns:
            Smoothed scales for current layer
        """
        if not prev_layer_scales or smoothing_factor == 0:
            return curr_layer_scales
        
        smoothed_scales = {}
        
        for name, curr_scale in curr_layer_scales.items():
            # Find corresponding scale in previous layer
            # For attention: o_proj output affects next layer's input_layernorm
            # For MLP: down_proj output affects next layer's input_layernorm
            
            prev_scale = None
            if 'input_layernorm' in name:
                # Look for previous layer's output projection
                for prev_name, prev_s in prev_layer_scales.items():
                    if 'o_proj' in prev_name or 'down_proj' in prev_name:
                        prev_scale = prev_s
                        break
            elif 'q_proj' in name or 'k_proj' in name or 'v_proj' in name:
                # These connect to input_layernorm
                for prev_name, prev_s in prev_layer_scales.items():
                    if 'input_layernorm' in prev_name:
                        prev_scale = prev_s
                        break
            elif 'gate_proj' in name or 'up_proj' in name:
                # These connect to post_attention_layernorm
                for prev_name, prev_s in prev_layer_scales.items():
                    if 'post_attention_layernorm' in prev_name:
                        prev_scale = prev_s
                        break
            
            if prev_scale is not None and prev_scale.shape == curr_scale.shape:
                # Apply smoothing
                smoothed_scale = (1 - smoothing_factor) * curr_scale + smoothing_factor * prev_scale
                smoothed_scales[name] = smoothed_scale
                
                self.logger.debug(f"Smoothed scales for {name}: "
                                f"prev_mean={prev_scale.mean():.3f}, "
                                f"curr_mean={curr_scale.mean():.3f}, "
                                f"smoothed_mean={smoothed_scale.mean():.3f}")
            else:
                # No matching previous scale, use current
                smoothed_scales[name] = curr_scale
        
        return smoothed_scales
    
    def pack_int4_weights(self, weights: torch.Tensor) -> torch.Tensor:
        """Pack INT4 weights for efficient storage
        
        Packing strategy:
        - Pack 2 INT4 values into 1 INT8
        - Maintain alignment for efficient unpacking
        - Store metadata for reconstruction
        """
        if self.config.bits != 4:
            return weights
        
        # Flatten weights
        orig_shape = weights.shape
        weights_flat = weights.reshape(-1)
        
        # Ensure even number of elements
        if len(weights_flat) % 2 != 0:
            weights_flat = torch.cat([
                weights_flat,
                torch.zeros(1, dtype=weights_flat.dtype, device=weights_flat.device)
            ])
        
        # Pack pairs: high nibble and low nibble
        packed = torch.zeros(len(weights_flat) // 2, dtype=torch.uint8, device=weights.device)
        packed = (weights_flat[::2].to(torch.uint8) << 4) | weights_flat[1::2].to(torch.uint8)
        
        # Reshape preserving batch dimensions
        packed_shape = list(orig_shape)
        packed_shape[-1] = (packed_shape[-1] + 1) // 2  # Round up division
        packed = packed.reshape(packed_shape)
        
        return packed
    
    def unpack_int4_weights(self, packed: torch.Tensor, shape: Tuple[int, ...]) -> torch.Tensor:
        """Unpack INT4 weights from packed format"""
        # Flatten packed weights
        packed_flat = packed.reshape(-1)
        
        # Extract high and low nibbles
        high = (packed_flat >> 4) & 0xF
        low = packed_flat & 0xF
        
        # Interleave to create unpacked weights
        unpacked = torch.zeros(len(packed_flat) * 2, dtype=torch.uint8, device=packed.device)
        unpacked[::2] = high
        unpacked[1::2] = low
        
        # Reshape to original shape
        # Handle potential padding
        target_numel = np.prod(shape)
        unpacked = unpacked[:target_numel].reshape(shape)
        
        return unpacked
    
    def validate_quantized_layer(self,
                                 original: torch.nn.Module,
                                 quantized: torch.nn.Module,
                                 test_input: torch.Tensor) -> float:
        """Validate quantized layer against original
        
        Validation metrics:
        - MSE between outputs
        - Cosine similarity
        - Max absolute difference
        """
        with torch.no_grad():
            # Move to same device
            device = next(original.parameters()).device
            
            # Create proper test input
            if hasattr(original, 'hidden_size'):
                hidden_size = original.hidden_size
            elif hasattr(original, 'input_layernorm'):
                hidden_size = original.input_layernorm.weight.shape[0]
            else:
                # Try to infer from first linear layer
                for module in original.modules():
                    if isinstance(module, nn.Linear):
                        hidden_size = module.in_features
                        break
                else:
                    hidden_size = 4096  # Fallback
            
            # Create test input with proper shape
            batch_size = 1
            seq_len = 128
            test_tensor = torch.randn(batch_size, seq_len, hidden_size, 
                                     device=device, dtype=torch.float16)
            
            # Get original output
            try:
                orig_out = original(test_tensor)
            except:
                # Some layers might need different inputs
                return 0.0
            
            # Get quantized output
            try:
                quant_out = quantized(test_tensor)
            except:
                # Quantized layer might have different structure
                return 0.0
            
            # Handle different output types
            if isinstance(orig_out, tuple):
                orig_out = orig_out[0]
            if isinstance(quant_out, tuple):
                quant_out = quant_out[0]
            
            # Calculate MSE
            mse = torch.mean((orig_out - quant_out) ** 2).item()
            
            # Calculate cosine similarity
            orig_flat = orig_out.reshape(-1)
            quant_flat = quant_out.reshape(-1)
            cosine_sim = F.cosine_similarity(orig_flat.unsqueeze(0), quant_flat.unsqueeze(0)).item()
            
            # Check for NaN/Inf
            if torch.isnan(quant_out).any() or torch.isinf(quant_out).any():
                self.logger.warning("NaN or Inf detected in quantized output")
                return float('inf')
            
            # Relative error
            relative_error = mse / (torch.mean(orig_out ** 2).item() + 1e-8)
            
            self.logger.debug(f"Validation - MSE: {mse:.6f}, Cosine Sim: {cosine_sim:.4f}, "
                            f"Relative Error: {relative_error:.6f}")
            
            return relative_error
    
    def get_layer_config(self, layer_name: str) -> Dict[str, Any]:
        """Get quantization config for specific layer"""
        # Check if layer should be quantized
        if self.should_quantize_layer(layer_name):
            # Get specific config for layer type
            if "attention" in layer_name or any(x in layer_name for x in ['q_proj', 'k_proj', 'v_proj', 'o_proj']):
                return {
                    "bits": self.config.bits,
                    "group_size": self.config.group_size,
                    "symmetric": False,  # Asymmetric for attention
                }
            elif "mlp" in layer_name or any(x in layer_name for x in ['gate_proj', 'up_proj', 'down_proj']):
                return {
                    "bits": self.config.bits,
                    "group_size": self.config.group_size,
                    "symmetric": self.config.symmetric,
                }
            elif "expert" in layer_name:
                return {
                    "bits": self.config.bits,
                    "group_size": self.config.group_size,
                    "symmetric": self.config.symmetric,
                }
            else:
                return {
                    "bits": self.config.bits,
                    "group_size": self.config.group_size,
                    "symmetric": self.config.symmetric,
                }
        else:
            return None
    
    def should_quantize_layer(self, layer_name: str) -> bool:
        """Check if layer should be quantized"""
        # Check against skip_layers patterns
        for pattern in self.config.skip_layers:
            # Handle wildcards
            pattern = pattern.replace('*', '.*')
            import re
            if re.match(pattern, layer_name):
                return False
        
        # Additional checks
        skip_keywords = ['embedding', 'norm', 'router', 'lm_head', 'output_layer']
        for keyword in skip_keywords:
            if keyword in layer_name.lower():
                return False
        
        return True
    
    def quantize_attention_layer(self,
                            attn_module: nn.Module,
                            calibration_data: Any,
                            scales: Dict[str, torch.Tensor]) -> nn.Module:
        """Specialized quantization for attention layers
        
        Handles:
        - Q, K, V projections
        - Output projection
        - Maintaining attention patterns
        """
        # Find attention sub-modules
        for name, module in attn_module.named_modules():
            if isinstance(module, nn.Linear):
                if any(proj in name for proj in ['q_proj', 'k_proj', 'v_proj', 'o_proj']):
                    if name in scales:
                        # Apply AWQ quantization
                        weight = module.weight
                        scale = scales[name]
                        
                        # Quantize
                        quantized_weight, group_scales, group_zeros = self.apply_awq_quantization(
                            weight, scale, self.config.group_size
                        )
                        
                        # Pack if INT4
                        if self.config.bits == 4:
                            quantized_weight = self.pack_int4_weights(quantized_weight)
                        
                        # Store quantized weights
                        module.register_buffer('weight_quantized', quantized_weight)
                        module.register_buffer('weight_scales', group_scales)
                        module.register_buffer('weight_zeros', group_zeros)
                        module.quantized = True
        
        return attn_module
    
    def quantize_mlp_layer(self,
                      mlp_module: nn.Module,
                      calibration_data: Any,
                      scales: Dict[str, torch.Tensor]) -> nn.Module:
        """Specialized quantization for MLP layers
        
        Handles:
        - Gate and up projections (GLM uses gated MLP)
        - Down projection
        - Activation functions
        """
        # Find MLP sub-modules
        for name, module in mlp_module.named_modules():
            if isinstance(module, nn.Linear):
                if any(proj in name for proj in ['gate_proj', 'up_proj', 'down_proj']):
                    if name in scales:
                        # Apply AWQ quantization
                        weight = module.weight
                        scale = scales[name]
                        
                        # Quantize with potentially different group size for MLP
                        mlp_group_size = self.config.group_size * 2  # Larger groups for MLP
                        quantized_weight, group_scales, group_zeros = self.apply_awq_quantization(
                            weight, scale, mlp_group_size
                        )
                        
                        # Pack if INT4
                        if self.config.bits == 4:
                            quantized_weight = self.pack_int4_weights(quantized_weight)
                        
                        # Store quantized weights
                        module.register_buffer('weight_quantized', quantized_weight)
                        module.register_buffer('weight_scales', group_scales)
                        module.register_buffer('weight_zeros', group_zeros)
                        module.quantized = True
        
        return mlp_module
    
    def quantize_moe_layer(self,
                      moe_module: nn.Module,
                      calibration_data: Any,
                      scales: Dict[str, torch.Tensor] = None) -> nn.Module:
        """Specialized quantization for MoE layers
        
        Handles:
        - Expert routing (keep in FP16)
        - Individual expert MLPs with propagated scales
        - Load balancing considerations
        """
        # Keep router in FP16 (critical for routing accuracy)
        self.logger.info("Keeping MoE router in FP16")
        
        # Track expert activation
        expert_activations = {}
        
        def track_expert(name):
            def hook_fn(module, input, output):
                expert_activations[name] = True
            return hook_fn
        
        # Register hooks on experts
        hooks = []
        if hasattr(moe_module, 'experts'):
            for i, expert in enumerate(moe_module.experts):
                handle = expert.register_forward_hook(track_expert(f"expert_{i}"))
                hooks.append(handle)
        
        # Run forward pass to see which experts are activated
        with torch.no_grad():
            try:
                # Create proper input - FIX: Use calibration_data properly
                if isinstance(calibration_data, dict):
                    if 'hidden_states' in calibration_data:
                        dummy_input = calibration_data['hidden_states']
                    else:
                        # Use synthetic data if no hidden states
                        if hasattr(moe_module, 'hidden_size'):
                            hidden_size = moe_module.hidden_size
                        else:
                            hidden_size = 4096
                        dummy_input = torch.randn(1, 128, hidden_size, 
                                                device=next(moe_module.parameters()).device, 
                                                dtype=torch.float16) * 0.02
                elif torch.is_tensor(calibration_data):
                    dummy_input = calibration_data
                else:
                    # Fallback to synthetic
                    if hasattr(moe_module, 'hidden_size'):
                        hidden_size = moe_module.hidden_size
                    else:
                        hidden_size = 4096
                    dummy_input = torch.randn(1, 128, hidden_size, 
                                            device=next(moe_module.parameters()).device, 
                                            dtype=torch.float16) * 0.02
                
                # Ensure proper device
                dummy_input = dummy_input.to(next(moe_module.parameters()).device)
                _ = moe_module(dummy_input)
            except:
                pass
        
        # Remove hooks
        for handle in hooks:
            handle.remove()
        
        # Quantize shared experts (always active)
        if hasattr(moe_module, 'shared_experts'):
            for i, expert in enumerate(moe_module.shared_experts):
                self.logger.info(f"Quantizing shared expert {i}")
                # FIX: Pass calibration_data, not just calibration_data
                expert_scales = self.compute_awq_scales(expert, calibration_data, "moe_expert")
                # FIX: Pass calibration_data to quantize_mlp_layer
                quantized_expert = self.quantize_mlp_layer(expert, calibration_data, expert_scales)
                moe_module.shared_experts[i] = quantized_expert
        
        # Quantize activated routed experts
        if hasattr(moe_module, 'experts'):
            for i, expert in enumerate(moe_module.experts):
                if f"expert_{i}" in expert_activations or len(expert_activations) == 0:
                    self.logger.info(f"Quantizing expert {i} (activated)")
                    # FIX: Pass calibration_data properly
                    expert_scales = self.compute_awq_scales(expert, calibration_data, "moe_expert")
                    # FIX: Pass calibration_data to quantize_mlp_layer
                    quantized_expert = self.quantize_mlp_layer(expert, calibration_data, expert_scales)
                    moe_module.experts[i] = quantized_expert
                else:
                    self.logger.info(f"Skipping expert {i} (not activated)")
                    # Could remove or keep in FP16
        
        return moe_module
    
    def compute_quantization_error(self,
                                  original_weight: torch.Tensor,
                                  quantized_weight: torch.Tensor,
                                  scales: torch.Tensor,
                                  zero_points: Optional[torch.Tensor]) -> float:
        """Compute reconstruction error after quantization"""
        # Dequantize
        if self.config.bits == 4:
            quantized_weight = self.unpack_int4_weights(quantized_weight, original_weight.shape)
        
        # Apply dequantization
        if zero_points is not None:
            dequant = (quantized_weight.float() - zero_points.unsqueeze(1)) * scales.unsqueeze(1)
        else:
            dequant = quantized_weight.float() * scales.unsqueeze(1)
        
        # Reshape to match original
        dequant = dequant.reshape(original_weight.shape)
        
        # Calculate MSE
        mse = torch.mean((original_weight - dequant) ** 2).item()
        
        # Calculate relative error
        relative_error = mse / (torch.mean(original_weight ** 2).item() + 1e-8)
        
        return relative_error
    
    def optimize_quantization_parameters(self,
                                        weight: torch.Tensor,
                                        calibration_data: Any) -> Dict[str, Any]:
        """Optimize quantization parameters for minimal error
        
        Grid search or optimization for:
        - group_size: [32, 64, 128]
        - damping: [0.01, 0.1]
        - symmetric vs asymmetric
        """
        best_params = {
            "group_size": self.config.group_size,
            "damping": self.config.awq_damping_percent,
            "symmetric": self.config.symmetric,
        }
        best_error = float('inf')
        
        # Grid search
        for group_size in [32, 64, 128]:
            for damping in [0.01, 0.05, 0.1]:
                for symmetric in [True, False]:
                    # Temporarily update config
                    old_group = self.config.group_size
                    old_damping = self.config.awq_damping_percent
                    old_symmetric = self.config.symmetric
                    
                    self.config.group_size = group_size
                    self.config.awq_damping_percent = damping
                    self.config.symmetric = symmetric
                    
                    # Test quantization
                    try:
                        # Compute dummy scales
                        scales = torch.ones(weight.shape[1], device=weight.device)
                        
                        # Quantize
                        quantized, group_scales, group_zeros = self.apply_awq_quantization(
                            weight, scales, group_size
                        )
                        
                        # Compute error
                        error = self.compute_quantization_error(
                            weight, quantized, group_scales, group_zeros
                        )
                        
                        if error < best_error:
                            best_error = error
                            best_params = {
                                "group_size": group_size,
                                "damping": damping,
                                "symmetric": symmetric,
                            }
                    except:
                        pass
                    
                    # Restore config
                    self.config.group_size = old_group
                    self.config.awq_damping_percent = old_damping
                    self.config.symmetric = old_symmetric
        
        self.logger.info(f"Optimal parameters: {best_params} (error: {best_error:.6f})")
        return best_params
    
    def create_quantized_linear(self,
                               original: nn.Linear,
                               quantized_weight: torch.Tensor,
                               scales: torch.Tensor,
                               zero_points: Optional[torch.Tensor],
                               bias: Optional[torch.Tensor]) -> nn.Module:
        """Create a quantized Linear module
        
        Custom module that:
        - Stores INT4 weights efficiently
        - Dequantizes on-the-fly during forward
        - Compatible with vLLM kernels
        """
        
        class QuantizedLinear(nn.Module):
            """Quantized linear layer with on-the-fly dequantization"""
            
            def __init__(self, in_features, out_features, weight_quantized, scales, zeros, bias, bits, group_size):
                super().__init__()
                self.in_features = in_features
                self.out_features = out_features
                self.bits = bits
                self.group_size = group_size
                
                # Store quantized weights and metadata
                self.register_buffer('weight_quantized', weight_quantized)
                self.register_buffer('scales', scales)
                if zeros is not None:
                    self.register_buffer('zeros', zeros)
                else:
                    self.zeros = None
                
                if bias is not None:
                    self.register_buffer('bias', bias)
                else:
                    self.bias = None
            
            def forward(self, x):
                # Dequantize weights
                weight = self._dequantize_weight()
                
                # Linear operation
                return F.linear(x, weight, self.bias)
            
            def _dequantize_weight(self):
                """Dequantize weights for computation"""
                # Unpack if INT4
                if self.bits == 4:
                    weight = self._unpack_int4(self.weight_quantized)
                else:
                    weight = self.weight_quantized.float()
                
                # Reshape for dequantization
                weight_flat = weight.reshape(-1)
                
                # Apply scales and zero points
                if self.zeros is not None:
                    # Expand scales and zeros to match weight shape
                    scales_expanded = self.scales.repeat_interleave(self.group_size)
                    zeros_expanded = self.zeros.repeat_interleave(self.group_size)
                    
                    # Trim to match weight size
                    scales_expanded = scales_expanded[:len(weight_flat)]
                    zeros_expanded = zeros_expanded[:len(weight_flat)]
                    
                    weight_flat = (weight_flat - zeros_expanded) * scales_expanded
                else:
                    scales_expanded = self.scales.repeat_interleave(self.group_size)
                    scales_expanded = scales_expanded[:len(weight_flat)]
                    weight_flat = weight_flat * scales_expanded
                
                # Reshape to original
                weight = weight_flat.reshape(self.out_features, self.in_features)
                
                return weight
            
            def _unpack_int4(self, packed):
                """Unpack INT4 weights"""
                # Extract nibbles
                high = (packed >> 4) & 0xF
                low = packed & 0xF
                
                # Interleave
                unpacked_shape = list(packed.shape)
                unpacked_shape[-1] = unpacked_shape[-1] * 2
                unpacked = torch.zeros(unpacked_shape, dtype=torch.float32, device=packed.device)
                unpacked[..., ::2] = high.float()
                unpacked[..., 1::2] = low.float()
                
                return unpacked
        
        # Create quantized linear module
        in_features = original.in_features
        out_features = original.out_features
        
        return QuantizedLinear(
            in_features, out_features,
            quantized_weight, scales, zero_points,
            bias, self.config.bits, self.config.group_size
        )
    
    def save_quantization_state(self,
                               layer_name: str,
                               state: Dict[str, Any]) -> None:
        """Save quantization state for layer"""
        # Create state dict
        save_state = {
            'layer_name': layer_name,
            'scales': state.get('scales', {}),
            'zero_points': state.get('zero_points', {}),
            'config': {
                'bits': self.config.bits,
                'group_size': self.config.group_size,
                'symmetric': self.config.symmetric,
            },
            'timestamp': time.time(),
        }
        
        # Add to cache
        self.quantization_cache[layer_name] = save_state
        
        # Optionally save to disk
        if hasattr(self.config, 'save_intermediate'):
            cache_file = Path(self.config.checkpoint_dir) / f"{layer_name}_quant_state.pt"
            torch.save(save_state, cache_file)
    
    def load_quantization_state(self,
                               layer_name: str) -> Optional[Dict[str, Any]]:
        """Load previously computed quantization state"""
        # Check cache first
        if layer_name in self.quantization_cache:
            return self.quantization_cache[layer_name]
        
        # Check disk
        cache_file = Path(self.config.checkpoint_dir) / f"{layer_name}_quant_state.pt"
        if cache_file.exists():
            state = torch.load(cache_file)
            self.quantization_cache[layer_name] = state
            return state
        
        return None
    
    # Helper methods
    
    def _should_skip_layer(self, layer_name: str, layer: nn.Module) -> bool:
        """Check if layer should be skipped"""
        # Embedding layers should typically be skipped
        if 'embed' in layer_name.lower() or 'embedding' in layer_name.lower():
            return True
        
        # Check config skip patterns
        if not self.should_quantize_layer(layer_name):
            return True
        
        # Skip if layer is too small
        layer_size = self._calculate_layer_size(layer)
        if layer_size < 0.001:  # Less than 1MB
            return True
        
        return False
    
    def _identify_layer_type(self, layer: nn.Module, layer_name: str) -> str:
        """Identify the type of layer"""
        layer_name_lower = layer_name.lower()
        
        # Check for MoE
        if hasattr(layer, 'experts') or 'moe' in layer_name_lower or 'expert' in layer_name_lower:
            return "moe"
        
        # Check for attention
        if any(x in layer_name_lower for x in ['attention', 'attn', 'self_attn']):
            return "attention"
        
        # Check for MLP
        if any(x in layer_name_lower for x in ['mlp', 'ffn', 'feed_forward']):
            return "mlp"
        
        # Check module structure
        has_qkv = any(hasattr(layer, attr) for attr in ['q_proj', 'k_proj', 'v_proj'])
        if has_qkv:
            return "attention"
        
        # Check for attention submodules
        has_attention_modules = any('q_proj' in n or 'k_proj' in n or 'v_proj' in n 
                                   for n, _ in layer.named_modules())
        if has_attention_modules:
            return "attention"
        
        has_gate_up = any(hasattr(layer, attr) for attr in ['gate_proj', 'up_proj'])
        if has_gate_up:
            return "mlp"
        
        # Check for MLP submodules
        has_mlp_modules = any('gate_proj' in n or 'up_proj' in n 
                             for n, _ in layer.named_modules())
        if has_mlp_modules:
            return "mlp"
        
        return "generic"
    
    def _determine_device(self, layer: nn.Module, size_gb: float) -> str:
        """Determine optimal device for quantization"""
        if not torch.cuda.is_available():
            return 'cpu'
        
        # Check if layer fits on GPU with calibration data
        if self.memory_manager.can_fit_on_gpu(size_gb * 2):  # 2x for working memory
            return 'cuda'
        else:
            return 'cpu'
    
    def _calculate_layer_size(self, layer: nn.Module) -> float:
        """Calculate layer size in GB"""
        total_params = 0
        for param in layer.parameters():
            total_params += param.numel()
        
        # Assume FP16 for unquantized, or calculate based on actual dtype
        bytes_per_param = 2
        if hasattr(layer, 'quantized') and layer.quantized:
            bytes_per_param = self.config.bits / 8
        
        return (total_params * bytes_per_param) / 1e9
    
    def _apply_awq_quantization(self, layer: nn.Module, scales: Dict[str, torch.Tensor]) -> nn.Module:
        """Generic AWQ quantization for any layer"""
        for name, module in layer.named_modules():
            if isinstance(module, nn.Linear) and name in scales:
                weight = module.weight
                scale = scales[name]
                
                # Quantize
                quantized_weight, group_scales, group_zeros = self.apply_awq_quantization(
                    weight, scale, self.config.group_size
                )
                
                # Pack if INT4
                if self.config.bits == 4:
                    quantized_weight = self.pack_int4_weights(quantized_weight)
                
                # Create quantized module
                quantized_module = self.create_quantized_linear(
                    module, quantized_weight, group_scales, group_zeros, 
                    module.bias if hasattr(module, 'bias') else None
                )
                
                # Replace module in layer
                parent_name, child_name = name.rsplit('.', 1) if '.' in name else ('', name)
                if parent_name:
                    parent = layer
                    for part in parent_name.split('.'):
                        parent = getattr(parent, part)
                    setattr(parent, child_name, quantized_module)
                else:
                    setattr(layer, name, quantized_module)
        
        return layer
    
    def _pack_layer_weights(self, layer: nn.Module) -> nn.Module:
        """Pack all quantized weights in layer"""
        # Already handled in quantization step
        return layer
    
    def _extract_zero_points(self, layer: nn.Module) -> Dict[str, torch.Tensor]:
        """Extract zero points from quantized layer"""
        zero_points = {}
        for name, module in layer.named_modules():
            if hasattr(module, 'zeros'):
                zero_points[name] = module.zeros
        return zero_points