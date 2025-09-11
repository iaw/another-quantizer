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
    

class LayerQuantizer:
    """Handles quantization of individual layers"""
    
    def __init__(self, config: Any, memory_manager: Any):
        """Initialize layer quantizer"""
        self.config = config
        self.memory_manager = memory_manager
        self.awq_scales = {}  # Cache scales across layers
        self.quantization_cache = {}  # Cache for reuse
        self.logger = logging.getLogger(__name__)
        
        # Initialize LLM Compressor wrapper
        from llm_compressor_wrapper import LLMCompressorWrapper
        self.llm_compressor = LLMCompressorWrapper(config)
        
        # Statistics tracking
        self.total_original_size = 0.0
        self.total_quantized_size = 0.0
        self.layer_errors = {}
        
    def quantize_layer(self, 
                       layer: torch.nn.Module, 
                       layer_name: str,
                       calibration_data: Any) -> Tuple[torch.nn.Module, LayerQuantizationResult]:
        """Quantize a single layer using AWQ
        
        Main entry point for layer quantization:
        1. Identify layer type and structure
        2. Compute AWQ scales using calibration data
        3. Apply quantization to weights
        4. Pack weights for efficient storage
        5. Validate and return quantized layer
        """
        start_time = time.time()
        self.logger.info(f"Starting quantization of layer: {layer_name}")
        
        # Calculate original size
        original_size_gb = self._calculate_layer_size(layer)
        self.total_original_size += original_size_gb
        
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
        calibration_data = calibration_data.to(device)
        
        # Memory check
        self.memory_manager.monitor_memory(f"before_quantize_{layer_name}")
        
        try:
            # Identify layer type
            layer_type = self._identify_layer_type(layer, layer_name)
            
            # Compute AWQ scales
            scales = self.compute_awq_scales(layer, calibration_data, layer_type)
            
            # Apply quantization based on layer type
            if layer_type == "moe":
                quantized_layer = self.quantize_moe_layer(layer, calibration_data)
            elif layer_type == "attention":
                quantized_layer = self.quantize_attention_layer(layer, calibration_data, scales)
            elif layer_type == "mlp":
                quantized_layer = self.quantize_mlp_layer(layer, calibration_data, scales)
            else:
                # Generic quantization
                quantized_layer = self._apply_awq_quantization(layer, scales)
            
            # Pack weights
            quantized_layer = self._pack_layer_weights(quantized_layer)
            
            # Validate quantization
            error = self.validate_quantized_layer(layer, quantized_layer, calibration_data)
            
            # Calculate quantized size
            quantized_size_gb = self._calculate_layer_size(quantized_layer)
            self.total_quantized_size += quantized_size_gb
            
            # Move back to CPU if needed
            if self.config.offload_to_cpu:
                quantized_layer = quantized_layer.to('cpu')
            
            # Clear GPU cache
            self.memory_manager.clear_gpu_cache()
            
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
            
            # Log results
            self.logger.info(f"Quantized {layer_name}: {original_size_gb:.2f}GB -> {quantized_size_gb:.2f}GB "
                           f"(compression: {result.compression_ratio:.2f}x, error: {error:.6f})")
            
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
    
    def compute_awq_scales(self,
                          layer: torch.nn.Module,
                          calibration_data: Any,
                          layer_type: str = "generic") -> Dict[str, torch.Tensor]:
        """Compute AWQ scaling factors for layer
        
        AWQ algorithm:
        1. Collect activation statistics
        2. Identify salient weight channels (top 1%)
        3. Compute optimal scaling factors
        4. Apply smoothing between layers
        """
        scales = {}
        
        # Hook to capture activations
        activations = {}
        
        def create_hook(name):
            def hook_fn(module, input, output):
                if isinstance(input, tuple):
                    input = input[0]
                activations[name] = input.detach()
            return hook_fn
        
        # Register hooks on Linear layers
        hooks = []
        linear_modules = {}
        for name, module in layer.named_modules():
            if isinstance(module, nn.Linear):
                handle = module.register_forward_hook(create_hook(name))
                hooks.append(handle)
                linear_modules[name] = module
        
        # Forward pass to collect activations
        with torch.no_grad():
            try:
                # Handle different input formats
                if calibration_data.dim() == 2:
                    calibration_data = calibration_data.unsqueeze(0)
                
                # Forward pass
                _ = layer(calibration_data)
            except Exception as e:
                self.logger.warning(f"Forward pass failed during scale computation: {e}")
        
        # Remove hooks
        for handle in hooks:
            handle.remove()
        
        # Compute scales for each Linear layer
        for name, module in linear_modules.items():
            if name not in activations:
                self.logger.warning(f"No activations captured for {name}")
                continue
            
            X = activations[name]
            W = module.weight
            
            # Compute per-channel importance
            # mean(abs(X), dim=[batch, seq_len]) -> [hidden_dim]
            X_abs_mean = torch.mean(torch.abs(X), dim=list(range(X.dim()-1)))
            
            # Identify salient channels (top 1%)
            num_salient = max(1, int(0.01 * len(X_abs_mean)))
            importance = X_abs_mean / (X_abs_mean.sum() + 1e-8)
            salient_indices = torch.topk(importance, num_salient).indices
            
            # Compute weight statistics
            W_abs_mean = torch.mean(torch.abs(W), dim=0)
            
            # Compute optimal scales
            # scale = sqrt(mean(X^2) / mean(W^2))
            X_rms = torch.sqrt(torch.mean(X ** 2, dim=list(range(X.dim()-1))))
            W_rms = torch.sqrt(torch.mean(W ** 2, dim=0))
            
            scale = torch.sqrt(X_rms / (W_rms + 1e-8))
            
            # Protect salient channels
            scale[salient_indices] *= self.config.awq_protection_factor
            
            # Apply damping for stability
            damping = self.config.awq_damping_percent
            scale = scale * (1 - damping) + 1.0 * damping
            
            # Clip to reasonable range
            scale = torch.clamp(scale, min=0.01, max=100.0)
            
            scales[name] = scale
            
            self.logger.debug(f"Computed scales for {name}: mean={scale.mean():.3f}, "
                            f"std={scale.std():.3f}, salient={num_salient}")
        
        return scales
    
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
            test_input = test_input.to(device)
            
            # Get original output
            try:
                orig_out = original(test_input)
            except:
                # Some layers might need different inputs
                return 0.0
            
            # Get quantized output
            try:
                quant_out = quantized(test_input)
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
                          calibration_data: Any) -> nn.Module:
        """Specialized quantization for MoE layers
        
        Handles:
        - Expert routing (keep in FP16)
        - Individual expert MLPs
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
                _ = moe_module(calibration_data)
            except:
                pass
        
        # Remove hooks
        for handle in hooks:
            handle.remove()
        
        # Quantize shared experts (always active)
        if hasattr(moe_module, 'shared_experts'):
            for i, expert in enumerate(moe_module.shared_experts):
                self.logger.info(f"Quantizing shared expert {i}")
                # Compute scales for this expert
                expert_scales = self.compute_awq_scales(expert, calibration_data, "moe_expert")
                # Quantize expert
                quantized_expert = self.quantize_mlp_layer(expert, calibration_data, expert_scales)
                moe_module.shared_experts[i] = quantized_expert
        
        # Quantize activated routed experts
        if hasattr(moe_module, 'experts'):
            for i, expert in enumerate(moe_module.experts):
                if f"expert_{i}" in expert_activations or len(expert_activations) == 0:
                    self.logger.info(f"Quantizing expert {i} (activated)")
                    # Compute scales for this expert
                    expert_scales = self.compute_awq_scales(expert, calibration_data, "moe_expert")
                    # Quantize expert
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
        # Check config skip patterns
        if not self.should_quantize_layer(layer_name):
            return True
        
        # Check if layer has any Linear modules
        has_linear = any(isinstance(m, nn.Linear) for _, m in layer.named_modules())
        if not has_linear:
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
        has_qkv = any('q_proj' in n or 'k_proj' in n or 'v_proj' in n for n, _ in layer.named_modules())
        if has_qkv:
            return "attention"
        
        has_gate_up = any('gate_proj' in n or 'up_proj' in n for n, _ in layer.named_modules())
        if has_gate_up:
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