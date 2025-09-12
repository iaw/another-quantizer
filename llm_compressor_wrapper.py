# llm_compressor_wrapper.py
"""Wrapper for LLM Compressor AWQ functionality"""

from typing import Dict, Any, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from pathlib import Path
import logging
import numpy as np
import gc
import re  # FIXED: Added missing import

# LLM Compressor imports
try:
    from llmcompressor.modifiers.quantization import AWQModifier
    from llmcompressor.transformers import oneshot
    from llmcompressor.modifiers.quantization.utils import create_mapping
except ImportError:
    logging.warning("LLM Compressor not installed. Install with: pip install llmcompressor")
    AWQModifier = None
    oneshot = None
    create_mapping = None


@dataclass
class AWQConfig:
    """Configuration for AWQ quantization"""
    bits: int = 4
    group_size: int = 128
    symmetric: bool = False
    damping_percent: float = 0.01
    true_sequential: bool = True
    desc_act: bool = False  # Activation order
    clip_ratio: float = 1.0  # Weight clipping
    protection_factor: float = 1.5  # For salient channels
    
    
@dataclass
class GLMLayerMapping:
    """Mapping configuration for GLM layers"""
    input_pattern: str
    output_patterns: List[str]
    layer_type: str  # 'attention', 'mlp', 'moe'
    

class LLMCompressorWrapper:
    """Wrapper for LLM Compressor AWQ operations"""
    
    def __init__(self, config: Any):
        """Initialize LLM Compressor wrapper"""
        self.config = config
        self.awq_modifier = None
        self.layer_mappings = self._create_glm_mappings()
        self.logger = logging.getLogger(__name__)
        self.calibration_cache = {}
        self.awq_scales = {}  # Store computed scales
        
        # Create AWQ configuration
        self.awq_config = AWQConfig(
            bits=config.bits,
            group_size=config.group_size,
            symmetric=config.symmetric,
            damping_percent=config.awq_damping_percent,
            true_sequential=config.awq_true_sequential,
            desc_act=config.awq_desc_act,
            protection_factor=config.awq_protection_factor,
        )
        
    def create_awq_modifier(self) -> AWQModifier:
        """Create AWQ modifier with GLM-specific settings"""
        if AWQModifier is None:
            raise ImportError("LLM Compressor not installed")
        
        # Get GLM-specific mappings
        mappings = self.get_glm_mappings()
        
        # Create AWQ modifier
        self.awq_modifier = AWQModifier(
            mappings=mappings,
            bits=self.awq_config.bits,
            group_size=self.awq_config.group_size,
            symmetric=self.awq_config.symmetric,
            damping_percent=self.awq_config.damping_percent,
            targets=["Linear"],  # Target Linear layers
            scheme="W4A16_ASYM" if self.awq_config.bits == 4 else "W8A16",
            ignore=self.config.skip_layers,  # Skip patterns from config
        )
        
        self.logger.info(f"Created AWQ modifier with {len(mappings)} layer mappings")
        
        return self.awq_modifier
    
    def get_glm_mappings(self) -> List[List[str]]:
        """Get GLM-specific layer mappings for AWQ"""
        # GLM4.5 layer connections for AWQ scale propagation
        mappings = [
            # Attention input scaling
            ["re:.*input_layernorm", 
             ["re:.*q_proj", "re:.*k_proj", "re:.*v_proj"]],
            
            # Attention output scaling
            ["re:.*v_proj", 
             ["re:.*o_proj"]],
            
            # MLP input scaling
            ["re:.*post_attention_layernorm",
             ["re:.*gate_proj", "re:.*up_proj"]],
            
            # MLP output scaling
            ["re:.*up_proj", 
             ["re:.*down_proj"]],
            
            # MoE expert scaling (if applicable)
            ["re:.*expert.*gate",
             ["re:.*expert.*up"]],
            
            ["re:.*expert.*up",
             ["re:.*expert.*down"]],
            
            # Shared expert scaling
            ["re:.*shared_expert.*gate",
             ["re:.*shared_expert.*up"]],
            
            ["re:.*shared_expert.*up",
             ["re:.*shared_expert.*down"]],
        ]
        
        return mappings
    
    def _create_glm_mappings(self) -> List[GLMLayerMapping]:
        """Create GLM layer mappings (internal)"""
        # Return empty list as this is just internal bookkeeping
        return []
    
    def quantize_layer_with_awq(self,
                               layer: torch.nn.Module,
                               layer_name: str,
                               calibration_inputs: torch.Tensor) -> torch.nn.Module:
        """Quantize single layer using AWQ"""
        self.logger.info(f"Quantizing layer: {layer_name}")
        
        # Move layer to GPU if available and possible
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        layer = layer.to(device)
        calibration_inputs = calibration_inputs.to(device)
        
        # Compute AWQ scales
        scales = self.compute_scaling_factors(layer, calibration_inputs)
        
        # Apply scales to weights
        self._apply_scales_to_layer(layer, scales)
        
        # Quantize weights
        quantized_layer = self._quantize_layer_weights(layer, scales)
        
        # Move back to CPU if needed
        if self.config.offload_to_cpu:
            quantized_layer = quantized_layer.to('cpu')
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return quantized_layer
    
    def compute_scaling_factors(self,
                               layer: torch.nn.Module,
                               inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute AWQ scaling factors"""
        scales = {}
        
        # Hook to capture activations
        activations = {}
        
        def hook_fn(name):
            def hook(module, input, output):
                # Store input activations
                if isinstance(input, tuple):
                    input = input[0]
                activations[name] = input.detach()
            return hook
        
        # Register hooks on Linear layers
        hooks = []
        for name, module in layer.named_modules():
            if isinstance(module, nn.Linear):
                handle = module.register_forward_hook(hook_fn(name))
                hooks.append(handle)
        
        # Forward pass to collect activations
        with torch.no_grad():
            # Handle different input shapes
            if inputs.dim() == 2:
                # Add batch dimension if needed
                inputs = inputs.unsqueeze(0)
            
            # Forward pass
            try:
                _ = layer(inputs)
            except:
                # Some layers might need different forward signature
                pass
        
        # Remove hooks
        for handle in hooks:
            handle.remove()
        
        # Compute scales for each Linear layer
        for name, module in layer.named_modules():
            if isinstance(module, nn.Linear) and name in activations:
                X = activations[name]
                W = module.weight
                
                # Compute importance based on activation magnitude
                # Shape: [batch, seq_len, hidden] -> [hidden]
                X_mean = torch.mean(torch.abs(X), dim=list(range(X.dim()-1)))
                
                # Identify salient channels (top 1%)
                k = max(1, int(0.01 * len(X_mean)))
                importance = X_mean / (X_mean.sum() + 1e-8)
                salient_indices = torch.topk(importance, k).indices
                
                # Compute scales
                W_mean = torch.mean(torch.abs(W), dim=0)
                scale = torch.sqrt(X_mean / (W_mean + 1e-8))
                
                # Protect salient channels
                scale[salient_indices] *= self.awq_config.protection_factor
                
                # Apply damping
                damping = self.awq_config.damping_percent
                scale = scale * (1 - damping) + 1.0 * damping
                
                # Clip scales to reasonable range
                scale = torch.clamp(scale, min=0.1, max=10.0)
                
                scales[name] = scale
        
        return scales
    
    def apply_quantization(self,
                         weights: torch.Tensor,
                         scales: torch.Tensor,
                         group_size: int = 128) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Apply quantization with scaling factors"""
        # Apply AWQ scales
        W_scaled = weights * scales.unsqueeze(0)
        
        # Prepare for group-wise quantization
        orig_shape = W_scaled.shape
        W_scaled = W_scaled.reshape(-1, group_size)
        
        # Quantize each group
        num_groups = W_scaled.shape[0]
        bits = self.awq_config.bits
        max_val = 2**bits - 1
        
        group_scales = torch.zeros(num_groups, dtype=torch.float16, device=W_scaled.device)
        group_zeros = torch.zeros(num_groups, dtype=torch.float16, device=W_scaled.device)
        W_quant = torch.zeros_like(W_scaled, dtype=torch.uint8)
        
        for i in range(num_groups):
            group = W_scaled[i]
            
            # Find min and max
            min_val = group.min().item()
            max_val_group = group.max().item()
            
            # Compute scale and zero point
            scale = (max_val_group - min_val) / max_val if max_val_group != min_val else 1.0
            zero_point = round(-min_val / scale) if scale != 0 else 0
            
            # Quantize
            group_quant = torch.round(group / scale + zero_point)
            group_quant = torch.clamp(group_quant, 0, max_val)
            
            W_quant[i] = group_quant.to(torch.uint8)
            group_scales[i] = scale
            group_zeros[i] = zero_point
        
        # Pack weights if INT4
        if bits == 4:
            W_packed = self.pack_weights(W_quant.reshape(orig_shape), bits=4)
        else:
            W_packed = W_quant.reshape(orig_shape)
        
        metadata = {
            "scales": group_scales,
            "zeros": group_zeros,
            "group_size": group_size,
            "bits": bits,
            "shape": orig_shape,
        }
        
        return W_packed, metadata
    
    def pack_weights(self, 
                     quantized_weights: torch.Tensor,
                     bits: int = 4) -> torch.Tensor:
        """Pack quantized weights for storage"""
        if bits == 4:
            # Pack 2 INT4 values per byte
            orig_shape = quantized_weights.shape
            quantized_weights = quantized_weights.reshape(-1)
            
            # Ensure even number of elements
            if len(quantized_weights) % 2 != 0:
                quantized_weights = torch.cat([
                    quantized_weights, 
                    torch.zeros(1, dtype=quantized_weights.dtype, device=quantized_weights.device)
                ])
            
            # Pack pairs
            packed = torch.zeros(len(quantized_weights) // 2, dtype=torch.uint8, device=quantized_weights.device)
            packed = (quantized_weights[::2] << 4) | quantized_weights[1::2]
            
            # Reshape to maintain dimensions (except last which is halved)
            packed_shape = list(orig_shape)
            packed_shape[-1] = packed_shape[-1] // 2
            packed = packed.reshape(packed_shape)
            
            return packed
        
        elif bits == 8:
            return quantized_weights.to(torch.int8)
        
        else:
            raise ValueError(f"Unsupported bit width: {bits}")
    
    def create_layer_config(self, layer_name: str) -> Dict[str, Any]:
        """Create quantization config for specific layer"""
        # Check if layer should be quantized
        for skip_pattern in self.config.skip_layers:
            if skip_pattern.replace('*', '') in layer_name:
                return None  # Skip this layer
        
        # Different configs for different layer types
        if 'attention' in layer_name or 'q_proj' in layer_name or 'k_proj' in layer_name:
            # Attention layers: preserve accuracy
            return {
                "bits": self.awq_config.bits,
                "group_size": self.awq_config.group_size,
                "symmetric": False,
            }
        elif 'mlp' in layer_name or 'gate_proj' in layer_name or 'up_proj' in layer_name:
            # MLP layers: can be more aggressive
            return {
                "bits": self.awq_config.bits,
                "group_size": self.awq_config.group_size * 2,  # Larger groups for MLP
                "symmetric": False,
            }
        elif 'expert' in layer_name:
            # MoE experts: standard quantization
            return {
                "bits": self.awq_config.bits,
                "group_size": self.awq_config.group_size,
                "symmetric": False,
            }
        else:
            # Default config
            return {
                "bits": self.awq_config.bits,
                "group_size": self.awq_config.group_size,
                "symmetric": self.awq_config.symmetric,
            }
    
    def should_skip_layer(self, layer_name: str) -> bool:
        """Check if layer should be skipped"""
        skip_patterns = [
            "lm_head",
            "embed",
            "router",  # MoE routers
            "layernorm",
            "ln_f",
            "norm",
        ]
        
        for pattern in skip_patterns:
            if pattern in layer_name.lower():
                return True
        
        # Check config skip patterns
        for pattern in self.config.skip_layers:
            pattern = pattern.replace('*', '.*')
            if re.match(pattern, layer_name):
                return True
        
        return False
    
    def prepare_oneshot_config(self,
                              model: nn.Module,
                              calibration_data: Any) -> Dict[str, Any]:
        """Prepare configuration for oneshot quantization"""
        if AWQModifier is None:
            raise ImportError("LLM Compressor not installed")
        
        # Create AWQ modifier
        awq_modifier = self.create_awq_modifier()
        
        # Create recipe
        recipe = [awq_modifier]
        
        # Configuration dictionary
        config = {
            "recipe": recipe,
            "dataset": calibration_data,
            "max_seq_length": self.config.max_position_embeddings,
            "num_calibration_samples": self.config.calibration_samples,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        }
        
        return config
    
    def run_oneshot_quantization(self,
                                model: nn.Module,
                                config: Dict[str, Any]) -> nn.Module:
        """Run oneshot quantization on entire model"""
        if oneshot is None:
            raise ImportError("LLM Compressor oneshot not available")
        
        # Run oneshot quantization
        quantized_model = oneshot(
            model=model,
            dataset=config["dataset"],
            recipe=config["recipe"],
            max_seq_length=config["max_seq_length"],
            num_calibration_samples=config["num_calibration_samples"],
            device=config["device"],
        )
        
        return quantized_model
    
    def extract_quantized_state(self,
                               layer: nn.Module) -> Dict[str, torch.Tensor]:
        """Extract quantized weights and metadata"""
        state = {}
        
        for name, module in layer.named_modules():
            if isinstance(module, nn.Linear):
                # Check if module has been quantized
                if hasattr(module, 'weight_packed'):
                    state[f"{name}.weight_packed"] = module.weight_packed
                    state[f"{name}.weight_scale"] = module.weight_scale
                    state[f"{name}.weight_zp"] = module.weight_zp
                    state[f"{name}.group_size"] = torch.tensor(module.group_size)
                    state[f"{name}.bits"] = torch.tensor(module.bits)
                else:
                    # Unquantized weight
                    state[f"{name}.weight"] = module.weight
                    if module.bias is not None:
                        state[f"{name}.bias"] = module.bias
        
        return state
    
    def create_quantized_linear(self,
                               in_features: int,
                               out_features: int,
                               quantized_state: Dict[str, torch.Tensor]) -> nn.Module:
        """Create custom quantized Linear module"""
        
        class QuantizedLinear(nn.Module):
            """Custom quantized linear layer"""
            
            def __init__(self, in_features, out_features, weight_packed, scales, zeros, group_size, bits):
                super().__init__()
                self.in_features = in_features
                self.out_features = out_features
                self.group_size = group_size
                self.bits = bits
                
                # Store quantized weights and metadata
                self.register_buffer('weight_packed', weight_packed)
                self.register_buffer('scales', scales)
                self.register_buffer('zeros', zeros)
            
            def forward(self, x):
                # Dequantize weights on the fly
                weight = self.dequantize_weights()
                return F.linear(x, weight, None)
            
            def dequantize_weights(self):
                """Dequantize weights for computation"""
                # Unpack if INT4
                if self.bits == 4:
                    weight = self.unpack_int4(self.weight_packed)
                else:
                    weight = self.weight_packed.float()
                
                # Apply scales and zero points
                weight = weight.reshape(-1, self.group_size)
                weight = (weight - self.zeros.unsqueeze(1)) * self.scales.unsqueeze(1)
                weight = weight.reshape(self.out_features, self.in_features)
                
                return weight
            
            def unpack_int4(self, packed):
                """Unpack INT4 weights"""
                # Unpack 2 values per byte
                high = (packed >> 4) & 0xF
                low = packed & 0xF
                
                unpacked = torch.zeros(packed.shape[0], packed.shape[1] * 2, dtype=torch.uint8, device=packed.device)
                unpacked[:, ::2] = high
                unpacked[:, 1::2] = low
                
                return unpacked
        
        # Extract quantized state
        weight_packed = quantized_state.get('weight_packed')
        scales = quantized_state.get('scales')
        zeros = quantized_state.get('zeros')
        group_size = quantized_state.get('group_size', 128)
        bits = quantized_state.get('bits', 4)
        
        return QuantizedLinear(in_features, out_features, weight_packed, scales, zeros, group_size, bits)
    
    def validate_quantization(self,
                            original: torch.Tensor,
                            quantized: torch.Tensor,
                            scales: torch.Tensor,
                            zero_points: torch.Tensor) -> Dict[str, float]:
        """Validate quantization quality"""
        # Dequantize for comparison
        if self.awq_config.bits == 4:
            quantized = self.unpack_int4(quantized)
        
        dequantized = (quantized.float() - zero_points) * scales
        
        # Calculate metrics
        mse = torch.mean((original - dequantized) ** 2).item()
        relative_error = mse / (torch.mean(original ** 2).item() + 1e-8)
        
        # Check value ranges
        max_val = 2**self.awq_config.bits - 1
        in_range = (quantized >= 0) & (quantized <= max_val)
        range_compliance = in_range.float().mean().item()
        
        return {
            "mse": mse,
            "relative_error": relative_error,
            "range_compliance": range_compliance,
        }
    
    def optimize_group_size(self,
                          layer: nn.Module,
                          calibration_data: Any) -> int:
        """Find optimal group size for layer"""
        group_sizes = [32, 64, 128, 256]
        best_group_size = 128
        best_error = float('inf')
        
        for group_size in group_sizes:
            # Try quantization with this group size
            self.awq_config.group_size = group_size
            
            # Compute scales
            scales = self.compute_scaling_factors(layer, calibration_data)
            
            # Test quantization on a sample weight
            for name, module in layer.named_modules():
                if isinstance(module, nn.Linear):
                    weight = module.weight
                    
                    # Apply quantization
                    quantized, metadata = self.apply_quantization(weight, scales.get(name, torch.ones_like(weight[0])), group_size)
                    
                    # Validate
                    metrics = self.validate_quantization(
                        weight, 
                        quantized, 
                        metadata['scales'], 
                        metadata['zeros']
                    )
                    
                    if metrics['relative_error'] < best_error:
                        best_error = metrics['relative_error']
                        best_group_size = group_size
                    
                    break  # Just test one weight
        
        self.logger.info(f"Optimal group size: {best_group_size} (error: {best_error:.6f})")
        return best_group_size
    
    def handle_moe_quantization(self,
                              moe_layer: nn.Module,
                              calibration_data: Any) -> nn.Module:
        """Special handling for MoE layers"""
        # Keep router in FP16
        # Router should not be quantized for accuracy
        
        # Track which experts were activated during calibration
        activated_experts = set()
        
        def track_expert_activation(module, input, output):
            # Track which experts are selected
            if hasattr(module, 'expert_indices'):
                activated_experts.update(module.expert_indices.cpu().numpy().tolist())
        
        # Register hook on router
        if hasattr(moe_layer, 'router'):
            hook = moe_layer.router.register_forward_hook(track_expert_activation)
            
            # Run forward pass to see which experts are used
            with torch.no_grad():
                _ = moe_layer(calibration_data)
            
            hook.remove()
        
        # Quantize shared experts (always active)
        if hasattr(moe_layer, 'shared_experts'):
            for i, expert in enumerate(moe_layer.shared_experts):
                self.logger.info(f"Quantizing shared expert {i}")
                quantized_expert = self.quantize_layer_with_awq(expert, f"shared_expert_{i}", calibration_data)
                moe_layer.shared_experts[i] = quantized_expert
        
        # Quantize only activated routed experts
        if hasattr(moe_layer, 'experts'):
            for i, expert in enumerate(moe_layer.experts):
                if i in activated_experts or len(activated_experts) == 0:
                    self.logger.info(f"Quantizing expert {i} (activated)")
                    quantized_expert = self.quantize_layer_with_awq(expert, f"expert_{i}", calibration_data)
                    moe_layer.experts[i] = quantized_expert
                else:
                    self.logger.info(f"Skipping expert {i} (not activated)")
                    # Could remove or keep in FP16
        
        return moe_layer
    
    def create_vllm_config(self,
                         quantized_model: nn.Module) -> Dict[str, Any]:
        """Create vLLM-compatible configuration"""
        config = {
            "quantization": "awq",
            "weight_bits": self.awq_config.bits,
            "group_size": self.awq_config.group_size,
            "symmetric": self.awq_config.symmetric,
            "version": "GEMM",
            "kernel": "awq_gemm",
            "zero_point": not self.awq_config.symmetric,
        }
        
        return config
    
    def benchmark_quantized_layer(self,
                                 layer: nn.Module,
                                 test_input: torch.Tensor,
                                 num_runs: int = 100) -> Dict[str, float]:
        """Benchmark quantized layer performance"""
        import time
        
        # Warmup runs
        for _ in range(10):
            with torch.no_grad():
                _ = layer(test_input)
        
        # Time forward passes
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        
        for _ in range(num_runs):
            with torch.no_grad():
                _ = layer(test_input)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()
        
        # Measure memory
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / 1e9
        else:
            memory_used = 0
        
        # Calculate throughput
        total_time = end_time - start_time
        avg_time = total_time / num_runs
        throughput = 1 / avg_time
        
        return {
            "avg_latency_ms": avg_time * 1000,
            "throughput_fps": throughput,
            "memory_gb": memory_used,
        }
    
    # Helper methods
    
    def _apply_scales_to_layer(self, layer: nn.Module, scales: Dict[str, torch.Tensor]) -> None:
        """Apply AWQ scales to layer weights"""
        for name, module in layer.named_modules():
            if isinstance(module, nn.Linear) and name in scales:
                scale = scales[name]
                # Apply scale to weights
                module.weight.data *= scale.unsqueeze(0)
    
    def _quantize_layer_weights(self, layer: nn.Module, scales: Dict[str, torch.Tensor]) -> nn.Module:
        """Quantize all Linear layers in the module"""
        for name, module in layer.named_modules():
            if isinstance(module, nn.Linear):
                # Skip if should not be quantized
                if self.should_skip_layer(name):
                    continue
                
                # Get scale for this layer
                scale = scales.get(name, torch.ones(module.weight.shape[1], device=module.weight.device))
                
                # Quantize weights
                quantized_weight, metadata = self.apply_quantization(
                    module.weight.data,
                    scale,
                    self.awq_config.group_size
                )
                
                # Replace with quantized version
                # Store quantized weights and metadata in module
                module.weight_packed = quantized_weight
                module.weight_scale = metadata['scales']
                module.weight_zp = metadata['zeros']
                module.group_size = metadata['group_size']
                module.bits = metadata['bits']
                
                # Mark as quantized
                module.quantized = True
        
        return layer
    
    def unpack_int4(self, packed: torch.Tensor) -> torch.Tensor:
        """Unpack INT4 weights"""
        # Extract high and low nibbles
        high = (packed >> 4) & 0xF
        low = packed & 0xF
        
        # Create unpacked tensor
        unpacked_shape = list(packed.shape)
        unpacked_shape[-1] = unpacked_shape[-1] * 2
        unpacked = torch.zeros(unpacked_shape, dtype=torch.uint8, device=packed.device)
        
        # Interleave values
        unpacked[..., ::2] = high
        unpacked[..., 1::2] = low
        
        return unpacked