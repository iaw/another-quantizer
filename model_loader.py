# model_loader.py
"""Sequential model loading and layer iteration"""

import torch
import torch.nn as nn
from typing import Iterator, Tuple, Dict, Any, Optional, List, Union
from pathlib import Path
import safetensors.torch
import json
import gc
from dataclasses import dataclass
import logging
import re
from collections import OrderedDict


@dataclass
class LayerInfo:
    """Information about a model layer"""
    name: str
    layer_type: str  # 'embedding', 'transformer', 'moe', 'lm_head'
    layer_index: Optional[int]
    weight_files: List[str]  # Which files contain this layer's weights
    estimated_size_gb: float
    sub_modules: List[str]  # e.g., ['attention.q_proj', 'attention.k_proj']
    is_moe: bool = False  # Flag for MoE layers
    

class SequentialModelLoader:
    """Load and iterate through model layers sequentially"""
    
    def __init__(self, model_path: str, memory_manager: Any):
        """Initialize model loader"""
        self.model_path = Path(model_path)
        self.memory_manager = memory_manager
        self.weight_map = {}  # Maps tensor names to files
        self.layer_names = []
        self.layer_info = {}  # Dict[str, LayerInfo]
        self.config = None
        self.weight_files = []  # List of all weight files
        self.current_loaded_files = {}  # Track currently loaded files
        self.logger = logging.getLogger(__name__)
        self.dtype = torch.float16  # Default dtype
        
        # Initialize by loading model configuration
        self.load_model_config()
        self.map_model_weights()
        self.build_layer_info()
        
    def load_model_config(self) -> Dict[str, Any]:
        """Load only the model configuration"""
        config_path = self.model_path / "config.json"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Model config not found: {config_path}")
        
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Extract key configuration
        self.num_layers = self.config.get('num_hidden_layers', self.config.get('num_layers'))
        self.hidden_size = self.config.get('hidden_size')
        self.num_attention_heads = self.config.get('num_attention_heads')
        self.intermediate_size = self.config.get('intermediate_size', self.config.get('ffn_hidden_size'))
        self.vocab_size = self.config.get('vocab_size', self.config.get('padded_vocab_size'))
        
        # MoE configuration for GLM4.5
        self.num_experts = self.config.get('num_experts', None)
        self.num_shared_experts = self.config.get('num_shared_experts', None)
        self.top_k_experts = self.config.get('num_experts_per_tok', 8)  # GLM4.5 uses top-8
        
        # Set dtype based on config
        torch_dtype = self.config.get('torch_dtype', 'float16')
        if torch_dtype == 'float16':
            self.dtype = torch.float16
        elif torch_dtype == 'bfloat16':
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float32
        
        self.logger.info(f"Loaded model config: {self.num_layers} layers, "
                        f"hidden_size={self.hidden_size}, "
                        f"num_experts={self.num_experts}")
        
        return self.config
    
    def map_model_weights(self) -> Dict[str, str]:
        """Create mapping of layer names to weight files"""
        # Check for index files (sharded models)
        safetensors_index = self.model_path / "model.safetensors.index.json"
        pytorch_index = self.model_path / "pytorch_model.bin.index.json"
        
        if safetensors_index.exists():
            # Safetensors format (preferred)
            with open(safetensors_index, 'r') as f:
                index_data = json.load(f)
            
            self.weight_map = index_data.get('weight_map', {})
            self.weight_files = list(set(self.weight_map.values()))
            self.use_safetensors = True
            
            self.logger.info(f"Found safetensors model with {len(self.weight_files)} shards")
            
        elif pytorch_index.exists():
            # PyTorch format
            with open(pytorch_index, 'r') as f:
                index_data = json.load(f)
            
            self.weight_map = index_data.get('weight_map', {})
            self.weight_files = list(set(self.weight_map.values()))
            self.use_safetensors = False
            
            self.logger.info(f"Found PyTorch model with {len(self.weight_files)} shards")
            
        else:
            # Single file model
            safetensors_file = self.model_path / "model.safetensors"
            pytorch_file = self.model_path / "pytorch_model.bin"
            
            if safetensors_file.exists():
                self.weight_files = ["model.safetensors"]
                self.use_safetensors = True
                # Build weight map by loading metadata
                with safetensors.safe_open(str(safetensors_file), framework="pt", device="cpu") as f:
                    for key in f.keys():
                        self.weight_map[key] = "model.safetensors"
            elif pytorch_file.exists():
                self.weight_files = ["pytorch_model.bin"]
                self.use_safetensors = False
                # Would need to load to get keys, skip for now
                self.logger.warning("Single PyTorch file model - weight mapping may be incomplete")
            else:
                raise FileNotFoundError(f"No model weights found in {self.model_path}")
        
        return self.weight_map
    
    def build_layer_info(self) -> Dict[str, LayerInfo]:
        """Build comprehensive layer information"""
        self.layer_info = {}
        
        # Group weights by layer
        layer_weights = {}
        for weight_name, weight_file in self.weight_map.items():
            # Parse layer name from weight name
            layer_name = self._extract_layer_name(weight_name)
            
            if layer_name not in layer_weights:
                layer_weights[layer_name] = {
                    'weights': [],
                    'files': set()
                }
            
            layer_weights[layer_name]['weights'].append(weight_name)
            layer_weights[layer_name]['files'].add(weight_file)
        
        # Create LayerInfo for each layer
        for layer_name, info in layer_weights.items():
            layer_type = self._determine_layer_type(layer_name)
            layer_index = self._extract_layer_index(layer_name)
            
            # Check if it's an MoE layer component
            is_moe = 'experts' in layer_name or 'shared_expert' in layer_name
            
            # For MoE experts, mark them specially
            if 'mlp.experts.' in layer_name:
                # This is a single expert
                expert_num = int(re.search(r'experts\.(\d+)', layer_name).group(1))
                layer_type = 'moe_expert'
            elif 'shared_expert' in layer_name:
                layer_type = 'shared_expert'
            
            # Estimate layer size
            size_gb = self._estimate_layer_size(layer_name, info['weights'])
            
            # Extract sub-modules
            sub_modules = self._extract_sub_modules(info['weights'], layer_name)
            
            self.layer_info[layer_name] = LayerInfo(
                name=layer_name,
                layer_type=layer_type,
                layer_index=layer_index,
                weight_files=list(info['files']),
                estimated_size_gb=size_gb,
                sub_modules=sub_modules,
                is_moe=is_moe
            )
        
        # Sort layers by index for sequential processing
        self.layer_names = self._sort_layers(list(self.layer_info.keys()))
        
        self.logger.info(f"Built layer info for {len(self.layer_info)} layers")
        
        return self.layer_info
    
    def iterate_layers(self) -> Iterator[Tuple[str, nn.Module]]:
        """Iterate through model layers one at a time
        
        Returns proper nn.Module objects for each layer
        """
        self.logger.info("Starting sequential layer iteration")
        
        for layer_name in self.layer_names:
            try:
                # Memory check before loading
                self.memory_manager.monitor_memory(f"before_{layer_name}")
                
                # Load layer from disk (returns nn.Module)
                layer_module = self.load_layer(layer_name)
                
                # Yield to caller for processing
                self.logger.info(f"Yielding layer: {layer_name}")
                yield (layer_name, layer_module)
                
                # Cleanup after processing
                self.cleanup_layer(layer_module)
                self.memory_manager.clear_gpu_cache()
                
                # Memory check after cleanup
                self.memory_manager.monitor_memory(f"after_{layer_name}")
                
            except Exception as e:
                self.logger.error(f"Error loading layer {layer_name}: {e}")
                raise
    
    def load_layer(self, layer_name: str) -> nn.Module:
        """Load a single layer from disk with validation
        
        Returns proper nn.Module constructed from weights
        """
        if layer_name not in self.layer_info:
            self.logger.error(f"Layer {layer_name} not found in layer_info")
            # Try to find similar layer name
            similar = [name for name in self.layer_info.keys() if layer_name in name or name in layer_name]
            if similar:
                self.logger.info(f"Similar layers found: {similar}")
                if len(similar) == 1:
                    layer_name = similar[0]
                    self.logger.info(f"Using similar layer: {layer_name}")
                else:
                    raise KeyError(f"Layer {layer_name} not found. Similar: {similar}")
            else:
                raise KeyError(f"Layer {layer_name} not found")
        
        layer_info = self.layer_info[layer_name]
        self.logger.info(f"Loading layer: {layer_name} (type: {layer_info.layer_type}, "
                        f"size: {layer_info.estimated_size_gb:.2f}GB)")
        
        # Validate layer info
        if not layer_info.weight_files:
            raise ValueError(f"No weight files specified for {layer_name}")
        
        # Load required weight files with validation
        weights = {}
        missing_files = []
        for weight_file in layer_info.weight_files:
            file_path = self.model_path / weight_file
            
            if not file_path.exists():
                self.logger.warning(f"Weight file not found: {file_path}")
                continue
            
            # Load only the weights we need for this layer
            layer_weight_names = [w for w, f in self.weight_map.items() 
                                 if f == weight_file and self._extract_layer_name(w) == layer_name]
            
            if not layer_weight_names:
                self.logger.debug(f"No weights to load from {weight_file} for layer {layer_name}")
                continue
            
            try:
                if self.use_safetensors:
                    loaded_weights = self.load_safetensors_partial(file_path, layer_weight_names)
                else:
                    loaded_weights = self.load_pytorch_partial(file_path, layer_weight_names)
                
                weights.update(loaded_weights)
            except Exception as e:
                self.logger.error(f"Error loading weights from {file_path}: {e}")
                continue
        
        if not weights:
            self.logger.warning(f"No weights loaded for layer {layer_name}")
            raise ValueError(f"No weights found for layer {layer_name}")
        
        # Construct appropriate module based on layer type
        if layer_info.layer_type == 'embedding':
            return self._construct_embedding_layer(weights)
        elif layer_info.layer_type == 'lm_head':
            return self._construct_output_layer(weights)
        elif layer_info.layer_type == 'layernorm':
            return self._construct_layernorm(weights)
        elif layer_info.is_moe:
            return self.construct_moe_layer(layer_info.layer_index, weights)
        else:
            return self.construct_transformer_layer(layer_info.layer_index, weights)
    
    def load_safetensors_partial(self, 
                                 file_path: Path,
                                 tensor_names: List[str]) -> Dict[str, torch.Tensor]:
        """Load specific tensors from safetensors file with error handling"""
        tensors = {}
        
        if not file_path.exists():
            self.logger.warning(f"Safetensors file not found: {file_path}")
            return tensors
        
        try:
            with safetensors.safe_open(str(file_path), framework="pt", device="cpu") as f:
                available_keys = f.keys()
                for name in tensor_names:
                    if name in available_keys:
                        tensors[name] = f.get_tensor(name)
                    else:
                        self.logger.debug(f"Tensor not found in file: {name}")
        except Exception as e:
            self.logger.error(f"Error loading safetensors file {file_path}: {e}")
        
        return tensors
    
    def load_pytorch_partial(self,
                           file_path: Path,
                           tensor_names: List[str]) -> Dict[str, torch.Tensor]:
        """Load specific tensors from PyTorch .bin file with error handling"""
        tensors = {}
        
        if not file_path.exists():
            self.logger.warning(f"PyTorch file not found: {file_path}")
            return tensors
        
        try:
            # Load entire file (less efficient than safetensors)
            state_dict = torch.load(file_path, map_location='cpu')
            
            # Filter to requested tensors
            for name in tensor_names:
                if name in state_dict:
                    tensors[name] = state_dict[name]
                else:
                    self.logger.debug(f"Tensor not found in file: {name}")
            
            # Clean up the full state dict
            del state_dict
            gc.collect()
            
        except Exception as e:
            self.logger.error(f"Error loading PyTorch file {file_path}: {e}")
        
        return tensors
    
    def _validate_and_reshape_input(self, hidden_states: torch.Tensor, hidden_size: int) -> torch.Tensor:
        """Validate and reshape input tensor for layer processing
        
        Args:
            hidden_states: Input tensor
            hidden_size: Expected hidden dimension
            
        Returns:
            Properly shaped tensor [batch, seq_len, hidden_size]
        """
        # Handle different input shapes
        if hidden_states.dim() == 1:
            # [hidden_size] -> [1, 1, hidden_size]
            if hidden_states.shape[0] == hidden_size:
                hidden_states = hidden_states.unsqueeze(0).unsqueeze(0)
            else:
                raise ValueError(f"1D input shape {hidden_states.shape} doesn't match hidden_size {hidden_size}")
                
        elif hidden_states.dim() == 2:
            if hidden_states.shape[-1] == hidden_size:
                # [seq_len, hidden_size] -> [1, seq_len, hidden_size]
                hidden_states = hidden_states.unsqueeze(0)
            elif hidden_states.shape[0] == hidden_size:
                # [hidden_size, seq_len] -> [1, seq_len, hidden_size] (transpose)
                hidden_states = hidden_states.t().unsqueeze(0)
            else:
                # Assume [batch, hidden_size] -> [batch, 1, hidden_size]
                if hidden_states.shape[1] == hidden_size:
                    hidden_states = hidden_states.unsqueeze(1)
                else:
                    raise ValueError(f"2D input shape {hidden_states.shape} incompatible with hidden_size {hidden_size}")
                    
        elif hidden_states.dim() == 3:
            # Already correct shape [batch, seq_len, hidden_size]
            if hidden_states.shape[-1] != hidden_size:
                raise ValueError(f"Hidden size mismatch: expected {hidden_size}, got {hidden_states.shape[-1]}")
        else:
            raise ValueError(f"Unexpected input dimensions: {hidden_states.dim()}D, shape {hidden_states.shape}")
        
        # Final validation
        if hidden_states.shape[-1] != hidden_size:
            raise ValueError(f"Final hidden size mismatch: expected {hidden_size}, got {hidden_states.shape[-1]}")
        
        return hidden_states
    
    def construct_transformer_layer(self,
                               layer_idx: int,
                               weights: Dict[str, torch.Tensor]) -> nn.Module:
        """Construct a transformer layer from weights
        
        Returns a proper nn.Module with attention and MLP components
        """
        
        class TransformerLayer(nn.Module):
            """GLM Transformer layer with GQA support"""
            
            def __init__(self, config: Dict[str, Any]):
                super().__init__()
                self.hidden_size = config['hidden_size']
                self.intermediate_size = config.get('intermediate_size', 10944)
                self.num_heads = config['num_attention_heads']
                self.num_kv_heads = config.get('num_key_value_heads', self.num_heads)
                self.head_dim = config.get('head_dim', 128)  # Use explicit head_dim from config
                
                # For GLM-4, the attention projections have different sizes
                self.q_size = self.num_heads * self.head_dim  # 96 * 128 = 12,288
                self.kv_size = self.num_kv_heads * self.head_dim  # 8 * 128 = 1,024
                
                # Attention components
                self.input_layernorm = nn.LayerNorm(self.hidden_size, eps=1e-5)
                self.attention = nn.ModuleDict({
                    'q_proj': nn.Linear(self.hidden_size, self.q_size, bias=False),
                    'k_proj': nn.Linear(self.hidden_size, self.kv_size, bias=False),
                    'v_proj': nn.Linear(self.hidden_size, self.kv_size, bias=False),
                    'o_proj': nn.Linear(self.q_size, self.hidden_size, bias=False),
                })
                
                # MLP components
                self.post_attention_layernorm = nn.LayerNorm(self.hidden_size, eps=1e-5)
                self.mlp = nn.ModuleDict({
                    'gate_proj': nn.Linear(self.hidden_size, self.intermediate_size, bias=False),
                    'up_proj': nn.Linear(self.hidden_size, self.intermediate_size, bias=False),
                    'down_proj': nn.Linear(self.intermediate_size, self.hidden_size, bias=False),
                })
            
            @property
            def device(self):
                """Get device of the layer"""
                return next(self.parameters()).device
            
            def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
                """Forward pass with GQA attention"""
                batch_size, seq_len, _ = hidden_states.shape
                residual = hidden_states
                
                # Input layer norm
                normed = self.input_layernorm(hidden_states)
                
                # Compute Q, K, V projections with different sizes
                q = self.attention['q_proj'](normed)  # [batch, seq, 12288]
                k = self.attention['k_proj'](normed)  # [batch, seq, 1024]
                v = self.attention['v_proj'](normed)  # [batch, seq, 1024]
                
                # Reshape for attention computation
                q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
                k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
                v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
                
                # Repeat K,V heads to match Q heads for GQA
                if self.num_kv_heads < self.num_heads:
                    repeat_factor = self.num_heads // self.num_kv_heads
                    k = k.repeat_interleave(repeat_factor, dim=1)
                    v = v.repeat_interleave(repeat_factor, dim=1)
                
                # Compute attention scores
                scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
                attn_weights = torch.softmax(scores, dim=-1)
                attn_output = torch.matmul(attn_weights, v)
                
                # Reshape back
                attn_output = attn_output.transpose(1, 2).contiguous().view(
                    batch_size, seq_len, self.q_size
                )
                
                # Output projection
                attn_output = self.attention['o_proj'](attn_output)
                
                # Residual connection
                hidden_states = residual + attn_output
                residual = hidden_states
                
                # Post-attention layer norm
                normed = self.post_attention_layernorm(hidden_states)
                
                # MLP with SwiGLU activation
                gate = self.mlp['gate_proj'](normed)
                up = self.mlp['up_proj'](normed)
                gate = gate * torch.sigmoid(gate)
                mlp_output = self.mlp['down_proj'](gate * up)
                
                # Final residual connection
                hidden_states = residual + mlp_output
                
                return hidden_states
        
        # Create layer module with actual config
        layer = TransformerLayer(self.config)
        
        # Store layer name for reference
        layer._layer_name = f"transformer.layers.{layer_idx}"
        
        # Load weights into the module (rest remains the same)
        for weight_name, weight_tensor in weights.items():
            # Validate tensor
            if weight_tensor is None:
                self.logger.warning(f"None tensor for {weight_name}")
                continue
            
            if torch.isnan(weight_tensor).any() or torch.isinf(weight_tensor).any():
                self.logger.warning(f"NaN/Inf detected in {weight_name}")
                continue
            
            # Extract component name and load weight
            if f"layers.{layer_idx}." in weight_name:
                component_name = weight_name.split(f"layers.{layer_idx}.")[-1]
                
                # Load attention weights
                if 'input_layernorm' in component_name:
                    if 'weight' in component_name:
                        layer.input_layernorm.weight.data = weight_tensor
                    elif 'bias' in component_name:
                        layer.input_layernorm.bias.data = weight_tensor
                
                elif 'q_proj' in component_name and 'weight' in component_name:
                    layer.attention['q_proj'].weight.data = weight_tensor
                elif 'k_proj' in component_name and 'weight' in component_name:
                    layer.attention['k_proj'].weight.data = weight_tensor
                elif 'v_proj' in component_name and 'weight' in component_name:
                    layer.attention['v_proj'].weight.data = weight_tensor
                elif 'o_proj' in component_name and 'weight' in component_name:
                    layer.attention['o_proj'].weight.data = weight_tensor
                
                # Load MLP weights
                elif 'post_attention_layernorm' in component_name:
                    if 'weight' in component_name:
                        layer.post_attention_layernorm.weight.data = weight_tensor
                    elif 'bias' in component_name:
                        layer.post_attention_layernorm.bias.data = weight_tensor
                
                elif 'gate_proj' in component_name and 'weight' in component_name:
                    layer.mlp['gate_proj'].weight.data = weight_tensor
                elif 'up_proj' in component_name and 'weight' in component_name:
                    layer.mlp['up_proj'].weight.data = weight_tensor
                elif 'down_proj' in component_name and 'weight' in component_name:
                    layer.mlp['down_proj'].weight.data = weight_tensor
        
        self.logger.debug(f"Constructed transformer layer {layer_idx}")
        return layer
    
    def construct_moe_layer(self,
                      layer_idx: int,
                      weights: Dict[str, torch.Tensor]) -> nn.Module:
        """Construct an MoE layer from weights"""
        
        class MoELayer(nn.Module):
            """GLM MoE layer"""
            
            def __init__(self, config: Dict[str, Any]):
                super().__init__()
                self.hidden_size = config['hidden_size']
                self.moe_intermediate_size = config.get('moe_intermediate_size', 1408)
                self.num_experts = config.get('n_routed_experts', 128)
                self.num_shared_experts = config.get('n_shared_experts', 1)
                self.num_experts_per_tok = config.get('num_experts_per_tok', 8)
                self.num_heads = config['num_attention_heads']
                self.num_kv_heads = config.get('num_key_value_heads', self.num_heads)
                self.head_dim = config.get('head_dim', 128)
                
                # Attention sizes
                self.q_size = self.num_heads * self.head_dim
                self.kv_size = self.num_kv_heads * self.head_dim
                assert self.num_experts > 0, f"num_experts must be positive, got {self.num_experts}"
                
                # Attention components (same as transformer)
                self.input_layernorm = nn.LayerNorm(self.hidden_size, eps=1e-5)
                self.attention = nn.ModuleDict({
                    'q_proj': nn.Linear(self.hidden_size, self.hidden_size, bias=False),
                    'k_proj': nn.Linear(self.hidden_size, self.hidden_size, bias=False),
                    'v_proj': nn.Linear(self.hidden_size, self.hidden_size, bias=False),
                    'o_proj': nn.Linear(self.hidden_size, self.hidden_size, bias=False),
                })
                
                # MoE components
                self.post_attention_layernorm = nn.LayerNorm(self.hidden_size, eps=1e-5)
                self.router = nn.Linear(self.hidden_size, self.num_experts, bias=False)
                
                # Create experts
                self.experts = nn.ModuleList()
                for _ in range(self.num_experts):
                    expert = nn.ModuleDict({
                        'gate_proj': nn.Linear(self.hidden_size, self.moe_intermediate_size, bias=False),
                        'up_proj': nn.Linear(self.hidden_size, self.moe_intermediate_size, bias=False),
                        'down_proj': nn.Linear(self.moe_intermediate_size, self.hidden_size, bias=False),
                    })
                    self.experts.append(expert)
                
                # Shared experts if applicable
                if self.num_shared_experts > 0:
                    self.shared_experts = nn.ModuleList()
                    for _ in range(self.num_shared_experts):
                        expert = nn.ModuleDict({
                            'gate_proj': nn.Linear(self.hidden_size, self.moe_intermediate_size, bias=False),
                            'up_proj': nn.Linear(self.hidden_size, self.moe_intermediate_size, bias=False),
                            'down_proj': nn.Linear(self.moe_intermediate_size, self.hidden_size, bias=False),
                        })
                        self.shared_experts.append(expert)
            @property
            def device(self):
                """Get device of the layer"""
                return next(self.parameters()).device
            
            def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
                """Forward pass for MoE layer with proper routing
                
                Implements actual expert selection and weighted combination
                """
                # Validate and reshape input
                if hasattr(self, '_parent_loader'):
                    hidden_states = self._parent_loader._validate_and_reshape_input(hidden_states, self.hidden_size)
                
                # Ensure we have proper shape
                if hidden_states.dim() != 3:
                    if hidden_states.dim() == 2:
                        hidden_states = hidden_states.unsqueeze(0)
                    else:
                        raise ValueError(f"Expected 3D input, got {hidden_states.dim()}D")
                
                batch_size, seq_len, hidden_size = hidden_states.shape
                residual = hidden_states
                
                # Input layer norm
                normed = self.input_layernorm(hidden_states)
                
                # Attention computation (same as TransformerLayer)
                q = self.attention['q_proj'](normed)
                k = self.attention['k_proj'](normed)  
                v = self.attention['v_proj'](normed)
                
                # Reshape for attention
                head_dim = hidden_size // self.num_heads
                q = q.view(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)
                k = k.view(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)
                v = v.view(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)
                
                # Compute attention
                scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
                attn_weights = torch.softmax(scores, dim=-1)
                attn_output = torch.matmul(attn_weights, v)
                
                # Reshape back
                attn_output = attn_output.transpose(1, 2).contiguous().view(
                    batch_size, seq_len, hidden_size
                )
                attn_output = self.attention['o_proj'](attn_output)
                
                # Residual connection
                hidden_states = residual + attn_output
                residual = hidden_states
                
                # Post-attention layer norm
                normed = self.post_attention_layernorm(hidden_states)
                
                # MoE routing
                router_logits = self.router(normed)  # [batch, seq, num_experts]
                router_probs = torch.softmax(router_logits, dim=-1)
                
                # Get top-k experts (use top-2 for efficiency)
                top_k = min(2, self.num_experts)
                topk_probs, topk_indices = torch.topk(router_probs, top_k, dim=-1)
                
                # Normalize top-k probabilities
                topk_probs = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-6)
                
                # Flatten for expert processing
                normed_flat = normed.view(-1, hidden_size)  # [batch*seq, hidden]
                topk_indices_flat = topk_indices.view(-1, top_k)  # [batch*seq, k]
                topk_probs_flat = topk_probs.view(-1, top_k)  # [batch*seq, k]
                
                # Initialize expert output
                expert_output = torch.zeros_like(normed_flat)
                
                # Apply selected experts with proper weighting
                for expert_idx in range(self.num_experts):
                    # Find tokens that selected this expert
                    expert_mask = (topk_indices_flat == expert_idx).any(dim=-1)
                    
                    if expert_mask.any():
                        expert = self.experts[expert_idx]
                        expert_input = normed_flat[expert_mask]
                        
                        # Compute expert output
                        gate = expert['gate_proj'](expert_input)
                        up = expert['up_proj'](expert_input)
                        gate = gate * torch.sigmoid(gate)  # SwiGLU activation
                        expert_out = expert['down_proj'](gate * up)
                        
                        # Get weights for this expert
                        expert_weights = torch.zeros(len(normed_flat), device=normed.device)
                        for k in range(top_k):
                            mask_k = topk_indices_flat[:, k] == expert_idx
                            expert_weights[mask_k] = topk_probs_flat[mask_k, k]
                        
                        # Add weighted expert output
                        expert_output[expert_mask] += expert_out * expert_weights[expert_mask].unsqueeze(-1)
                
                # Reshape back to [batch, seq, hidden]
                expert_output = expert_output.view(batch_size, seq_len, hidden_size)
                
                # Apply shared experts if available (always active)
                if self.num_shared_experts > 0 and hasattr(self, 'shared_experts'):
                    shared_output = torch.zeros_like(normed)
                    for shared_expert in self.shared_experts:
                        gate = shared_expert['gate_proj'](normed)
                        up = shared_expert['up_proj'](normed)
                        gate = gate * torch.sigmoid(gate)
                        shared_out = shared_expert['down_proj'](gate * up)
                        shared_output += shared_out / self.num_shared_experts
                    
                    # Combine routed and shared experts
                    expert_output = expert_output * 0.5 + shared_output * 0.5
                
                # Final residual connection
                hidden_states = residual + expert_output
                
                return hidden_states
        
        # Create MoE layer
        layer = MoELayer(self.config) 
        
        # Load weights into the module
        for weight_name, weight_tensor in weights.items():
            # Validate tensor
            if weight_tensor is None:
                self.logger.warning(f"None tensor for {weight_name}")
                continue
            
            if torch.isnan(weight_tensor).any() or torch.isinf(weight_tensor).any():
                self.logger.warning(f"NaN/Inf detected in {weight_name}")
                continue
            
            # Extract component name and load weight
            if f"layers.{layer_idx}." in weight_name:
                component_name = weight_name.split(f"layers.{layer_idx}.")[-1]
                
                # Load attention weights (same as transformer)
                if 'input_layernorm' in component_name:
                    if 'weight' in component_name:
                        layer.input_layernorm.weight.data = weight_tensor
                    elif 'bias' in component_name:
                        layer.input_layernorm.bias.data = weight_tensor
                
                elif 'q_proj' in component_name and 'weight' in component_name:
                    layer.attention['q_proj'].weight.data = weight_tensor
                elif 'k_proj' in component_name and 'weight' in component_name:
                    layer.attention['k_proj'].weight.data = weight_tensor
                elif 'v_proj' in component_name and 'weight' in component_name:
                    layer.attention['v_proj'].weight.data = weight_tensor
                elif 'o_proj' in component_name and 'weight' in component_name:
                    layer.attention['o_proj'].weight.data = weight_tensor
                
                # Load post attention layernorm
                elif 'post_attention_layernorm' in component_name:
                    if 'weight' in component_name:
                        layer.post_attention_layernorm.weight.data = weight_tensor
                    elif 'bias' in component_name:
                        layer.post_attention_layernorm.bias.data = weight_tensor
                
                # Load router
                elif 'router' in component_name and 'weight' in component_name:
                    layer.router.weight.data = weight_tensor
                
                # Load expert weights
                elif 'expert' in component_name:
                    # Parse expert index
                    expert_match = re.search(r'expert\.(\d+)', component_name)
                    if expert_match:
                        expert_idx = int(expert_match.group(1))
                        
                        # Determine if shared or regular expert
                        if 'shared' in component_name and hasattr(layer, 'shared_experts'):
                            if expert_idx < len(layer.shared_experts):
                                expert = layer.shared_experts[expert_idx]
                                if 'gate_proj' in component_name and 'weight' in component_name:
                                    expert['gate_proj'].weight.data = weight_tensor
                                elif 'up_proj' in component_name and 'weight' in component_name:
                                    expert['up_proj'].weight.data = weight_tensor
                                elif 'down_proj' in component_name and 'weight' in component_name:
                                    expert['down_proj'].weight.data = weight_tensor
                        else:
                            # Regular expert
                            if expert_idx < len(layer.experts):
                                expert = layer.experts[expert_idx]
                                if 'gate_proj' in component_name and 'weight' in component_name:
                                    expert['gate_proj'].weight.data = weight_tensor
                                elif 'up_proj' in component_name and 'weight' in component_name:
                                    expert['up_proj'].weight.data = weight_tensor
                                elif 'down_proj' in component_name and 'weight' in component_name:
                                    expert['down_proj'].weight.data = weight_tensor
        
        self.logger.debug(f"Constructed MoE layer {layer_idx} with {self.num_experts} experts")
        return layer
    
    def save_quantized_layer(self, 
                           layer_name: str,
                           layer: torch.nn.Module,
                           output_path: Path) -> None:
        """Save quantized layer to disk"""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Extract quantized weights and metadata
        state_dict = {}
        metadata = {}
        
        for name, param in layer.named_parameters():
            if hasattr(param, 'quantized'):
                # Save quantized weights and scales separately
                state_dict[f"{layer_name}.{name}.qweight"] = param.data
                if hasattr(param, 'scale'):
                    state_dict[f"{layer_name}.{name}.scale"] = param.scale
                if hasattr(param, 'zero_point'):
                    state_dict[f"{layer_name}.{name}.zero_point"] = param.zero_point
                
                metadata[f"{layer_name}.{name}"] = {
                    'bits': getattr(param, 'bits', 4),
                    'group_size': getattr(param, 'group_size', 128),
                }
            else:
                # Save unquantized parameters as-is
                state_dict[f"{layer_name}.{name}"] = param.data
        
        # Save in safetensors format
        output_file = output_path / f"{layer_name.replace('/', '_')}.safetensors"
        safetensors.torch.save_file(state_dict, str(output_file), metadata=metadata)
        
        self.logger.info(f"Saved quantized layer to {output_file}")
        
        # Update index file
        self._update_index_file(output_path, layer_name, output_file.name)
    
    def load_embeddings(self) -> torch.nn.Module:
        """Load embedding layers"""
        embedding_weights = {}
        
        # Find embedding weights
        for weight_name, weight_file in self.weight_map.items():
            if 'embedding' in weight_name.lower() or 'embed' in weight_name.lower():
                file_path = self.model_path / weight_file
                if self.use_safetensors:
                    weights = self.load_safetensors_partial(file_path, [weight_name])
                else:
                    weights = self.load_pytorch_partial(file_path, [weight_name])
                embedding_weights.update(weights)
        
        return self._construct_embedding_layer(embedding_weights)
    
    def _construct_embedding_layer(self, weights: Dict[str, torch.Tensor]) -> nn.Module:
        """Construct embedding layer from weights"""
        # Find the main embedding weight
        embed_weight = None
        for key, tensor in weights.items():
            if 'weight' in key:
                embed_weight = tensor
                break
        
        if embed_weight is None:
            raise ValueError("No embedding weights found")
        
        vocab_size, hidden_size = embed_weight.shape
        embedding = nn.Embedding(vocab_size, hidden_size)
        embedding.weight.data = embed_weight
        
        return embedding
    
    def _construct_output_layer(self, weights: Dict[str, torch.Tensor]) -> nn.Module:
        """Construct output/lm_head layer from weights"""
        # Find output projection weight
        output_weight = None
        output_bias = None
        
        for key, tensor in weights.items():
            if 'weight' in key:
                output_weight = tensor
            elif 'bias' in key:
                output_bias = tensor
        
        if output_weight is None:
            raise ValueError("No output head weights found")
        
        vocab_size, hidden_size = output_weight.shape
        lm_head = nn.Linear(hidden_size, vocab_size, bias=output_bias is not None)
        lm_head.weight.data = output_weight
        if output_bias is not None:
            lm_head.bias.data = output_bias
        
        return lm_head
    
    def _construct_layernorm(self, weights: Dict[str, torch.Tensor]) -> nn.Module:
        """Construct layer normalization from weights"""
        norm_weight = None
        norm_bias = None
        
        for key, tensor in weights.items():
            if 'weight' in key:
                norm_weight = tensor
            elif 'bias' in key:
                norm_bias = tensor
        
        if norm_weight is None:
            raise ValueError("No layer norm weights found")
        
        hidden_size = norm_weight.shape[0]
        layer_norm = nn.LayerNorm(hidden_size, elementwise_affine=True)
        layer_norm.weight.data = norm_weight
        if norm_bias is not None:
            layer_norm.bias.data = norm_bias
        
        return layer_norm
    
    def cleanup_layer(self, layer: torch.nn.Module) -> None:
        """Clean up layer from memory"""
        # Delete all parameters
        for param in layer.parameters():
            param.data = torch.empty(0)
        
        # Clear module buffers
        for buffer in layer.buffers():
            buffer.data = torch.empty(0)
        
        # Delete the module
        del layer
        
        # Force garbage collection
        gc.collect()
        
        # Clear CUDA cache if applicable
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_layer_size(self, layer_name: str) -> float:
        """Get size of layer in GB"""
        if layer_name not in self.layer_info:
            return 0.0
        
        return self.layer_info[layer_name].estimated_size_gb
    
    def estimate_total_layers(self) -> int:
        """Estimate total number of layers to process"""
        # Count: embedding + transformer layers + layer norms + output head
        total = 1  # embedding
        total += self.num_layers  # transformer/MoE layers
        total += 2  # final layer norm + output head
        return total
    
    def get_layer_device_placement(self, layer_name: str) -> str:
        """Determine optimal device for layer"""
        layer_info = self.layer_info.get(layer_name)
        if not layer_info:
            return 'cpu'
        
        # Check if layer fits on GPU
        if self.memory_manager.can_fit_on_gpu(layer_info.estimated_size_gb):
            return 'cuda:0'
        else:
            return 'cpu'
    
    def verify_layer_integrity(self, layer: nn.Module) -> bool:
        """Verify layer loaded correctly"""
        try:
            # Check all parameters
            for name, param in layer.named_parameters():
                # Check for NaN/Inf
                if torch.isnan(param).any() or torch.isinf(param).any():
                    self.logger.error(f"NaN/Inf found in parameter: {name}")
                    return False
                
                # Check shapes are reasonable
                if param.numel() == 0:
                    self.logger.error(f"Empty parameter: {name}")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error verifying layer: {e}")
            return False
    
    def create_layer_checkpoint(self,
                              layer_name: str,
                              layer: nn.Module) -> Dict[str, Any]:
        """Create checkpoint data for a layer"""
        checkpoint = {
            'layer_name': layer_name,
            'state_dict': layer.state_dict(),
            'layer_info': self.layer_info[layer_name].__dict__,
        }
        return checkpoint
    
    def load_from_checkpoint(self,
                           checkpoint_path: Path,
                           layer_name: str) -> Optional[nn.Module]:
        """Load a layer from checkpoint"""
        checkpoint_file = checkpoint_path / f"{layer_name.replace('/', '_')}.pt"
        
        if not checkpoint_file.exists():
            return None
        
        checkpoint = torch.load(checkpoint_file, map_location='cpu')
        
        # Reconstruct layer based on type
        layer_info = self.layer_info[layer_name]
        
        # Load weights for this layer
        weights = checkpoint['state_dict']
        
        # Construct appropriate module
        if layer_info.is_moe:
            layer = self.construct_moe_layer(layer_info.layer_index, weights)
        else:
            layer = self.construct_transformer_layer(layer_info.layer_index, weights)
        
        layer.load_state_dict(checkpoint['state_dict'])
        
        return layer
    
    # Helper methods
    
    def _extract_layer_name(self, weight_name: str) -> str:
        """Extract layer name from weight name"""
        
        # For MoE layers, treat each expert as a separate "layer" for memory efficiency
        if 'mlp.experts.' in weight_name:
            # Extract up to the expert number: model.layers.N.mlp.experts.M
            match = re.search(r'(model\.layers\.\d+\.mlp\.experts\.\d+)', weight_name)
            if match:
                return match.group(1)
        
        # For shared experts
        if 'mlp.shared_expert' in weight_name:
            match = re.search(r'(model\.layers\.\d+\.mlp\.shared_expert)', weight_name)
            if match:
                return match.group(1)
        
        # For non-MoE layers (attention, layer norms), group them together
        patterns = [
            r'(model\.embed_tokens)',           # Embedding
            r'(model\.layers\.\d+\.self_attn)', # Attention as a unit
            r'(model\.layers\.\d+\.mlp)(?!\.experts)',  # MLP but not experts
            r'(model\.layers\.\d+\.input_layernorm)',   # Layer norms
            r'(model\.layers\.\d+\.post_attention_layernorm)',
            r'(model\.norm)',                   # Final norm
            r'(lm_head)',                      # Output head
        ]
        
        for pattern in patterns:
            match = re.search(pattern, weight_name)
            if match:
                return match.group(1)
        
        # Fallback
        return weight_name
    
    def _extract_layer_index(self, layer_name: str) -> Optional[int]:
        """Extract layer index from layer name"""
        match = re.search(r'layers\.(\d+)', layer_name)
        if match:
            return int(match.group(1))
        return None
    
    def _determine_layer_type(self, layer_name: str) -> str:
        """Determine the type of layer"""
        layer_name_lower = layer_name.lower()
        
        # Check for embedding - need to handle 'embed_tokens' specifically
        if 'embedding' in layer_name_lower or 'embed' in layer_name_lower:
            return 'embedding'
        elif 'lm_head' in layer_name.lower() or 'output' in layer_name.lower():
            return 'lm_head'
        elif 'layernorm' in layer_name.lower() or 'ln' in layer_name.lower():
            return 'layernorm'
        elif 'layers' in layer_name:
            # Check if it's MoE based on weights
            return 'transformer'  # Will be refined when checking weights
        else:
            return 'other'
    
    def _estimate_layer_size(self, layer_name: str, weight_names: List[str]) -> float:
        """Estimate layer size in GB"""
        total_params = 0
        
        # Get actual sizes from weight tensors if available
        for weight_name in weight_names:
            # Check for specific components
            if 'experts.' in weight_name and 'mlp.experts.' in layer_name:
                # Single expert size (not all 128!)
                # Each expert has gate, up, down projections
                if 'gate' in weight_name or 'up' in weight_name:
                    # hidden_size -> moe_intermediate_size
                    total_params += self.hidden_size * self.config.get('moe_intermediate_size', 1408)
                elif 'down' in weight_name:
                    # moe_intermediate_size -> hidden_size
                    total_params += self.config.get('moe_intermediate_size', 1408) * self.hidden_size
            elif 'shared_expert' in weight_name:
                # Shared expert (larger than regular experts)
                if 'gate' in weight_name or 'up' in weight_name:
                    total_params += self.hidden_size * self.intermediate_size
                elif 'down' in weight_name:
                    total_params += self.intermediate_size * self.hidden_size
            elif any(x in weight_name for x in ['q_proj', 'k_proj', 'v_proj', 'o_proj']):
                # Attention weights - use actual sizes from config
                if 'q_proj' in weight_name:
                    total_params += self.hidden_size * (self.num_attention_heads * self.config.get('head_dim', 128))
                elif 'k_proj' in weight_name or 'v_proj' in weight_name:
                    total_params += self.hidden_size * (self.config.get('num_key_value_heads', 8) * self.config.get('head_dim', 128))
                elif 'o_proj' in weight_name:
                    total_params += (self.num_attention_heads * self.config.get('head_dim', 128)) * self.hidden_size
            elif 'gate_proj' in weight_name or 'up_proj' in weight_name:
                total_params += self.hidden_size * self.intermediate_size
            elif 'down_proj' in weight_name:
                total_params += self.intermediate_size * self.hidden_size
            elif 'norm' in weight_name:
                total_params += self.hidden_size * 2  # weight and bias
        
        # Convert to GB (assuming BF16 from config)
        dtype_size = 2  # BF16 = 2 bytes
        size_gb = (total_params * dtype_size) / 1e9
        
        return size_gb
    
    def _extract_sub_modules(self, weight_names: List[str], layer_name: str) -> List[str]:
        """Extract sub-module names from weight names"""
        sub_modules = set()
        
        for weight_name in weight_names:
            # Remove layer prefix
            if layer_name in weight_name:
                sub_name = weight_name.replace(layer_name + '.', '')
                # Remove .weight or .bias suffix
                sub_name = sub_name.replace('.weight', '').replace('.bias', '')
                sub_modules.add(sub_name)
        
        return sorted(list(sub_modules))
    
    def _sort_layers(self, layer_names: List[str]) -> List[str]:
        """Sort layers in processing order"""
        sorted_layers = []
        
        # 1. Embedding layers first
        for name in layer_names:
            if 'embed' in name.lower():
                sorted_layers.append(name)
        
        # 2. Process each transformer layer completely
        layer_indices = set()
        for name in layer_names:
            match = re.search(r'layers\.(\d+)', name)
            if match:
                layer_indices.add(int(match.group(1)))
        
        for idx in sorted(layer_indices):
            # For each layer, process in order:
            # a. Layer norm
            for name in layer_names:
                if f'layers.{idx}.input_layernorm' in name:
                    sorted_layers.append(name)
            
            # b. Self attention
            for name in layer_names:
                if f'layers.{idx}.self_attn' in name and name not in sorted_layers:
                    sorted_layers.append(name)
            
            # c. Post attention layer norm
            for name in layer_names:
                if f'layers.{idx}.post_attention_layernorm' in name:
                    sorted_layers.append(name)
            
            # d. Shared expert (if exists)
            for name in layer_names:
                if f'layers.{idx}.mlp.shared_expert' in name and name not in sorted_layers:
                    sorted_layers.append(name)
            
            # e. Individual experts (process one by one)
            expert_layers = [n for n in layer_names if f'layers.{idx}.mlp.experts.' in n]
            expert_layers.sort(key=lambda x: int(re.search(r'experts\.(\d+)', x).group(1)))
            sorted_layers.extend(expert_layers)
            
            # f. Other MLP components
            for name in layer_names:
                if f'layers.{idx}.mlp' in name and 'expert' not in name and name not in sorted_layers:
                    sorted_layers.append(name)
        
        # 3. Final layer norm
        for name in layer_names:
            if 'norm' in name.lower() and 'layers' not in name:
                sorted_layers.append(name)
        
        # 4. Output head
        for name in layer_names:
            if 'lm_head' in name or 'output' in name:
                sorted_layers.append(name)
        
        # Add any remaining layers
        for name in layer_names:
            if name not in sorted_layers:
                sorted_layers.append(name)
        
        return sorted_layers
    
    def _update_index_file(self, output_path: Path, layer_name: str, file_name: str) -> None:
        """Update the index file with new layer information"""
        index_file = output_path / "model.safetensors.index.json"
        
        if index_file.exists():
            with open(index_file, 'r') as f:
                index_data = json.load(f)
        else:
            index_data = {'weight_map': {}, 'metadata': {}}
        
        # Add layer to weight map
        index_data['weight_map'][layer_name] = file_name
        
        # Save updated index
        with open(index_file, 'w') as f:
            json.dump(index_data, f, indent=2)