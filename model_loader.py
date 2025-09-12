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
            
            # Check if it's an MoE layer
            is_moe = False
            if layer_index is not None and self.num_experts is not None:
                # GLM4.5 specific: check if this layer has experts
                is_moe = any('expert' in w for w in info['weights'])
            
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
        """Load a single layer from disk
        
        Returns proper nn.Module constructed from weights
        """
        if layer_name not in self.layer_info:
            self.logger.error(f"Layer {layer_name} not found in layer_info")
            raise KeyError(f"Layer {layer_name} not found")
        
        layer_info = self.layer_info[layer_name]
        self.logger.info(f"Loading layer: {layer_name} (type: {layer_info.layer_type}, "
                        f"size: {layer_info.estimated_size_gb:.2f}GB)")
        
        # Load required weight files
        weights = {}
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
    
    def construct_transformer_layer(self,
                                   layer_idx: int,
                                   weights: Dict[str, torch.Tensor]) -> nn.Module:
        """Construct a transformer layer from weights
        
        Returns a proper nn.Module with attention and MLP components
        """
        
        class TransformerLayer(nn.Module):
            """GLM Transformer layer"""
            
            def __init__(self, hidden_size: int, intermediate_size: int, num_heads: int):
                super().__init__()
                self.hidden_size = hidden_size
                self.intermediate_size = intermediate_size
                self.num_heads = num_heads
                
                # Attention components
                self.input_layernorm = nn.LayerNorm(hidden_size, eps=1e-5)
                self.attention = nn.ModuleDict({
                    'q_proj': nn.Linear(hidden_size, hidden_size, bias=False),
                    'k_proj': nn.Linear(hidden_size, hidden_size, bias=False),
                    'v_proj': nn.Linear(hidden_size, hidden_size, bias=False),
                    'o_proj': nn.Linear(hidden_size, hidden_size, bias=False),
                })
                
                # MLP components
                self.post_attention_layernorm = nn.LayerNorm(hidden_size, eps=1e-5)
                self.mlp = nn.ModuleDict({
                    'gate_proj': nn.Linear(hidden_size, intermediate_size, bias=False),
                    'up_proj': nn.Linear(hidden_size, intermediate_size, bias=False),
                    'down_proj': nn.Linear(intermediate_size, hidden_size, bias=False),
                })
            
            def forward(self, x):
                # Simplified forward pass (actual implementation would be more complex)
                return x
        
        # Create layer module
        layer = TransformerLayer(
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_heads=self.num_attention_heads
        )
        
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
        """Construct an MoE layer from weights
        
        Returns a proper nn.Module with router and experts
        """
        
        class MoELayer(nn.Module):
            """GLM MoE layer"""
            
            def __init__(self, hidden_size: int, intermediate_size: int, 
                        num_experts: int, num_shared_experts: int, num_heads: int):
                super().__init__()
                self.hidden_size = hidden_size
                self.intermediate_size = intermediate_size
                self.num_experts = num_experts
                self.num_shared_experts = num_shared_experts
                
                # Attention components (same as transformer)
                self.input_layernorm = nn.LayerNorm(hidden_size, eps=1e-5)
                self.attention = nn.ModuleDict({
                    'q_proj': nn.Linear(hidden_size, hidden_size, bias=False),
                    'k_proj': nn.Linear(hidden_size, hidden_size, bias=False),
                    'v_proj': nn.Linear(hidden_size, hidden_size, bias=False),
                    'o_proj': nn.Linear(hidden_size, hidden_size, bias=False),
                })
                
                # MoE components
                self.post_attention_layernorm = nn.LayerNorm(hidden_size, eps=1e-5)
                self.router = nn.Linear(hidden_size, num_experts, bias=False)
                
                # Create experts
                self.experts = nn.ModuleList()
                for _ in range(num_experts):
                    expert = nn.ModuleDict({
                        'gate_proj': nn.Linear(hidden_size, intermediate_size, bias=False),
                        'up_proj': nn.Linear(hidden_size, intermediate_size, bias=False),
                        'down_proj': nn.Linear(intermediate_size, hidden_size, bias=False),
                    })
                    self.experts.append(expert)
                
                # Shared experts if applicable
                if num_shared_experts > 0:
                    self.shared_experts = nn.ModuleList()
                    for _ in range(num_shared_experts):
                        expert = nn.ModuleDict({
                            'gate_proj': nn.Linear(hidden_size, intermediate_size, bias=False),
                            'up_proj': nn.Linear(hidden_size, intermediate_size, bias=False),
                            'down_proj': nn.Linear(intermediate_size, hidden_size, bias=False),
                        })
                        self.shared_experts.append(expert)
            
            def forward(self, x):
                # Simplified forward pass
                return x
        
        # Create MoE layer
        layer = MoELayer(
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_experts=self.num_experts or 8,
            num_shared_experts=self.num_shared_experts or 0,
            num_heads=self.num_attention_heads
        )
        
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
        # GLM4.5 patterns
        patterns = [
            r'(transformer\.embedding)',
            r'(transformer\.layers\.\d+)',
            r'(transformer\.final_layernorm)',
            r'(lm_head)',
            r'(output_layer)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, weight_name)
            if match:
                return match.group(1)
        
        # Fallback: use everything before the last dot
        parts = weight_name.rsplit('.', 2)
        if len(parts) > 2:
            return parts[0]
        
        return weight_name
    
    def _extract_layer_index(self, layer_name: str) -> Optional[int]:
        """Extract layer index from layer name"""
        match = re.search(r'layers\.(\d+)', layer_name)
        if match:
            return int(match.group(1))
        return None
    
    def _determine_layer_type(self, layer_name: str) -> str:
        """Determine the type of layer"""
        if 'embedding' in layer_name.lower():
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
        
        # Estimate based on weight names and expected shapes
        for weight_name in weight_names:
            if 'q_proj' in weight_name or 'k_proj' in weight_name or 'v_proj' in weight_name or 'o_proj' in weight_name:
                # Attention weights
                total_params += self.hidden_size * self.hidden_size
            elif 'gate_proj' in weight_name or 'up_proj' in weight_name:
                total_params += self.hidden_size * self.intermediate_size
            elif 'down_proj' in weight_name:
                total_params += self.intermediate_size * self.hidden_size
            elif 'expert' in weight_name and self.num_experts:
                # MoE expert weights
                if 'gate' in weight_name or 'up' in weight_name:
                    total_params += self.hidden_size * self.intermediate_size
                else:
                    total_params += self.intermediate_size * self.hidden_size
            elif 'embedding' in weight_name:
                total_params += self.vocab_size * self.hidden_size
            elif 'lm_head' in weight_name:
                total_params += self.vocab_size * self.hidden_size
            elif 'norm' in weight_name:
                total_params += self.hidden_size * 2  # weight and bias
        
        # Convert to GB (assuming FP16)
        size_gb = (total_params * 2) / 1e9
        
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
            if 'embedding' in name.lower():
                sorted_layers.append(name)
        
        # 2. Transformer layers in order
        transformer_layers = []
        for name in layer_names:
            if 'layers' in name:
                transformer_layers.append(name)
        
        # Sort by layer index
        transformer_layers.sort(key=lambda x: self._extract_layer_index(x) or 0)
        sorted_layers.extend(transformer_layers)
        
        # 3. Final layer norm
        for name in layer_names:
            if 'final' in name or ('layernorm' in name and 'layers' not in name):
                sorted_layers.append(name)
        
        # 4. Output head last
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