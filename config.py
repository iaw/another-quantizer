# config.py
"""Configuration settings for GLM sequential quantization"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
from pathlib import Path
import yaml
import json
import torch
import psutil
import logging


@dataclass
class QuantizationConfig:
    """Main configuration for quantization process"""
    
    # Model paths
    model_path: str
    output_path: str
    checkpoint_dir: str
    
    # Quantization settings
    bits: int = 4
    group_size: int = 128  # AWQ recommended default
    symmetric: bool = False
    
    # Dtype settings
    compute_dtype: Optional[str] = None  # 'float16', 'bfloat16', 'float32', or None for auto-detect
    force_dtype_conversion: bool = False  # Force all computations to use compute_dtype
    mixed_precision: bool = True  # Allow different dtypes for different layers
    
    # Memory settings
    max_gpu_memory: int = 20  # GB
    max_cpu_memory: int = 200  # GB
    offload_to_cpu: bool = True
    
    # Processing settings
    calibration_samples: int = 128
    calibration_batch_size: int = 4
    checkpoint_every_n_layers: int = 5
    
    # Layer settings - GLM4.5 specific patterns
    skip_layers: List[str] = field(default_factory=lambda: [
        "lm_head",
        "output_layer",
        "*.layernorm",
        "*.ln_f",
        "*.input_layernorm",
        "*.post_attention_layernorm",
        "embedding*",
        "*.router",  # MoE routers stay in FP16
    ])
    
    # Model configuration defaults
    max_position_embeddings: int = 2048  # Default sequence length
    
    # AWQ specific settings
    awq_damping_percent: float = 0.01
    awq_desc_act: bool = False
    awq_true_sequential: bool = True
    awq_protection_factor: float = 1.5  # For salient channels
    
    def __post_init__(self):
        """Initialize default values and validate config"""
        # Convert paths to Path objects
        self.model_path = Path(self.model_path)
        self.output_path = Path(self.output_path)
        self.checkpoint_dir = Path(self.checkpoint_dir)
        
        # Auto-detect available memory if not set
        if self.max_gpu_memory == 20 and torch.cuda.is_available():
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            # Use 90% of available GPU memory
            self.max_gpu_memory = int(gpu_mem * 0.9)
            
        if self.max_cpu_memory == 200:
            cpu_mem = psutil.virtual_memory().total / 1e9
            # Use 80% of available CPU memory
            self.max_cpu_memory = int(cpu_mem * 0.8)
        
        # Set max_position_embeddings from model config if available
        if hasattr(self, 'model_path') and self.model_path.exists():
            config_file = self.model_path / "config.json"
            if config_file.exists():
                try:
                    with open(config_file, 'r') as f:
                        model_config = json.load(f)
                    # Try to get max_position_embeddings from model config
                    self.max_position_embeddings = model_config.get(
                        'max_position_embeddings',
                        model_config.get('max_sequence_length', 2048)
                    )
                except:
                    # Keep default if loading fails
                    pass
        
        # Validate configuration
        self.validate()
    
    def validate(self) -> bool:
        """Validate configuration settings"""
        errors = []
        warnings = []
        
        # Validate dtype settings
        if self.compute_dtype is not None:
            valid_dtypes = ['float16', 'bfloat16', 'float32', 'auto']
            if self.compute_dtype not in valid_dtypes:
                errors.append(f"Invalid compute_dtype: {self.compute_dtype}. Must be one of {valid_dtypes}")
        
        # Check if model_path exists
        if not self.model_path.exists():
            errors.append(f"Model path does not exist: {self.model_path}")
        
        # Verify output_path parent exists and is writable
        if not self.output_path.parent.exists():
            # Try to create it
            try:
                self.output_path.parent.mkdir(parents=True, exist_ok=True)
            except:
                errors.append(f"Cannot create output directory parent: {self.output_path.parent}")
        
        # Ensure memory limits are reasonable
        if self.max_gpu_memory < 4:
            warnings.append(f"GPU memory limit may be too low: {self.max_gpu_memory}GB")
        
        if self.max_cpu_memory < 16:
            warnings.append(f"CPU memory limit may be too low: {self.max_cpu_memory}GB")
        
        # Validate quantization parameters
        if self.bits not in [3, 4, 8]:
            errors.append(f"Bits must be 3, 4, or 8, got: {self.bits}")
        
        if self.group_size not in [-1, 32, 64, 128, 256]:
            errors.append(f"Group size must be -1, 32, 64, 128, or 256, got: {self.group_size}")
        
        # Check GPU availability if not offloading to CPU
        if not self.offload_to_cpu and not torch.cuda.is_available():
            errors.append("GPU not available but offload_to_cpu is False")
        
        # Check calibration settings
        if self.calibration_samples < 1:
            errors.append(f"Calibration samples must be >= 1, got: {self.calibration_samples}")
        
        if self.calibration_batch_size < 1:
            errors.append(f"Calibration batch size must be >= 1, got: {self.calibration_batch_size}")
        
        # Validate max_position_embeddings
        if self.max_position_embeddings < 128:
            warnings.append(f"max_position_embeddings seems too small: {self.max_position_embeddings}")
        
        # Check checkpoint frequency
        if self.checkpoint_every_n_layers < 1:
            self.checkpoint_every_n_layers = 5  # Auto-fix
            warnings.append("checkpoint_every_n_layers was < 1, set to 5")
        
        # Print warnings
        for warning in warnings:
            logging.warning(warning)
        
        if errors:
            raise ConfigError("\n".join(errors))
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization"""
        config_dict = asdict(self)
        
        # Convert Path objects to strings
        config_dict['model_path'] = str(self.model_path)
        config_dict['output_path'] = str(self.output_path)
        config_dict['checkpoint_dir'] = str(self.checkpoint_dir)
        
        # Add metadata
        config_dict['_metadata'] = {
            'version': '1.0.0',
            'model_type': 'GLM4.5',
            'quantization_method': 'AWQ',
            'timestamp': None  # Will be set when saving
        }
        
        return config_dict
    
    @classmethod
    def from_yaml(cls, path: str) -> 'QuantizationConfig':
        """Load config from YAML file"""
        path = Path(path)
        if not path.exists():
            raise ConfigError(f"Config file not found: {path}")
        
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Remove metadata if present
        config_dict.pop('_metadata', None)
        
        # Ensure required fields have defaults
        config_dict.setdefault('max_position_embeddings', 2048)
        
        return cls(**config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'QuantizationConfig':
        """Create config from dictionary"""
        # Remove metadata if present
        config_dict = config_dict.copy()
        config_dict.pop('_metadata', None)
        
        # Ensure required fields have defaults
        config_dict.setdefault('max_position_embeddings', 2048)
        
        return cls(**config_dict)
    
    def save(self, path: str) -> None:
        """Save config to YAML file"""
        from datetime import datetime
        
        path = Path(path)
        config_dict = self.to_dict()
        
        # Update timestamp
        config_dict['_metadata']['timestamp'] = datetime.now().isoformat()
        
        # Create parent directory if needed
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
    
    def get_awq_config(self) -> Dict[str, Any]:
        """Get AWQ-specific configuration"""
        return {
            "bits": self.bits,
            "group_size": self.group_size,
            "symmetric": self.symmetric,
            "damping_percent": self.awq_damping_percent,
            "desc_act": self.awq_desc_act,
            "true_sequential": self.awq_true_sequential,
            "protection_factor": self.awq_protection_factor,
        }


@dataclass
class GLMModelConfig:
    """GLM model-specific configuration"""
    
    num_layers: int
    hidden_size: int
    num_attention_heads: int
    vocab_size: int
    num_experts: Optional[int] = None  # For MoE models
    num_shared_experts: Optional[int] = None  # GLM4.5 has shared experts
    intermediate_size: Optional[int] = None
    max_position_embeddings: int = 131072  # GLM4.5 default
    layer_names: List[str] = field(default_factory=list)
    dtype: str = "float16"
    rope_theta: float = 10000.0
    partial_rotary_factor: float = 0.5  # GLM4.5 uses partial RoPE
    
    @classmethod
    def from_model_config(cls, model_path: str) -> 'GLMModelConfig':
        """Extract config from model files"""
        model_path = Path(model_path)
        config_path = model_path / "config.json"
        
        if not config_path.exists():
            raise ConfigError(f"Model config not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Extract GLM4.5-specific fields
        num_layers = config.get('num_hidden_layers', config.get('num_layers'))
        hidden_size = config.get('hidden_size')
        num_attention_heads = config.get('num_attention_heads')
        vocab_size = config.get('vocab_size', config.get('padded_vocab_size'))
        
        # MoE configuration for GLM4.5
        num_experts = config.get('num_experts', None)
        num_shared_experts = config.get('num_shared_experts', None)
        
        # Intermediate size (FFN)
        intermediate_size = config.get('intermediate_size', config.get('ffn_hidden_size'))
        
        # Other parameters
        max_position_embeddings = config.get('max_position_embeddings', 131072)
        rope_theta = config.get('rope_theta', 10000.0)
        partial_rotary_factor = config.get('partial_rotary_factor', 0.5)
        
        # Generate layer names
        layer_names = []
        layer_names.append("transformer.embedding")
        for i in range(num_layers):
            layer_names.append(f"transformer.layers.{i}")
        layer_names.append("transformer.final_layernorm")
        layer_names.append("lm_head")
        
        return cls(
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            vocab_size=vocab_size,
            num_experts=num_experts,
            num_shared_experts=num_shared_experts,
            intermediate_size=intermediate_size,
            max_position_embeddings=max_position_embeddings,
            layer_names=layer_names,
            rope_theta=rope_theta,
            partial_rotary_factor=partial_rotary_factor,
        )
    
    def get_layer_memory_estimate(self, layer_idx: int) -> float:
        """Estimate memory requirement for a specific layer in GB"""
        bytes_per_param = 2 if self.dtype == "float16" else 4
        
        if layer_idx == -1:  # Embedding layer
            params = self.vocab_size * self.hidden_size
        elif layer_idx == self.num_layers:  # Output head
            params = self.vocab_size * self.hidden_size
        else:  # Transformer layer
            # Attention: Q, K, V, O projections
            attention_params = 4 * self.hidden_size * self.hidden_size
            
            # MLP or MoE
            if self.num_experts is not None:
                # MoE layer: router + experts
                router_params = self.hidden_size * self.num_experts
                expert_params = self.num_experts * 2 * self.hidden_size * self.intermediate_size
                if self.num_shared_experts:
                    expert_params += self.num_shared_experts * 2 * self.hidden_size * self.intermediate_size
                mlp_params = router_params + expert_params
            else:
                # Standard MLP: gate, up, down projections
                mlp_params = 3 * self.hidden_size * self.intermediate_size
            
            # Layer norms (negligible but included)
            norm_params = 2 * self.hidden_size
            
            params = attention_params + mlp_params + norm_params
        
        return (params * bytes_per_param) / 1e9
    
    def get_total_params(self) -> int:
        """Calculate total number of parameters"""
        total = 0
        
        # Embeddings
        total += self.vocab_size * self.hidden_size
        
        # Transformer layers
        for i in range(self.num_layers):
            # Attention
            total += 4 * self.hidden_size * self.hidden_size
            
            # MLP/MoE
            if self.num_experts is not None:
                total += self.hidden_size * self.num_experts  # Router
                total += self.num_experts * 2 * self.hidden_size * self.intermediate_size
                if self.num_shared_experts:
                    total += self.num_shared_experts * 2 * self.hidden_size * self.intermediate_size
            else:
                total += 3 * self.hidden_size * self.intermediate_size
            
            # Layer norms
            total += 2 * self.hidden_size
        
        # Final layer norm
        total += self.hidden_size
        
        # Output head
        total += self.vocab_size * self.hidden_size
        
        return total
    
    def get_layer_pattern(self, layer_idx: int) -> str:
        """Get naming pattern for layer at index"""
        if layer_idx == -1:
            return "transformer.embedding"
        elif layer_idx == self.num_layers:
            return "lm_head"
        elif layer_idx == self.num_layers - 1:
            return "transformer.final_layernorm"
        else:
            return f"transformer.layers.{layer_idx}"
    
    def estimate_quantized_size(self, bits: int = 4) -> float:
        """Estimate final model size after quantization in GB"""
        total_params = self.get_total_params()
        
        # Quantized layers (most weights)
        quantized_params = total_params * 0.9  # Assume 90% of weights are quantized
        quantized_size = (quantized_params * bits / 8) / 1e9
        
        # Non-quantized layers (embeddings, norms, etc.)
        unquantized_params = total_params * 0.1
        unquantized_size = (unquantized_params * 2) / 1e9  # FP16
        
        # Scales and metadata overhead (approximately 3% for group_size=128)
        overhead = quantized_size * 0.03
        
        return quantized_size + unquantized_size + overhead
    
    def get_moe_config(self) -> Optional[Dict[str, Any]]:
        """Get MoE-specific configuration if applicable"""
        if self.num_experts is None:
            return None
        
        return {
            "num_experts": self.num_experts,
            "num_shared_experts": self.num_shared_experts,
            "expert_intermediate_size": self.intermediate_size,
            "router_type": "sigmoid",  # GLM4.5 uses sigmoid gates
            "top_k": 8,  # GLM4.5 typically uses top-8 routing
        }


@dataclass
class AWQLayerConfig:
    """Configuration for AWQ quantization of a specific layer"""
    
    layer_name: str
    input_layer: str  # Layer that provides input for scaling
    output_layers: List[str]  # Layers that receive output
    quantize: bool = True
    bits: int = 4
    group_size: int = 128
    
    def to_mapping(self) -> List[str]:
        """Convert to LLM Compressor mapping format"""
        return [self.input_layer, self.output_layers]


class ConfigError(Exception):
    """Custom exception for configuration errors"""
    pass


def create_default_config(model_path: str, output_path: str) -> QuantizationConfig:
    """Create a default configuration for GLM model"""
    # Detect model size from path or config
    model_config = GLMModelConfig.from_model_config(model_path)
    
    # Set appropriate memory limits based on model size
    estimated_size = model_config.estimate_quantized_size(bits=4)
    
    # Determine if CPU offloading is needed
    gpu_available = torch.cuda.is_available()
    if gpu_available:
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        offload_needed = estimated_size > gpu_mem * 0.8
    else:
        offload_needed = True
    
    # Create config
    config = QuantizationConfig(
        model_path=model_path,
        output_path=output_path,
        checkpoint_dir=Path(output_path).parent / "checkpoints",
        offload_to_cpu=offload_needed,
    )
    
    # Adjust batch size based on model size - ensure minimum for AWQ
    if model_config.num_layers > 50:  # Large model
        config.calibration_batch_size = 2  # Minimum 2 for sufficient samples
    elif model_config.num_layers > 30:  # Medium model
        config.calibration_batch_size = 4  # Better statistics with more samples
    else:  # Small model
        config.calibration_batch_size = 8  # Can afford more samples
    
    return config


def get_glm_layer_mappings() -> List[AWQLayerConfig]:
    """Get standard AWQ layer mappings for GLM architecture"""
    # GLM4.5-specific layer connections for AWQ scale propagation
    return [
        # Attention input scaling
        AWQLayerConfig(
            layer_name="input_layernorm",
            input_layer="re:.*input_layernorm",
            output_layers=["re:.*q_proj", "re:.*k_proj", "re:.*v_proj"]
        ),
        # Attention output scaling
        AWQLayerConfig(
            layer_name="v_proj",
            input_layer="re:.*v_proj",
            output_layers=["re:.*o_proj"]
        ),
        # MLP input scaling
        AWQLayerConfig(
            layer_name="post_attention_layernorm",
            input_layer="re:.*post_attention_layernorm",
            output_layers=["re:.*gate_proj", "re:.*up_proj"]
        ),
        # MLP output scaling
        AWQLayerConfig(
            layer_name="up_proj",
            input_layer="re:.*up_proj",
            output_layers=["re:.*down_proj"]
        ),
        # MoE expert scaling (if applicable)
        AWQLayerConfig(
            layer_name="expert_gate",
            input_layer="re:.*expert.*gate",
            output_layers=["re:.*expert.*up"]
        ),
        AWQLayerConfig(
            layer_name="expert_up",
            input_layer="re:.*expert.*up",
            output_layers=["re:.*expert.*down"]
        ),
    ]