# vllm_export.py
"""Export quantized model in vLLM-compatible format"""

import torch
import json
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import safetensors.torch
import numpy as np
from dataclasses import dataclass
import logging
from tqdm import tqdm


@dataclass
class VLLMExportConfig:
    """Configuration for vLLM export"""
    quantization: str = "awq"
    weight_bits: int = 4
    group_size: int = 128
    zero_point: bool = True
    version: str = "GEMM"  # GEMM or GEMV
    desc_act: bool = False  # Activation order
    clip_qkv: Optional[float] = None
    num_bits: int = 4
    


class VLLMExporter:
    """Export quantized model for vLLM deployment"""
    
    def __init__(self, quantized_path: str, output_path: str):
        """Initialize vLLM exporter
        
        Args:
            quantized_path: Path to quantized model
            output_path: Path for vLLM-compatible output
        """
        self.quantized_path = Path(quantized_path)
        self.output_path = Path(output_path)
        self.logger = logging.getLogger(__name__)
        
        # Load quantization metadata
        self.quantization_metadata = self._load_quantization_metadata()
        
        # Create export config
        self.export_config = VLLMExportConfig(
            weight_bits=self.quantization_metadata.get('bits', 4),
            group_size=self.quantization_metadata.get('group_size', 128),
            zero_point=not self.quantization_metadata.get('symmetric', False)
        )
        
    def export(self, num_shards: Optional[int] = None) -> Dict[str, Any]:
        """Export model in vLLM format
        
        Args:
            num_shards: Number of shards for tensor parallel (None for auto)
            
        Returns:
            Export status dictionary
        """
        self.logger.info(f"Exporting model to vLLM format: {self.output_path}")
        
        # Create output directory
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        status = {
            'success': False,
            'weights_converted': False,
            'config_created': False,
            'index_created': False,
            'errors': []
        }
        
        try:
            # Step 1: Convert and reorganize weights
            self.logger.info("Converting weights to vLLM format...")
            weight_info = self._convert_weights_to_vllm_format()
            status['weights_converted'] = True
            
            # Step 2: Create vLLM-compatible config
            self.logger.info("Creating vLLM configuration...")
            self._create_vllm_config(weight_info)
            status['config_created'] = True
            
            # Step 3: Handle sharding if needed
            if num_shards and num_shards > 1:
                self.logger.info(f"Sharding model for {num_shards} GPUs...")
                self._shard_model_weights(num_shards, weight_info)
            
            # Step 4: Create index files
            self.logger.info("Creating index files...")
            self._create_index_files(weight_info, num_shards)
            status['index_created'] = True
            
            # Step 5: Copy tokenizer files
            self._copy_tokenizer_files()
            
            # Step 6: Validate export
            validation_result = self._validate_vllm_export()
            status['validation'] = validation_result
            
            status['success'] = validation_result['valid']
            
        except Exception as e:
            self.logger.error(f"Export failed: {e}")
            status['errors'].append(str(e))
        
        return status
    
    def _convert_weights_to_vllm_format(self) -> Dict[str, Any]:
        """Convert quantized weights to vLLM format
        
        Returns:
            Dictionary with weight information
        """
        weight_info = {
            'total_size': 0,
            'num_layers': 0,
            'layer_shapes': {},
            'quantized_layers': [],
            'files_created': []
        }
        
        # Load existing weights
        weight_files = list(self.quantized_path.glob("*.safetensors"))
        if not weight_files:
            weight_files = list(self.quantized_path.glob("*.pt"))
        
        if not weight_files:
            raise FileNotFoundError(f"No weight files found in {self.quantized_path}")
        
        # Process and convert weights
        all_weights = {}
        qweight_dict = {}
        scales_dict = {}
        zeros_dict = {}
        
        for weight_file in tqdm(weight_files, desc="Loading weights"):
            if weight_file.suffix == '.safetensors':
                weights = safetensors.torch.load_file(str(weight_file))
            else:
                weights = torch.load(weight_file, map_location='cpu')
            
            for name, tensor in weights.items():
                # vLLM expects specific naming conventions
                vllm_name = self._convert_to_vllm_naming(name)
                
                # Separate quantized weights, scales, and zeros
                if 'qweight' in name or 'weight_quantized' in name:
                    qweight_dict[vllm_name] = tensor
                    weight_info['quantized_layers'].append(vllm_name)
                elif 'scales' in name or 'weight_scale' in name:
                    scales_dict[vllm_name.replace('.qweight', '.scales')] = tensor
                elif 'zeros' in name or 'weight_zero' in name or 'qzeros' in name:
                    zeros_dict[vllm_name.replace('.qweight', '.qzeros')] = tensor
                else:
                    # Non-quantized weights
                    all_weights[vllm_name] = tensor
        
        # Pack weights in vLLM format
        self.logger.info("Packing weights in vLLM format...")
        packed_weights = self._pack_awq_weights(qweight_dict, scales_dict, zeros_dict)
        
        # Merge all weights
        all_weights.update(packed_weights)
        
        # Save in vLLM format
        output_file = self.output_path / "pytorch_model.bin"
        torch.save(all_weights, output_file)
        weight_info['files_created'].append(str(output_file))
        
        # Calculate total size and shapes
        for name, tensor in all_weights.items():
            weight_info['total_size'] += tensor.numel() * tensor.element_size()
            weight_info['layer_shapes'][name] = list(tensor.shape)
        
        weight_info['num_layers'] = len(set(self._extract_layer_index(n) for n in all_weights.keys() 
                                           if self._extract_layer_index(n) is not None))
        
        self.logger.info(f"Converted {len(all_weights)} weight tensors, "
                        f"total size: {weight_info['total_size']/1e9:.2f}GB")
        
        return weight_info
    
    def _pack_awq_weights(self, 
                         qweights: Dict[str, torch.Tensor],
                         scales: Dict[str, torch.Tensor],
                         zeros: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Pack AWQ weights in vLLM format
        
        vLLM expects AWQ weights in specific format:
        - INT4 weights packed as INT32
        - Scales and zeros interleaved
        - Specific tensor layouts
        """
        packed = {}
        
        for name, qweight in qweights.items():
            scale_name = name.replace('.qweight', '.scales')
            zero_name = name.replace('.qweight', '.qzeros')
            
            if scale_name in scales:
                # Pack quantized weight for vLLM
                if self.export_config.weight_bits == 4:
                    # Pack INT4 -> INT32 for vLLM CUDA kernels
                    packed_qweight = self._pack_int4_to_int32(qweight)
                    packed[name] = packed_qweight
                else:
                    packed[name] = qweight
                
                # Add scales
                packed[scale_name] = scales[scale_name]
                
                # Add zeros if present
                if zero_name in zeros:
                    if self.export_config.weight_bits == 4:
                        # Pack zeros as well
                        packed_zeros = self._pack_int4_to_int32(zeros[zero_name])
                        packed[zero_name] = packed_zeros
                    else:
                        packed[zero_name] = zeros[zero_name]
            else:
                # No scales found, keep original
                packed[name] = qweight
        
        return packed
    
    def _pack_int4_to_int32(self, tensor: torch.Tensor) -> torch.Tensor:
        """Pack INT4 weights to INT32 for vLLM
        
        vLLM CUDA kernels expect INT4 weights packed as INT32
        """
        # Ensure tensor is on CPU
        tensor = tensor.cpu()
        
        # Get original shape
        original_shape = tensor.shape
        
        # Flatten tensor
        flat = tensor.flatten()
        
        # Pad if necessary (need multiple of 8 for INT4->INT32)
        remainder = len(flat) % 8
        if remainder != 0:
            padding = 8 - remainder
            flat = torch.cat([flat, torch.zeros(padding, dtype=flat.dtype)])
        
        # Pack 8 INT4 values into 1 INT32
        num_int32 = len(flat) // 8
        packed = torch.zeros(num_int32, dtype=torch.int32)
        
        for i in range(8):
            # Extract 4-bit values and shift into position
            values = flat[i::8].to(torch.int32) & 0xF
            packed |= values << (4 * i)
        
        # Reshape to match original dimensions (adjusted for packing)
        if len(original_shape) == 2:
            packed = packed.reshape(original_shape[0], -1)
        
        return packed
    
    def _create_vllm_config(self, weight_info: Dict[str, Any]) -> None:
        """Create vLLM-compatible configuration files"""
        
        # Load original config
        original_config_path = self.quantized_path / "config.json"
        if not original_config_path.exists():
            raise FileNotFoundError(f"Original config not found: {original_config_path}")
        
        with open(original_config_path, 'r') as f:
            config = json.load(f)
        
        # Update with vLLM quantization config
        config['quantization_config'] = {
            'bits': self.export_config.weight_bits,
            'group_size': self.export_config.group_size,
            'damp_percent': self.quantization_metadata.get('awq_damping_percent', 0.01),
            'desc_act': self.export_config.desc_act,
            'static_groups': False,
            'sym': not self.export_config.zero_point,
            'true_sequential': True,
            'model_name_or_path': str(self.quantized_path),
            'quant_method': 'awq',
            'version': self.export_config.version,
        }
        
        # Add vLLM-specific fields
        config['torch_dtype'] = 'float16'
        config['_name_or_path'] = str(self.quantized_path)
        
        # Save config
        output_config = self.output_path / "config.json"
        with open(output_config, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Create quantization config file (some vLLM versions look for this)
        quant_config_file = self.output_path / "quantize_config.json"
        with open(quant_config_file, 'w') as f:
            json.dump(config['quantization_config'], f, indent=2)
        
        self.logger.info(f"Created vLLM configuration: {output_config}")
    
    def _shard_model_weights(self, num_shards: int, weight_info: Dict[str, Any]) -> None:
        """Shard model weights for tensor parallel
        
        Args:
            num_shards: Number of shards
            weight_info: Weight information
        """
        # Load the combined weights
        weights_file = self.output_path / "pytorch_model.bin"
        weights = torch.load(weights_file, map_location='cpu')
        
        # Identify which tensors to shard
        shardable_layers = [
            'q_proj', 'k_proj', 'v_proj',  # Attention
            'o_proj',
            'gate_proj', 'up_proj',  # MLP
            'down_proj',
        ]
        
        # Create shards
        shards = [{} for _ in range(num_shards)]
        
        for name, tensor in weights.items():
            should_shard = any(layer in name for layer in shardable_layers)
            
            if should_shard and len(tensor.shape) >= 2:
                # Determine shard dimension (usually dim 0 for column parallel)
                if any(x in name for x in ['q_proj', 'k_proj', 'v_proj', 'gate_proj', 'up_proj']):
                    # Column parallel - shard dim 0
                    shard_size = tensor.shape[0] // num_shards
                    for i in range(num_shards):
                        start = i * shard_size
                        end = (i + 1) * shard_size if i < num_shards - 1 else tensor.shape[0]
                        shards[i][name] = tensor[start:end].clone()
                elif any(x in name for x in ['o_proj', 'down_proj']):
                    # Row parallel - shard dim 1
                    shard_size = tensor.shape[1] // num_shards
                    for i in range(num_shards):
                        start = i * shard_size
                        end = (i + 1) * shard_size if i < num_shards - 1 else tensor.shape[1]
                        shards[i][name] = tensor[:, start:end].clone()
                else:
                    # Replicate across all shards
                    for i in range(num_shards):
                        shards[i][name] = tensor.clone()
            else:
                # Replicate non-shardable tensors
                for i in range(num_shards):
                    shards[i][name] = tensor.clone()
        
        # Save shards
        for i, shard in enumerate(shards):
            shard_file = self.output_path / f"pytorch_model-{i+1:05d}-of-{num_shards:05d}.bin"
            torch.save(shard, shard_file)
            self.logger.info(f"Created shard: {shard_file}")
        
        # Remove original combined file
        weights_file.unlink()
    
    def _create_index_files(self, weight_info: Dict[str, Any], num_shards: Optional[int]) -> None:
        """Create index files for sharded model"""
        
        if num_shards and num_shards > 1:
            # Create index for sharded model
            weight_map = {}
            
            for i in range(num_shards):
                shard_file = f"pytorch_model-{i+1:05d}-of-{num_shards:05d}.bin"
                shard_path = self.output_path / shard_file
                
                if shard_path.exists():
                    shard_weights = torch.load(shard_path, map_location='cpu')
                    for name in shard_weights.keys():
                        weight_map[name] = shard_file
            
            index = {
                'metadata': {
                    'total_size': weight_info['total_size']
                },
                'weight_map': weight_map
            }
            
            index_file = self.output_path / "pytorch_model.bin.index.json"
            with open(index_file, 'w') as f:
                json.dump(index, f, indent=2)
            
            self.logger.info(f"Created index file: {index_file}")
    
    def _copy_tokenizer_files(self) -> None:
        """Copy tokenizer files to output"""
        tokenizer_files = [
            "tokenizer.json",
            "tokenizer_config.json", 
            "tokenizer.model",
            "special_tokens_map.json",
            "added_tokens.json"
        ]
        
        for file_name in tokenizer_files:
            src = self.quantized_path / file_name
            if src.exists():
                dst = self.output_path / file_name
                shutil.copy2(src, dst)
                self.logger.debug(f"Copied {file_name}")
    
    def _validate_vllm_export(self) -> Dict[str, Any]:
        """Validate the export is vLLM compatible"""
        validation = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Check required files
        required_files = ['config.json']
        if (self.output_path / "pytorch_model.bin").exists():
            required_files.append('pytorch_model.bin')
        elif list(self.output_path.glob("pytorch_model-*.bin")):
            required_files.append('pytorch_model.bin.index.json')
        else:
            validation['errors'].append("No model weights found")
            validation['valid'] = False
        
        for file_name in required_files:
            if not (self.output_path / file_name).exists():
                validation['errors'].append(f"Missing required file: {file_name}")
                validation['valid'] = False
        
        # Check config has quantization info
        config_file = self.output_path / "config.json"
        if config_file.exists():
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            if 'quantization_config' not in config:
                validation['errors'].append("Missing quantization_config in config.json")
                validation['valid'] = False
            else:
                quant_config = config['quantization_config']
                if quant_config.get('quant_method') != 'awq':
                    validation['warnings'].append(f"Unexpected quant_method: {quant_config.get('quant_method')}")
        
        return validation
    
    def _load_quantization_metadata(self) -> Dict[str, Any]:
        """Load quantization metadata from source model"""
        metadata_file = self.quantized_path / "quantization_metadata.json"
        
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                return json.load(f)
        
        # Fallback to defaults
        return {
            'bits': 4,
            'group_size': 128,
            'symmetric': False,
            'method': 'awq'
        }
    
    def _convert_to_vllm_naming(self, name: str) -> str:
        """Convert weight names to vLLM conventions"""
        # vLLM expects certain naming patterns
        vllm_name = name
        
        # Convert GLM naming to more standard naming if needed
        replacements = {
            'transformer.': 'model.',
            '.self_attn.': '.attn.',
            'lm_head': 'lm_head',
            'embedding': 'embed_tokens',
        }
        
        for old, new in replacements.items():
            vllm_name = vllm_name.replace(old, new)
        
        return vllm_name
    
    def _extract_layer_index(self, name: str) -> Optional[int]:
        """Extract layer index from weight name"""
        import re
        match = re.search(r'layers?\.(\d+)', name)
        if match:
            return int(match.group(1))
        return None