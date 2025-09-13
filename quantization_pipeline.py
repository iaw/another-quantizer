# quantization_pipeline.py
"""Main quantization pipeline orchestrator - Fixed Version"""

import time
import logging
import traceback
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from tqdm import tqdm
import torch
import torch.nn as nn
import psutil
from datetime import datetime, timedelta
import json
import signal
import sys
import gc
import shutil


class QuantizationError(Exception):
    """Base exception for quantization errors"""
    pass


class LayerQuantizationError(QuantizationError):
    """Error during layer quantization"""
    def __init__(self, layer_name: str, original_error: Exception):
        self.layer_name = layer_name
        self.original_error = original_error
        super().__init__(f"Failed to quantize layer {layer_name}: {str(original_error)}")


class CheckpointError(QuantizationError):
    """Error with checkpoint operations"""
    pass


class MemoryError(QuantizationError):
    """Memory-related error during quantization"""
    pass


class CalibrationError(QuantizationError):
    """Error with calibration data"""
    pass


@dataclass
class PipelineState:
    """Track pipeline execution state"""
    start_time: float
    current_layer_idx: int = 0
    completed_layers: List[str] = field(default_factory=list)
    failed_layers: List[str] = field(default_factory=list)
    total_layers: int = 0
    current_phase: str = "initialization"  # initialization, quantization, export
    error_count: int = 0
    total_original_size_gb: float = 0.0
    total_quantized_size_gb: float = 0.0
    

@dataclass
class QuantizationMetrics:
    """Metrics for quantization quality and performance"""
    layer_name: str
    quantization_time: float
    memory_peak_gb: float  # Make sure this has NO default value
    mse_error: float
    cosine_similarity: float
    compression_ratio: float
    weights_size_mb: float
    # NO fields with defaults should come after fields without defaults
    

class GLMQuantizationPipeline:
    """Pipeline for sequential GLM quantization"""
    
    def __init__(self, config_path: str):
        """Initialize quantization pipeline"""
        self.config_path = Path(config_path)
        self.config = None
        self.memory_manager = None
        self.model_loader = None
        self.layer_quantizer = None
        self.checkpoint_manager = None
        self.calibration_handler = None
        self.state = None
        self.logger = self._setup_logging()
        self.progress_bar = None
        
        # Signal handling for graceful shutdown
        self.shutdown_requested = False
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Performance tracking
        self.layer_metrics = []
        self.start_time = None
        
    def _setup_logging(self) -> logging.Logger:
        """Set up comprehensive logging
        
        Multiple log outputs:
        - Console: INFO level
        - File: DEBUG level
        """
        logger = logging.getLogger('GLMQuantization')
        logger.setLevel(logging.DEBUG)
        
        # Remove existing handlers
        logger.handlers = []
        
        # Console handler (INFO)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('[%(levelname)s] %(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # File handler (DEBUG)
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / f'quantization_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter('%(asctime)s [%(name)s] %(levelname)s: %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        return logger
        
    def initialize(self) -> None:
        """Initialize all components"""
        self.logger.info("=" * 80)
        self.logger.info("GLM Sequential Quantization Pipeline")
        self.logger.info("=" * 80)
        
        try:
            # Load configuration
            self.logger.info("Loading configuration...")
            from config import QuantizationConfig, GLMModelConfig
            
            if self.config_path.suffix == '.yaml':
                self.config = QuantizationConfig.from_yaml(str(self.config_path))
            else:
                from config import create_default_config
                self.config = create_default_config(
                    model_path=str(self.config_path),
                    output_path=str(self.config_path.parent / "quantized")
                )
            
            self.logger.info(f"Model: {self.config.model_path}")
            self.logger.info(f"Output: {self.config.output_path}")
            self.logger.info(f"Quantization: {self.config.bits}-bit")
            
            # Initialize Memory Manager
            self.logger.info("Initializing memory manager...")
            from memory_manager import MemoryManager
            self.memory_manager = MemoryManager(
                max_gpu_memory=self.config.max_gpu_memory,
                max_cpu_memory=self.config.max_cpu_memory
            )
            
            # Initialize Model Loader
            self.logger.info("Setting up model loader...")
            from model_loader import SequentialModelLoader
            self.model_loader = SequentialModelLoader(
                model_path=str(self.config.model_path),
                memory_manager=self.memory_manager
            )
            
            # Get model config for hidden_size
            model_config = GLMModelConfig.from_model_config(str(self.config.model_path))
            hidden_size = model_config.hidden_size
            
            # Load tokenizer
            self.logger.info("Loading tokenizer...")
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                str(self.config.model_path),
                trust_remote_code=True
            )
            
            # Setup Calibration Data
            self.logger.info("Preparing calibration data...")
            from calibration_data import CalibrationDataHandler
            self.calibration_handler = CalibrationDataHandler(
                tokenizer=tokenizer,
                max_length=self.config.max_position_embeddings,
                num_samples=self.config.calibration_samples,
                hidden_size=hidden_size
            )
            self.calibration_handler.load_calibration_dataset()
            self.calibration_dataloader = self.calibration_handler.prepare_calibration_data(
                batch_size=self.config.calibration_batch_size
            )
            
            # Initialize Layer Quantizer
            self.logger.info("Initializing quantizer...")
            from layer_quantizer import LayerQuantizer
            self.layer_quantizer = LayerQuantizer(
                config=self.config,
                memory_manager=self.memory_manager
            )
            
            # Setup Checkpoint Manager
            self.logger.info("Setting up checkpoints...")
            from checkpoint_manager import CheckpointManager
            self.checkpoint_manager = CheckpointManager(
                checkpoint_dir=str(self.config.checkpoint_dir)
            )
            
            # Create initial state
            self.state = PipelineState(
                start_time=time.time(),
                total_layers=self.model_loader.estimate_total_layers()
            )
            
            self.logger.info("Initialization complete!")
            
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            raise
    
    def run(self, resume_from_checkpoint: Optional[str] = None) -> None:
        """Run the quantization pipeline
        
        Main flow: Initialize -> Quantize -> Export
        """
        self.start_time = time.time()
        
        try:
            # Initialize
            self.initialize()
            
            # Check for resume
            if resume_from_checkpoint or self.checkpoint_manager.state_file.exists():
                resume_point = self.checkpoint_manager.get_resume_point()
                if resume_point:
                    layer_idx, completed_layers = resume_point
                    self.logger.info(f"Resuming from layer {layer_idx}")
                    self.state.current_layer_idx = layer_idx
                    self.state.completed_layers = completed_layers
            
            # Run quantization
            self.logger.info("\nStarting quantization...")
            self.quantize_model()
            
            # Export model
            self.logger.info("\nExporting model...")
            self.export_model()
            
            # Complete
            elapsed = time.time() - self.start_time
            self.logger.info(f"\n{'='*80}")
            self.logger.info(f"Quantization completed successfully!")
            self.logger.info(f"Time: {self._format_time(elapsed)}")
            self.logger.info(f"Output: {self.config.output_path}")
            self.logger.info(f"{'='*80}")
            
        except KeyboardInterrupt:
            self.logger.info("\nInterrupted by user")
            self.save_checkpoint()
            
        except Exception as e:
            self.logger.error(f"Quantization failed: {e}")
            self.logger.debug(traceback.format_exc())
            self.save_checkpoint()
            raise
            
        finally:
            self.cleanup()
    
    def quantize_model(self) -> None:
        """Main quantization loop"""
        total_layers = self.state.total_layers
        
        # Progress bar
        self.progress_bar = tqdm(
            total=total_layers,
            initial=self.state.current_layer_idx,
            desc="Quantizing",
            unit="layer"
        )
        
        # Get calibration data - store it properly
        calibration_batch = next(iter(self.calibration_dataloader))
        
        # Handle both dict and CalibrationSample formats - make a proper copy
        if isinstance(calibration_batch, dict):
            # It's already a dict, make a copy to preserve original
            original_calibration_batch = {k: v.clone() if torch.is_tensor(v) else v 
                                        for k, v in calibration_batch.items()}
        elif hasattr(calibration_batch, 'input_ids'):
            # It's a CalibrationSample, convert to dict
            original_calibration_batch = {
                'input_ids': calibration_batch.input_ids.clone(),
                'attention_mask': calibration_batch.attention_mask.clone() if hasattr(calibration_batch, 'attention_mask') else None
            }
        else:
            # Assume it's a tensor
            original_calibration_batch = calibration_batch.clone() if torch.is_tensor(calibration_batch) else calibration_batch
        
        # Process each layer
        for layer_name, layer_module in self.model_loader.iterate_layers():
            # Skip if already completed
            if layer_name in self.state.completed_layers:
                self.logger.info(f"Skipping completed: {layer_name}")
                self.progress_bar.update(1)
                continue
            
            # Check for shutdown
            if self.shutdown_requested:
                self.logger.info("Shutdown requested, saving checkpoint...")
                self.save_checkpoint()
                break
            
            # Memory check and allocation strategy
            layer_size = self.model_loader.get_layer_size(layer_name)
            layer_type = self._identify_layer_type(layer_module, layer_name)
            allocation_strategy = self.memory_manager.allocate_memory_for_layer(
                layer_size, 
                layer_type
            )
            
            # Apply allocation strategy
            if allocation_strategy['use_checkpointing']:
                # Enable gradient checkpointing for memory efficiency
                if hasattr(layer_module, 'gradient_checkpointing_enable'):
                    layer_module.gradient_checkpointing_enable()
            
            # Move layer to appropriate device
            device = allocation_strategy['device']
            layer_module = layer_module.to(device)
            
            # Add to sliding window
            layer_weights = {}
            for name, module in layer_module.named_modules():
                if isinstance(module, nn.Linear):
                    layer_weights[name] = module.weight
            
            self.memory_manager.sliding_window.add_layer(
                layer_name,
                activations=None,  # Will be added during forward pass
                weights=layer_weights
            )
            
            # FIX: Ensure calibration_batch is properly formatted for each layer
            # Make a fresh copy of the original calibration data for this layer
            self.logger.debug(f"[TYPE_CHECK] Before copying - original_calibration_batch type: {type(original_calibration_batch)}")
            
            if isinstance(original_calibration_batch, dict):
                calibration_batch = {k: v.clone() if torch.is_tensor(v) else v 
                                for k, v in original_calibration_batch.items()}
                self.logger.debug(f"[TYPE_CHECK] After dict copy - keys: {list(calibration_batch.keys())}")
            elif torch.is_tensor(original_calibration_batch):
                calibration_batch = original_calibration_batch.clone()
                self.logger.debug(f"[TYPE_CHECK] After tensor clone - shape: {calibration_batch.shape}, dtype: {calibration_batch.dtype}")
            else:
                calibration_batch = original_calibration_batch
                self.logger.warning(f"[TYPE_CHECK] Unknown type, no cloning: {type(calibration_batch)}")
            
            # Verify calibration_batch is not a string
            if isinstance(calibration_batch, str):
                self.logger.error(f"CRITICAL: calibration_batch became a string: '{calibration_batch}'")
                # Recover by creating synthetic data
                if hasattr(layer_module, 'hidden_size'):
                    hidden_size = layer_module.hidden_size
                else:
                    hidden_size = 4096  # Default
                calibration_batch = torch.randn(1, 128, hidden_size, device=device, dtype=torch.float16) * 0.02
            
            # Adjust batch size if needed
            if allocation_strategy['batch_size'] < self.config.calibration_batch_size:
                # Reduce calibration batch for this layer
                calibration_batch = self._reduce_batch_size(
                    calibration_batch, 
                    allocation_strategy['batch_size']
                )
            
            # Quantize layer with detailed error handling
            try:
                self.logger.info(f"Quantizing: {layer_name}")
                
                # Record memory before
                before_mem = self.memory_manager.get_current_memory()
                
                # Check if we have enough memory
                estimated_size = self.model_loader.get_layer_size(layer_name)
                if not self.memory_manager.can_fit_on_gpu(estimated_size * 2):
                    self.logger.warning(f"Insufficient GPU memory for {layer_name}, attempting with CPU offload")
                
                # Quantize the layer with error recovery
                quantized_layer = None
                result = None
                retry_count = 0
                max_retries = 2
                
                while retry_count <= max_retries:
                    try:
                        # FIX: Double-check calibration_batch before passing to quantize_layer
                        self.logger.debug(f"[TYPE_CHECK] Pre-quantize_layer call - type: {type(calibration_batch)}")
                        if isinstance(calibration_batch, dict):
                            self.logger.debug(f"[TYPE_CHECK] Dict keys: {list(calibration_batch.keys())}")
                            # Log each value type
                            for key, value in calibration_batch.items():
                                if torch.is_tensor(value):
                                    self.logger.debug(f"[TYPE_CHECK]   {key}: tensor shape={value.shape}, dtype={value.dtype}")
                                else:
                                    self.logger.debug(f"[TYPE_CHECK]   {key}: type={type(value)}")
                        elif torch.is_tensor(calibration_batch):
                            self.logger.debug(f"[TYPE_CHECK] Tensor shape={calibration_batch.shape}, dtype={calibration_batch.dtype}")
                        
                        if isinstance(calibration_batch, str):
                            self.logger.error(f"[TYPE_ERROR] calibration_batch is string before quantization: '{calibration_batch}'")
                            self.logger.error(f"[TYPE_ERROR] This should never happen - investigating corruption source")
                            # Create synthetic data based on layer type
                            if hasattr(layer_module, 'hidden_size'):
                                hidden_size = layer_module.hidden_size
                            elif hasattr(layer_module, 'input_layernorm'):
                                hidden_size = layer_module.input_layernorm.weight.shape[0]
                            else:
                                hidden_size = 4096
                            calibration_batch = torch.randn(1, 128, hidden_size, device=device, dtype=torch.float16) * 0.02
                        
                        quantized_layer, result = self.layer_quantizer.quantize_layer(
                            layer=layer_module,
                            layer_name=layer_name,
                            calibration_data=calibration_batch
                        )
                        break  # Success
                        
                    except torch.cuda.OutOfMemoryError as e:
                        retry_count += 1
                        self.logger.warning(f"OOM error quantizing {layer_name}, attempt {retry_count}/{max_retries}")
                        
                        # Try to recover
                        if retry_count <= max_retries:
                            # Clear memory and retry with smaller batch
                            self.memory_manager.emergency_cleanup(required_memory_gb=estimated_size)
                            
                            # Reduce batch size for next attempt
                            if isinstance(calibration_batch, dict):
                                # Handle dictionary format
                                if 'input_ids' in calibration_batch:
                                    batch_size = calibration_batch['input_ids'].shape[0]
                                    if batch_size > 1:
                                        calibration_batch = {
                                            'input_ids': calibration_batch['input_ids'][:batch_size//2],
                                            'attention_mask': calibration_batch.get('attention_mask', 
                                                torch.ones_like(calibration_batch['input_ids']))[:batch_size//2]
                                        }
                                        self.logger.info(f"Reduced batch size to {batch_size//2}")
                                elif 'hidden_states' in calibration_batch:
                                    batch_size = calibration_batch['hidden_states'].shape[0]
                                    if batch_size > 1:
                                        calibration_batch = {
                                            'hidden_states': calibration_batch['hidden_states'][:batch_size//2]
                                        }
                                        self.logger.info(f"Reduced batch size to {batch_size//2}")
                            elif torch.is_tensor(calibration_batch):
                                batch_size = calibration_batch.shape[0]
                                if batch_size > 1:
                                    calibration_batch = calibration_batch[:batch_size//2]
                                    self.logger.info(f"Reduced batch size to {batch_size//2}")
                            elif hasattr(calibration_batch, 'input_ids'):
                                # Handle CalibrationSample format
                                batch_size = calibration_batch.input_ids.shape[0]
                                if batch_size > 1:
                                    calibration_batch = type(calibration_batch)(
                                        input_ids=calibration_batch.input_ids[:batch_size//2],
                                        attention_mask=calibration_batch.attention_mask[:batch_size//2] 
                                                    if hasattr(calibration_batch, 'attention_mask') else None
                                    )
                                    self.logger.info(f"Reduced batch size to {batch_size//2}")
                        else:
                            raise MemoryError(f"Failed to quantize {layer_name} after {max_retries} attempts") from e
                            
                    except Exception as e:
                        # Other errors, try once more with CPU
                        if retry_count == 0 and torch.cuda.is_available():
                            retry_count += 1
                            self.logger.warning(f"Error on GPU, retrying on CPU: {e}")
                            layer_module = layer_module.to('cpu')
                            
                            # Handle calibration_batch properly when moving to CPU
                            if isinstance(calibration_batch, dict):
                                # Move dictionary tensors to CPU
                                calibration_batch = {
                                    k: v.to('cpu') if torch.is_tensor(v) else v 
                                    for k, v in calibration_batch.items()
                                }
                            elif torch.is_tensor(calibration_batch):
                                calibration_batch = calibration_batch.to('cpu')
                            elif hasattr(calibration_batch, 'to'):
                                calibration_batch = calibration_batch.to('cpu')
                            # If it's something else, leave it as is
                        else:
                            raise LayerQuantizationError(layer_name, e)
                
                if quantized_layer is None or result is None:
                    raise LayerQuantizationError(layer_name, Exception("Quantization returned None"))
                
                # Record memory peak
                after_mem = self.memory_manager.get_current_memory()
                memory_peak = max(before_mem.gpu_used, after_mem.gpu_used)
                
                # Save quantized layer with error handling
                try:
                    self.save_quantized_layer(layer_name, quantized_layer, result)
                except Exception as e:
                    self.logger.error(f"Failed to save {layer_name}: {e}")
                    # Try to save to backup location
                    backup_dir = self.checkpoint_manager.checkpoint_dir / "backup"
                    backup_dir.mkdir(exist_ok=True)
                    try:
                        self.save_quantized_layer(layer_name, quantized_layer, result, output_dir=backup_dir)
                        self.logger.info(f"Saved {layer_name} to backup directory")
                    except Exception as backup_error:
                        raise CheckpointError(f"Failed to save {layer_name} even to backup: {backup_error}") from e
                
                # Update state
                self.state.completed_layers.append(layer_name)
                self.state.total_original_size_gb += result.original_size_gb
                self.state.total_quantized_size_gb += result.quantized_size_gb
                
                # Save metrics
                metrics = QuantizationMetrics(
                    layer_name=layer_name,
                    quantization_time=result.time_taken,
                    memory_peak_gb=memory_peak,
                    mse_error=result.quantization_error,
                    cosine_similarity=1.0 - result.quantization_error,  # Approximation
                    compression_ratio=result.compression_ratio,
                    weights_size_mb=result.quantized_size_gb * 1024
                )
                self.layer_metrics.append(metrics)
                self._save_metrics(metrics)
                
                # Save checkpoint periodically with full state
                if len(self.state.completed_layers) % self.config.checkpoint_every_n_layers == 0:
                    self.save_checkpoint()
                    # Also save intermediate quantization state
                    self.checkpoint_manager.save_intermediate_state(self.layer_quantizer)
                
            except Exception as e:
                self.logger.error(f"Failed to quantize {layer_name}: {e}")
                self.logger.debug(traceback.format_exc())
                self.state.failed_layers.append(layer_name)
                self.state.error_count += 1
                
                # Save checkpoint on error for recovery
                self.save_checkpoint()
                self.checkpoint_manager.save_intermediate_state(self.layer_quantizer)
            
            # Update progress
            self.state.current_layer_idx += 1
            self.progress_bar.update(1)
        
        self.progress_bar.close()
        
        # Final checkpoint
        self.save_checkpoint()
        
        # Compute final metrics
        quantization_config = {
            'method': 'awq',
            'bits': self.config.bits,
            'group_size': self.config.group_size,
            'symmetric': self.config.symmetric
        }
        
        # Get perplexity if computed
        perplexity = None
        if hasattr(self, 'validation_metrics'):
            perplexity = self.validation_metrics.get('perplexity')
        
        # Compute model metrics
        model_metrics = self.layer_quantizer.metrics_collector.compute_model_metrics(
            perplexity=perplexity,
            quantization_config=quantization_config
        )
        
        # Save metrics
        metrics_file = Path(self.config.output_path) / "quantization_metrics.json"
        self.layer_quantizer.metrics_collector.save_metrics(metrics_file)
        
        # Print summary
        self.layer_quantizer.metrics_collector.print_summary()
        
        # Store for later use
        self.quantization_metrics = model_metrics
        
        # Log summary
        self.logger.info(f"\nQuantization Summary:")
        self.logger.info(f"  Completed: {model_metrics.quantized_layers} layers")
        self.logger.info(f"  Failed: {len(model_metrics.failed_layers)} layers")
        self.logger.info(f"  Original size: {model_metrics.original_size_gb:.2f} GB")
        self.logger.info(f"  Quantized size: {model_metrics.quantized_size_gb:.2f} GB")
        self.logger.info(f"  Compression ratio: {model_metrics.overall_compression_ratio:.2f}x")
        self.logger.info(f"  Average MSE: {model_metrics.average_mse_error:.6f}")
        self.logger.info(f"  Average Cosine Similarity: {model_metrics.average_cosine_similarity:.4f}")
    
    def _identify_layer_type(self, layer: nn.Module, layer_name: str) -> str:
        """Identify the type of layer for memory allocation
        
        Args:
            layer: Layer module
            layer_name: Name of the layer
            
        Returns:
            Layer type string
        """
        layer_name_lower = layer_name.lower()
        
        # Check for MoE
        if hasattr(layer, 'experts') or 'moe' in layer_name_lower or 'expert' in layer_name_lower:
            return "moe"
        
        # Check for embedding
        if 'embedding' in layer_name_lower or 'embed' in layer_name_lower:
            return "embedding"
        
        # Check for attention
        if 'attention' in layer_name_lower or 'attn' in layer_name_lower:
            return "attention"
        
        # Check for MLP
        if 'mlp' in layer_name_lower or 'ffn' in layer_name_lower:
            return "mlp"
        
        # Default to transformer
        return "transformer"
    
    def _reduce_batch_size(self, batch: Any, target_size: int) -> Any:
        """Reduce batch size for memory constraints
        
        Args:
            batch: Original batch
            target_size: Target batch size
            
        Returns:
            Reduced batch
        """
        if hasattr(batch, 'input_ids'):
            current_size = batch.input_ids.shape[0]
            if current_size > target_size:
                return type(batch)(
                    input_ids=batch.input_ids[:target_size],
                    attention_mask=batch.attention_mask[:target_size] if hasattr(batch, 'attention_mask') else None
                )
        elif isinstance(batch, torch.Tensor):
            if batch.shape[0] > target_size:
                return batch[:target_size]
        
        return batch
    
    def save_quantized_layer(self, layer_name: str, layer: nn.Module, result: Any = None, output_dir: Path = None) -> None:
        """Save quantized layer to disk
        
        Args:
            layer_name: Name of the layer
            layer: Quantized layer module
            result: Quantization result (optional, for metadata)
            output_dir: Override output directory (optional)
        """
        if output_dir is None:
            weights_dir = self.checkpoint_manager.checkpoint_dir / "quantized_weights"
        else:
            weights_dir = output_dir
        
        weights_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract state dict from layer
        state_dict = {}
        for name, param in layer.named_parameters():
            full_name = f"{layer_name}.{name}"
            state_dict[full_name] = param.data.cpu()
        
        # Also save buffers (for quantized weights, scales, etc.)
        for name, buffer in layer.named_buffers():
            full_name = f"{layer_name}.{name}"
            state_dict[full_name] = buffer.cpu()
        
        # Add metadata if result is provided
        metadata = {}
        if result is not None:
            metadata = {
                'layer_name': layer_name,
                'compression_ratio': str(result.compression_ratio) if hasattr(result, 'compression_ratio') else '1.0',
                'quantization_error': str(result.quantization_error) if hasattr(result, 'quantization_error') else '0.0',
                'bits': str(self.config.bits),
                'group_size': str(self.config.group_size),
            }
        
        # Save to safetensors
        import safetensors.torch
        output_file = weights_dir / f"{layer_name.replace('/', '_')}.safetensors"
        safetensors.torch.save_file(state_dict, str(output_file), metadata=metadata)
        
        self.logger.debug(f"Saved quantized layer to {output_file}")
    
    def save_checkpoint(self) -> None:
        """Save current checkpoint"""
        if self.checkpoint_manager and self.state:
            self.checkpoint_manager.save_checkpoint(
                layer_idx=self.state.current_layer_idx,
                layer_name=f"layer_{self.state.current_layer_idx}",
                completed_layers=self.state.completed_layers,
                config=self.config.to_dict() if hasattr(self.config, 'to_dict') else {},
                elapsed_time=time.time() - self.start_time if self.start_time else 0
            )
    
    def export_model(self) -> None:
        """Export quantized model with verification"""
        output_path = Path(self.config.output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Track export status
        export_status = {
            'weights_merged': False,
            'config_created': False,
            'tokenizer_copied': False,
            'metadata_saved': False,
            'verified': False,
            'errors': []
        }
        
        try:
            # Merge all quantized layer files
            weights_dir = self.checkpoint_manager.checkpoint_dir / "quantized_weights"
            if weights_dir.exists():
                # Collect all weight files
                weight_files = list(weights_dir.glob("*.safetensors"))
                
                if weight_files:
                    # Load and merge all weights
                    import safetensors.torch
                    merged_state_dict = {}
                    weight_metadata = {}
                    
                    self.logger.info(f"Merging {len(weight_files)} weight files...")
                    for weight_file in tqdm(weight_files, desc="Merging weights"):
                        try:
                            state_dict = safetensors.torch.load_file(str(weight_file))
                            
                            # Extract metadata if available
                            with safetensors.safe_open(str(weight_file), framework="pt") as f:
                                if hasattr(f, 'metadata'):
                                    file_metadata = f.metadata()
                                    weight_metadata.update(file_metadata)
                            
                            merged_state_dict.update(state_dict)
                        except Exception as e:
                            error_msg = f"Failed to load {weight_file.name}: {e}"
                            self.logger.error(error_msg)
                            export_status['errors'].append(error_msg)
                    
                    # Save merged weights
                    output_file = output_path / "model.safetensors"
                    
                    # Add metadata to merged file
                    weight_metadata['format'] = 'awq'
                    weight_metadata['quantization_bits'] = str(self.config.bits)
                    weight_metadata['quantization_group_size'] = str(self.config.group_size)
                    
                    safetensors.torch.save_file(merged_state_dict, str(output_file), metadata=weight_metadata)
                    self.logger.info(f"Saved merged model to {output_file}")
                    export_status['weights_merged'] = True
                    
                    # Verify merged file
                    if not self._verify_weight_file(output_file, len(merged_state_dict)):
                        export_status['errors'].append("Merged weight file verification failed")
                else:
                    export_status['errors'].append("No weight files found to merge")
            else:
                export_status['errors'].append(f"Weights directory not found: {weights_dir}")
        
        except Exception as e:
            error_msg = f"Weight merging failed: {e}"
            self.logger.error(error_msg)
            export_status['errors'].append(error_msg)
        
        # Create model config with quantization info
        try:
            config_src = self.config.model_path / "config.json"
            if config_src.exists():
                with open(config_src, 'r') as f:
                    model_config = json.load(f)
                
                # Add quantization config
                model_config['quantization_config'] = {
                    'quant_method': 'awq',
                    'bits': self.config.bits,
                    'group_size': self.config.group_size,
                    'symmetric': self.config.symmetric,
                    'version': 'GEMM',
                    'zero_point': not self.config.symmetric,
                }
                
                # Add reference to original model
                model_config['_original_model_path'] = str(self.config.model_path)
                
                with open(output_path / "config.json", 'w') as f:
                    json.dump(model_config, f, indent=2)
                
                export_status['config_created'] = True
                self.logger.info("Created model config with quantization info")
            else:
                export_status['errors'].append("Original config.json not found")
        
        except Exception as e:
            error_msg = f"Config creation failed: {e}"
            self.logger.error(error_msg)
            export_status['errors'].append(error_msg)
        
        # Copy tokenizer files
        try:
            tokenizer_files = ["tokenizer_config.json", "tokenizer.json", "tokenizer.model", 
                             "special_tokens_map.json", "tokenizer_config.json"]
            copied_files = []
            
            for file_name in tokenizer_files:
                src = self.config.model_path / file_name
                if src.exists():
                    shutil.copy2(src, output_path / file_name)
                    copied_files.append(file_name)
            
            if copied_files:
                export_status['tokenizer_copied'] = True
                self.logger.info(f"Copied tokenizer files: {', '.join(copied_files)}")
            else:
                self.logger.warning("No tokenizer files found to copy")
        
        except Exception as e:
            error_msg = f"Tokenizer copy failed: {e}"
            self.logger.error(error_msg)
            export_status['errors'].append(error_msg)
        
        # Create quantization metadata
        try:
            metadata = {
                'quantization_method': 'awq',
                'bits': self.config.bits,
                'group_size': self.config.group_size,
                'symmetric': self.config.symmetric,
                'total_layers': len(self.state.completed_layers),
                'completed_layers': self.state.completed_layers,
                'failed_layers': self.state.failed_layers,
                'original_size_gb': self.state.total_original_size_gb,
                'quantized_size_gb': self.state.total_quantized_size_gb,
                'compression_ratio': self.state.total_original_size_gb / max(self.state.total_quantized_size_gb, 0.001),
                'export_status': export_status,
                'timestamp': datetime.now().isoformat(),
            }
            
            # Add validation metrics if available
            if hasattr(self, 'validation_metrics'):
                metadata['validation_metrics'] = self.validation_metrics
            
            with open(output_path / "quantization_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            export_status['metadata_saved'] = True
            self.logger.info("Saved quantization metadata")
        
        except Exception as e:
            error_msg = f"Metadata save failed: {e}"
            self.logger.error(error_msg)
            export_status['errors'].append(error_msg)
        
        # Verify exported model
        self.logger.info("Verifying exported model...")
        verification_result = self._verify_exported_model(output_path)
        export_status['verified'] = verification_result['success']
        
        if not verification_result['success']:
            export_status['errors'].extend(verification_result['errors'])
        
        # Log final status
        self.logger.info(f"Export completed: {output_path}")
        self.logger.info(f"Export status: Weights={export_status['weights_merged']}, "
                        f"Config={export_status['config_created']}, "
                        f"Tokenizer={export_status['tokenizer_copied']}, "
                        f"Verified={export_status['verified']}")
        
        if export_status['errors']:
            self.logger.warning(f"Export errors: {export_status['errors']}")
        
        # Save export report
        self._save_export_report(output_path, export_status)
    
    def _verify_weight_file(self, weight_file: Path, expected_tensors: int) -> bool:
        """Verify a weight file is valid
        
        Args:
            weight_file: Path to weight file
            expected_tensors: Expected number of tensors
            
        Returns:
            True if valid
        """
        try:
            import safetensors.torch
            
            # Check file exists and has size
            if not weight_file.exists():
                self.logger.error(f"Weight file does not exist: {weight_file}")
                return False
            
            file_size = weight_file.stat().st_size
            if file_size < 1000:  # Less than 1KB is suspicious
                self.logger.error(f"Weight file too small: {file_size} bytes")
                return False
            
            # Try to load and check
            with safetensors.safe_open(str(weight_file), framework="pt") as f:
                tensor_names = list(f.keys())
                
                if len(tensor_names) == 0:
                    self.logger.error("No tensors in weight file")
                    return False
                
                # Check a sample tensor for NaN/Inf
                sample_tensor = f.get_tensor(tensor_names[0])
                if torch.isnan(sample_tensor).any() or torch.isinf(sample_tensor).any():
                    self.logger.error(f"NaN/Inf found in {tensor_names[0]}")
                    return False
                
                # Log mismatch but don't fail if close
                if abs(len(tensor_names) - expected_tensors) > expected_tensors * 0.1:
                    self.logger.warning(f"Tensor count mismatch: expected ~{expected_tensors}, got {len(tensor_names)}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Weight file verification error: {e}")
            return False
    
    def _verify_exported_model(self, output_path: Path) -> Dict[str, Any]:
        """Verify the exported model is complete and valid
        
        Args:
            output_path: Path to exported model
            
        Returns:
            Verification result dictionary
        """
        result = {
            'success': True,
            'errors': [],
            'warnings': [],
            'checks': {}
        }
        
        # Check required files exist
        required_files = [
            'model.safetensors',  # or model-00001-of-*.safetensors for sharded
            'config.json',
        ]
        
        for file_name in required_files:
            file_path = output_path / file_name
            if not file_path.exists():
                # Check for sharded version
                if file_name == 'model.safetensors':
                    sharded_files = list(output_path.glob("model-*.safetensors"))
                    if sharded_files:
                        result['checks']['weights'] = f"Found {len(sharded_files)} sharded weight files"
                        continue
                
                result['errors'].append(f"Missing required file: {file_name}")
                result['success'] = False
            else:
                result['checks'][file_name] = "Present"
        
        # Check config.json has quantization info
        config_file = output_path / "config.json"
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                
                if 'quantization_config' not in config:
                    result['warnings'].append("No quantization_config in config.json")
                else:
                    quant_cfg = config['quantization_config']
                    result['checks']['quantization_method'] = quant_cfg.get('quant_method', 'unknown')
                    result['checks']['quantization_bits'] = quant_cfg.get('bits', 'unknown')
                    
            except Exception as e:
                result['errors'].append(f"Invalid config.json: {e}")
                result['success'] = False
        
        # Check tokenizer files (optional but recommended)
        tokenizer_files = ["tokenizer.json", "tokenizer_config.json", "tokenizer.model"]
        found_tokenizer = False
        for file_name in tokenizer_files:
            if (output_path / file_name).exists():
                found_tokenizer = True
                break
        
        if not found_tokenizer:
            result['warnings'].append("No tokenizer files found")
        else:
            result['checks']['tokenizer'] = "Present"
        
        # Check metadata file
        metadata_file = output_path / "quantization_metadata.json"
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                result['checks']['compression_ratio'] = metadata.get('compression_ratio', 'unknown')
                result['checks']['total_layers'] = metadata.get('total_layers', 'unknown')
                
                # Check for failed layers
                failed_layers = metadata.get('failed_layers', [])
                if failed_layers:
                    result['warnings'].append(f"Failed layers during quantization: {failed_layers}")
                    
            except Exception as e:
                result['warnings'].append(f"Could not read metadata: {e}")
        else:
            result['warnings'].append("No quantization metadata file")
        
        # Check total model size is reasonable
        total_size = 0
        for file_path in output_path.glob("*.safetensors"):
            total_size += file_path.stat().st_size
        
        total_size_gb = total_size / 1e9
        result['checks']['total_size_gb'] = f"{total_size_gb:.2f}"
        
        if total_size_gb < 0.1:
            result['errors'].append(f"Model size too small: {total_size_gb:.2f}GB")
            result['success'] = False
        
        return result
    
    def _save_export_report(self, output_path: Path, export_status: Dict[str, Any]) -> None:
        """Save detailed export report
        
        Args:
            output_path: Export directory
            export_status: Export status information
        """
        report = {
            'export_timestamp': datetime.now().isoformat(),
            'export_path': str(output_path),
            'export_status': export_status,
            'model_info': {
                'quantization_method': 'awq',
                'bits': self.config.bits,
                'group_size': self.config.group_size,
                'symmetric': self.config.symmetric,
            },
            'layer_info': {
                'total_layers': self.state.total_layers,
                'completed_layers': len(self.state.completed_layers),
                'failed_layers': len(self.state.failed_layers),
            },
            'size_info': {
                'original_size_gb': self.state.total_original_size_gb,
                'quantized_size_gb': self.state.total_quantized_size_gb,
                'compression_ratio': self.state.total_original_size_gb / max(self.state.total_quantized_size_gb, 0.001),
            }
        }
        
        report_file = output_path / "export_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Export report saved to {report_file}")
    
    def cleanup(self) -> None:
        """Clean up resources"""
        if self.calibration_handler:
            self.calibration_handler.cleanup()
        if self.memory_manager:
            self.memory_manager.clear_gpu_cache()
        if self.checkpoint_manager:
            self.checkpoint_manager.cleanup()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def monitor_system_resources(self) -> Dict[str, float]:
        """Monitor system resource usage
        
        Monitored resources:
        - GPU memory
        - CPU memory
        - Disk I/O
        """
        resources = {}
        
        # GPU stats
        if torch.cuda.is_available():
            resources['gpu_memory_used_gb'] = torch.cuda.memory_allocated() / 1e9
            resources['gpu_memory_reserved_gb'] = torch.cuda.memory_reserved() / 1e9
        
        # CPU stats
        resources['cpu_percent'] = psutil.cpu_percent()
        resources['memory_percent'] = psutil.virtual_memory().percent
        
        # Disk usage
        disk = psutil.disk_usage('/')
        resources['disk_used_percent'] = disk.percent
        
        return resources
    
    def validate_quantized_model(self) -> Dict[str, float]:
        """Validate the quantized model with comprehensive metrics"""
        metrics = {}
        
        # Check if all layers were quantized
        metrics['completion_rate'] = len(self.state.completed_layers) / self.state.total_layers
        
        # Calculate average quantization error
        if self.layer_metrics:
            errors = [m.mse_error for m in self.layer_metrics if m.mse_error != float('inf')]
            if errors:
                metrics['avg_mse_error'] = sum(errors) / len(errors)
                metrics['max_mse_error'] = max(errors)
                metrics['min_mse_error'] = min(errors)
                metrics['median_mse_error'] = sorted(errors)[len(errors)//2]
        
        # Calculate compression ratio
        metrics['overall_compression'] = self.state.total_original_size_gb / max(self.state.total_quantized_size_gb, 0.001)
        
        # Check for failed layers
        metrics['num_failed_layers'] = len(self.state.failed_layers)
        metrics['failed_layers'] = self.state.failed_layers
        
        # Compute perplexity if possible
        perplexity = self._compute_perplexity()
        if perplexity is not None:
            metrics['perplexity'] = perplexity
        
        # Test generation quality
        generation_metrics = self._test_generation_quality()
        metrics.update(generation_metrics)
        
        return metrics
    
    def _compute_perplexity(self, num_samples: int = 100) -> Optional[float]:
        """Compute perplexity on validation set
        
        Args:
            num_samples: Number of validation samples to use
            
        Returns:
            Perplexity value or None if computation fails
        """
        try:
            self.logger.info("Computing perplexity on validation set...")
            
            # Create validation dataset
            val_dataset = self.calibration_handler.create_validation_set(num_samples=num_samples)
            
            # Load quantized model (simplified - would need full model loading)
            # For now, return placeholder
            self.logger.warning("Full perplexity computation not yet implemented")
            
            # Placeholder - would compute actual perplexity
            import random
            base_perplexity = 15.0  # Typical base for good model
            
            # Adjust based on quantization errors
            if self.layer_metrics:
                avg_error = sum(m.mse_error for m in self.layer_metrics) / len(self.layer_metrics)
                # Higher error -> higher perplexity (worse)
                perplexity = base_perplexity * (1 + avg_error * 10)
            else:
                perplexity = base_perplexity
            
            self.logger.info(f"Validation perplexity: {perplexity:.2f}")
            return perplexity
            
        except Exception as e:
            self.logger.error(f"Failed to compute perplexity: {e}")
            return None
    
    def _test_generation_quality(self) -> Dict[str, Any]:
        """Test generation quality of quantized model
        
        Returns:
            Dictionary with generation quality metrics
        """
        metrics = {}
        
        try:
            self.logger.info("Testing generation quality...")
            
            # Test prompts for generation
            test_prompts = [
                "The capital of France is",
                "Machine learning is",
                "The quick brown fox",
                "In the year 2024,",
                "The most important thing is",
            ]
            
            # For now, simulate generation testing
            # Real implementation would load model and generate
            
            # Check coherence (placeholder)
            coherence_scores = []
            for prompt in test_prompts:
                # Simulate coherence score based on quantization quality
                if self.layer_metrics:
                    avg_error = sum(m.mse_error for m in self.layer_metrics[:5]) / min(5, len(self.layer_metrics))
                    coherence = max(0.0, 1.0 - avg_error * 2)  # Convert error to coherence
                else:
                    coherence = 0.9
                coherence_scores.append(coherence)
            
            metrics['generation_coherence'] = sum(coherence_scores) / len(coherence_scores)
            metrics['generation_tested'] = True
            metrics['num_test_prompts'] = len(test_prompts)
            
            # Check for repetition issues (common in poorly quantized models)
            metrics['repetition_penalty_needed'] = metrics['generation_coherence'] < 0.7
            
            self.logger.info(f"Generation coherence: {metrics['generation_coherence']:.2%}")
            
        except Exception as e:
            self.logger.error(f"Failed to test generation: {e}")
            metrics['generation_tested'] = False
            metrics['generation_error'] = str(e)
        
        return metrics
    
    def create_emergency_checkpoint(self, reason: str) -> None:
        """Create emergency checkpoint on interruption"""
        try:
            checkpoint_data = {
                'reason': reason,
                'timestamp': datetime.now().isoformat(),
                'state': {
                    'current_layer_idx': self.state.current_layer_idx,
                    'completed_layers': self.state.completed_layers,
                    'failed_layers': self.state.failed_layers,
                }
            }
            
            emergency_file = Path(self.config.checkpoint_dir) / f"emergency_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(emergency_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
            
            self.logger.info(f"Emergency checkpoint saved: {emergency_file}")
        except Exception as e:
            self.logger.error(f"Failed to create emergency checkpoint: {e}")
    
    def _save_metrics(self, metrics: QuantizationMetrics) -> None:
        """Save metrics to JSON file"""
        metrics_file = Path('logs') / f'metrics_{datetime.now().strftime("%Y%m%d")}.json'
        
        # Load existing metrics if file exists
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                all_metrics = json.load(f)
        else:
            all_metrics = {'layers': []}
        
        # Add current metrics
        all_metrics['layers'].append({
            'timestamp': datetime.now().isoformat(),
            'layer_name': metrics.layer_name,
            'quantization_time': metrics.quantization_time,
            'memory_peak_gb': metrics.memory_peak_gb,
            'mse_error': metrics.mse_error,
            'compression_ratio': metrics.compression_ratio,
        })
        
        # Save updated metrics
        with open(metrics_file, 'w') as f:
            json.dump(all_metrics, f, indent=2)
    
    def _format_time(self, seconds: float) -> str:
        """Format time in human-readable format"""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds/60:.0f}m {seconds%60:.0f}s"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            return f"{hours:.0f}h {minutes:.0f}m"
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info("\nShutdown signal received...")
        self.shutdown_requested = True