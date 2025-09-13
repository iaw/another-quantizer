# pipeline.py
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
    memory_peak_gb: float
    mse_error: float
    cosine_similarity: float
    compression_ratio: float
    weights_size_mb: float
    

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
        
        # Get calibration data
        calibration_batch = next(iter(self.calibration_dataloader))
        
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
            
            # Quantize layer
            try:
                self.logger.info(f"Quantizing: {layer_name}")
                
                # Record memory before
                before_mem = self.memory_manager.get_current_memory()
                
                # Quantize the layer (now properly works with nn.Module)
                quantized_layer, result = self.layer_quantizer.quantize_layer(
                    layer=layer_module,
                    layer_name=layer_name,
                    calibration_data=calibration_batch
                )
                
                # Record memory peak
                after_mem = self.memory_manager.get_current_memory()
                memory_peak = max(before_mem.gpu_used, after_mem.gpu_used)
                
                # Save quantized layer
                self.save_quantized_layer(layer_name, quantized_layer)
                
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
                
                # Save checkpoint periodically
                if len(self.state.completed_layers) % self.config.checkpoint_every_n_layers == 0:
                    self.save_checkpoint()
                
            except Exception as e:
                self.logger.error(f"Failed to quantize {layer_name}: {e}")
                self.logger.debug(traceback.format_exc())
                self.state.failed_layers.append(layer_name)
                self.state.error_count += 1
            
            # Update progress
            self.state.current_layer_idx += 1
            self.progress_bar.update(1)
        
        self.progress_bar.close()
        
        # Final checkpoint
        self.save_checkpoint()
        
        # Log summary
        self.logger.info(f"\nQuantization Summary:")
        self.logger.info(f"  Completed: {len(self.state.completed_layers)} layers")
        self.logger.info(f"  Failed: {len(self.state.failed_layers)} layers")
        self.logger.info(f"  Original size: {self.state.total_original_size_gb:.2f} GB")
        self.logger.info(f"  Quantized size: {self.state.total_quantized_size_gb:.2f} GB")
        compression_ratio = self.state.total_original_size_gb / max(self.state.total_quantized_size_gb, 0.001)
        self.logger.info(f"  Compression ratio: {compression_ratio:.2f}x")
    
    def save_quantized_layer(self, layer_name: str, layer: nn.Module) -> None:
        """Save quantized layer to disk"""
        weights_dir = self.checkpoint_manager.checkpoint_dir / "quantized_weights"
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
        
        # Save to safetensors
        import safetensors.torch
        output_file = weights_dir / f"{layer_name.replace('/', '_')}.safetensors"
        safetensors.torch.save_file(state_dict, str(output_file))
        
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
        """Export quantized model"""
        output_path = Path(self.config.output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Merge all quantized layer files
        weights_dir = self.checkpoint_manager.checkpoint_dir / "quantized_weights"
        if weights_dir.exists():
            # Collect all weight files
            weight_files = list(weights_dir.glob("*.safetensors"))
            
            if weight_files:
                # Load and merge all weights
                import safetensors.torch
                merged_state_dict = {}
                
                for weight_file in weight_files:
                    state_dict = safetensors.torch.load_file(str(weight_file))
                    merged_state_dict.update(state_dict)
                
                # Save merged weights
                output_file = output_path / "model.safetensors"
                safetensors.torch.save_file(merged_state_dict, str(output_file))
                self.logger.info(f"Saved merged model to {output_file}")
        
        # Create model config with quantization info
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
            }
            
            with open(output_path / "config.json", 'w') as f:
                json.dump(model_config, f, indent=2)
        
        # Copy tokenizer files
        for file_name in ["tokenizer_config.json", "tokenizer.json", "tokenizer.model", "special_tokens_map.json"]:
            src = self.config.model_path / file_name
            if src.exists():
                shutil.copy2(src, output_path / file_name)
        
        # Create quantization metadata
        metadata = {
            'quantization_method': 'awq',
            'bits': self.config.bits,
            'group_size': self.config.group_size,
            'symmetric': self.config.symmetric,
            'total_layers': len(self.state.completed_layers),
            'failed_layers': self.state.failed_layers,
            'original_size_gb': self.state.total_original_size_gb,
            'quantized_size_gb': self.state.total_quantized_size_gb,
            'compression_ratio': self.state.total_original_size_gb / max(self.state.total_quantized_size_gb, 0.001),
            'timestamp': datetime.now().isoformat(),
        }
        
        with open(output_path / "quantization_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Model exported to: {output_path}")
    
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
        """Validate the quantized model"""
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
        
        # Calculate compression ratio
        metrics['overall_compression'] = self.state.total_original_size_gb / max(self.state.total_quantized_size_gb, 0.001)
        
        # Check for failed layers
        metrics['num_failed_layers'] = len(self.state.failed_layers)
        
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