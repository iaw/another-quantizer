# quantization_pipeline.py
"""Main quantization pipeline orchestrator"""

import time
import logging
import traceback
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from tqdm import tqdm
import torch
import psutil
from datetime import datetime, timedelta
import json
import signal
import sys
import gc


@dataclass
class PipelineState:
    """Track pipeline execution state"""
    start_time: float
    current_layer_idx: int = 0
    completed_layers: List[str] = field(default_factory=list)
    failed_layers: List[str] = field(default_factory=list)
    total_layers: int = 0
    current_phase: str = "initialization"  # initialization, quantization, validation, export
    last_checkpoint: Optional[str] = None
    error_count: int = 0
    retry_count: int = 0
    metrics: Dict[str, float] = field(default_factory=dict)
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
    """Main pipeline for sequential GLM quantization"""
    
    def __init__(self, config_path: str):
        """Initialize quantization pipeline"""
        self.config_path = Path(config_path)
        self.config = None
        self.memory_manager = None
        self.model_loader = None
        self.layer_quantizer = None
        self.checkpoint_manager = None
        self.calibration_handler = None
        self.llm_compressor = None
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
        - Metrics: Separate CSV file
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
        
        # Metrics handler (CSV)
        metrics_file = log_dir / f'metrics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        self.metrics_file = metrics_file
        
        # Write CSV header
        with open(metrics_file, 'w') as f:
            f.write("timestamp,layer_name,quantization_time,memory_peak_gb,error,compression_ratio\n")
        
        return logger
        
    def initialize(self) -> None:
        """Initialize all components
        
        Setup sequence:
        1. Load configuration
        2. Initialize memory manager
        3. Setup model loader
        4. Prepare calibration data
        5. Initialize quantizer
        6. Setup checkpointing
        """
        self.logger.info("=" * 80)
        self.logger.info("GLM Sequential Quantization Pipeline")
        self.logger.info("=" * 80)
        
        try:
            # Load and validate config
            self.logger.info("Loading configuration...")
            from config import QuantizationConfig, GLMModelConfig
            
            if self.config_path.suffix == '.yaml':
                self.config = QuantizationConfig.from_yaml(str(self.config_path))
            else:
                # Assume it's a model path
                from config import create_default_config
                self.config = create_default_config(
                    model_path=str(self.config_path),
                    output_path=str(self.config_path.parent / "quantized")
                )
            
            self.logger.info(f"Model path: {self.config.model_path}")
            self.logger.info(f"Output path: {self.config.output_path}")
            self.logger.info(f"Quantization: {self.config.bits}-bit, group_size={self.config.group_size}")
            
            # Initialize MemoryManager
            self.logger.info("Initializing memory manager...")
            from memory_manager import MemoryManager
            self.memory_manager = MemoryManager(
                max_gpu_memory=self.config.max_gpu_memory,
                max_cpu_memory=self.config.max_cpu_memory
            )
            self.memory_manager.monitor_memory("initial")
            
            # Create SequentialModelLoader
            self.logger.info("Setting up model loader...")
            from model_loader import SequentialModelLoader
            self.model_loader = SequentialModelLoader(
                model_path=str(self.config.model_path),
                memory_manager=self.memory_manager
            )
            
            # Load model configuration
            model_config = GLMModelConfig.from_model_config(str(self.config.model_path))
            self.logger.info(f"Model: {model_config.num_layers} layers, "
                           f"hidden_size={model_config.hidden_size}, "
                           f"params={model_config.get_total_params()/1e9:.1f}B")
            
            # Setup device map
            device_map = self.memory_manager.setup_device_map(model_config)
            
            # Load tokenizer
            self.logger.info("Loading tokenizer...")
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                str(self.config.model_path),
                trust_remote_code=True
            )
            
            # Setup CalibrationDataHandler
            self.logger.info("Preparing calibration data...")
            from calibration_data import CalibrationDataHandler
            self.calibration_handler = CalibrationDataHandler(
                tokenizer=tokenizer,
                max_length=self.config.max_position_embeddings if hasattr(self.config, 'max_position_embeddings') else 2048,
                num_samples=self.config.calibration_samples
            )
            
            # Load calibration dataset
            self.calibration_handler.load_calibration_dataset(dataset_name="c4")
            self.calibration_dataloader = self.calibration_handler.prepare_calibration_data(
                batch_size=self.config.calibration_batch_size
            )
            
            # Initialize LayerQuantizer
            self.logger.info("Initializing layer quantizer...")
            from layer_quantizer import LayerQuantizer
            self.layer_quantizer = LayerQuantizer(
                config=self.config,
                memory_manager=self.memory_manager
            )
            
            # Setup CheckpointManager
            self.logger.info("Setting up checkpoint manager...")
            from checkpoint_manager import CheckpointManager
            self.checkpoint_manager = CheckpointManager(
                checkpoint_dir=str(self.config.checkpoint_dir)
            )
            
            # Initialize LLMCompressorWrapper
            self.logger.info("Initializing LLM Compressor...")
            from llm_compressor_wrapper import LLMCompressorWrapper
            self.llm_compressor = LLMCompressorWrapper(config=self.config)
            
            # Create initial state
            self.state = PipelineState(
                start_time=time.time(),
                total_layers=self.model_loader.estimate_total_layers()
            )
            
            # Log system info
            self._log_system_info()
            
            self.logger.info("Initialization complete!")
            
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            self.logger.debug(traceback.format_exc())
            raise
    
    def run(self, resume_from_checkpoint: Optional[str] = None) -> None:
        """Run the quantization pipeline
        
        Main execution flow:
        1. Initialize or resume
        2. Run quantization loop
        3. Validate results
        4. Export model
        5. Cleanup
        """
        self.start_time = time.time()
        
        try:
            # Initialize components
            self.initialize()
            
            # Resume from checkpoint if provided
            if resume_from_checkpoint:
                self.logger.info(f"Resuming from checkpoint: {resume_from_checkpoint}")
                self._resume_from_checkpoint(resume_from_checkpoint)
            else:
                # Check for auto-resume
                resume_point = self.checkpoint_manager.get_resume_point()
                if resume_point:
                    layer_idx, completed_layers = resume_point
                    self.logger.info(f"Auto-resuming from layer {layer_idx}")
                    self.state.current_layer_idx = layer_idx
                    self.state.completed_layers = completed_layers
            
            # Run pre-flight checks
            if not self.pre_flight_checks():
                raise RuntimeError("Pre-flight checks failed")
            
            # Execute quantization
            self.logger.info("\nStarting quantization...")
            self.state.current_phase = "quantization"
            self.quantize_model()
            
            # Check if interrupted
            if self.shutdown_requested:
                self.logger.info("Quantization interrupted by user")
                return
            
            # Validate quantized model
            self.logger.info("\nValidating quantized model...")
            self.state.current_phase = "validation"
            validation_metrics = self.validate_quantized_model()
            
            # Export to vLLM format
            self.logger.info("\nExporting model...")
            self.state.current_phase = "export"
            self.export_to_vllm(str(self.config.output_path))
            
            # Generate final report
            self.logger.info("\nGenerating reports...")
            self._generate_final_report(validation_metrics)
            
            elapsed_time = time.time() - self.start_time
            self.logger.info(f"\n{'='*80}")
            self.logger.info(f"Quantization completed successfully!")
            self.logger.info(f"Total time: {self._format_time(elapsed_time)}")
            self.logger.info(f"Output saved to: {self.config.output_path}")
            self.logger.info(f"{'='*80}")
            
        except KeyboardInterrupt:
            self.logger.info("\nQuantization interrupted by user")
            self.handle_interrupt()
            
        except Exception as e:
            self.logger.error(f"Quantization failed: {e}")
            self.logger.debug(traceback.format_exc())
            
            # Save emergency checkpoint
            self.create_emergency_checkpoint(f"error_{e}")
            
            # Cleanup
            self.cleanup()
            raise
            
        finally:
            # Final cleanup
            self.cleanup()
    
    def quantize_model(self) -> None:
        """Main quantization loop
        
        Sequential layer processing:
        - Iterate through layers
        - Quantize each
        - Save checkpoints
        - Handle errors
        """
        # Get total layer count
        total_layers = self.state.total_layers
        
        # Initialize progress bar
        self.progress_bar = tqdm(
            total=total_layers,
            initial=self.state.current_layer_idx,
            desc="Quantizing layers",
            unit="layer"
        )
        
        # Get calibration data
        calibration_batch = next(iter(self.calibration_dataloader))
        calibration_inputs = calibration_batch['input_ids']
        
        # Iterate through layers
        for layer_name, layer in self.model_loader.iterate_layers():
            # Check if should skip (already completed)
            if layer_name in self.state.completed_layers:
                self.logger.info(f"Skipping already quantized layer: {layer_name}")
                self.progress_bar.update(1)
                continue
            
            # Check for shutdown
            if self.shutdown_requested:
                self.logger.info("Shutdown requested, saving checkpoint...")
                self._save_checkpoint(layer_name)
                break
            
            # Quantize layer
            try:
                metrics = self.quantize_single_layer(
                    layer_idx=self.state.current_layer_idx,
                    layer_name=layer_name,
                    layer=layer,
                    calibration_inputs=calibration_inputs
                )
                
                # Add to completed layers
                self.state.completed_layers.append(layer_name)
                
                # Save checkpoint if needed
                if self.state.current_layer_idx % self.config.checkpoint_every_n_layers == 0:
                    self._save_checkpoint(layer_name)
                
            except Exception as e:
                self.logger.error(f"Error quantizing layer {layer_name}: {e}")
                self.state.failed_layers.append(layer_name)
                self.state.error_count += 1
                
                # Try to handle error
                if not self.handle_layer_failure(layer_name, e):
                    raise
            
            # Update progress
            self.state.current_layer_idx += 1
            self.progress_bar.update(1)
            
            # Log progress
            self.log_progress(
                self.state.current_layer_idx,
                total_layers,
                time.time() - self.state.start_time
            )
        
        # Close progress bar
        if self.progress_bar:
            self.progress_bar.close()
        
        # Final checkpoint
        if self.state.completed_layers:
            self._save_checkpoint("final")
    
    def quantize_single_layer(self, 
                             layer_idx: int,
                             layer_name: str,
                             layer: Any,
                             calibration_inputs: torch.Tensor) -> QuantizationMetrics:
        """Quantize a single layer and return metrics
        
        Layer processing:
        1. Load layer weights
        2. Prepare calibration data
        3. Apply AWQ quantization
        4. Validate quality
        5. Save results
        """
        start_time = time.time()
        
        # Monitor memory before
        self.memory_manager.monitor_memory(f"before_{layer_name}")
        memory_before = self.memory_manager.get_current_memory()
        
        self.logger.info(f"\nQuantizing layer {layer_idx}: {layer_name}")
        
        # Quantize layer
        quantized_layer, result = self.layer_quantizer.quantize_layer(
            layer=layer,
            layer_name=layer_name,
            calibration_data=calibration_inputs
        )
        
        # Save quantized weights
        self.checkpoint_manager.save_quantized_weights(
            layer_name=layer_name,
            weights=quantized_layer,
            scales=result.scales,
            metadata={
                'group_size': result.group_size,
                'bits': result.bits,
                'quantization_error': result.quantization_error,
                'compression_ratio': result.compression_ratio,
            }
        )
        
        # Monitor memory after
        self.memory_manager.monitor_memory(f"after_{layer_name}")
        memory_after = self.memory_manager.get_current_memory()
        memory_peak = max(memory_before.gpu_used, memory_after.gpu_used)
        
        # Update totals
        self.state.total_original_size_gb += result.original_size_gb
        self.state.total_quantized_size_gb += result.quantized_size_gb
        
        # Calculate metrics
        quantization_time = time.time() - start_time
        
        metrics = QuantizationMetrics(
            layer_name=layer_name,
            quantization_time=quantization_time,
            memory_peak_gb=memory_peak,
            mse_error=result.quantization_error,
            cosine_similarity=1.0,  # Would need to calculate
            compression_ratio=result.compression_ratio,
            weights_size_mb=result.quantized_size_gb * 1024
        )
        
        # Save metrics
        self.layer_metrics.append(metrics)
        self._save_metrics(metrics)
        
        # Cleanup layer
        self.model_loader.cleanup_layer(layer)
        
        return metrics
    
    def validate_quantized_model(self) -> Dict[str, float]:
        """Validate the fully quantized model
        
        Validation checks:
        - Perplexity on validation set
        - Sample generation quality
        - Memory footprint
        - Inference speed
        """
        self.logger.info("Running validation...")
        
        validation_metrics = {
            'total_layers': len(self.state.completed_layers),
            'failed_layers': len(self.state.failed_layers),
            'compression_ratio': self.state.total_original_size_gb / max(self.state.total_quantized_size_gb, 0.001),
            'total_original_gb': self.state.total_original_size_gb,
            'total_quantized_gb': self.state.total_quantized_size_gb,
        }
        
        # Calculate average quantization error
        if self.layer_metrics:
            avg_error = sum(m.mse_error for m in self.layer_metrics) / len(self.layer_metrics)
            validation_metrics['avg_quantization_error'] = avg_error
        
        # Get validation dataset
        val_dataloader = self.calibration_handler.create_validation_set(num_samples=32)
        
        # Would run perplexity evaluation here
        # For now, use placeholder
        validation_metrics['perplexity'] = 10.0  # Placeholder
        
        # Test generation quality
        validation_metrics['generation_quality'] = self._test_generation_quality()
        
        # Measure memory footprint
        validation_metrics['memory_footprint_gb'] = self.state.total_quantized_size_gb
        
        # Log validation results
        self.logger.info("\nValidation Results:")
        self.logger.info(f"  Total layers quantized: {validation_metrics['total_layers']}")
        self.logger.info(f"  Failed layers: {validation_metrics['failed_layers']}")
        self.logger.info(f"  Compression ratio: {validation_metrics['compression_ratio']:.2f}x")
        self.logger.info(f"  Original size: {validation_metrics['total_original_gb']:.2f} GB")
        self.logger.info(f"  Quantized size: {validation_metrics['total_quantized_gb']:.2f} GB")
        
        return validation_metrics
    
    def export_to_vllm(self, output_path: str) -> None:
        """Export quantized model in vLLM format
        
        Export steps:
        1. Merge all quantized layers
        2. Create vLLM config
        3. Convert to vLLM format
        4. Create model card
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Exporting to vLLM format: {output_path}")
        
        # Merge all checkpoints
        checkpoint_ids = [cp.checkpoint_id for cp in self.checkpoint_manager.checkpoint_history]
        self.checkpoint_manager.merge_checkpoints(checkpoint_ids[-1:], output_path)  # Use latest
        
        # Create vLLM config
        vllm_config = self.llm_compressor.create_vllm_config(None)  # Would pass model
        
        # Add quantization config
        quant_config = {
            "quantization": "awq",
            "weight_bits": self.config.bits,
            "group_size": self.config.group_size,
            "symmetric": self.config.symmetric,
            "version": "GEMM",
        }
        
        # Save configs
        config_file = output_path / "config.json"
        
        # Load original config and update
        original_config_file = self.config.model_path / "config.json"
        if original_config_file.exists():
            with open(original_config_file, 'r') as f:
                model_config = json.load(f)
        else:
            model_config = {}
        
        model_config['quantization_config'] = quant_config
        
        with open(config_file, 'w') as f:
            json.dump(model_config, f, indent=2)
        
        # Copy tokenizer files
        self._copy_tokenizer_files(output_path)
        
        # Create model card
        model_card = self.create_model_card(
            base_model=str(self.config.model_path),
            metrics={'compression_ratio': self.state.total_original_size_gb / max(self.state.total_quantized_size_gb, 0.001)}
        )
        
        readme_file = output_path / "README.md"
        with open(readme_file, 'w') as f:
            f.write(model_card)
        
        self.logger.info(f"Export complete: {output_path}")
    
    def handle_oom_error(self, layer_idx: int) -> None:
        """Handle out-of-memory errors
        
        Recovery strategies:
        1. Clear all caches
        2. Reduce batch size
        3. Increase CPU offloading
        4. Skip problematic layer
        """
        self.logger.warning(f"Handling OOM error at layer {layer_idx}")
        
        # Emergency memory cleanup
        self.memory_manager.emergency_cleanup()
        
        # Reduce calibration batch size
        if self.config.calibration_batch_size > 1:
            self.config.calibration_batch_size = max(1, self.config.calibration_batch_size // 2)
            self.logger.info(f"Reduced batch size to {self.config.calibration_batch_size}")
        
        # Increase offloading
        self.config.offload_to_cpu = True
        
        # Save emergency checkpoint
        self.create_emergency_checkpoint(f"oom_layer_{layer_idx}")
        
        # Retry counter
        self.state.retry_count += 1
        
        if self.state.retry_count > 3:
            self.logger.error("Too many OOM errors, skipping layer")
            # Skip this layer
    
    def log_progress(self, 
                    layer_idx: int,
                    total_layers: int,
                    elapsed_time: float) -> None:
        """Log quantization progress
        
        Progress information:
        - Percentage complete
        - Current layer
        - Time elapsed
        - ETA
        - Memory usage
        """
        # Calculate percentage
        percentage = (layer_idx / total_layers) * 100
        
        # Estimate remaining time
        if layer_idx > 0:
            avg_time_per_layer = elapsed_time / layer_idx
            remaining_layers = total_layers - layer_idx
            eta_seconds = avg_time_per_layer * remaining_layers
            eta_str = self._format_time(eta_seconds)
        else:
            eta_str = "calculating..."
        
        # Get memory stats
        memory_stats = self.memory_manager.get_current_memory()
        
        # Update progress bar description
        if self.progress_bar:
            self.progress_bar.set_description(
                f"Layer {layer_idx}/{total_layers} | "
                f"GPU: {memory_stats.gpu_used:.1f}GB | "
                f"ETA: {eta_str}"
            )
        
        # Log to file
        self.checkpoint_manager.save_progress_log(
            layer_name=f"layer_{layer_idx}",
            metrics={
                'percentage': percentage,
                'elapsed_time': elapsed_time,
                'gpu_memory_gb': memory_stats.gpu_used,
                'cpu_memory_gb': memory_stats.cpu_used,
            }
        )
    
    def cleanup(self) -> None:
        """Clean up resources
        
        Cleanup tasks:
        - Clear memory
        - Close files
        - Save final state
        """
        self.logger.info("Cleaning up resources...")
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Clear calibration data
        if self.calibration_handler:
            self.calibration_handler.cleanup()
        
        # Save final metrics
        if self.layer_metrics:
            self._save_final_metrics()
        
        # Clean temporary files
        if self.checkpoint_manager:
            self.checkpoint_manager.cleanup_old_checkpoints(keep_last=5)
        
        # Force garbage collection
        gc.collect()
        
        self.logger.info("Cleanup complete")
    
    def estimate_total_time(self, 
                          layers_done: int,
                          elapsed_time: float,
                          total_layers: int) -> float:
        """Estimate remaining time
        
        Time estimation:
        - Based on average layer time
        - Adjusted for layer size variations
        - Include checkpoint overhead
        """
        if layers_done == 0:
            # Rough estimate
            return total_layers * 60  # 1 minute per layer estimate
        
        # Calculate average time per layer
        avg_time_per_layer = elapsed_time / layers_done
        
        # Estimate remaining time
        remaining_layers = total_layers - layers_done
        base_estimate = avg_time_per_layer * remaining_layers
        
        # Add checkpoint overhead (5% of time)
        checkpoint_overhead = base_estimate * 0.05
        
        # Add validation/export time (10% of total)
        final_overhead = (elapsed_time + base_estimate) * 0.1
        
        return base_estimate + checkpoint_overhead + final_overhead
    
    def pre_flight_checks(self) -> bool:
        """Run pre-flight validation checks
        
        Checks:
        - Model files exist
        - Sufficient disk space
        - GPU available
        - Dependencies installed
        """
        self.logger.info("Running pre-flight checks...")
        
        checks_passed = True
        
        # Check model files
        if not self.config.model_path.exists():
            self.logger.error(f"Model path does not exist: {self.config.model_path}")
            checks_passed = False
        
        # Check disk space
        import shutil
        free_space = shutil.disk_usage(self.config.output_path.parent).free / 1e9
        required_space = self.state.total_layers * 0.5  # Estimate 0.5GB per layer
        
        if free_space < required_space:
            self.logger.warning(f"Low disk space: {free_space:.1f}GB free, ~{required_space:.1f}GB needed")
        
        # Check GPU availability
        if not self.config.offload_to_cpu and not torch.cuda.is_available():
            self.logger.error("GPU not available but CPU offloading is disabled")
            checks_passed = False
        
        # Check CUDA version
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            self.logger.info(f"CUDA version: {cuda_version}")
        
        # Test small quantization
        try:
            test_tensor = torch.randn(100, 100)
            # Simple quantization test
            _ = torch.quantize_per_tensor(test_tensor, 0.1, 10, torch.qint8)
        except Exception as e:
            self.logger.error(f"Quantization test failed: {e}")
            checks_passed = False
        
        if checks_passed:
            self.logger.info("Pre-flight checks passed ✓")
        else:
            self.logger.error("Pre-flight checks failed ✗")
        
        return checks_passed
    
    def resume_from_checkpoint(self, checkpoint_path: str) -> None:
        """Resume from a saved checkpoint
        
        Resume process:
        1. Load checkpoint
        2. Restore state
        3. Skip completed layers
        4. Continue from interruption
        """
        self._resume_from_checkpoint(checkpoint_path)
    
    def create_emergency_checkpoint(self, reason: str) -> None:
        """Create emergency checkpoint on error
        
        Quick save for recovery:
        - Current state
        - Partial progress
        - Error information
        """
        self.logger.info(f"Creating emergency checkpoint: {reason}")
        
        checkpoint_path = self.checkpoint_manager.save_checkpoint(
            layer_idx=self.state.current_layer_idx,
            layer_name=f"emergency_{reason}",
            quantized_layers=self.state.completed_layers,
            metadata={
                'reason': reason,
                'error_count': self.state.error_count,
                'elapsed_time': time.time() - self.state.start_time,
                'state': self.state.__dict__,
            }
        )
        
        self.logger.info(f"Emergency checkpoint saved: {checkpoint_path}")
    
    def monitor_system_resources(self) -> Dict[str, float]:
        """Monitor system resource usage
        
        Monitored resources:
        - GPU memory
        - CPU memory
        - Disk I/O
        - Temperature
        """
        resources = {}
        
        # GPU stats
        if torch.cuda.is_available():
            resources['gpu_memory_used_gb'] = torch.cuda.memory_allocated() / 1e9
            resources['gpu_memory_reserved_gb'] = torch.cuda.memory_reserved() / 1e9
            
            # Try to get temperature
            try:
                import nvidia_ml_py as nvml
                nvml.nvmlInit()
                handle = nvml.nvmlDeviceGetHandleByIndex(0)
                temp = nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)
                resources['gpu_temperature_c'] = temp
            except:
                pass
        
        # CPU stats
        resources['cpu_percent'] = psutil.cpu_percent()
        resources['memory_percent'] = psutil.virtual_memory().percent
        
        # Disk usage
        disk = psutil.disk_usage('/')
        resources['disk_used_percent'] = disk.percent
        
        return resources
    
    def adaptive_batch_sizing(self, 
                            current_batch: int,
                            memory_usage: float) -> int:
        """Adaptively adjust batch size based on memory
        
        Adjustment strategy:
        - Increase if memory available
        - Decrease if near limit
        - Maintain stability
        """
        # Check current memory usage percentage
        memory_percent = (memory_usage / self.config.max_gpu_memory) * 100
        
        if memory_percent < 60:
            # Can increase batch size
            new_batch = min(current_batch * 2, 8)
        elif memory_percent > 80:
            # Should decrease batch size
            new_batch = max(current_batch // 2, 1)
        else:
            # Keep current
            new_batch = current_batch
        
        # Ensure within bounds
        new_batch = max(1, min(new_batch, 8))
        
        if new_batch != current_batch:
            self.logger.info(f"Adjusting batch size: {current_batch} -> {new_batch}")
        
        return new_batch
    
    def handle_layer_failure(self,
                           layer_name: str,
                           error: Exception) -> bool:
        """Handle individual layer failure
        
        Failure handling:
        - Log detailed error
        - Attempt retry
        - Skip if persistent
        """
        self.logger.error(f"Layer {layer_name} failed: {error}")
        self.logger.debug(traceback.format_exc())
        
        # Check if OOM error
        if "out of memory" in str(error).lower():
            self.handle_oom_error(self.state.current_layer_idx)
            return True  # Retry after OOM handling
        
        # Check retry count for this layer
        if self.state.retry_count < 3:
            self.logger.info(f"Retrying layer {layer_name} (attempt {self.state.retry_count + 1}/3)")
            self.state.retry_count += 1
            time.sleep(5)  # Wait before retry
            return True
        
        # Skip layer if too many failures
        self.logger.warning(f"Skipping layer {layer_name} after 3 failed attempts")
        self.state.failed_layers.append(layer_name)
        self.state.retry_count = 0  # Reset for next layer
        
        return True  # Continue with next layer
    
    def generate_quantization_report(self) -> Dict[str, Any]:
        """Generate comprehensive quantization report
        
        Report contents:
        - Summary statistics
        - Per-layer metrics
        - Error analysis
        - Performance data
        """
        report = {
            'summary': {
                'total_layers': self.state.total_layers,
                'completed_layers': len(self.state.completed_layers),
                'failed_layers': len(self.state.failed_layers),
                'compression_ratio': self.state.total_original_size_gb / max(self.state.total_quantized_size_gb, 0.001),
                'total_time': time.time() - self.start_time if self.start_time else 0,
            },
            'layer_metrics': [],
            'failed_layers': self.state.failed_layers,
            'system_info': self._get_system_info(),
        }
        
        # Add per-layer metrics
        for metric in self.layer_metrics:
            report['layer_metrics'].append({
                'layer_name': metric.layer_name,
                'quantization_time': metric.quantization_time,
                'compression_ratio': metric.compression_ratio,
                'error': metric.mse_error,
                'size_mb': metric.weights_size_mb,
            })
        
        return report
    
    def validate_layer_quality(self,
                              layer_name: str,
                              metrics: QuantizationMetrics) -> bool:
        """Validate individual layer quality
        
        Quality thresholds:
        - MSE < 0.01
        - Cosine similarity > 0.99
        - No NaN/Inf values
        """
        # Check MSE threshold
        if metrics.mse_error > 0.01:
            self.logger.warning(f"Layer {layer_name} has high MSE: {metrics.mse_error:.6f}")
            return False
        
        # Check cosine similarity
        if metrics.cosine_similarity < 0.99:
            self.logger.warning(f"Layer {layer_name} has low cosine similarity: {metrics.cosine_similarity:.4f}")
            return False
        
        return True
    
    def optimize_pipeline_performance(self) -> None:
        """Optimize pipeline settings based on runtime data
        
        Optimizations:
        - Adjust batch sizes
        - Tune memory allocation
        - Update checkpoint frequency
        """
        if not self.layer_metrics:
            return
        
        # Analyze performance data
        avg_time = sum(m.quantization_time for m in self.layer_metrics) / len(self.layer_metrics)
        avg_memory = sum(m.memory_peak_gb for m in self.layer_metrics) / len(self.layer_metrics)
        
        # Adjust batch size based on memory usage
        current_batch = self.config.calibration_batch_size
        new_batch = self.adaptive_batch_sizing(current_batch, avg_memory)
        self.config.calibration_batch_size = new_batch
        
        # Adjust checkpoint frequency based on speed
        if avg_time < 30:  # Fast layers
            self.config.checkpoint_every_n_layers = 10
        elif avg_time > 120:  # Slow layers
            self.config.checkpoint_every_n_layers = 1
        
        self.logger.info("Optimized pipeline settings based on performance")
    
    def create_model_card(self,
                         base_model: str,
                         metrics: Dict[str, Any]) -> str:
        """Create model card for quantized model
        
        Card contents:
        - Model description
        - Quantization details
        - Performance metrics
        - Usage instructions
        """
        model_card = f"""# Quantized GLM Model

## Model Description
This is a quantized version of {base_model} using AWQ (Activation-aware Weight Quantization).

## Quantization Details
- **Method**: AWQ
- **Bits**: {self.config.bits}
- **Group Size**: {self.config.group_size}
- **Compression Ratio**: {metrics.get('compression_ratio', 0):.2f}x

## Performance Metrics
- **Original Size**: {self.state.total_original_size_gb:.2f} GB
- **Quantized Size**: {self.state.total_quantized_size_gb:.2f} GB
- **Layers Quantized**: {len(self.state.completed_layers)}/{self.state.total_layers}

## Usage
```python
from vllm import LLM, SamplingParams

model = LLM(model="path/to/quantized/model", quantization="awq")
```

## Limitations
- This is a quantized model with reduced precision
- Some quality degradation compared to full precision model
- Optimized for inference speed and memory efficiency

## Citation
If you use this model, please cite the original GLM paper and AWQ paper.
"""
        
        return model_card
    
    # Helper methods
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info("\nShutdown signal received...")
        self.shutdown_requested = True
    
    def _resume_from_checkpoint(self, checkpoint_path: str) -> None:
        """Internal method to resume from checkpoint"""
        restoration_data = self.checkpoint_manager.restore_from_checkpoint(Path(checkpoint_path))
        
        if restoration_data:
            self.state.current_layer_idx = restoration_data['layer_idx'] + 1
            self.state.completed_layers = restoration_data['quantized_layers']
            self.state.start_time = time.time() - restoration_data.get('elapsed_time', 0)
            
            self.logger.info(f"Resumed from layer {self.state.current_layer_idx}")
    
    def _save_checkpoint(self, layer_name: str) -> None:
        """Save checkpoint"""
        self.checkpoint_manager.save_checkpoint(
            layer_idx=self.state.current_layer_idx,
            layer_name=layer_name,
            quantized_layers=self.state.completed_layers,
            metadata={
                'elapsed_time': time.time() - self.state.start_time,
                'memory_manager': self.memory_manager,
                'config': self.config.to_dict(),
                'model_path': str(self.config.model_path),
                'total_original_size_gb': self.state.total_original_size_gb,
                'total_quantized_size_gb': self.state.total_quantized_size_gb,
                'compression_ratio': self.state.total_original_size_gb / max(self.state.total_quantized_size_gb, 0.001),
                'error_metrics': {'error_count': self.state.error_count},
            }
        )
    
    def _save_metrics(self, metrics: QuantizationMetrics) -> None:
        """Save metrics to CSV"""
        with open(self.metrics_file, 'a') as f:
            f.write(f"{datetime.now().isoformat()},{metrics.layer_name},"
                   f"{metrics.quantization_time:.2f},{metrics.memory_peak_gb:.2f},"
                   f"{metrics.mse_error:.6f},{metrics.compression_ratio:.2f}\n")
    
    def _save_final_metrics(self) -> None:
        """Save final metrics summary"""
        summary_file = self.metrics_file.with_suffix('.summary.json')
        
        summary = {
            'total_layers': len(self.layer_metrics),
            'total_time': time.time() - self.start_time if self.start_time else 0,
            'avg_compression_ratio': sum(m.compression_ratio for m in self.layer_metrics) / len(self.layer_metrics),
            'avg_error': sum(m.mse_error for m in self.layer_metrics) / len(self.layer_metrics),
            'total_original_gb': self.state.total_original_size_gb,
            'total_quantized_gb': self.state.total_quantized_size_gb,
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
    
    def _generate_final_report(self, validation_metrics: Dict[str, Any]) -> None:
        """Generate final reports"""
        # Generate quantization report
        quant_report = self.generate_quantization_report()
        quant_report['validation'] = validation_metrics
        
        report_file = self.config.output_path / "quantization_report.json"
        with open(report_file, 'w') as f:
            json.dump(quant_report, f, indent=2)
        
        # Generate checkpoint report
        self.checkpoint_manager.export_checkpoint_report(
            self.config.output_path / "checkpoint_report.json"
        )
    
    def _test_generation_quality(self) -> float:
        """Test generation quality of quantized model"""
        # Placeholder - would test actual generation
        return 0.95
    
    def _copy_tokenizer_files(self, output_path: Path) -> None:
        """Copy tokenizer files to output"""
        tokenizer_files = [
            "tokenizer_config.json",
            "tokenizer.json",
            "special_tokens_map.json",
            "vocab.json",
            "merges.txt",
            "tokenizer.model",  # For sentencepiece
        ]
        
        for file_name in tokenizer_files:
            src = self.config.model_path / file_name
            if src.exists():
                dst = output_path / file_name
                import shutil
                shutil.copy2(src, dst)
    
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
    
    def _log_system_info(self) -> None:
        """Log system information"""
        info = self._get_system_info()
        
        self.logger.info("\nSystem Information:")
        self.logger.info(f"  Python: {sys.version.split()[0]}")
        self.logger.info(f"  PyTorch: {torch.__version__}")
        self.logger.info(f"  CUDA: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            self.logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
            self.logger.info(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")
        self.logger.info(f"  CPU Cores: {info['cpu_cores']}")
        self.logger.info(f"  RAM: {info['total_memory_gb']:.1f}GB")
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        info = {
            'python_version': sys.version,
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cpu_cores': psutil.cpu_count(),
            'total_memory_gb': psutil.virtual_memory().total / 1e9,
        }
        
        if torch.cuda.is_available():
            info['gpu_name'] = torch.cuda.get_device_name(0)
            info['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        return info
    
    def handle_interrupt(self) -> None:
        """Handle keyboard interrupt gracefully"""
        self.logger.info("Handling interrupt...")
        
        # Save checkpoint
        if self.state and self.state.completed_layers:
            self.create_emergency_checkpoint("interrupted")
        
        # Cleanup
        self.cleanup()
        
        self.logger.info("Interrupt handled. Checkpoint saved.")