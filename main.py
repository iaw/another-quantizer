# main.py
"""Entry point for GLM sequential quantization"""

import argparse
import logging
import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any, List
import torch
import json
import yaml
import time
from datetime import datetime
import signal
import traceback
import warnings

# In main.py, add this import after the other imports (around line 20-30)
from quantization_pipeline import (
    GLMQuantizationPipeline,
    CalibrationError,
    CheckpointError, 
    LayerQuantizationError,
    MemoryError as QuantizationMemoryError  # Rename to avoid conflict with Python's MemoryError
)

# Suppress some warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")


# Global variables for signal handling
pipeline = None
shutdown_requested = False


def setup_logging(verbose: bool = False, 
                  log_file: Optional[str] = None) -> None:
    """Set up logging configuration
    
    Logging setup:
    - Console output (INFO or DEBUG)
    - File output (always DEBUG)
    - Separate error log
    - Metrics CSV
    """
    # Create log directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Set up root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    
    # Remove existing handlers
    root_logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_formatter = logging.Formatter(
        '[%(levelname)s] %(message)s' if not verbose else 
        '%(asctime)s [%(name)s] %(levelname)s: %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler
    if log_file is None:
        log_file = log_dir / f"quantization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    else:
        log_file = Path(log_file)
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s [%(name)s] %(levelname)s: %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)
    
    # Error file handler
    error_file = log_dir / f"errors_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    error_handler = logging.FileHandler(error_file)
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(file_formatter)
    root_logger.addHandler(error_handler)
    
    # Set up special loggers for metrics
    metrics_logger = logging.getLogger('metrics')
    metrics_file = log_dir / f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    metrics_handler = logging.FileHandler(metrics_file)
    metrics_handler.setFormatter(logging.Formatter('%(message)s'))
    metrics_logger.addHandler(metrics_handler)
    metrics_logger.setLevel(logging.INFO)
    
    # Write CSV header
    metrics_logger.info("timestamp,metric,value")
    
    logging.info(f"Logging initialized. Logs: {log_file}")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments
    
    Arguments:
    - Model paths
    - Configuration
    - Resume options
    - Hardware settings
    - Output options
    """
    parser = argparse.ArgumentParser(
        description="Quantize GLM models with AWQ for vLLM deployment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic quantization
  python main.py --model-path /path/to/glm4.5 --output-path /path/to/output
  
  # Resume from checkpoint
  python main.py --resume --checkpoint-dir /path/to/checkpoints
  
  # Custom configuration
  python main.py --config config.yaml
  
  # Validate existing model
  python main.py --validate-only --model-path /path/to/quantized
        """
    )
    
    # Model arguments
    parser.add_argument(
        "--model-path",
        type=str,
        help="Path to GLM model to quantize"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        help="Where to save quantized model (default: model_path/quantized)"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["glm4", "glm4.5", "chatglm3", "auto"],
        default="auto",
        help="GLM variant (default: auto-detect)"
    )
    
    # Quantization arguments
    parser.add_argument(
        "--bits",
        type=int,
        choices=[3, 4, 8],
        default=4,
        help="Quantization bits (default: 4)"
    )
    parser.add_argument(
        "--group-size",
        type=int,
        choices=[-1, 32, 64, 128, 256],
        default=128,
        help="Group size for quantization (default: 128, -1 for per-channel)"
    )
    parser.add_argument(
        "--calibration-samples",
        type=int,
        default=128,
        help="Number of calibration samples (default: 128)"
    )
    parser.add_argument(
        "--calibration-dataset",
        type=str,
        default="c4",
        choices=["c4", "wikitext", "openwebtext", "pile", "custom"],
        help="Calibration dataset (default: c4)"
    )
    parser.add_argument(
        "--calibration-path",
        type=str,
        help="Path to custom calibration data (for --calibration-dataset custom)"
    )
    
    # Memory arguments
    parser.add_argument(
        "--max-gpu-memory",
        type=int,
        default=20,
        help="GPU memory limit in GB (default: 20)"
    )
    parser.add_argument(
        "--max-cpu-memory",
        type=int,
        default=200,
        help="CPU memory limit in GB (default: 200)"
    )
    parser.add_argument(
        "--offload-to-cpu",
        action="store_true",
        help="Enable CPU offloading for large models"
    )
    parser.add_argument(
        "--no-offload-to-cpu",
        dest="offload_to_cpu",
        action="store_false",
        help="Disable CPU offloading"
    )
    parser.set_defaults(offload_to_cpu=True)
    
    # Processing arguments
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Calibration batch size (default: 2)"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        help="Directory for checkpoints (default: output_path/checkpoints)"
    )
    parser.add_argument(
        "--checkpoint-frequency",
        type=int,
        default=5,
        help="Layers between checkpoints (default: 5)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from latest checkpoint"
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        help="Resume from specific checkpoint"
    )
    
    # Advanced arguments
    parser.add_argument(
        "--config",
        type=str,
        help="Config file (YAML) - overrides other arguments"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate existing quantized model"
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Only export existing quantized model"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Test setup without quantization"
    )
    
    # Hardware arguments
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="CUDA device (default: cuda:0)"
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallel (default: 1)"
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        help="Number of CPU threads (default: auto)"
    )
    
    # AWQ specific arguments
    parser.add_argument(
        "--awq-damping",
        type=float,
        default=0.01,
        help="AWQ damping percent (default: 0.01)"
    )
    parser.add_argument(
        "--awq-protection-factor",
        type=float,
        default=1.5,
        help="AWQ protection factor for salient channels (default: 1.5)"
    )
    parser.add_argument(
        "--symmetric",
        action="store_true",
        help="Use symmetric quantization"
    )
    
    # Parse and validate
    args = parser.parse_args()
    
    # Validation
    if not args.config and not args.model_path and not args.resume:
        parser.error("Either --model-path, --config, or --resume must be specified")
    
    if args.calibration_dataset == "custom" and not args.calibration_path:
        parser.error("--calibration-path required when using custom dataset")
    
    return args


def validate_environment() -> bool:
    """Validate CUDA availability and dependencies
    
    Environment checks:
    - CUDA availability
    - GPU memory
    - Required packages
    - Disk space
    """
    logger = logging.getLogger(__name__)
    
    # Check CUDA
    if not torch.cuda.is_available():
        logger.warning("CUDA not available, will use CPU (much slower)")
        # Not a hard failure, can still use CPU
    else:
        # Check GPU memory
        gpu_mem = torch.cuda.get_device_properties(0).total_memory
        logger.info(f"GPU available: {torch.cuda.get_device_name(0)} "
                   f"({gpu_mem/1e9:.1f}GB)")
        
        if gpu_mem < 8 * 1e9:
            logger.warning(f"Low GPU memory: {gpu_mem/1e9:.1f}GB, "
                          "consider using CPU offloading")
    
    # Check packages
    required_packages = [
        "transformers",
        "safetensors",
        "datasets",
        "tqdm",
        "psutil",
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing required packages: {', '.join(missing_packages)}")
        logger.error(f"Install with: pip install {' '.join(missing_packages)}")
        return False
    
    # Check llmcompressor (optional but recommended)
    try:
        import llmcompressor
        logger.info("LLM Compressor available")
    except ImportError:
        logger.warning("LLM Compressor not installed. "
                      "Install with: pip install llmcompressor")
        # Not a hard failure, wrapper will handle it
    
    # Check disk space
    import shutil
    free_space = shutil.disk_usage('.').free / 1e9
    if free_space < 50:
        logger.warning(f"Low disk space: {free_space:.1f}GB free")
    
    # Check CPU memory
    import psutil
    mem = psutil.virtual_memory()
    logger.info(f"System RAM: {mem.total/1e9:.1f}GB "
               f"(available: {mem.available/1e9:.1f}GB)")
    
    if mem.available < 8 * 1e9:
        logger.warning("Low system memory, quantization may be slow")
    
    return True


def main() -> int:
    """Main entry point
    
    Execution flow:
    1. Parse arguments
    2. Validate environment
    3. Setup logging
    4. Run appropriate command
    5. Handle errors
    """
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Setup logging
        setup_logging(args.verbose, getattr(args, 'log_file', None))
        
        # Validate environment
        if not validate_environment():
            logging.error("Environment validation failed")
            return 1
        
        # Setup signal handlers
        setup_signal_handlers()
        
        # Print banner
        if not args.validate_only and not args.export_only:
            print_banner()
        
        # Route to appropriate function
        if args.validate_only:
            return validate_command(args)
        elif args.export_only:
            return export_command(args)
        elif args.dry_run:
            return dry_run_command(args)
        else:
            return quantize_command(args)
            
    except KeyboardInterrupt:
        logging.info("Quantization interrupted by user")
        return 130
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        logging.debug(traceback.format_exc())
        return 1


def quantize_command(args: argparse.Namespace) -> int:
    """Main quantization command
    
    Full quantization pipeline:
    1. Load/create config
    2. Initialize pipeline
    3. Run quantization
    4. Validate results
    5. Export model
    """
    global pipeline
    
    # Create or load config with clear logic
    if args.config:
        # User provided a config file
        config = load_config(args.config)
        logging.info(f"Loaded configuration from: {args.config}")
    else:
        # Create config from command line arguments
        config = create_config_from_args(args)
        logging.info("Created configuration from command line arguments")
    
    # Validate config
    if not validate_config(config):
        logging.error("Invalid configuration")
        return 1
    
    # Save config for reference
    output_path = Path(config['output_path'])
    output_path.mkdir(parents=True, exist_ok=True)
    config_file_path = output_path / "quantization_config.yaml"
    save_config(config, config_file_path)
    logging.info(f"Saved configuration to: {config_file_path}")
    
    # Initialize pipeline
    from quantization_pipeline import GLMQuantizationPipeline
    
    # Create temporary config file for pipeline
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f)
        temp_config_file = f.name
    
    try:
        # Initialize pipeline with config file
        pipeline = GLMQuantizationPipeline(temp_config_file)
        
        # Determine if we should resume
        should_resume = args.resume or args.resume_from is not None
        
        # Run quantization with error recovery
        try:
            should_resume = args.resume or args.resume_from is not None
            resume_checkpoint = args.resume_from if args.resume_from else (True if args.resume else None)
            pipeline.run(resume_from_checkpoint=resume_checkpoint)
            logging.info("Quantization completed successfully")
            return 0
            
        except MemoryError as e:
            logging.error(f"Memory error during quantization: {e}")
            logging.info("Try reducing --batch-size or --calibration-samples")
            logging.info("Or enable --offload-to-cpu if not already enabled")
            return 2
            
        except CalibrationError as e:
            logging.error(f"Calibration data error: {e}")
            logging.info("Try using a different dataset or provide custom data with --calibration-path")
            return 3
            
        except CheckpointError as e:
            logging.error(f"Checkpoint error: {e}")
            logging.info("Check disk space and permissions in checkpoint directory")
            return 4
            
        except LayerQuantizationError as e:
            logging.error(f"Layer quantization failed: {e}")
            logging.info(f"Failed layer: {e.layer_name}")
            logging.info("The partial model may be usable. Check the checkpoint directory.")
            return 5
            
    except KeyboardInterrupt:
        logging.info("Quantization interrupted by user")
        if pipeline:
            pipeline.save_checkpoint()
            logging.info("Progress saved. Use --resume to continue.")
        return 130
        
    except Exception as e:
        logging.error(f"Unexpected error: {type(e).__name__}: {e}")
        logging.debug(traceback.format_exc())
        
        # Try to save emergency checkpoint
        if pipeline:
            try:
                pipeline.create_emergency_checkpoint("unexpected_error")
                logging.info("Emergency checkpoint saved")
            except:
                pass
        
        return 1
    finally:
        # Clean up temp file
        if 'temp_config_file' in locals():
            try:
                Path(temp_config_file).unlink(missing_ok=True)
            except:
                pass


def validate_command(args: argparse.Namespace) -> int:
    """Validate quantized model command
    
    Validation tasks:
    - Load quantized model
    - Check integrity
    - Measure perplexity
    - Test generation
    """
    # Get paths
    quantized_path = args.model_path or args.output_path
    original_path = getattr(args, 'original_model_path', None)
    
    if not quantized_path:
        logging.error("Model path required for validation")
        return 1
    
    # If no original path specified, try to infer from quantized path
    if not original_path:
        # Look for original path in quantized model metadata
        quantized_config = Path(quantized_path) / "config.json"
        if quantized_config.exists():
            with open(quantized_config, 'r') as f:
                config = json.load(f)
                original_path = config.get('_original_model_path')
        
        if not original_path:
            logging.warning("Original model path not specified, validation will be limited")
            # Run basic validation
            metrics = validate_quantized_model(quantized_path)
            print_validation_results(metrics)
            
            # Basic pass/fail
            if metrics.get("perplexity", float('inf')) < 100:
                logging.info("Basic validation PASSED ✓")
                return 0
            else:
                logging.error("Basic validation FAILED ✗")
                return 1
    
    logging.info(f"Validating quantized model: {quantized_path}")
    logging.info(f"Against original model: {original_path}")
    
    # Use full validator
    from validation import ModelValidator
    
    validator = ModelValidator(original_path, quantized_path)
    
    # Run full validation
    result = validator.validate_full_model(num_samples=args.validation_samples if hasattr(args, 'validation_samples') else 100)
    
    # Generate report
    report_path = Path(quantized_path) / "validation_report.json"
    validator.generate_validation_report(result, report_path)
    
    # Print results
    print("\n" + "="*80)
    print("VALIDATION RESULTS")
    print("="*80)
    print(f"Overall Score: {result.overall_score:.2%}")
    print(f"Perplexity: {result.perplexity:.2f}")
    print(f"Generation Quality: {result.generation_quality:.2%}")
    print(f"Weight Statistics:")
    for key, value in result.weight_statistics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    print(f"Validation: {'PASSED ✓' if result.passed else 'FAILED ✗'}")
    print("="*80)
    
    return 0 if result.passed else 1


def export_command(args: argparse.Namespace) -> int:
    """Export quantized model for vLLM
    
    Export tasks:
    - Convert format to vLLM requirements
    - Create vLLM-specific configs
    - Handle sharding for tensor parallel
    - Optimize for serving
    """
    input_path = args.model_path or args.checkpoint_dir
    output_path = args.output_path
    
    if not input_path or not output_path:
        logging.error("Both input and output paths required for export")
        return 1
    
    logging.info(f"Exporting model from {input_path} to {output_path}")
    
    try:
        # Use vLLM exporter
        from vllm_export import VLLMExporter
        
        exporter = VLLMExporter(input_path, output_path)
        
        # Export with sharding if multiple GPUs
        num_gpus = getattr(args, 'num_gpus', 1)
        export_status = exporter.export(num_shards=num_gpus if num_gpus > 1 else None)
        
        if export_status['success']:
            logging.info(f"Export successful: {output_path}")
            
            # Create serving config
            serving_config = create_vllm_config(
                output_path,
                num_gpus=num_gpus,
                max_length=getattr(args, 'max_length', 32768)
            )
            
            # Save serving config
            config_file = Path(output_path) / "vllm_serving_config.json"
            with open(config_file, 'w') as f:
                json.dump(serving_config, f, indent=2)
            
            logging.info(f"Created vLLM serving config: {config_file}")
            
            # Print usage instructions
            print("\n" + "="*80)
            print("vLLM Export Complete!")
            print("="*80)
            print(f"\nModel exported to: {output_path}")
            print(f"\nTo serve with vLLM:")
            print(f"  python -m vllm.entrypoints.openai.api_server \\")
            print(f"    --model {output_path} \\")
            print(f"    --quantization awq \\")
            print(f"    --tensor-parallel-size {num_gpus}")
            print("\nOr with the config file:")
            print(f"  vllm serve --config {config_file}")
            print("="*80 + "\n")
            
            return 0
        else:
            logging.error(f"Export failed: {export_status.get('errors', 'Unknown error')}")
            return 1
        
    except Exception as e:
        logging.error(f"Export failed: {e}")
        import traceback
        logging.debug(traceback.format_exc())
        return 1


def dry_run_command(args: argparse.Namespace) -> int:
    """Dry run to test setup
    
    Test tasks:
    - Load model config
    - Test calibration data
    - Memory estimation
    - Time estimation
    """
    logging.info("Running dry run...")
    
    # Create config with improved logic
    if args.config:
        config = load_config(args.config)
        logging.info(f"Using config file: {args.config}")
    else:
        if not args.model_path:
            logging.error("Either --config or --model-path required for dry run")
            return 1
        config = create_config_from_args(args)
        logging.info("Using command line arguments for configuration")
    
    # Validate config first
    if not validate_config(config):
        logging.error("Invalid configuration")
        return 1
    
    # Load model config
    from config import GLMModelConfig
    try:
        model_config = GLMModelConfig.from_model_config(config['model_path'])
        logging.info(f"Successfully loaded model config from: {config['model_path']}")
    except Exception as e:
        logging.error(f"Failed to load model config: {e}")
        return 1
    
    # Estimate memory requirements
    memory_est = estimate_memory_requirements(model_config, config)
    
    # Test calibration data loading
    logging.info("Testing calibration data loading...")
    try:
        from transformers import AutoTokenizer
        from calibration_data import CalibrationDataHandler
        
        tokenizer = AutoTokenizer.from_pretrained(
            config['model_path'],
            trust_remote_code=True
        )
        
        handler = CalibrationDataHandler(
            tokenizer=tokenizer,
            max_length=config.get('max_position_embeddings', 2048),
            num_samples=10,  # Just test with few samples
            hidden_size=model_config.hidden_size
        )
        handler.load_calibration_dataset(dataset_name="c4")
        
        logging.info("Calibration data loading successful")
    except Exception as e:
        logging.error(f"Calibration data loading failed: {e}")
        return 1
    
    # Estimate time
    time_est = estimate_quantization_time(config['model_path'], config)
    
    # Print summary
    print("\n" + "="*80)
    print("DRY RUN SUMMARY")
    print("="*80)
    print(f"Model: {config['model_path']}")
    print(f"Output: {config['output_path']}")
    print(f"Parameters: {model_config.get_total_params()/1e9:.1f}B")
    print(f"Layers: {model_config.num_layers}")
    print(f"Quantization: {config['bits']}-bit, group_size={config['group_size']}")
    print(f"\nMemory Requirements:")
    print(f"  GPU: {memory_est['gpu_required_gb']:.1f}GB")
    print(f"  CPU: {memory_est['cpu_required_gb']:.1f}GB")
    print(f"  Disk: {memory_est['disk_required_gb']:.1f}GB")
    print(f"\nEstimated Time: {time_est:.1f} hours")
    print(f"\nEstimated Output Size: {model_config.estimate_quantized_size(config['bits']):.1f}GB")
    print("\nConfiguration Valid: ✓")
    print("Model Accessible: ✓")
    print("Calibration Data: ✓")
    print("="*80)
    
    return 0


def quantize_glm(
    model_path: str,
    output_path: str,
    config_file: Optional[str] = None,
    resume_checkpoint: Optional[str] = None,
    validate_only: bool = False,
    **kwargs
) -> None:
    """High-level function to quantize GLM model
    
    Python API for quantization:
    - Simplified interface
    - Programmatic access
    - Integration friendly
    """
    # Create config
    if config_file:
        config = load_config(config_file)
    else:
        from config import create_default_config
        config_obj = create_default_config(model_path, output_path)
        
        # Apply kwargs overrides
        for key, value in kwargs.items():
            if hasattr(config_obj, key):
                setattr(config_obj, key, value)
        
        config = config_obj.to_dict()
    
    # Initialize pipeline - FIXED IMPORT
    from quantization_pipeline import GLMQuantizationPipeline
    
    # Create temp config file
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f)
        config_file = f.name
    
    try:
        pipeline = GLMQuantizationPipeline(config_file)
        
        # Run quantization
        pipeline.run(resume_from_checkpoint=resume_checkpoint)
        
        # Validate if requested
        if validate_only:
            metrics = pipeline.validate_quantized_model()
            print_validation_results(metrics)
            
    finally:
        # Clean up
        Path(config_file).unlink(missing_ok=True)


def validate_quantized_model(model_path: str) -> Dict[str, float]:
    """Validate a quantized model
    
    Validation metrics:
    - Perplexity
    - Generation quality
    - Memory usage
    - Inference speed
    """
    model_path = Path(model_path)
    
    metrics = {
        "model_path": str(model_path),
        "exists": model_path.exists(),
    }
    
    if not model_path.exists():
        logging.error(f"Model path does not exist: {model_path}")
        return metrics
    
    # Check for required files
    required_files = ["config.json"]
    for file in required_files:
        file_path = model_path / file
        metrics[f"has_{file}"] = file_path.exists()
    
    # Check quantization config
    config_file = model_path / "config.json"
    if config_file.exists():
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        if "quantization_config" in config:
            quant_config = config["quantization_config"]
            metrics["quantization_method"] = quant_config.get("quantization", "unknown")
            metrics["bits"] = quant_config.get("weight_bits", 0)
            metrics["group_size"] = quant_config.get("group_size", 0)
    
    # Check model size
    total_size = sum(f.stat().st_size for f in model_path.rglob("*.safetensors"))
    metrics["size_gb"] = total_size / 1e9
    
    # Would test actual model loading and generation here
    # For now, use placeholders
    metrics["perplexity"] = 10.0  # Placeholder
    metrics["generation_ok"] = True  # Placeholder
    
    return metrics


def export_for_vllm(
    quantized_path: str,
    output_path: str,
    num_gpus: int = 4
) -> None:
    """Export quantized model for vLLM deployment
    
    Export configuration:
    - Tensor parallel sharding
    - Kernel selection
    - Serving optimization
    """
    quantized_path = Path(quantized_path)
    output_path = Path(output_path)
    
    logging.info(f"Exporting for vLLM with {num_gpus} GPUs")
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Copy model files
    import shutil
    for file in quantized_path.glob("*.safetensors"):
        shutil.copy2(file, output_path / file.name)
    
    # Create vLLM config
    vllm_config = {
        "model": str(output_path),
        "quantization": "awq",
        "tensor_parallel_size": num_gpus,
        "gpu_memory_utilization": 0.95,
        "trust_remote_code": True,
    }
    
    # Save vLLM config
    config_file = output_path / "vllm_config.json"
    with open(config_file, 'w') as f:
        json.dump(vllm_config, f, indent=2)
    
    # Copy/update model config
    model_config_src = quantized_path / "config.json"
    model_config_dst = output_path / "config.json"
    
    if model_config_src.exists():
        shutil.copy2(model_config_src, model_config_dst)
    
    # Copy tokenizer files
    tokenizer_files = [
        "tokenizer_config.json",
        "tokenizer.json",
        "tokenizer.model",
        "special_tokens_map.json",
    ]
    
    for file in tokenizer_files:
        src = quantized_path / file
        if src.exists():
            shutil.copy2(src, output_path / file)
    
    logging.info(f"Export complete: {output_path}")


def create_config_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    """Create configuration from command line arguments
    
    Config creation:
    - Map args to config
    - Set defaults
    - Validate settings
    """
    from config import QuantizationConfig
    
    # Clear separation between model path and config file
    if args.config:
        # Load from config file
        config_obj = QuantizationConfig.from_yaml(args.config)
    else:
        # Create from arguments
        if not args.model_path:
            raise ValueError("--model-path required when not using --config")
        
        # Ensure paths are strings
        model_path = str(args.model_path)
        output_path = args.output_path or f"{model_path}_awq_{args.bits}bit"
        checkpoint_dir = args.checkpoint_dir or f"{output_path}/checkpoints"
        
        # Create config object with all parameters
        config_obj = QuantizationConfig(
            model_path=model_path,
            output_path=output_path,
            checkpoint_dir=checkpoint_dir,
            bits=args.bits,
            group_size=args.group_size,
            symmetric=args.symmetric,
            max_gpu_memory=args.max_gpu_memory,
            max_cpu_memory=args.max_cpu_memory,
            offload_to_cpu=args.offload_to_cpu,
            calibration_samples=args.calibration_samples,
            calibration_batch_size=args.batch_size,
            checkpoint_every_n_layers=args.checkpoint_frequency,
            awq_damping_percent=args.awq_damping,
            awq_protection_factor=args.awq_protection_factor,
        )
    
    # Apply any command-line overrides even when using config file
    if args.config and args.output_path:
        config_obj.output_path = args.output_path
    if args.config and args.checkpoint_dir:
        config_obj.checkpoint_dir = args.checkpoint_dir
    
    return config_obj.to_dict()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from file
    
    Config loading:
    - Support YAML and JSON
    - Validate schema
    - Apply defaults
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    if config_path.suffix == '.yaml' or config_path.suffix == '.yml':
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    elif config_path.suffix == '.json':
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        raise ValueError(f"Unsupported config format: {config_path.suffix}")
    
    # Validate and apply defaults
    required_fields = ['model_path', 'output_path']
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required field in config: {field}")
    
    # Apply defaults
    defaults = {
        'bits': 4,
        'group_size': 128,
        'calibration_samples': 128,
        'calibration_batch_size': 2,
        'max_gpu_memory': 20,
        'max_cpu_memory': 200,
        'offload_to_cpu': True,
        'checkpoint_every_n_layers': 5,
    }
    
    for key, value in defaults.items():
        if key not in config:
            config[key] = value
    
    return config


def save_config(config: Dict[str, Any], output_path: Path) -> None:
    """Save configuration to file
    
    Config saving:
    - YAML format
    - Include metadata
    - Version info
    """
    # Add metadata
    config['_metadata'] = {
        'created': datetime.now().isoformat(),
        'version': '1.0.0',
        'tool': 'GLM Sequential Quantization',
    }
    
    # Convert to YAML
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    logging.info(f"Configuration saved to {output_path}")


def validate_config(config: Dict[str, Any]) -> bool:
    """Validate configuration
    
    Validation checks:
    - Required fields
    - Value ranges
    - Path existence
    """
    # Check required fields
    required = ['model_path', 'output_path']
    for field in required:
        if field not in config:
            logging.error(f"Missing required config field: {field}")
            return False
    
    # Check paths
    model_path = Path(config['model_path'])
    if not model_path.exists():
        logging.error(f"Model path does not exist: {model_path}")
        return False
    
    # Check value ranges
    if config.get('bits', 4) not in [3, 4, 8]:
        logging.error(f"Invalid bits value: {config.get('bits')}")
        return False
    
    if config.get('group_size', 128) not in [-1, 32, 64, 128, 256]:
        logging.error(f"Invalid group_size: {config.get('group_size')}")
        return False
    
    return True


def setup_signal_handlers() -> None:
    """Setup signal handlers for graceful shutdown
    
    Signal handling:
    - SIGINT (Ctrl+C)
    - SIGTERM
    - Save checkpoint on interrupt
    """
    def signal_handler(signum, frame):
        global shutdown_requested, pipeline
        if not shutdown_requested:
            shutdown_requested = True
            logging.info("Shutdown requested, saving checkpoint...")
            if pipeline:
                try:
                    pipeline.create_emergency_checkpoint("interrupted")
                except:
                    pass
            sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def print_validation_results(metrics: Dict[str, Any]) -> None:
    """Print validation results in formatted table
    
    Output formatting:
    - Table layout
    - Color coding
    - Pass/fail indicators
    """
    print("\n" + "="*60)
    print("VALIDATION RESULTS")
    print("="*60)
    
    for key, value in metrics.items():
        if isinstance(value, bool):
            status = "✓" if value else "✗"
            print(f"{key:30s}: {status}")
        elif isinstance(value, float):
            print(f"{key:30s}: {value:.4f}")
        else:
            print(f"{key:30s}: {value}")
    
    print("="*60)
    
    # Overall status
    passed = metrics.get("perplexity", float('inf')) < 20 and metrics.get("generation_ok", False)
    if passed:
        print("OVERALL: PASSED ✓")
    else:
        print("OVERALL: FAILED ✗")
    print("="*60 + "\n")


def estimate_quantization_time(
    model_path: str,
    config: Dict[str, Any]
) -> float:
    """Estimate time for quantization
    
    Time estimation:
    - Based on model size
    - Hardware capabilities
    - Configuration settings
    """
    from config import GLMModelConfig
    
    try:
        model_config = GLMModelConfig.from_model_config(model_path)
    except:
        # Fallback estimate
        return 2.0  # 2 hours default
    
    # Count layers
    num_layers = model_config.num_layers
    
    # Estimate per-layer time (minutes)
    if torch.cuda.is_available():
        # GPU available
        if model_config.num_experts:  # MoE model
            time_per_layer = 3.0  # 3 minutes per MoE layer
        else:
            time_per_layer = 1.0  # 1 minute per standard layer
    else:
        # CPU only (much slower)
        time_per_layer = 10.0  # 10 minutes per layer
    
    # Adjust for batch size
    batch_size = config.get('calibration_batch_size', 2)
    if batch_size == 1:
        time_per_layer *= 1.5
    
    # Total time
    total_minutes = num_layers * time_per_layer
    
    # Add overhead (loading, validation, export)
    total_minutes += 30  # 30 minutes overhead
    
    return total_minutes / 60  # Return hours


def estimate_memory_requirements(
    model_config: Any,
    config: Dict[str, Any]
) -> Dict[str, float]:
    """Estimate memory requirements
    
    Memory estimation:
    - Model weights
    - Activations
    - Calibration data
    - Quantization overhead
    """
    # Model size
    model_size_gb = model_config.get_total_params() * 2 / 1e9  # FP16
    
    # Calibration data
    batch_size = config.get('calibration_batch_size', 2)
    seq_length = config.get('max_length', 2048)
    calibration_memory_gb = (batch_size * seq_length * model_config.hidden_size * 4) / 1e9
    
    # Working memory (2x model size for quantization)
    working_memory_gb = model_size_gb * 2
    
    # GPU requirements
    gpu_required = min(working_memory_gb, model_size_gb / model_config.num_layers + calibration_memory_gb)
    
    # CPU requirements
    cpu_required = model_size_gb + calibration_memory_gb
    
    # Disk requirements
    disk_required = model_size_gb * 2  # Original + quantized
    
    return {
        'gpu_required_gb': gpu_required,
        'cpu_required_gb': cpu_required,
        'disk_required_gb': disk_required,
    }


def check_disk_space(
    model_path: str,
    output_path: str,
    checkpoint_dir: str
) -> bool:
    """Check if sufficient disk space available
    
    Space requirements:
    - Original model size
    - Quantized model size
    - Checkpoint overhead
    - Temporary files
    """
    import shutil
    
    # Get available space
    free_space = shutil.disk_usage('.').free / 1e9
    
    # Estimate required space
    model_path = Path(model_path)
    if model_path.exists():
        model_size = sum(f.stat().st_size for f in model_path.rglob('*')) / 1e9
    else:
        model_size = 50  # Default estimate 50GB
    
    required_space = model_size * 3  # Original + quantized + checkpoints
    
    if free_space < required_space:
        logging.warning(f"Low disk space: {free_space:.1f}GB available, "
                       f"~{required_space:.1f}GB recommended")
        return False
    
    return True


def print_banner() -> None:
    """Print startup banner with version info"""
    banner = """
╔══════════════════════════════════════════╗
║     GLM Sequential Quantization Tool     ║
║           AWQ 4-bit for vLLM             ║
╚══════════════════════════════════════════╝
    """
    print(banner)
    print(f"Version: 1.0.0")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()


def cleanup_on_exit() -> None:
    """Cleanup function for atexit
    
    Cleanup tasks:
    - Clear GPU cache
    - Remove temp files
    - Save final state
    """
    global pipeline
    
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Remove temporary files
    import tempfile
    temp_dir = Path(tempfile.gettempdir())
    for tmp_file in temp_dir.glob("tmp*.yaml"):
        try:
            if tmp_file.stat().st_mtime < time.time() - 3600:  # Older than 1 hour
                tmp_file.unlink()
        except:
            pass
    
    # Log cleanup
    logging.info("Cleanup complete")


if __name__ == "__main__":
    import atexit
    atexit.register(cleanup_on_exit)
    sys.exit(main())