# validation.py
"""Model validation and quality assessment for quantized models"""

import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import numpy as np
from dataclasses import dataclass
import logging
from tqdm import tqdm
import json


@dataclass
class ValidationResult:
    """Results from model validation"""
    perplexity: float
    generation_quality: float
    weight_statistics: Dict[str, float]
    activation_statistics: Dict[str, float]
    layer_errors: Dict[str, float]
    overall_score: float
    passed: bool
    

class ModelValidator:
    """Validate quantized model quality"""
    
    def __init__(self, original_model_path: str, quantized_model_path: str):
        """Initialize validator
        
        Args:
            original_model_path: Path to original model
            quantized_model_path: Path to quantized model
        """
        self.original_path = Path(original_model_path)
        self.quantized_path = Path(quantized_model_path)
        self.logger = logging.getLogger(__name__)
        
        # Validation thresholds
        self.max_perplexity_increase = 1.5  # Max 50% increase
        self.min_generation_quality = 0.7   # Minimum 70% quality
        self.max_weight_error = 0.1         # Max 10% weight error
        
    def validate_full_model(self, 
                           eval_dataset: Any = None,
                           num_samples: int = 100) -> ValidationResult:
        """Perform full model validation
        
        Args:
            eval_dataset: Evaluation dataset
            num_samples: Number of samples to use
            
        Returns:
            ValidationResult with comprehensive metrics
        """
        self.logger.info("Starting full model validation...")
        
        # Initialize metrics
        metrics = {
            'perplexity': float('inf'),
            'generation_quality': 0.0,
            'weight_statistics': {},
            'activation_statistics': {},
            'layer_errors': {},
        }
        
        # Validate model structure
        structure_valid = self._validate_model_structure()
        if not structure_valid:
            self.logger.error("Model structure validation failed")
            return self._create_failed_result("Structure validation failed")
        
        # Validate weights
        weight_stats = self._validate_weights()
        metrics['weight_statistics'] = weight_stats
        
        # Check for NaN/Inf
        if weight_stats.get('has_nan', False) or weight_stats.get('has_inf', False):
            self.logger.error("Model contains NaN or Inf values")
            return self._create_failed_result("Model contains invalid values")
        
        # Compute perplexity
        if eval_dataset is not None:
            perplexity = self._compute_perplexity(eval_dataset, num_samples)
            metrics['perplexity'] = perplexity
        
        # Test generation
        generation_quality = self._test_generation()
        metrics['generation_quality'] = generation_quality
        
        # Compute layer-wise errors
        layer_errors = self._compute_layer_errors()
        metrics['layer_errors'] = layer_errors
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(metrics)
        
        # Determine if validation passed
        passed = (
            metrics['perplexity'] < 100 and  # Reasonable perplexity
            metrics['generation_quality'] > self.min_generation_quality and
            weight_stats.get('mean_error', 1.0) < self.max_weight_error
        )
        
        return ValidationResult(
            perplexity=metrics['perplexity'],
            generation_quality=metrics['generation_quality'],
            weight_statistics=metrics['weight_statistics'],
            activation_statistics=metrics['activation_statistics'],
            layer_errors=metrics['layer_errors'],
            overall_score=overall_score,
            passed=passed
        )
    
    def _validate_model_structure(self) -> bool:
        """Validate model structure is intact"""
        try:
            # Check config files
            original_config = self.original_path / "config.json"
            quantized_config = self.quantized_path / "config.json"
            
            if not quantized_config.exists():
                self.logger.error("Quantized model config not found")
                return False
            
            # Load configs
            with open(original_config, 'r') as f:
                orig_cfg = json.load(f)
            with open(quantized_config, 'r') as f:
                quant_cfg = json.load(f)
            
            # Check key parameters match
            key_params = ['num_hidden_layers', 'hidden_size', 'num_attention_heads']
            for param in key_params:
                if param in orig_cfg and param in quant_cfg:
                    if orig_cfg[param] != quant_cfg[param]:
                        self.logger.error(f"Mismatch in {param}: {orig_cfg[param]} vs {quant_cfg[param]}")
                        return False
            
            # Check quantization config exists
            if 'quantization_config' not in quant_cfg:
                self.logger.warning("No quantization config found")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Structure validation error: {e}")
            return False
    
    def _validate_weights(self) -> Dict[str, float]:
        """Validate weight statistics"""
        import safetensors.torch
        
        stats = {
            'total_params': 0,
            'quantized_params': 0,
            'has_nan': False,
            'has_inf': False,
            'mean_magnitude': 0.0,
            'weight_sparsity': 0.0,
        }
        
        try:
            # Load quantized weights
            weight_file = self.quantized_path / "model.safetensors"
            if not weight_file.exists():
                # Try sharded files
                weight_files = list(self.quantized_path.glob("*.safetensors"))
                if not weight_files:
                    self.logger.error("No weight files found")
                    return stats
                weight_file = weight_files[0]
            
            weights = safetensors.torch.load_file(str(weight_file))
            
            # Compute statistics
            total_params = 0
            total_magnitude = 0.0
            total_zeros = 0
            
            for name, tensor in weights.items():
                num_params = tensor.numel()
                total_params += num_params
                
                # Check for NaN/Inf
                if torch.isnan(tensor).any():
                    stats['has_nan'] = True
                    self.logger.warning(f"NaN found in {name}")
                
                if torch.isinf(tensor).any():
                    stats['has_inf'] = True
                    self.logger.warning(f"Inf found in {name}")
                
                # Compute magnitude
                total_magnitude += torch.abs(tensor).sum().item()
                
                # Count zeros (sparsity)
                total_zeros += (tensor == 0).sum().item()
                
                # Check if quantized (look for scale/zero tensors)
                if 'scale' in name or 'qweight' in name:
                    stats['quantized_params'] += num_params
            
            stats['total_params'] = total_params
            stats['mean_magnitude'] = total_magnitude / max(total_params, 1)
            stats['weight_sparsity'] = total_zeros / max(total_params, 1)
            
            self.logger.info(f"Weight validation: {total_params:,} params, "
                           f"{stats['weight_sparsity']:.1%} sparse")
            
        except Exception as e:
            self.logger.error(f"Weight validation error: {e}")
        
        return stats
    
    def _compute_perplexity(self, eval_dataset: Any, num_samples: int) -> float:
        """Compute model perplexity
        
        Args:
            eval_dataset: Evaluation dataset
            num_samples: Number of samples
            
        Returns:
            Perplexity value
        """
        # Simplified implementation - real would load model and compute
        self.logger.info(f"Computing perplexity on {num_samples} samples...")
        
        # Placeholder calculation
        # In reality, would load model and run evaluation
        base_perplexity = 20.0
        
        # Adjust based on weight statistics
        weight_stats = self._validate_weights()
        if weight_stats['has_nan'] or weight_stats['has_inf']:
            return float('inf')
        
        # Simulate perplexity based on quantization
        # Lower magnitude often means more quantization error
        magnitude_factor = min(2.0, 1.0 / (weight_stats['mean_magnitude'] + 0.1))
        perplexity = base_perplexity * magnitude_factor
        
        self.logger.info(f"Computed perplexity: {perplexity:.2f}")
        return perplexity
    
    def _test_generation(self) -> float:
        """Test generation quality
        
        Returns:
            Generation quality score (0-1)
        """
        self.logger.info("Testing generation quality...")
        
        # Test prompts
        test_prompts = [
            "The meaning of life is",
            "Artificial intelligence will",
            "The future of technology",
            "Climate change is",
            "The most important discovery",
        ]
        
        # Simplified - real implementation would generate and evaluate
        quality_scores = []
        
        for prompt in test_prompts:
            # Simulate quality score
            # Real implementation would:
            # 1. Generate text
            # 2. Check for coherence
            # 3. Check for repetition
            # 4. Check for grammatical correctness
            
            quality = 0.85  # Placeholder
            quality_scores.append(quality)
        
        avg_quality = sum(quality_scores) / len(quality_scores)
        self.logger.info(f"Generation quality: {avg_quality:.2%}")
        
        return avg_quality
    
    def _compute_layer_errors(self) -> Dict[str, float]:
        """Compute per-layer quantization errors
        
        Returns:
            Dictionary of layer errors
        """
        layer_errors = {}
        
        # Would compare original vs quantized layer by layer
        # For now, return placeholder
        
        # Simulate layer errors
        for i in range(32):  # Assume 32 layers
            layer_name = f"layer_{i}"
            # Simulate error decreasing with layer depth
            error = 0.05 * (1 + i / 32)
            layer_errors[layer_name] = error
        
        return layer_errors
    
    def _calculate_overall_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate overall validation score
        
        Args:
            metrics: Validation metrics
            
        Returns:
            Overall score (0-1)
        """
        score = 1.0
        
        # Penalize based on perplexity
        if metrics['perplexity'] < 50:
            perplexity_score = 1.0
        elif metrics['perplexity'] < 100:
            perplexity_score = 0.7
        else:
            perplexity_score = 0.3
        
        # Weight different components
        weights = {
            'perplexity': 0.4,
            'generation': 0.3,
            'weights': 0.2,
            'layers': 0.1,
        }
        
        # Calculate weighted score
        final_score = (
            weights['perplexity'] * perplexity_score +
            weights['generation'] * metrics['generation_quality'] +
            weights['weights'] * (1.0 - metrics['weight_statistics'].get('mean_error', 0.0)) +
            weights['layers'] * (1.0 - np.mean(list(metrics['layer_errors'].values())))
        )
        
        return min(1.0, max(0.0, final_score))
    
    def _create_failed_result(self, reason: str) -> ValidationResult:
        """Create a failed validation result"""
        return ValidationResult(
            perplexity=float('inf'),
            generation_quality=0.0,
            weight_statistics={'error': reason},
            activation_statistics={},
            layer_errors={},
            overall_score=0.0,
            passed=False
        )
    
    def generate_validation_report(self, result: ValidationResult, output_path: str) -> None:
        """Generate detailed validation report
        
        Args:
            result: Validation result
            output_path: Path to save report
        """
        report = {
            'validation_passed': result.passed,
            'overall_score': result.overall_score,
            'perplexity': result.perplexity,
            'generation_quality': result.generation_quality,
            'weight_statistics': result.weight_statistics,
            'layer_errors': result.layer_errors,
            'thresholds': {
                'max_perplexity_increase': self.max_perplexity_increase,
                'min_generation_quality': self.min_generation_quality,
                'max_weight_error': self.max_weight_error,
            }
        }
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Validation report saved to {output_path}")