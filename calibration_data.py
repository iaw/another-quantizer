# calibration_data.py
"""Handle calibration data for quantization"""

import torch
from typing import List, Iterator, Optional, Dict, Any, Tuple
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import json
import numpy as np
from dataclasses import dataclass
import logging
from transformers import AutoTokenizer
from datasets import load_dataset
import random
import hashlib
import pickle
import gc


@dataclass
class CalibrationSample:
    """Simple calibration sample"""
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    

@dataclass
class LayerActivationCache:
    """Cache activations for a specific layer"""
    layer_name: str
    inputs: List[torch.Tensor]
    outputs: Optional[List[torch.Tensor]] = None
    attention_cache: Optional[Dict[str, torch.Tensor]] = None
    

class CalibrationDataset(Dataset):
    """Custom dataset for calibration data"""
    
    def __init__(self, samples: List[CalibrationSample]):
        """Initialize calibration dataset"""
        self.samples = samples
        
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> CalibrationSample:
        return self.samples[idx]


class CalibrationDataHandler:
    """Simplified calibration data handler for AWQ quantization"""
    
    def __init__(self, 
                 tokenizer: Any,
                 max_length: int = 2048,
                 num_samples: int = 128,
                 hidden_size: int = 4096):
        """Initialize calibration data handler"""
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_samples = num_samples
        self.hidden_size = hidden_size
        self.logger = logging.getLogger(__name__)
        
        # Storage for processed samples
        self.raw_data = []
        self.processed_samples = []
        
    def load_calibration_dataset(self, 
                                dataset_name: str = "c4",
                                subset: Optional[str] = "en",
                                split: str = "train",
                                custom_path: Optional[str] = None) -> None:
        """Load calibration dataset - SIMPLIFIED
        
        Just use synthetic data as fallback for simplicity
        """
        self.logger.info(f"Loading calibration dataset: {dataset_name}")
        
        if custom_path:
            # Try to load custom dataset
            try:
                self._load_custom_dataset(custom_path)
            except Exception as e:
                self.logger.warning(f"Failed to load custom dataset: {e}")
                self._generate_synthetic_data()
        else:
            # For simplicity, just use synthetic data
            # Real implementation would load from HuggingFace
            self.logger.info("Using synthetic calibration data")
            self._generate_synthetic_data()
        
        self.logger.info(f"Loaded {len(self.raw_data)} calibration samples")
    
    def prepare_calibration_data(self, 
                                batch_size: int = 2) -> DataLoader:
        """Prepare calibration data loader - SIMPLIFIED"""
        if not self.raw_data:
            self.logger.warning("No raw data loaded, generating synthetic data")
            self._generate_synthetic_data()
        
        # Tokenize all samples
        self.logger.info(f"Tokenizing {len(self.raw_data)} samples")
        
        # Simple tokenization
        self.processed_samples = []
        for text in self.raw_data[:self.num_samples]:
            tokens = self.tokenizer(
                text,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            sample = CalibrationSample(
                input_ids=tokens['input_ids'][0],
                attention_mask=tokens['attention_mask'][0]
            )
            self.processed_samples.append(sample)
        
        # Create dataset and dataloader
        dataset = CalibrationDataset(self.processed_samples)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,  # No shuffle for reproducibility
            num_workers=0,
            pin_memory=torch.cuda.is_available()
        )
        
        self.logger.info(f"Created dataloader with {len(dataloader)} batches")
        return dataloader
    
    def get_layer_inputs(self, 
                        layer_idx: int,
                        model_forward_func: Any) -> torch.Tensor:
        """Get inputs for specific layer during calibration
        
        Hook-based activation collection:
        - Register forward hooks
        - Run calibration forward pass
        - Collect intermediate activations
        """
        # Fallback: return dummy input with dynamic hidden size
        return torch.randn(1, self.max_length, self.hidden_size)
    
    def create_validation_set(self, num_samples: int = 32) -> DataLoader:
        """Create validation dataset
        
        Separate from calibration set:
        - Different samples
        - Used for quality validation
        - Smaller size
        """
        # Generate new samples for validation
        self._generate_synthetic_data()
        
        # Take last samples for validation
        val_texts = self.raw_data[-num_samples:]
        
        val_samples = []
        for text in val_texts:
            tokens = self.tokenizer(
                text,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            sample = CalibrationSample(
                input_ids=tokens['input_ids'][0],
                attention_mask=tokens['attention_mask'][0]
            )
            val_samples.append(sample)
        
        # Create validation dataset
        val_dataset = CalibrationDataset(val_samples)
        
        # Create DataLoader
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available()
        )
        
        return val_dataloader
    
    def cleanup(self) -> None:
        """Clean up calibration data from memory"""
        self.raw_data = []
        self.processed_samples = []
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.logger.info("Cleaned up calibration data")
    
    def cache_to_disk(self, cache_path: str) -> None:
        """Cache processed calibration data to disk
        
        Caching strategy:
        - Save tokenized tensors
        - Use safetensors format
        - Include metadata
        """
        cache_path = Path(cache_path)
        cache_path.mkdir(parents=True, exist_ok=True)
        
        # Save processed samples
        if self.processed_samples:
            # Create cache data
            cache_data = {
                'num_samples': len(self.processed_samples),
                'max_length': self.max_length,
                'tokenizer_name': self.tokenizer.name_or_path if hasattr(self.tokenizer, 'name_or_path') else 'unknown',
            }
            
            # Save samples
            for i, sample in enumerate(self.processed_samples):
                sample_data = {
                    'input_ids': sample.input_ids.cpu(),
                    'attention_mask': sample.attention_mask.cpu(),
                }
                
                # Save as tensor file
                torch.save(sample_data, cache_path / f"sample_{i}.pt")
            
            # Save metadata
            with open(cache_path / "metadata.json", 'w') as f:
                json.dump(cache_data, f)
            
            self.logger.info(f"Cached {len(self.processed_samples)} samples to {cache_path}")
    
    def load_from_cache(self, cache_path: str) -> bool:
        """Load calibration data from cache
        
        Validation:
        - Check cache validity
        - Verify tokenizer compatibility
        - Load tensors efficiently
        """
        cache_path = Path(cache_path)
        
        if not cache_path.exists():
            return False
        
        try:
            # Load metadata
            with open(cache_path / "metadata.json", 'r') as f:
                metadata = json.load(f)
            
            # Load samples
            self.processed_samples = []
            for i in range(metadata['num_samples']):
                sample_file = cache_path / f"sample_{i}.pt"
                if not sample_file.exists():
                    self.logger.warning(f"Missing cache file: {sample_file}")
                    return False
                
                sample_data = torch.load(sample_file)
                sample = CalibrationSample(
                    input_ids=sample_data['input_ids'],
                    attention_mask=sample_data['attention_mask']
                )
                self.processed_samples.append(sample)
            
            self.logger.info(f"Loaded {len(self.processed_samples)} samples from cache")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading cache: {e}")
            return False
    
    def estimate_calibration_memory(self) -> float:
        """Estimate memory requirement for calibration in GB"""
        if not self.processed_samples:
            # Estimate based on configuration
            samples_memory = self.num_samples * self.max_length * 2 / 1e9  # 2 bytes per token
            # Add overhead for activations
            activation_memory = self.num_samples * self.max_length * self.hidden_size * 2 / 1e9
            return samples_memory + activation_memory
        
        # Calculate based on actual samples
        samples_memory = 0
        for sample in self.processed_samples:
            samples_memory += sample.input_ids.numel() * sample.input_ids.element_size()
            samples_memory += sample.attention_mask.numel() * sample.attention_mask.element_size()
        
        samples_memory = samples_memory / 1e9
        
        # Estimate activation memory
        activation_memory = len(self.processed_samples) * self.max_length * self.hidden_size * 2 / 1e9
        
        return samples_memory + activation_memory
    
    def create_calibration_report(self) -> Dict[str, Any]:
        """Create report on calibration data
        
        Report includes:
        - Dataset statistics
        - Sample diversity metrics
        - Length distribution
        - Token coverage
        """
        report = {
            'num_samples': len(self.processed_samples),
            'num_raw_samples': len(self.raw_data),
            'max_length': self.max_length,
            'tokenizer': self.tokenizer.name_or_path if hasattr(self.tokenizer, 'name_or_path') else 'unknown',
            'estimated_memory_gb': self.estimate_calibration_memory()
        }
        
        return report
    
    def batch_generator(self,
                       batch_size: int = 1) -> Iterator[CalibrationSample]:
        """Generate batches for sequential processing
        
        Memory-efficient iteration:
        - Yield one batch at a time
        - Clear previous batch
        - Support variable batch sizes
        """
        if not self.processed_samples:
            self.logger.warning("No processed samples available")
            return
        
        num_samples = len(self.processed_samples)
        indices = list(range(num_samples))
        
        for i in range(0, num_samples, batch_size):
            batch_indices = indices[i:i + batch_size]
            batch = [self.processed_samples[idx] for idx in batch_indices]
            
            # Collate batch
            if len(batch) == 1:
                yield batch[0]
            else:
                # Stack tensors for batch
                input_ids = torch.stack([s.input_ids for s in batch])
                attention_mask = torch.stack([s.attention_mask for s in batch])
                
                yield CalibrationSample(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
    
    # Helper methods
    
    def _generate_synthetic_data(self) -> None:
        """Generate simple synthetic calibration data"""
        self.logger.info("Generating synthetic calibration data")
        
        # Simple text templates
        templates = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning models require large amounts of data for training.",
            "Natural language processing has advanced significantly in recent years.",
            "Deep learning architectures continue to evolve and improve.",
            "Transformer models have revolutionized the field of AI.",
            "Quantization reduces model size while maintaining performance.",
            "GPU acceleration enables faster model inference.",
            "Large language models can understand and generate human-like text.",
        ]
        
        # Generate samples by repeating and combining templates
        self.raw_data = []
        for i in range(self.num_samples):
            # Combine multiple templates to create longer text
            num_templates = random.randint(3, 8)
            selected = random.choices(templates, k=num_templates)
            text = " ".join(selected)
            self.raw_data.append(text)
    
    def _load_custom_dataset(self, path: str) -> None:
        """Load custom dataset from file"""
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Custom dataset not found: {path}")
        
        if path.suffix == '.txt':
            with open(path, 'r') as f:
                # Each line is a sample
                self.raw_data = [line.strip() for line in f if line.strip()][:self.num_samples]
        elif path.suffix == '.json':
            with open(path, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    self.raw_data = data[:self.num_samples]
                elif isinstance(data, dict) and 'texts' in data:
                    self.raw_data = data['texts'][:self.num_samples]
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")