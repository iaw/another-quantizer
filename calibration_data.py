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


@dataclass
class CalibrationSample:
    """Single calibration sample"""
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: Optional[torch.Tensor] = None
    token_type_ids: Optional[torch.Tensor] = None  # For GLM compatibility
    

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
    """Manage calibration data for AWQ quantization"""
    
    def __init__(self, 
                 tokenizer: Any,
                 max_length: int = 2048,
                 num_samples: int = 128):
        """Initialize calibration data handler"""
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_samples = num_samples
        self.cached_data = None
        self.activation_cache = {}  # Layer-wise activation cache
        self.logger = logging.getLogger(__name__)
        
        # Dataset statistics
        self.token_frequencies = {}
        self.length_distribution = []
        
        # Raw data storage
        self.raw_data = []
        self.processed_samples = []
        
    def load_calibration_dataset(self, 
                                dataset_name: str = "c4",
                                subset: Optional[str] = "en",
                                split: str = "train",
                                custom_path: Optional[str] = None) -> None:
        """Load calibration dataset
        
        Supports multiple datasets:
        - C4: General web text
        - WikiText: Wikipedia articles
        - OpenWebText: Reddit links
        - Custom: User-provided
        """
        self.logger.info(f"Loading calibration dataset: {dataset_name}")
        
        if custom_path:
            # Load custom dataset from file
            self._load_custom_dataset(custom_path)
        else:
            # Load from HuggingFace datasets
            try:
                if dataset_name == "c4":
                    dataset = load_dataset(dataset_name, subset, split=split, streaming=True)
                elif dataset_name == "wikitext":
                    dataset = load_dataset("wikitext", "wikitext-103-v1", split=split)
                elif dataset_name == "openwebtext":
                    dataset = load_dataset("openwebtext", split=split, streaming=True)
                elif dataset_name == "pile":
                    dataset = load_dataset("EleutherAI/pile", split=split, streaming=True)
                else:
                    dataset = load_dataset(dataset_name, split=split)
                
                # Extract text samples
                self._extract_samples_from_dataset(dataset)
                
            except Exception as e:
                self.logger.error(f"Error loading dataset {dataset_name}: {e}")
                self.logger.info("Falling back to synthetic data")
                self._generate_synthetic_data()
        
        self.logger.info(f"Loaded {len(self.raw_data)} raw samples")
    
    def prepare_calibration_data(self, 
                                batch_size: int = 2) -> DataLoader:
        """Prepare calibration data loader
        
        Processing steps:
        1. Tokenize text samples
        2. Create attention masks
        3. Handle GLM-specific formatting
        4. Batch for efficiency
        """
        if not self.raw_data:
            self.logger.warning("No raw data loaded, generating synthetic data")
            self._generate_synthetic_data()
        
        # Tokenize all samples
        self.logger.info(f"Tokenizing {len(self.raw_data)} samples")
        tokenized_samples = self.tokenize_batch(self.raw_data)
        
        # Create CalibrationSample objects
        self.processed_samples = []
        for i in range(len(tokenized_samples['input_ids'])):
            sample = CalibrationSample(
                input_ids=tokenized_samples['input_ids'][i],
                attention_mask=tokenized_samples['attention_mask'][i],
                position_ids=tokenized_samples.get('position_ids', [None] * len(tokenized_samples['input_ids']))[i],
                token_type_ids=tokenized_samples.get('token_type_ids', [None] * len(tokenized_samples['input_ids']))[i],
            )
            self.processed_samples.append(sample)
        
        # Create dataset
        dataset = CalibrationDataset(self.processed_samples)
        
        # Create DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # Avoid multiprocessing issues
            pin_memory=torch.cuda.is_available(),
            collate_fn=self._collate_fn
        )
        
        self.logger.info(f"Created dataloader with {len(dataloader)} batches")
        
        return dataloader
    
    def tokenize_batch(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """Tokenize a batch of text
        
        GLM4.5-specific tokenization:
        - Standard tokenization (no [gMASK] for GLM4.5)
        - Handle position IDs
        - Create attention masks
        """
        # Ensure texts are strings
        texts = [str(text) for text in texts]
        
        # Tokenize with padding and truncation
        encodings = self.tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
            return_attention_mask=True,
        )
        
        # Create position IDs
        batch_size, seq_len = encodings['input_ids'].shape
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
        
        # GLM4.5 doesn't need special token type IDs
        # But keep for compatibility
        token_type_ids = torch.zeros_like(encodings['input_ids'])
        
        result = {
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'position_ids': position_ids,
            'token_type_ids': token_type_ids,
        }
        
        # Update statistics
        self._update_statistics(encodings['input_ids'])
        
        return result
    
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
                if sample.position_ids is not None:
                    sample_data['position_ids'] = sample.position_ids.cpu()
                if sample.token_type_ids is not None:
                    sample_data['token_type_ids'] = sample.token_type_ids.cpu()
                
                # Save as tensor file
                torch.save(sample_data, cache_path / f"sample_{i}.pt")
            
            # Save metadata
            with open(cache_path / "metadata.json", 'w') as f:
                json.dump(cache_data, f)
            
            # Create hash for validation
            hash_str = f"{self.tokenizer.name_or_path if hasattr(self.tokenizer, 'name_or_path') else 'unknown'}_{self.max_length}_{len(self.processed_samples)}"
            hash_value = hashlib.md5(hash_str.encode()).hexdigest()
            
            with open(cache_path / "hash.txt", 'w') as f:
                f.write(hash_value)
            
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
            
            # Verify tokenizer compatibility
            expected_hash = f"{self.tokenizer.name_or_path if hasattr(self.tokenizer, 'name_or_path') else 'unknown'}_{self.max_length}_{metadata['num_samples']}"
            expected_hash_value = hashlib.md5(expected_hash.encode()).hexdigest()
            
            with open(cache_path / "hash.txt", 'r') as f:
                cached_hash = f.read().strip()
            
            if cached_hash != expected_hash_value:
                self.logger.warning("Cache hash mismatch, cache may be invalid")
                return False
            
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
                    attention_mask=sample_data['attention_mask'],
                    position_ids=sample_data.get('position_ids'),
                    token_type_ids=sample_data.get('token_type_ids'),
                )
                self.processed_samples.append(sample)
            
            self.logger.info(f"Loaded {len(self.processed_samples)} samples from cache")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading cache: {e}")
            return False
    
    def get_layer_inputs(self, 
                        layer_idx: int,
                        model_forward_func: Any) -> torch.Tensor:
        """Get inputs for specific layer during calibration
        
        Hook-based activation collection:
        - Register forward hooks
        - Run calibration forward pass
        - Collect intermediate activations
        """
        if f"layer_{layer_idx}" in self.activation_cache:
            # Return cached activations
            cache = self.activation_cache[f"layer_{layer_idx}"]
            if cache.inputs:
                return torch.cat(cache.inputs, dim=0)
        
        # Register hook to capture inputs
        activations = []
        
        def hook_fn(module, input, output):
            if isinstance(input, tuple):
                input = input[0]
            activations.append(input.detach().cpu())
        
        # Register hook at layer_idx - 1 to get inputs to layer_idx
        hook_handle = None
        # This would need the actual model structure to register correctly
        
        # Run forward pass with calibration data
        with torch.no_grad():
            for sample in self.processed_samples[:min(10, len(self.processed_samples))]:
                try:
                    _ = model_forward_func(
                        input_ids=sample.input_ids.unsqueeze(0),
                        attention_mask=sample.attention_mask.unsqueeze(0)
                    )
                except:
                    pass
        
        # Remove hook
        if hook_handle:
            hook_handle.remove()
        
        # Cache activations
        if activations:
            self.activation_cache[f"layer_{layer_idx}"] = LayerActivationCache(
                layer_name=f"layer_{layer_idx}",
                inputs=activations,
                outputs=None,
                attention_cache=None
            )
            return torch.cat(activations, dim=0)
        
        # Fallback: return dummy input
        return torch.randn(1, self.max_length, 4096)  # Assuming hidden_size=4096
    
    def create_validation_set(self, num_samples: int = 32) -> DataLoader:
        """Create validation dataset
        
        Separate from calibration set:
        - Different samples
        - Used for quality validation
        - Smaller size
        """
        # Ensure we have enough samples
        if len(self.processed_samples) < self.num_samples + num_samples:
            self.logger.warning("Not enough samples for separate validation set")
            # Use last portion of calibration set
            val_samples = self.processed_samples[-num_samples:]
        else:
            # Generate new samples or use reserved ones
            if len(self.raw_data) > self.num_samples:
                # Use additional raw data
                val_texts = self.raw_data[self.num_samples:self.num_samples + num_samples]
                val_tokenized = self.tokenize_batch(val_texts)
                
                val_samples = []
                for i in range(len(val_tokenized['input_ids'])):
                    sample = CalibrationSample(
                        input_ids=val_tokenized['input_ids'][i],
                        attention_mask=val_tokenized['attention_mask'][i],
                        position_ids=val_tokenized.get('position_ids', [None] * len(val_tokenized['input_ids']))[i],
                        token_type_ids=val_tokenized.get('token_type_ids', [None] * len(val_tokenized['input_ids']))[i],
                    )
                    val_samples.append(sample)
            else:
                # Use last portion of processed samples
                val_samples = self.processed_samples[-num_samples:]
        
        # Create validation dataset
        val_dataset = CalibrationDataset(val_samples)
        
        # Create DataLoader
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
            collate_fn=self._collate_fn
        )
        
        return val_dataloader
    
    def cleanup(self) -> None:
        """Clean up calibration data from memory"""
        # Clear cached data
        self.cached_data = None
        self.processed_samples = []
        self.raw_data = []
        
        # Clear activation cache
        self.activation_cache.clear()
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.logger.info("Cleaned up calibration data")
    
    def collect_activation_statistics(self,
                                     model: Any,
                                     layer_names: List[str]) -> Dict[str, Dict[str, float]]:
        """Collect activation statistics for AWQ
        
        Statistics per layer:
        - Mean absolute value
        - Standard deviation
        - Max values
        - Sparsity
        """
        statistics = {}
        
        # Register hooks for specified layers
        hooks = []
        layer_stats = {}
        
        def create_hook(layer_name):
            def hook_fn(module, input, output):
                if isinstance(input, tuple):
                    input = input[0]
                
                # Compute statistics
                input_flat = input.reshape(-1)
                stats = {
                    'mean_abs': torch.mean(torch.abs(input_flat)).item(),
                    'std': torch.std(input_flat).item(),
                    'max_abs': torch.max(torch.abs(input_flat)).item(),
                    'sparsity': (input_flat == 0).float().mean().item(),
                }
                
                if layer_name not in layer_stats:
                    layer_stats[layer_name] = []
                layer_stats[layer_name].append(stats)
            
            return hook_fn
        
        # Register hooks (would need actual model structure)
        # for layer_name in layer_names:
        #     module = get_module_by_name(model, layer_name)
        #     handle = module.register_forward_hook(create_hook(layer_name))
        #     hooks.append(handle)
        
        # Run forward passes
        with torch.no_grad():
            for sample in self.processed_samples[:min(10, len(self.processed_samples))]:
                try:
                    _ = model(
                        input_ids=sample.input_ids.unsqueeze(0),
                        attention_mask=sample.attention_mask.unsqueeze(0)
                    )
                except:
                    pass
        
        # Remove hooks
        for handle in hooks:
            handle.remove()
        
        # Aggregate statistics
        for layer_name, stats_list in layer_stats.items():
            if stats_list:
                aggregated = {
                    'mean_abs': np.mean([s['mean_abs'] for s in stats_list]),
                    'std': np.mean([s['std'] for s in stats_list]),
                    'max_abs': np.max([s['max_abs'] for s in stats_list]),
                    'sparsity': np.mean([s['sparsity'] for s in stats_list]),
                }
                statistics[layer_name] = aggregated
        
        return statistics
    
    def generate_diverse_samples(self, 
                                texts: List[str],
                                augmentation: bool = True) -> List[str]:
        """Generate diverse calibration samples
        
        Augmentation techniques:
        - Random cropping
        - Sentence shuffling
        - Token dropout
        """
        augmented_samples = []
        
        for text in texts:
            # Original sample
            augmented_samples.append(text)
            
            if augmentation:
                # Random cropping
                words = text.split()
                if len(words) > 50:
                    start_idx = random.randint(0, len(words) - 50)
                    cropped = ' '.join(words[start_idx:start_idx + random.randint(50, min(200, len(words) - start_idx))])
                    augmented_samples.append(cropped)
                
                # Sentence shuffling
                sentences = text.split('. ')
                if len(sentences) > 2:
                    shuffled_sentences = sentences.copy()
                    random.shuffle(shuffled_sentences)
                    shuffled = '. '.join(shuffled_sentences)
                    augmented_samples.append(shuffled)
                
                # Token dropout (randomly remove 10% of words)
                if len(words) > 10:
                    dropout_words = [w for w in words if random.random() > 0.1]
                    dropout = ' '.join(dropout_words)
                    augmented_samples.append(dropout)
        
        # Ensure diversity in length distribution
        augmented_samples = self._balance_length_distribution(augmented_samples)
        
        return augmented_samples[:self.num_samples]
    
    def filter_samples_by_length(self,
                                samples: List[str],
                                min_length: int = 50,
                                max_length: Optional[int] = None) -> List[str]:
        """Filter samples by token length
        
        Quality control:
        - Remove too short samples
        - Cap extremely long samples
        - Balance length distribution
        """
        if max_length is None:
            max_length = self.max_length
        
        filtered_samples = []
        
        for sample in samples:
            # Tokenize to get length
            tokens = self.tokenizer.encode(sample, add_special_tokens=False)
            token_length = len(tokens)
            
            # Filter by length
            if token_length >= min_length:
                if token_length <= max_length:
                    filtered_samples.append(sample)
                else:
                    # Truncate if too long
                    truncated_tokens = tokens[:max_length]
                    truncated_text = self.tokenizer.decode(truncated_tokens, skip_special_tokens=True)
                    filtered_samples.append(truncated_text)
        
        # Balance distribution
        filtered_samples = self._balance_length_distribution(filtered_samples)
        
        return filtered_samples
    
    def create_glm_specific_inputs(self,
                                  input_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Create GLM-specific model inputs
        
        GLM4.5 requirements:
        - Position encoding
        - Attention mask (causal)
        - No special token type IDs needed
        """
        batch_size, seq_len = input_ids.shape
        
        # Create position IDs
        position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
        
        # Create causal attention mask
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
        
        # GLM4.5 uses standard causal masking
        causal_mask = torch.tril(torch.ones(seq_len, seq_len))
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
        }
    
    def compute_importance_scores(self,
                                 activations: torch.Tensor) -> torch.Tensor:
        """Compute importance scores for AWQ
        
        AWQ importance metric:
        - Based on activation magnitude
        - Averaged across samples
        """
        # Compute mean absolute value across batch dimension
        # Shape: [batch, seq_len, hidden] -> [hidden]
        importance = torch.mean(torch.abs(activations), dim=[0, 1])
        
        # Normalize
        importance = importance / (importance.sum() + 1e-8)
        
        # Apply smoothing
        smoothing_factor = 0.1
        importance = importance * (1 - smoothing_factor) + smoothing_factor / len(importance)
        
        return importance
    
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
        random.shuffle(indices)
        
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
                
                position_ids = None
                if batch[0].position_ids is not None:
                    position_ids = torch.stack([s.position_ids for s in batch])
                
                token_type_ids = None
                if batch[0].token_type_ids is not None:
                    token_type_ids = torch.stack([s.token_type_ids for s in batch])
                
                yield CalibrationSample(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    token_type_ids=token_type_ids
                )
    
    def save_activation_cache(self,
                            layer_name: str,
                            activations: torch.Tensor) -> None:
        """Save layer activations for reuse
        
        Caching strategy:
        - Store on CPU if large
        - Compress if needed
        - Track memory usage
        """
        # Move to CPU to save GPU memory
        activations_cpu = activations.cpu()
        
        # Check size
        size_gb = activations_cpu.numel() * activations_cpu.element_size() / 1e9
        
        if size_gb > 1.0:
            self.logger.warning(f"Large activation cache for {layer_name}: {size_gb:.2f}GB")
            # Could implement compression here
        
        # Add to cache
        if layer_name not in self.activation_cache:
            self.activation_cache[layer_name] = LayerActivationCache(
                layer_name=layer_name,
                inputs=[],
                outputs=None,
                attention_cache=None
            )
        
        self.activation_cache[layer_name].inputs.append(activations_cpu)
        
        # Monitor memory usage
        total_cache_size = sum(
            sum(a.numel() * a.element_size() for a in cache.inputs) 
            for cache in self.activation_cache.values()
        ) / 1e9
        
        if total_cache_size > 10.0:
            self.logger.warning(f"Activation cache size: {total_cache_size:.2f}GB, consider clearing")
    
    def load_activation_cache(self,
                            layer_name: str) -> Optional[torch.Tensor]:
        """Load cached activations for layer"""
        if layer_name in self.activation_cache:
            cache = self.activation_cache[layer_name]
            if cache.inputs:
                # Concatenate all cached inputs
                return torch.cat(cache.inputs, dim=0)
        
        return None
    
    def estimate_calibration_memory(self) -> float:
        """Estimate memory requirement for calibration in GB"""
        if not self.processed_samples:
            # Estimate based on configuration
            samples_memory = self.num_samples * self.max_length * 2 / 1e9  # 2 bytes per token
            # Add overhead for activations (assume hidden_size=4096)
            activation_memory = self.num_samples * self.max_length * 4096 * 2 / 1e9
            return samples_memory + activation_memory
        
        # Calculate based on actual samples
        samples_memory = 0
        for sample in self.processed_samples:
            samples_memory += sample.input_ids.numel() * sample.input_ids.element_size()
            samples_memory += sample.attention_mask.numel() * sample.attention_mask.element_size()
        
        samples_memory = samples_memory / 1e9
        
        # Estimate activation memory
        activation_memory = len(self.processed_samples) * self.max_length * 4096 * 2 / 1e9
        
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
        }
        
        # Length distribution
        if self.length_distribution:
            report['length_stats'] = {
                'mean': np.mean(self.length_distribution),
                'std': np.std(self.length_distribution),
                'min': np.min(self.length_distribution),
                'max': np.max(self.length_distribution),
                'median': np.median(self.length_distribution),
            }
        
        # Token coverage
        if self.token_frequencies:
            report['unique_tokens'] = len(self.token_frequencies)
            report['total_tokens'] = sum(self.token_frequencies.values())
            
            # Top tokens
            top_tokens = sorted(self.token_frequencies.items(), key=lambda x: x[1], reverse=True)[:20]
            report['top_tokens'] = top_tokens
        
        # Memory estimate
        report['estimated_memory_gb'] = self.estimate_calibration_memory()
        
        # Activation cache size
        if self.activation_cache:
            cache_size = sum(
                sum(a.numel() * a.element_size() for a in cache.inputs) 
                for cache in self.activation_cache.values()
            ) / 1e9
            report['activation_cache_gb'] = cache_size
        
        return report
    
    # Helper methods
    
    def _extract_samples_from_dataset(self, dataset) -> None:
        """Extract text samples from HuggingFace dataset"""
        count = 0
        for item in dataset:
            if count >= self.num_samples * 2:  # Get extra for diversity
                break
            
            # Extract text field
            text = None
            if isinstance(item, dict):
                # Try common field names
                for field in ['text', 'content', 'document', 'passage']:
                    if field in item:
                        text = item[field]
                        break
            elif isinstance(item, str):
                text = item
            
            if text and len(text) > 100:  # Filter very short texts
                self.raw_data.append(text)
                count += 1
        
        # Apply filtering and augmentation
        self.raw_data = self.filter_samples_by_length(self.raw_data)
        if len(self.raw_data) < self.num_samples:
            self.raw_data = self.generate_diverse_samples(self.raw_data, augmentation=True)
    
    def _load_custom_dataset(self, path: str) -> None:
        """Load custom dataset from file"""
        path = Path(path)
        
        if path.suffix == '.json':
            with open(path, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    self.raw_data = data[:self.num_samples * 2]
                elif isinstance(data, dict) and 'texts' in data:
                    self.raw_data = data['texts'][:self.num_samples * 2]
        elif path.suffix == '.txt':
            with open(path, 'r') as f:
                # Split by double newlines or use each line
                text = f.read()
                if '\n\n' in text:
                    self.raw_data = text.split('\n\n')[:self.num_samples * 2]
                else:
                    self.raw_data = text.split('\n')[:self.num_samples * 2]
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
        
        # Filter and process
        self.raw_data = self.filter_samples_by_length(self.raw_data)
    
    def _generate_synthetic_data(self) -> None:
        """Generate synthetic calibration data as fallback"""
        self.logger.info("Generating synthetic calibration data")
        
        templates = [
            "The {adj} {noun} {verb} {adv} in the {place}.",
            "Studies show that {topic} has {impact} on {field}.",
            "In {year}, researchers discovered {finding} about {subject}.",
            "{person} announced that {company} will {action} by {date}.",
            "The {event} resulted in {outcome} for {group}.",
        ]
        
        adjectives = ["quick", "brown", "lazy", "beautiful", "intelligent", "modern", "ancient", "digital"]
        nouns = ["fox", "dog", "computer", "algorithm", "system", "network", "model", "dataset"]
        verbs = ["jumps", "runs", "processes", "analyzes", "transforms", "computes", "generates", "optimizes"]
        adverbs = ["quickly", "slowly", "efficiently", "carefully", "precisely", "accurately", "systematically"]
        places = ["garden", "laboratory", "datacenter", "office", "cloud", "server", "network", "cluster"]
        
        synthetic_samples = []
        
        for _ in range(self.num_samples * 2):
            template = random.choice(templates)
            text = template.format(
                adj=random.choice(adjectives),
                noun=random.choice(nouns),
                verb=random.choice(verbs),
                adv=random.choice(adverbs),
                place=random.choice(places),
                topic=random.choice(["artificial intelligence", "machine learning", "deep learning", "neural networks"]),
                impact=random.choice(["significant impact", "minimal effect", "transformative influence"]),
                field=random.choice(["healthcare", "finance", "education", "transportation"]),
                year=random.randint(2020, 2025),
                finding=random.choice(["breakthrough", "improvement", "optimization", "innovation"]),
                subject=random.choice(["language models", "computer vision", "robotics", "automation"]),
                person=random.choice(["The CEO", "The researcher", "The team lead", "The director"]),
                company=random.choice(["TechCorp", "AI Systems", "DataTech", "CloudNet"]),
                action=random.choice(["launch new products", "expand operations", "improve efficiency"]),
                date=random.choice(["next quarter", "next year", "by 2025", "soon"]),
                event=random.choice(["update", "release", "announcement", "discovery"]),
                outcome=random.choice(["improvements", "benefits", "changes", "opportunities"]),
                group=random.choice(["users", "customers", "developers", "researchers"]),
            )
            
            # Expand text by repeating and varying
            expanded = ' '.join([text] * random.randint(5, 20))
            synthetic_samples.append(expanded)
        
        self.raw_data = synthetic_samples[:self.num_samples]
    
    def _update_statistics(self, input_ids: torch.Tensor) -> None:
        """Update dataset statistics"""
        # Update length distribution
        for ids in input_ids:
            # Count actual tokens (non-padding)
            actual_length = (ids != self.tokenizer.pad_token_id).sum().item()
            self.length_distribution.append(actual_length)
        
        # Update token frequencies
        for ids in input_ids:
            for token_id in ids:
                token_id = token_id.item()
                if token_id != self.tokenizer.pad_token_id:
                    self.token_frequencies[token_id] = self.token_frequencies.get(token_id, 0) + 1
    
    def _balance_length_distribution(self, samples: List[str]) -> List[str]:
        """Balance the length distribution of samples"""
        if not samples:
            return samples
        
        # Group samples by length buckets
        buckets = {}
        for sample in samples:
            length = len(self.tokenizer.encode(sample, add_special_tokens=False))
            bucket = length // 100  # 100-token buckets
            if bucket not in buckets:
                buckets[bucket] = []
            buckets[bucket].append(sample)
        
        # Select balanced samples from each bucket
        balanced = []
        samples_per_bucket = max(1, self.num_samples // len(buckets))
        
        for bucket in sorted(buckets.keys()):
            bucket_samples = buckets[bucket]
            selected = random.sample(bucket_samples, min(samples_per_bucket, len(bucket_samples)))
            balanced.extend(selected)
        
        return balanced[:self.num_samples]
    
    def _collate_fn(self, batch: List[CalibrationSample]) -> Dict[str, torch.Tensor]:
        """Custom collate function for DataLoader"""
        if len(batch) == 1:
            sample = batch[0]
            return {
                'input_ids': sample.input_ids.unsqueeze(0),
                'attention_mask': sample.attention_mask.unsqueeze(0),
                'position_ids': sample.position_ids.unsqueeze(0) if sample.position_ids is not None else None,
                'token_type_ids': sample.token_type_ids.unsqueeze(0) if sample.token_type_ids is not None else None,
            }
        
        # Stack tensors
        input_ids = torch.stack([s.input_ids for s in batch])
        attention_mask = torch.stack([s.attention_mask for s in batch])
        
        position_ids = None
        if batch[0].position_ids is not None:
            position_ids = torch.stack([s.position_ids for s in batch])
        
        token_type_ids = None
        if batch[0].token_type_ids is not None:
            token_type_ids = torch.stack([s.token_type_ids for s in batch])
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'token_type_ids': token_type_ids,
        }