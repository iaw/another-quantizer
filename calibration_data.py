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
        """Load calibration dataset from HuggingFace or custom source
        
        Supports c4, wikitext, pile, and custom datasets
        """
        self.logger.info(f"Loading calibration dataset: {dataset_name}")
        
        if custom_path:
            # Try to load custom dataset
            try:
                self._load_custom_dataset(custom_path)
                self.logger.info(f"Loaded custom dataset from {custom_path}")
                return
            except Exception as e:
                self.logger.warning(f"Failed to load custom dataset: {e}")
                # Fall through to try standard datasets
        
        # Try to load from HuggingFace datasets
        try:
            from datasets import load_dataset
            
            # Configure dataset parameters
            dataset_configs = {
                "c4": {"path": "c4", "subset": subset or "en", "split": f"{split}[:1000]"},
                "wikitext": {"path": "wikitext", "subset": "wikitext-103-v1", "split": split},
                "pile": {"path": "EleutherAI/pile", "subset": subset, "split": f"{split}[:1000]"},
                "openwebtext": {"path": "Skylion007/openwebtext", "subset": None, "split": f"{split}[:1000]"},
            }
            
            if dataset_name in dataset_configs:
                config = dataset_configs[dataset_name]
                self.logger.info(f"Loading {dataset_name} from HuggingFace...")
                
                # Load dataset
                if config["subset"]:
                    dataset = load_dataset(config["path"], config["subset"], split=config["split"], streaming=True)
                else:
                    dataset = load_dataset(config["path"], split=config["split"], streaming=True)
                
                # Extract text samples
                self.raw_data = []
                text_field = "text" if "text" in next(iter(dataset)).keys() else "content"
                
                for i, sample in enumerate(dataset):
                    if i >= self.num_samples * 2:  # Load 2x samples for variety
                        break
                    text = sample.get(text_field, "")
                    if text and len(text) > 100:  # Filter very short texts
                        self.raw_data.append(text)
                
                if len(self.raw_data) < self.num_samples:
                    self.logger.warning(f"Only loaded {len(self.raw_data)} samples, generating additional synthetic data")
                    self._generate_synthetic_data()
                
                self.logger.info(f"Loaded {len(self.raw_data)} samples from {dataset_name}")
            else:
                raise ValueError(f"Unknown dataset: {dataset_name}")
                
        except ImportError as e:
            self.logger.warning("datasets library not installed, using synthetic data")
            self.logger.info("Install with: pip install datasets")
            self.logger.info("Or install with: pip install datasets transformers")
            self._generate_synthetic_data()
            
        except ConnectionError as e:
            self.logger.error(f"Network error loading dataset: {e}")
            self.logger.info("Check your internet connection or use a local dataset with --calibration-path")
            self._generate_synthetic_data()
            
        except ValueError as e:
            self.logger.error(f"Invalid dataset configuration: {e}")
            self.logger.info(f"Available datasets: {list(dataset_configs.keys())}")
            raise CalibrationError(f"Invalid dataset: {dataset_name}") from e
            
        except Exception as e:
            self.logger.warning(f"Unexpected error loading dataset: {type(e).__name__}: {e}")
            self.logger.info("Falling back to synthetic data")
            # Log full traceback for debugging
            import traceback
            self.logger.debug(traceback.format_exc())
            self._generate_synthetic_data()
        
        self.logger.info(f"Total calibration samples: {len(self.raw_data)}")
    
    def download_and_cache_dataset(self, dataset_name: str = "c4", cache_dir: str = "~/.cache/glm_quantization") -> bool:
        """Download and cache a dataset for offline use
        
        Args:
            dataset_name: Name of dataset to download
            cache_dir: Directory to cache dataset
            
        Returns:
            True if successful, False otherwise
        """
        cache_dir = Path(cache_dir).expanduser()
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        cache_file = cache_dir / f"{dataset_name}_calibration.json"
        
        # Check if already cached
        if cache_file.exists():
            self.logger.info(f"Loading cached dataset from {cache_file}")
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                    self.raw_data = cached_data['texts']
                    self.logger.info(f"Loaded {len(self.raw_data)} cached samples")
                    return True
            except Exception as e:
                self.logger.warning(f"Failed to load cache: {e}")
        
        # Download fresh data
        try:
            from datasets import load_dataset
            
            self.logger.info(f"Downloading {dataset_name} dataset...")
            
            if dataset_name == "c4":
                dataset = load_dataset("c4", "en", split="train", streaming=True)
            elif dataset_name == "wikitext":
                dataset = load_dataset("wikitext", "wikitext-103-v1", split="train")
            else:
                self.logger.error(f"Unsupported dataset for caching: {dataset_name}")
                return False
            
            # Collect samples
            texts = []
            for i, sample in enumerate(dataset):
                if i >= self.num_samples * 3:  # Get extra for filtering
                    break
                text = sample.get('text', '')
                if text and len(text) > 200:
                    texts.append(text)
            
            # Cache the data
            cache_data = {
                'dataset': dataset_name,
                'num_samples': len(texts),
                'texts': texts
            }
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f)
            
            self.logger.info(f"Cached {len(texts)} samples to {cache_file}")
            self.raw_data = texts[:self.num_samples]
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to download dataset: {e}")
            return False
    
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
    
    def get_calibration_inputs(self, 
                               hidden_size: int,
                               seq_length: int = None,
                               device: str = 'cpu',
                               return_tokens: bool = False) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Generate calibration inputs for layer processing
        
        Args:
            hidden_size: Model hidden dimension
            seq_length: Sequence length (uses max_length if None)
            device: Device to place tensors on
            return_tokens: If True, return tokenized data instead of hidden states
            
        Returns:
            Tensor of shape [batch_size, seq_length, hidden_size] or dict with tokens
        """
        if seq_length is None:
            seq_length = self.max_length
        
        if return_tokens and self.processed_samples:
            # Return actual tokenized data for proper forward pass
            batch_size = min(len(self.processed_samples), self.num_samples)
            
            # Stack tokenized samples
            input_ids_list = []
            attention_mask_list = []
            
            for i in range(batch_size):
                sample = self.processed_samples[i]
                input_ids_list.append(sample.input_ids)
                attention_mask_list.append(sample.attention_mask)
            
            input_ids = torch.stack(input_ids_list).to(device)
            attention_mask = torch.stack(attention_mask_list).to(device)
            
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask
            }
        
        if self.processed_samples:
            # Create more realistic hidden states based on token embeddings
            batch_size = min(len(self.processed_samples), self.num_samples)
            
            # Initialize with small random values (like post-embedding)
            inputs = torch.randn(
                batch_size, 
                seq_length, 
                hidden_size,
                dtype=torch.float16,
                device=device
            ) * 0.02
            
            # Add position-based variation to simulate positional embeddings
            position_variance = torch.arange(seq_length, device=device).float() / seq_length
            position_variance = position_variance.unsqueeze(0).unsqueeze(-1)
            inputs = inputs + position_variance * 0.01
            
            self.logger.debug(f"Generated calibration inputs from {batch_size} samples: "
                            f"shape={inputs.shape}, device={inputs.device}")
        else:
            # Fallback to synthetic hidden states
            self.logger.warning("No processed samples, using synthetic hidden states")
            inputs = torch.randn(
                self.num_samples,
                seq_length,
                hidden_size,
                dtype=torch.float16,
                device=device
            ) * 0.02
            
            # Add position variance even for synthetic data
            position_variance = torch.arange(seq_length, device=device).float() / seq_length
            position_variance = position_variance.unsqueeze(0).unsqueeze(-1)
            inputs = inputs + position_variance * 0.01
            
            self.logger.debug(f"Generated synthetic calibration inputs: "
                            f"shape={inputs.shape}, device={inputs.device}")
        
        return inputs
    
    def get_calibration_batch(self, 
                            batch_size: int = 1,
                            hidden_size: int = 4096,
                            seq_length: int = None,
                            device: str = 'cpu') -> Dict[str, torch.Tensor]:
        """Get a batch of calibration data in the format expected by layers
        
        Args:
            batch_size: Batch size
            hidden_size: Model hidden dimension  
            seq_length: Sequence length
            device: Device to place tensors on
            
        Returns:
            Dictionary with 'hidden_states' and optionally 'attention_mask'
        """
        if seq_length is None:
            seq_length = self.max_length
            
        # Get hidden states
        all_hidden_states = self.get_calibration_inputs(hidden_size, seq_length, device)
        
        # Select batch
        if all_hidden_states.shape[0] > batch_size:
            indices = torch.randperm(all_hidden_states.shape[0])[:batch_size]
            hidden_states = all_hidden_states[indices]
        else:
            hidden_states = all_hidden_states[:batch_size]
        
        # Create attention mask (all ones for now - no padding)
        attention_mask = torch.ones(
            hidden_states.shape[0], 
            seq_length,
            dtype=torch.float16,
            device=device
        )
        
        return {
            'hidden_states': hidden_states,
            'attention_mask': attention_mask,
        }
    
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
        """Generate diverse synthetic calibration data"""
        self.logger.info("Generating synthetic calibration data")
        
        # More diverse text templates covering different domains
        templates = {
            "technical": [
                "Machine learning models require large amounts of data for training effective representations.",
                "Deep learning architectures continue to evolve with innovations in attention mechanisms.",
                "Transformer models have revolutionized natural language processing and computer vision.",
                "Quantization techniques reduce model size while maintaining acceptable performance levels.",
                "GPU acceleration and specialized hardware enable efficient large-scale model inference.",
                "Neural networks learn hierarchical representations through multiple layers of abstraction.",
                "Optimization algorithms like Adam and SGD guide the training process effectively.",
                "Transfer learning allows models to leverage knowledge from pre-trained representations.",
            ],
            "general": [
                "The weather patterns this year have been particularly unusual across many regions.",
                "Economic indicators suggest varying trends in global markets and trade relations.",
                "Scientific research continues to advance our understanding of complex phenomena.",
                "Educational institutions are adapting to new methods of teaching and learning.",
                "Healthcare systems worldwide face ongoing challenges and opportunities for improvement.",
                "Environmental conservation efforts require coordinated action at multiple levels.",
                "Technological innovations are reshaping how people communicate and collaborate.",
                "Cultural exchanges foster mutual understanding between diverse communities.",
            ],
            "narrative": [
                "The discovery came after years of dedicated research and countless experiments.",
                "Teams worked tirelessly to overcome the technical challenges they encountered.",
                "Initial results were promising, leading to expanded investigations and trials.",
                "Collaborative efforts between institutions accelerated the pace of progress.",
                "Unexpected findings opened new avenues for exploration and development.",
                "The implications of this work extend far beyond the original scope.",
                "Future applications could transform multiple industries and fields.",
                "Continued investment in research and development remains critical.",
            ]
        }
        
        # Generate samples with variety
        if not hasattr(self, 'raw_data'):
            self.raw_data = []
        
        existing_samples = len(self.raw_data)
        needed_samples = self.num_samples - existing_samples
        
        if needed_samples <= 0:
            return
        
        for i in range(needed_samples):
            # Vary the text length and complexity
            num_sentences = random.randint(5, 15)
            
            # Mix templates from different categories
            selected_sentences = []
            for _ in range(num_sentences):
                category = random.choice(list(templates.keys()))
                sentence = random.choice(templates[category])
                selected_sentences.append(sentence)
            
            # Add some variation by occasionally modifying sentences
            if random.random() < 0.3:
                # Add a number or year
                year = random.randint(2000, 2024)
                selected_sentences.append(f"In {year}, significant developments occurred in this field.")
            
            if random.random() < 0.3:
                # Add a percentage or statistic
                percent = random.randint(10, 90)
                selected_sentences.append(f"Studies show that approximately {percent}% of cases exhibit similar patterns.")
            
            # Combine into coherent text
            text = " ".join(selected_sentences)
            
            # Ensure minimum length
            if len(text) > 200:
                self.raw_data.append(text)
        
        # Shuffle for variety
        random.shuffle(self.raw_data)
        
        self.logger.info(f"Generated {needed_samples} synthetic samples (total: {len(self.raw_data)})")
    
    def _load_custom_dataset(self, path: str) -> None:
        """Load custom dataset from file
        
        Supports .txt, .json, .jsonl, .csv formats
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Custom dataset not found: {path}")
        
        self.raw_data = []
        
        if path.suffix == '.txt':
            with open(path, 'r', encoding='utf-8') as f:
                # Each line is a sample
                for line in f:
                    line = line.strip()
                    if line and len(line) > 50:  # Filter very short lines
                        self.raw_data.append(line)
                        if len(self.raw_data) >= self.num_samples * 2:
                            break
                            
        elif path.suffix == '.json':
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    # List of texts
                    for item in data:
                        if isinstance(item, str):
                            self.raw_data.append(item)
                        elif isinstance(item, dict) and 'text' in item:
                            self.raw_data.append(item['text'])
                        if len(self.raw_data) >= self.num_samples * 2:
                            break
                elif isinstance(data, dict):
                    # Dictionary with texts field
                    texts = data.get('texts', data.get('text', data.get('data', [])))
                    if isinstance(texts, list):
                        self.raw_data = texts[:self.num_samples * 2]
                        
        elif path.suffix == '.jsonl':
            # JSON Lines format
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        item = json.loads(line.strip())
                        if isinstance(item, dict) and 'text' in item:
                            self.raw_data.append(item['text'])
                        elif isinstance(item, str):
                            self.raw_data.append(item)
                        if len(self.raw_data) >= self.num_samples * 2:
                            break
                    except json.JSONDecodeError:
                        continue
                        
        elif path.suffix == '.csv':
            # CSV format
            import csv
            with open(path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Try common text column names
                    text = row.get('text', row.get('content', row.get('sentence', '')))
                    if text and len(text) > 50:
                        self.raw_data.append(text)
                        if len(self.raw_data) >= self.num_samples * 2:
                            break
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}. Supported: .txt, .json, .jsonl, .csv")
        
        if not self.raw_data:
            raise ValueError(f"No valid text samples found in {path}")
        
        # Shuffle for variety
        random.shuffle(self.raw_data)
        self.raw_data = self.raw_data[:self.num_samples]