# checkpoint_manager.py
"""Checkpoint management for resumable quantization"""

import json
import shutil
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import torch
import safetensors.torch
from dataclasses import dataclass, asdict
import logging
import time
import os
import pickle


@dataclass
class CheckpointMetadata:
    """Metadata for a checkpoint"""
    checkpoint_id: str
    layer_idx: int
    layer_name: str
    timestamp: str
    elapsed_time: float
    layers_completed: List[str]
    layers_remaining: List[str]
    memory_stats: Dict[str, float]
    quantization_config: Dict[str, Any]
    model_hash: str  # Hash of original model for verification
    checkpoint_version: str = "1.0"
    error_metrics: Optional[Dict[str, float]] = None
    compression_ratio: float = 1.0
    total_original_size_gb: float = 0.0
    total_quantized_size_gb: float = 0.0
    

@dataclass
class LayerCheckpoint:
    """Information for a single quantized layer"""
    layer_name: str
    weight_file: str
    scales_file: str
    metadata: Dict[str, Any]
    size_bytes: int
    hash: str  # For integrity verification
    quantization_error: float = 0.0
    compression_ratio: float = 1.0


class CheckpointManager:
    """Manage checkpoints for resumable quantization"""
    
    def __init__(self, checkpoint_dir: str):
        """Initialize checkpoint manager"""
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.current_checkpoint = None
        self.checkpoint_history = []
        self.layer_checkpoints = {}  # Dict[str, LayerCheckpoint]
        self.logger = logging.getLogger(__name__)
        self.checkpoint_index_file = self.checkpoint_dir / "checkpoint_index.json"
        
        # Load existing checkpoint index if available
        self._load_checkpoint_index()
        
    def save_checkpoint(self,
                       layer_idx: int,
                       layer_name: str,
                       quantized_layers: List[str],
                       metadata: Dict[str, Any]) -> Path:
        """Save checkpoint after quantizing layer
        
        Creates a complete checkpoint with:
        - Checkpoint metadata
        - References to quantized layers
        - Recovery information
        - Progress tracking
        """
        # Generate checkpoint ID (timestamp-based)
        checkpoint_id = f"checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{layer_idx}"
        checkpoint_path = self.checkpoint_dir / checkpoint_id
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Saving checkpoint {checkpoint_id} for layer {layer_name}")
        
        # Extract memory statistics
        memory_stats = metadata.get('memory_stats', {})
        if not memory_stats and 'memory_manager' in metadata:
            memory_manager = metadata['memory_manager']
            current_stats = memory_manager.get_current_memory()
            memory_stats = {
                'gpu_used_gb': current_stats.gpu_used,
                'gpu_total_gb': current_stats.gpu_total,
                'cpu_used_gb': current_stats.cpu_used,
                'cpu_available_gb': current_stats.cpu_available,
            }
        
        # Create checkpoint metadata
        checkpoint_metadata = CheckpointMetadata(
            checkpoint_id=checkpoint_id,
            layer_idx=layer_idx,
            layer_name=layer_name,
            timestamp=datetime.now().isoformat(),
            elapsed_time=metadata.get('elapsed_time', 0.0),
            layers_completed=quantized_layers,
            layers_remaining=metadata.get('layers_remaining', []),
            memory_stats=memory_stats,
            quantization_config=metadata.get('config', {}),
            model_hash=metadata.get('model_hash', self._compute_model_hash(metadata.get('model_path', ''))),
            error_metrics=metadata.get('error_metrics'),
            compression_ratio=metadata.get('compression_ratio', 1.0),
            total_original_size_gb=metadata.get('total_original_size_gb', 0.0),
            total_quantized_size_gb=metadata.get('total_quantized_size_gb', 0.0),
        )
        
        # Save metadata
        metadata_file = checkpoint_path / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(asdict(checkpoint_metadata), f, indent=2)
        
        # Copy/link quantized weights for this layer
        if 'quantized_weights' in metadata:
            self._save_layer_weights(checkpoint_path, layer_name, metadata['quantized_weights'])
        
        # Create recovery manifest
        recovery_manifest = {
            'checkpoint_id': checkpoint_id,
            'layer_idx': layer_idx,
            'layer_name': layer_name,
            'quantized_layers': quantized_layers,
            'layer_checkpoints': {name: asdict(cp) for name, cp in self.layer_checkpoints.items()},
            'timestamp': checkpoint_metadata.timestamp,
        }
        
        manifest_file = checkpoint_path / "recovery_manifest.json"
        with open(manifest_file, 'w') as f:
            json.dump(recovery_manifest, f, indent=2)
        
        # Update checkpoint index
        self.checkpoint_history.append(checkpoint_metadata)
        self._update_checkpoint_index()
        
        # Clean old checkpoints if needed
        self.cleanup_old_checkpoints(keep_last=3)
        
        self.current_checkpoint = checkpoint_metadata
        
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        return checkpoint_path
    
    def load_latest_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Load the most recent valid checkpoint
        
        Finds and loads the latest checkpoint:
        - Verify integrity
        - Load metadata
        - Validate compatibility
        """
        if not self.checkpoint_history:
            self._load_checkpoint_index()
        
        if not self.checkpoint_history:
            self.logger.info("No checkpoints found")
            return None
        
        # Sort by timestamp (newest first)
        sorted_checkpoints = sorted(self.checkpoint_history, 
                                   key=lambda x: x.timestamp, 
                                   reverse=True)
        
        # Find most recent valid checkpoint
        for checkpoint_metadata in sorted_checkpoints:
            checkpoint_path = self.checkpoint_dir / checkpoint_metadata.checkpoint_id
            
            if self.verify_checkpoint(checkpoint_path):
                self.logger.info(f"Loading checkpoint: {checkpoint_metadata.checkpoint_id}")
                return self.restore_from_checkpoint(checkpoint_path)
            else:
                self.logger.warning(f"Invalid checkpoint: {checkpoint_metadata.checkpoint_id}")
        
        return None
    
    def get_resume_point(self) -> Optional[Tuple[int, List[str]]]:
        """Get layer index and completed layers to resume from
        
        Returns:
        - Layer index to start from
        - List of already completed layers
        """
        checkpoint_data = self.load_latest_checkpoint()
        
        if checkpoint_data is None:
            return None
        
        layer_idx = checkpoint_data.get('layer_idx', 0) + 1  # Start from next layer
        completed_layers = checkpoint_data.get('quantized_layers', [])
        
        # Verify completed layers exist
        for layer_name in completed_layers:
            if layer_name not in self.layer_checkpoints:
                self.logger.warning(f"Missing checkpoint for completed layer: {layer_name}")
        
        self.logger.info(f"Resuming from layer {layer_idx}, {len(completed_layers)} layers completed")
        
        return (layer_idx, completed_layers)
    
    def save_quantized_weights(self,
                              layer_name: str,
                              weights: Any,
                              scales: Dict[str, torch.Tensor],
                              metadata: Dict[str, Any]) -> LayerCheckpoint:
        """Save quantized weights with checkpoint
        
        Saves layer weights in recoverable format:
        - Packed INT4 weights
        - Scaling factors
        - Configuration metadata
        """
        # Create layer directory
        layer_dir = self.checkpoint_dir / "layers" / layer_name.replace('/', '_')
        layer_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare weights for saving
        weight_dict = {}
        scale_dict = {}
        
        if hasattr(weights, 'state_dict'):
            # It's a module
            state_dict = weights.state_dict()
            for key, value in state_dict.items():
                if 'weight' in key or 'quantized' in key:
                    weight_dict[key] = value
                elif 'scale' in key:
                    scale_dict[key] = value
                elif 'zero' in key or 'zp' in key:
                    scale_dict[key] = value
        else:
            # Direct tensor
            weight_dict['weight'] = weights
        
        # Add scales
        for key, value in scales.items():
            scale_dict[f"scale_{key}"] = value
        
        # Save weights in safetensors format
        weight_file = layer_dir / "weights.safetensors"
        safetensors.torch.save_file(weight_dict, weight_file)
        
        # Save scales separately
        scales_file = layer_dir / "scales.safetensors"
        if scale_dict:
            safetensors.torch.save_file(scale_dict, scales_file)
        
        # Save metadata
        layer_metadata = {
            'layer_name': layer_name,
            'group_size': metadata.get('group_size', 128),
            'bits': metadata.get('bits', 4),
            'symmetric': metadata.get('symmetric', False),
            'quantization_error': metadata.get('quantization_error', 0.0),
            'compression_ratio': metadata.get('compression_ratio', 1.0),
            'timestamp': datetime.now().isoformat(),
        }
        
        metadata_file = layer_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(layer_metadata, f, indent=2)
        
        # Calculate hash for integrity
        file_hash = self.compute_file_hash(weight_file)
        
        # Calculate size
        size_bytes = weight_file.stat().st_size
        if scales_file.exists():
            size_bytes += scales_file.stat().st_size
        
        # Create LayerCheckpoint object
        layer_checkpoint = LayerCheckpoint(
            layer_name=layer_name,
            weight_file=str(weight_file.relative_to(self.checkpoint_dir)),
            scales_file=str(scales_file.relative_to(self.checkpoint_dir)) if scales_file.exists() else "",
            metadata=layer_metadata,
            size_bytes=size_bytes,
            hash=file_hash,
            quantization_error=layer_metadata['quantization_error'],
            compression_ratio=layer_metadata['compression_ratio'],
        )
        
        # Add to layer_checkpoints dict
        self.layer_checkpoints[layer_name] = layer_checkpoint
        
        self.logger.info(f"Saved quantized weights for {layer_name}: {size_bytes/1e6:.2f}MB")
        
        return layer_checkpoint
    
    def cleanup_old_checkpoints(self, keep_last: int = 3) -> None:
        """Remove old checkpoints, keeping only recent ones
        
        Cleanup strategy:
        - Keep last N checkpoints
        - Keep milestone checkpoints (every 10 layers)
        - Remove temporary files
        """
        if len(self.checkpoint_history) <= keep_last:
            return
        
        # Sort by timestamp
        sorted_checkpoints = sorted(self.checkpoint_history, 
                                   key=lambda x: x.timestamp, 
                                   reverse=True)
        
        # Identify checkpoints to keep
        checkpoints_to_keep = set()
        
        # Keep last N checkpoints
        for checkpoint in sorted_checkpoints[:keep_last]:
            checkpoints_to_keep.add(checkpoint.checkpoint_id)
        
        # Keep milestone checkpoints (every 10 layers)
        for checkpoint in sorted_checkpoints:
            if checkpoint.layer_idx % 10 == 0:
                checkpoints_to_keep.add(checkpoint.checkpoint_id)
        
        # Remove old checkpoint directories
        for checkpoint in sorted_checkpoints:
            if checkpoint.checkpoint_id not in checkpoints_to_keep:
                checkpoint_path = self.checkpoint_dir / checkpoint.checkpoint_id
                if checkpoint_path.exists():
                    self.logger.info(f"Removing old checkpoint: {checkpoint.checkpoint_id}")
                    shutil.rmtree(checkpoint_path)
                    
                    # Remove from history
                    self.checkpoint_history = [
                        cp for cp in self.checkpoint_history 
                        if cp.checkpoint_id != checkpoint.checkpoint_id
                    ]
        
        # Update checkpoint index
        self._update_checkpoint_index()
        
        # Clean temporary files
        self._clean_temp_files()
    
    def create_checkpoint_metadata(self,
                                  layer_idx: int,
                                  layer_name: str,
                                  elapsed_time: float,
                                  memory_stats: Any,
                                  error_metrics: Optional[Dict[str, float]] = None) -> CheckpointMetadata:
        """Create metadata for checkpoint
        
        Captures complete state information:
        - Progress information
        - Configuration
        - Performance metrics
        - Error metrics
        """
        # Generate checkpoint ID
        checkpoint_id = f"checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{layer_idx}"
        
        # Calculate completed/remaining layers
        completed_layers = list(self.layer_checkpoints.keys())
        
        # Format timestamp
        timestamp = datetime.now().isoformat()
        
        # Extract config (would need actual config object)
        config = {
            'checkpoint_dir': str(self.checkpoint_dir),
            'version': '1.0',
        }
        
        # Create CheckpointMetadata
        metadata = CheckpointMetadata(
            checkpoint_id=checkpoint_id,
            layer_idx=layer_idx,
            layer_name=layer_name,
            timestamp=timestamp,
            elapsed_time=elapsed_time,
            layers_completed=completed_layers,
            layers_remaining=[],  # Would need full layer list
            memory_stats=memory_stats if isinstance(memory_stats, dict) else {},
            quantization_config=config,
            model_hash="",  # Would need model path
            error_metrics=error_metrics,
        )
        
        return metadata
    
    def verify_checkpoint(self, checkpoint_path: Path) -> bool:
        """Verify checkpoint integrity
        
        Validation checks:
        - File existence
        - Hash verification
        - Metadata consistency
        - Weight file integrity
        """
        if not checkpoint_path.exists():
            return False
        
        try:
            # Load metadata
            metadata_file = checkpoint_path / "metadata.json"
            if not metadata_file.exists():
                return False
            
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Check recovery manifest
            manifest_file = checkpoint_path / "recovery_manifest.json"
            if not manifest_file.exists():
                return False
            
            with open(manifest_file, 'r') as f:
                manifest = json.load(f)
            
            # Verify layer checkpoints
            for layer_name, checkpoint_info in manifest.get('layer_checkpoints', {}).items():
                weight_file = self.checkpoint_dir / checkpoint_info['weight_file']
                if not weight_file.exists():
                    self.logger.warning(f"Missing weight file: {weight_file}")
                    return False
                
                # Verify hash if available
                if 'hash' in checkpoint_info:
                    computed_hash = self.compute_file_hash(weight_file)
                    if computed_hash != checkpoint_info['hash']:
                        self.logger.warning(f"Hash mismatch for {weight_file}")
                        return False
            
            # Test load a sample weight
            if manifest.get('layer_checkpoints'):
                sample_layer = list(manifest['layer_checkpoints'].keys())[0]
                sample_checkpoint = manifest['layer_checkpoints'][sample_layer]
                weight_file = self.checkpoint_dir / sample_checkpoint['weight_file']
                
                try:
                    # Try loading with safetensors
                    _ = safetensors.torch.load_file(weight_file)
                except Exception as e:
                    self.logger.warning(f"Failed to load sample weight: {e}")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error verifying checkpoint: {e}")
            return False
    
    def list_checkpoints(self) -> List[CheckpointMetadata]:
        """List all available checkpoints
        
        Returns sorted list of checkpoints with metadata
        """
        # Read checkpoint index
        self._load_checkpoint_index()
        
        # Sort by timestamp
        sorted_checkpoints = sorted(self.checkpoint_history, 
                                   key=lambda x: x.timestamp, 
                                   reverse=True)
        
        # Filter out invalid checkpoints
        valid_checkpoints = []
        for checkpoint in sorted_checkpoints:
            checkpoint_path = self.checkpoint_dir / checkpoint.checkpoint_id
            if checkpoint_path.exists():
                valid_checkpoints.append(checkpoint)
        
        return valid_checkpoints
    
    def restore_from_checkpoint(self, checkpoint_path: Path) -> Dict[str, Any]:
        """Restore state from specific checkpoint
        
        Full restoration:
        - Load metadata
        - Verify integrity
        - Load quantized layers
        - Restore configuration
        """
        if not self.verify_checkpoint(checkpoint_path):
            raise ValueError(f"Invalid checkpoint: {checkpoint_path}")
        
        # Load metadata
        metadata_file = checkpoint_path / "metadata.json"
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Load recovery manifest
        manifest_file = checkpoint_path / "recovery_manifest.json"
        with open(manifest_file, 'r') as f:
            manifest = json.load(f)
        
        # Restore layer checkpoints
        self.layer_checkpoints = {}
        for layer_name, checkpoint_dict in manifest.get('layer_checkpoints', {}).items():
            self.layer_checkpoints[layer_name] = LayerCheckpoint(**checkpoint_dict)
        
        # Prepare restoration dict
        restoration_data = {
            'layer_idx': metadata['layer_idx'],
            'layer_name': metadata['layer_name'],
            'quantized_layers': metadata['layers_completed'],
            'config': metadata['quantization_config'],
            'layer_checkpoints': self.layer_checkpoints,
            'checkpoint_metadata': metadata,
            'elapsed_time': metadata.get('elapsed_time', 0.0),
            'error_metrics': metadata.get('error_metrics'),
        }
        
        self.logger.info(f"Restored checkpoint: {metadata['checkpoint_id']}")
        
        return restoration_data
    
    def create_checkpoint_index(self) -> None:
        """Create or update checkpoint index file
        
        Index structure:
        - List of all checkpoints
        - Metadata for each
        - Quick lookup information
        """
        index_data = {
            'version': '1.0',
            'created': datetime.now().isoformat(),
            'checkpoints': [],
            'layer_checkpoints': {},
        }
        
        # Add all checkpoints
        for checkpoint in self.checkpoint_history:
            index_data['checkpoints'].append(asdict(checkpoint))
        
        # Add layer checkpoints
        for layer_name, layer_checkpoint in self.layer_checkpoints.items():
            index_data['layer_checkpoints'][layer_name] = asdict(layer_checkpoint)
        
        # Save index
        with open(self.checkpoint_index_file, 'w') as f:
            json.dump(index_data, f, indent=2)
        
        self.logger.debug(f"Updated checkpoint index with {len(self.checkpoint_history)} checkpoints")
    
    def save_progress_log(self,
                         layer_name: str,
                         metrics: Dict[str, float]) -> None:
        """Save progress log entry
        
        Tracks quantization progress:
        - Layer completion times
        - Memory usage
        - Error metrics
        """
        progress_file = self.checkpoint_dir / "progress.log"
        
        # Create log entry
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'layer_name': layer_name,
            'metrics': metrics,
        }
        
        # Append to progress log file
        with open(progress_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        # Also save as structured JSON for easier parsing
        progress_json_file = self.checkpoint_dir / "progress.json"
        
        if progress_json_file.exists():
            with open(progress_json_file, 'r') as f:
                progress_data = json.load(f)
        else:
            progress_data = {'entries': []}
        
        progress_data['entries'].append(log_entry)
        
        with open(progress_json_file, 'w') as f:
            json.dump(progress_data, f, indent=2)
    
    def get_checkpoint_by_id(self, checkpoint_id: str) -> Optional[CheckpointMetadata]:
        """Get specific checkpoint by ID"""
        for checkpoint in self.checkpoint_history:
            if checkpoint.checkpoint_id == checkpoint_id:
                return checkpoint
        return None
    
    def merge_checkpoints(self,
                         checkpoint_ids: List[str],
                         output_path: Path) -> None:
        """Merge multiple checkpoints into single model
        
        For final model assembly:
        - Collect all quantized layers
        - Merge into single model
        - Create final index
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Merging {len(checkpoint_ids)} checkpoints to {output_path}")
        
        # Load all checkpoints
        all_layers = {}
        
        for checkpoint_id in checkpoint_ids:
            checkpoint = self.get_checkpoint_by_id(checkpoint_id)
            if not checkpoint:
                self.logger.warning(f"Checkpoint not found: {checkpoint_id}")
                continue
            
            checkpoint_path = self.checkpoint_dir / checkpoint_id
            if not checkpoint_path.exists():
                continue
            
            # Load manifest
            manifest_file = checkpoint_path / "recovery_manifest.json"
            if manifest_file.exists():
                with open(manifest_file, 'r') as f:
                    manifest = json.load(f)
                
                # Collect layer checkpoints
                for layer_name, checkpoint_info in manifest.get('layer_checkpoints', {}).items():
                    all_layers[layer_name] = checkpoint_info
        
        # Copy all layer files to output
        weight_map = {}
        
        for layer_name, checkpoint_info in all_layers.items():
            # Copy weight file
            src_weight = self.checkpoint_dir / checkpoint_info['weight_file']
            dst_weight = output_path / f"{layer_name.replace('/', '_')}_weights.safetensors"
            
            if src_weight.exists():
                shutil.copy2(src_weight, dst_weight)
                weight_map[layer_name] = str(dst_weight.name)
            
            # Copy scales file
            if checkpoint_info.get('scales_file'):
                src_scales = self.checkpoint_dir / checkpoint_info['scales_file']
                dst_scales = output_path / f"{layer_name.replace('/', '_')}_scales.safetensors"
                
                if src_scales.exists():
                    shutil.copy2(src_scales, dst_scales)
        
        # Create merged index
        index_data = {
            'weight_map': weight_map,
            'metadata': {
                'quantization_method': 'awq',
                'bits': 4,
                'group_size': 128,
                'merged_from': checkpoint_ids,
                'timestamp': datetime.now().isoformat(),
            }
        }
        
        index_file = output_path / "model.safetensors.index.json"
        with open(index_file, 'w') as f:
            json.dump(index_data, f, indent=2)
        
        self.logger.info(f"Merged {len(all_layers)} layers to {output_path}")
    
    def calculate_checkpoint_size(self) -> float:
        """Calculate total size of checkpoint data in GB"""
        total_size = 0
        
        # Sum all checkpoint directories
        for checkpoint in self.checkpoint_history:
            checkpoint_path = self.checkpoint_dir / checkpoint.checkpoint_id
            if checkpoint_path.exists():
                size = sum(f.stat().st_size for f in checkpoint_path.rglob('*') if f.is_file())
                total_size += size
        
        # Add layer checkpoint files
        layers_dir = self.checkpoint_dir / "layers"
        if layers_dir.exists():
            size = sum(f.stat().st_size for f in layers_dir.rglob('*') if f.is_file())
            total_size += size
        
        return total_size / 1e9
    
    def export_checkpoint_report(self, output_file: Path) -> None:
        """Export detailed checkpoint report
        
        Generates report with:
        - All checkpoints
        - Progress timeline
        - Error metrics
        - Memory usage over time
        """
        output_file = Path(output_file)
        
        report = {
            'generated': datetime.now().isoformat(),
            'checkpoint_dir': str(self.checkpoint_dir),
            'total_checkpoints': len(self.checkpoint_history),
            'total_layers': len(self.layer_checkpoints),
            'total_size_gb': self.calculate_checkpoint_size(),
            'checkpoints': [],
            'layer_summary': {},
            'progress_timeline': [],
        }
        
        # Add checkpoint details
        for checkpoint in self.checkpoint_history:
            checkpoint_data = asdict(checkpoint)
            checkpoint_path = self.checkpoint_dir / checkpoint.checkpoint_id
            checkpoint_data['exists'] = checkpoint_path.exists()
            checkpoint_data['valid'] = self.verify_checkpoint(checkpoint_path) if checkpoint_path.exists() else False
            report['checkpoints'].append(checkpoint_data)
        
        # Add layer summary
        for layer_name, layer_checkpoint in self.layer_checkpoints.items():
            report['layer_summary'][layer_name] = {
                'size_mb': layer_checkpoint.size_bytes / 1e6,
                'compression_ratio': layer_checkpoint.compression_ratio,
                'quantization_error': layer_checkpoint.quantization_error,
            }
        
        # Load progress timeline
        progress_json_file = self.checkpoint_dir / "progress.json"
        if progress_json_file.exists():
            with open(progress_json_file, 'r') as f:
                progress_data = json.load(f)
                report['progress_timeline'] = progress_data.get('entries', [])
        
        # Save report
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Also create markdown version
        md_file = output_file.with_suffix('.md')
        self._create_markdown_report(report, md_file)
        
        self.logger.info(f"Exported checkpoint report to {output_file}")
    
    def validate_resume_compatibility(self,
                                     checkpoint: CheckpointMetadata,
                                     current_config: Dict[str, Any]) -> bool:
        """Validate checkpoint compatibility with current config
        
        Checks:
        - Model architecture match
        - Quantization config compatibility
        - Version compatibility
        """
        # Check version compatibility
        if checkpoint.checkpoint_version != "1.0":
            self.logger.warning(f"Version mismatch: checkpoint={checkpoint.checkpoint_version}, expected=1.0")
            return False
        
        # Compare model hashes
        current_model_hash = self._compute_model_hash(current_config.get('model_path', ''))
        if checkpoint.model_hash and checkpoint.model_hash != current_model_hash:
            self.logger.warning(f"Model hash mismatch")
            # This might be okay if model structure is same
        
        # Check config compatibility
        checkpoint_config = checkpoint.quantization_config
        
        # Check critical parameters
        if checkpoint_config.get('bits') != current_config.get('bits'):
            self.logger.error(f"Bits mismatch: checkpoint={checkpoint_config.get('bits')}, "
                            f"current={current_config.get('bits')}")
            return False
        
        if checkpoint_config.get('group_size') != current_config.get('group_size'):
            self.logger.warning(f"Group size mismatch: checkpoint={checkpoint_config.get('group_size')}, "
                              f"current={current_config.get('group_size')}")
            # This might be okay
        
        return True
    
    def create_recovery_point(self,
                            layer_idx: int,
                            state: Dict[str, Any]) -> str:
        """Create lightweight recovery point
        
        Quick save for crash recovery:
        - Minimal metadata
        - Current state
        - Fast write
        """
        # Generate recovery ID
        recovery_id = f"recovery_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{layer_idx}"
        recovery_file = self.checkpoint_dir / f"{recovery_id}.recovery"
        
        # Save minimal state
        recovery_data = {
            'recovery_id': recovery_id,
            'layer_idx': layer_idx,
            'timestamp': datetime.now().isoformat(),
            'state': state,
        }
        
        # Use atomic write
        temp_file = recovery_file.with_suffix('.tmp')
        with open(temp_file, 'wb') as f:
            pickle.dump(recovery_data, f)
        
        # Atomic rename
        temp_file.rename(recovery_file)
        
        self.logger.debug(f"Created recovery point: {recovery_id}")
        
        return recovery_id
    
    def load_recovery_point(self, recovery_id: str) -> Optional[Dict[str, Any]]:
        """Load recovery point for crash recovery"""
        recovery_file = self.checkpoint_dir / f"{recovery_id}.recovery"
        
        if not recovery_file.exists():
            return None
        
        try:
            with open(recovery_file, 'rb') as f:
                recovery_data = pickle.load(f)
            
            # Validate integrity
            if recovery_data.get('recovery_id') != recovery_id:
                return None
            
            return recovery_data.get('state')
            
        except Exception as e:
            self.logger.error(f"Failed to load recovery point: {e}")
            return None
    
    def compute_file_hash(self, file_path: Path) -> str:
        """Compute hash of file for integrity checking"""
        if not file_path.exists():
            return ""
        
        # Use SHA256
        sha256_hash = hashlib.sha256()
        
        # Read in chunks for large files
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        
        return sha256_hash.hexdigest()
    
    # Helper methods
    
    def _load_checkpoint_index(self) -> None:
        """Load checkpoint index from file"""
        if not self.checkpoint_index_file.exists():
            return
        
        try:
            with open(self.checkpoint_index_file, 'r') as f:
                index_data = json.load(f)
            
            # Load checkpoint history
            self.checkpoint_history = []
            for checkpoint_dict in index_data.get('checkpoints', []):
                self.checkpoint_history.append(CheckpointMetadata(**checkpoint_dict))
            
            # Load layer checkpoints
            self.layer_checkpoints = {}
            for layer_name, checkpoint_dict in index_data.get('layer_checkpoints', {}).items():
                self.layer_checkpoints[layer_name] = LayerCheckpoint(**checkpoint_dict)
            
            self.logger.info(f"Loaded checkpoint index: {len(self.checkpoint_history)} checkpoints, "
                           f"{len(self.layer_checkpoints)} layers")
            
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint index: {e}")
    
    def _update_checkpoint_index(self) -> None:
        """Update the checkpoint index file"""
        self.create_checkpoint_index()
    
    def _save_layer_weights(self, checkpoint_path: Path, layer_name: str, weights: Any) -> None:
        """Save layer weights to checkpoint directory"""
        weights_dir = checkpoint_path / "weights"
        weights_dir.mkdir(exist_ok=True)
        
        weight_file = weights_dir / f"{layer_name.replace('/', '_')}.safetensors"
        
        if isinstance(weights, dict):
            safetensors.torch.save_file(weights, weight_file)
        elif hasattr(weights, 'state_dict'):
            safetensors.torch.save_file(weights.state_dict(), weight_file)
        else:
            # Wrap in dict
            safetensors.torch.save_file({'weight': weights}, weight_file)
    
    def _compute_model_hash(self, model_path: str) -> str:
        """Compute hash of model for verification"""
        if not model_path or not Path(model_path).exists():
            return ""
        
        # Hash config.json for model identity
        config_file = Path(model_path) / "config.json"
        if config_file.exists():
            return self.compute_file_hash(config_file)
        
        return ""
    
    def _clean_temp_files(self) -> None:
        """Clean temporary files from checkpoint directory"""
        # Remove .tmp files
        for tmp_file in self.checkpoint_dir.rglob("*.tmp"):
            try:
                tmp_file.unlink()
            except:
                pass
        
        # Remove old recovery files (older than 1 day)
        cutoff_time = time.time() - 86400  # 1 day
        for recovery_file in self.checkpoint_dir.glob("*.recovery"):
            if recovery_file.stat().st_mtime < cutoff_time:
                try:
                    recovery_file.unlink()
                except:
                    pass
    
    def _create_markdown_report(self, report: Dict[str, Any], output_file: Path) -> None:
        """Create markdown version of checkpoint report"""
        md_content = f"""# Checkpoint Report

Generated: {report['generated']}

## Summary
- Total Checkpoints: {report['total_checkpoints']}
- Total Layers: {report['total_layers']}
- Total Size: {report['total_size_gb']:.2f} GB

## Checkpoints

| ID | Layer | Time | Valid | Size |
|---|---|---|---|---|
"""
        
        for checkpoint in report['checkpoints']:
            valid = "✓" if checkpoint.get('valid', False) else "✗"
            md_content += f"| {checkpoint['checkpoint_id']} | {checkpoint['layer_name']} | {checkpoint['timestamp']} | {valid} | - |\n"
        
        md_content += "\n## Layer Summary\n\n"
        md_content += "| Layer | Size (MB) | Compression | Error |\n"
        md_content += "|---|---|---|---|\n"
        
        for layer_name, summary in report['layer_summary'].items():
            md_content += f"| {layer_name} | {summary['size_mb']:.2f} | {summary['compression_ratio']:.2f}x | {summary['quantization_error']:.6f} |\n"
        
        with open(output_file, 'w') as f:
            f.write(md_content)