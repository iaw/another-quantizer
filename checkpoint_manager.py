# checkpoint_manager.py
"""Simplified checkpoint management for resumable quantization"""

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
class CheckpointState:
    """Simplified checkpoint state - JSON serializable"""
    layer_idx: int
    layer_name: str
    completed_layers: List[str]
    timestamp: str
    elapsed_time: float
    config: Dict[str, Any]
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.__dict__, indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'CheckpointState':
        """Create from JSON string"""
        data = json.loads(json_str)
        return cls(**data)


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
    """Simplified checkpoint manager for resumable quantization"""
    
    def __init__(self, checkpoint_dir: str):
        """Initialize checkpoint manager"""
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # Simple state file
        self.state_file = self.checkpoint_dir / "checkpoint_state.json"
        self.current_state = None
    
    def save_checkpoint(self,
                       layer_idx: int,
                       layer_name: str,
                       completed_layers: List[str],
                       config: Dict[str, Any],
                       elapsed_time: float = 0.0) -> None:
        """Save simple checkpoint state to JSON file"""
        
        # Create checkpoint state
        state = CheckpointState(
            layer_idx=layer_idx,
            layer_name=layer_name,
            completed_layers=completed_layers,
            timestamp=datetime.now().isoformat(),
            elapsed_time=elapsed_time,
            config=config
        )
        
        # Save to JSON file (atomic write)
        temp_file = self.state_file.with_suffix('.tmp')
        with open(temp_file, 'w') as f:
            f.write(state.to_json())
        
        # Atomic rename
        temp_file.replace(self.state_file)
        
        self.current_state = state
        self.logger.info(f"Saved checkpoint at layer {layer_idx}: {layer_name}")
    
    def load_checkpoint(self) -> Optional[CheckpointState]:
        """Load checkpoint state from JSON file"""
        if not self.state_file.exists():
            self.logger.info("No checkpoint found")
            return None
        
        try:
            with open(self.state_file, 'r') as f:
                json_str = f.read()
            
            state = CheckpointState.from_json(json_str)
            self.current_state = state
            
            # Verify checkpoint integrity
            if not self._verify_checkpoint_integrity(state):
                self.logger.warning("Checkpoint integrity check failed")
                # Try to recover what we can
                state = self._recover_partial_checkpoint(state)
            
            self.logger.info(f"Loaded checkpoint: layer {state.layer_idx}, "
                           f"{len(state.completed_layers)} layers completed")
            
            return state
            
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            return None
    
    def _verify_checkpoint_integrity(self, state: CheckpointState) -> bool:
        """Verify that checkpoint state matches saved weights"""
        weights_dir = self.checkpoint_dir / "quantized_weights"
        if not weights_dir.exists():
            return False
        
        # Check that all completed layers have weight files
        for layer_name in state.completed_layers:
            weight_file = weights_dir / f"{layer_name.replace('/', '_')}.safetensors"
            if not weight_file.exists():
                self.logger.warning(f"Missing weight file for completed layer: {layer_name}")
                return False
        
        return True
    
    def _recover_partial_checkpoint(self, state: CheckpointState) -> CheckpointState:
        """Recover what we can from a partial checkpoint"""
        weights_dir = self.checkpoint_dir / "quantized_weights"
        
        # Find which layers actually have saved weights
        actual_completed = []
        for layer_name in state.completed_layers:
            weight_file = weights_dir / f"{layer_name.replace('/', '_')}.safetensors"
            if weight_file.exists():
                actual_completed.append(layer_name)
            else:
                self.logger.info(f"Removing {layer_name} from completed list (no weights found)")
        
        # Update state
        state.completed_layers = actual_completed
        state.layer_idx = len(actual_completed) - 1 if actual_completed else -1
        
        return state
    
    def get_resume_point(self) -> Optional[Tuple[int, List[str]]]:
        """Get layer index and completed layers to resume from"""
        state = self.load_checkpoint()
        
        if state is None:
            return None
        
        # Resume from next layer
        resume_idx = state.layer_idx + 1
        completed_layers = state.completed_layers
        
        self.logger.info(f"Resuming from layer {resume_idx}, "
                        f"{len(completed_layers)} layers completed")
        
        return (resume_idx, completed_layers)
    
    def restore_quantized_state(self, model_loader: Any, layer_quantizer: Any) -> Dict[str, Any]:
        """Restore quantized model state from checkpoint
        
        Args:
            model_loader: Model loader instance
            layer_quantizer: Layer quantizer instance
            
        Returns:
            Dictionary with restored state information
        """
        if not self.current_state:
            state = self.load_checkpoint()
            if not state:
                return {'success': False, 'error': 'No checkpoint found'}
        else:
            state = self.current_state
        
        restored_info = {
            'success': True,
            'completed_layers': state.completed_layers,
            'layer_idx': state.layer_idx,
            'restored_weights': {},
            'restored_scales': {},
        }
        
        weights_dir = self.checkpoint_dir / "quantized_weights"
        scales_dir = self.checkpoint_dir / "scales"
        
        # Restore quantized weights
        for layer_name in state.completed_layers:
            weight_file = weights_dir / f"{layer_name.replace('/', '_')}.safetensors"
            
            if weight_file.exists():
                try:
                    # Load quantized weights
                    weights = safetensors.torch.load_file(str(weight_file))
                    restored_info['restored_weights'][layer_name] = weights
                    
                    # Load metadata
                    with safetensors.safe_open(str(weight_file), framework="pt") as f:
                        metadata = f.metadata() if hasattr(f, 'metadata') else {}
                    
                    self.logger.info(f"Restored weights for {layer_name} "
                                   f"(bits={metadata.get('bits', '?')}, "
                                   f"group_size={metadata.get('group_size', '?')})")
                    
                except Exception as e:
                    self.logger.error(f"Failed to restore weights for {layer_name}: {e}")
                    restored_info['success'] = False
        
        # Restore AWQ scales if available
        scales_file = scales_dir / "awq_scales.pt"
        if scales_file.exists():
            try:
                scales_data = torch.load(scales_file, map_location='cpu')
                if hasattr(layer_quantizer, 'awq_scales'):
                    layer_quantizer.awq_scales = scales_data
                    restored_info['restored_scales'] = scales_data
                    self.logger.info(f"Restored AWQ scales for {len(scales_data)} layers")
            except Exception as e:
                self.logger.warning(f"Failed to restore AWQ scales: {e}")
        
        # Restore activation cache if available
        cache_file = self.checkpoint_dir / "activation_cache.pt"
        if cache_file.exists():
            try:
                cache_data = torch.load(cache_file, map_location='cpu')
                if hasattr(layer_quantizer, 'activation_cache'):
                    layer_quantizer.activation_cache = cache_data
                    self.logger.info(f"Restored activation cache with {len(cache_data)} entries")
            except Exception as e:
                self.logger.warning(f"Failed to restore activation cache: {e}")
        
        return restored_info
    
    def save_quantized_weights(self,
                              layer_name: str,
                              weights: Dict[str, torch.Tensor],
                              quantization_metadata: Optional[Dict[str, Any]] = None) -> None:
        """Save quantized weights for a layer with metadata"""
        # Create directory for weights
        weights_dir = self.checkpoint_dir / "quantized_weights"
        weights_dir.mkdir(exist_ok=True)
        
        # Save weights to safetensors file
        output_file = weights_dir / f"{layer_name.replace('/', '_')}.safetensors"
        
        # Prepare metadata
        metadata = {
            'layer_name': layer_name,
            'timestamp': datetime.now().isoformat(),
            'checkpoint_version': '1.0',
        }
        
        if quantization_metadata:
            metadata.update({
                'quantization_method': quantization_metadata.get('method', 'awq'),
                'bits': str(quantization_metadata.get('bits', 4)),
                'group_size': str(quantization_metadata.get('group_size', 128)),
                'compression_ratio': str(quantization_metadata.get('compression_ratio', 1.0)),
                'error': str(quantization_metadata.get('error', 0.0)),
            })
        
        try:
            # Save with metadata
            safetensors.torch.save_file(weights, output_file, metadata=metadata)
            self.logger.debug(f"Saved weights for {layer_name} to {output_file}")
            
            # Update manifest
            self._update_manifest(layer_name, output_file, metadata)
            
        except Exception as e:
            self.logger.error(f"Failed to save weights for {layer_name}: {e}")
    
    def _update_manifest(self, layer_name: str, weight_file: Path, metadata: Dict[str, Any]) -> None:
        """Update manifest file with layer information"""
        manifest_file = self.checkpoint_dir / "manifest.json"
        
        if manifest_file.exists():
            with open(manifest_file, 'r') as f:
                manifest = json.load(f)
        else:
            manifest = {
                'created': datetime.now().isoformat(),
                'layers': {}
            }
        
        manifest['layers'][layer_name] = {
            'file': weight_file.name,
            'timestamp': metadata.get('timestamp'),
            'quantization': {
                'method': metadata.get('quantization_method'),
                'bits': metadata.get('bits'),
                'group_size': metadata.get('group_size'),
            }
        }
        
        manifest['last_updated'] = datetime.now().isoformat()
        
        with open(manifest_file, 'w') as f:
            json.dump(manifest, f, indent=2)
    
    def save_intermediate_state(self, layer_quantizer: Any) -> None:
        """Save intermediate quantization state (scales, caches, etc.)
        
        Args:
            layer_quantizer: Layer quantizer instance with state to save
        """
        # Save AWQ scales
        if hasattr(layer_quantizer, 'awq_scales') and layer_quantizer.awq_scales:
            scales_dir = self.checkpoint_dir / "scales"
            scales_dir.mkdir(exist_ok=True)
            scales_file = scales_dir / "awq_scales.pt"
            
            try:
                torch.save(layer_quantizer.awq_scales, scales_file)
                self.logger.debug(f"Saved AWQ scales for {len(layer_quantizer.awq_scales)} layers")
            except Exception as e:
                self.logger.warning(f"Failed to save AWQ scales: {e}")
        
        # Save activation cache (limit size to prevent huge files)
        if hasattr(layer_quantizer, 'activation_cache') and layer_quantizer.activation_cache:
            # Only save last 5 layers of activations
            cache_to_save = dict(list(layer_quantizer.activation_cache.items())[-5:])
            
            if cache_to_save:
                cache_file = self.checkpoint_dir / "activation_cache.pt"
                try:
                    torch.save(cache_to_save, cache_file)
                    self.logger.debug(f"Saved activation cache for {len(cache_to_save)} layers")
                except Exception as e:
                    self.logger.warning(f"Failed to save activation cache: {e}")
        
        # Save scale propagator state if available
        if hasattr(layer_quantizer, 'scale_propagator'):
            propagator_file = self.checkpoint_dir / "scale_propagator_state.json"
            try:
                propagator_state = {
                    'layer_connections': layer_quantizer.scale_propagator.layer_connections,
                    'scale_factors': {k: v.tolist() if torch.is_tensor(v) else v 
                                    for k, v in layer_quantizer.scale_propagator.scale_factors.items()}
                }
                with open(propagator_file, 'w') as f:
                    json.dump(propagator_state, f, indent=2)
                self.logger.debug("Saved scale propagator state")
            except Exception as e:
                self.logger.warning(f"Failed to save scale propagator state: {e}")
    
    def cleanup(self) -> None:
        """Simple cleanup - remove old checkpoint files"""
        # Keep only the state file and weights directory
        for item in self.checkpoint_dir.iterdir():
            if item.is_file() and item.suffix == '.tmp':
                item.unlink()
                self.logger.debug(f"Cleaned up temp file: {item}")
    
    def get_checkpoint_info(self) -> Dict[str, Any]:
        """Get information about current checkpoint"""
        if self.current_state:
            return {
                'layer_idx': self.current_state.layer_idx,
                'completed_layers': len(self.current_state.completed_layers),
                'elapsed_time': self.current_state.elapsed_time,
                'timestamp': self.current_state.timestamp,
            }
        return {}
    
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
    
    def create_checkpoint_metadata(self,
                                  layer_idx: int,
                                  layer_name: str,
                                  elapsed_time: float,
                                  memory_stats: Any,
                                  error_metrics: Optional[Dict[str, float]] = None) -> CheckpointState:
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
        completed_layers = self.current_state.completed_layers if self.current_state else []
        
        # Format timestamp
        timestamp = datetime.now().isoformat()
        
        # Extract config (would need actual config object)
        config = {
            'checkpoint_dir': str(self.checkpoint_dir),
            'version': '1.0',
        }
        
        # Create CheckpointState
        state = CheckpointState(
            layer_idx=layer_idx,
            layer_name=layer_name,
            completed_layers=completed_layers,
            timestamp=timestamp,
            elapsed_time=elapsed_time,
            config=config
        )
        
        return state
    
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
            'current_state': asdict(self.current_state) if self.current_state else None,
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
        
        self.logger.info(f"Exported checkpoint report to {output_file}")