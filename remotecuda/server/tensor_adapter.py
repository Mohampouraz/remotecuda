"""
Tensor Adapter — PyTorch ↔ Remote GPU Bridge
============================================
Converts between PyTorch tensors (client) and remote GPU tensors.
Handles serialization, shape tracking, and device emulation.
"""

from typing import Dict, Optional
import torch
import numpy as np


class TensorAdapter:
    """
    Bridges PyTorch tensors to remote GPU tensors.
    
    Maintains a registry mapping local PyTorch tensor IDs to remote GPU tensor IDs.
    Provides transparent conversion between the two representations.
    """
    
    def __init__(self, connection):
        """
        Args:
            connection: GPUConnection instance.
        """
        self._conn = connection
        self._local_to_remote: Dict[int, int] = {}  # id(local_tensor) → remote_tensor_id
        self._remote_to_local: Dict[int, torch.Tensor] = {}  # remote_tensor_id → local_cache
        self._remote_info: Dict[int, dict] = {}  # remote_tensor_id → {shape, dtype}
    
    def send_tensor(self, tensor: torch.Tensor) -> int:
        """
        Transfer a PyTorch tensor to the remote GPU.
        
        Args:
            tensor: PyTorch tensor to send.
            
        Returns:
            int: Remote tensor ID.
        """
        local_id = id(tensor)
        
        # Check if already sent
        if local_id in self._local_to_remote:
            return self._local_to_remote[local_id]
        
        # Send to GPU
        remote_id = self._conn.allocate_tensor(tensor)
        
        # Register mapping
        self._local_to_remote[local_id] = remote_id
        self._remote_info[remote_id] = {
            'shape': tuple(tensor.shape),
            'dtype': str(tensor.dtype).replace('torch.', ''),
        }
        
        return remote_id
    
    def get_tensor(self, remote_id: int) -> torch.Tensor:
        """
        Retrieve a tensor from the remote GPU.
        
        Args:
            remote_id: Remote tensor ID.
            
        Returns:
            torch.Tensor: Retrieved tensor.
        """
        # Check cache
        if remote_id in self._remote_to_local:
            return self._remote_to_local[remote_id]
        
        # Fetch from GPU
        tensor = self._conn.get_tensor(remote_id)
        
        # Cache locally
        self._remote_to_local[remote_id] = tensor
        
        return tensor
    
    def get_remote_id(self, tensor: torch.Tensor) -> Optional[int]:
        """Get the remote tensor ID for a local tensor, if registered."""
        return self._local_to_remote.get(id(tensor))
    
    def get_remote_info(self, remote_id: int) -> Optional[dict]:
        """Get info about a remote tensor."""
        return self._remote_info.get(remote_id)
    
    def release_tensor(self, remote_id: int):
        """Release a remote tensor and clear local cache."""
        try:
            self._conn.free_tensor(remote_id)
        except Exception:
            pass
        
        self._remote_info.pop(remote_id, None)
        self._remote_to_local.pop(remote_id, None)
        
        # Clean reverse mapping
        to_remove = [lid for lid, rid in self._local_to_remote.items() if rid == remote_id]
        for lid in to_remove:
            del self._local_to_remote[lid]
    
    def clear(self):
        """Release all remote tensors and clear all caches."""
        for remote_id in list(self._remote_info.keys()):
            self.release_tensor(remote_id)
        
        self._local_to_remote.clear()
        self._remote_to_local.clear()
        self._remote_info.clear()