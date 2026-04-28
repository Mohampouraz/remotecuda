"""
Tensor Compression Module
=========================
High-performance tensor serialization with multiple compression algorithms.
Automatically selects the best algorithm based on tensor properties.
"""

import pickle
import zlib
import lzma
from typing import Optional

import torch
import numpy as np


class TensorCompressor:
    """
    Handles tensor compression and decompression for network transfer.
    
    Supports multiple compression algorithms:
        - zlib (fast, good general purpose)
        - lzma (best compression, slower)
        - None (no compression, for small tensors)
    
    Auto-selects the best method based on tensor size.
    """
    
    # Thresholds for algorithm selection (in bytes)
    SMALL_TENSOR_THRESHOLD = 1024       # 1KB: no compression
    MEDIUM_TENSOR_THRESHOLD = 1024 * 1024  # 1MB: zlib
    # Above 1MB: lzma for best compression
    
    @staticmethod
    def compress_tensor(tensor: torch.Tensor) -> bytes:
        """
        Compress a tensor for network transfer.
        
        Process:
        1. Move to CPU and detach
        2. Serialize metadata + raw bytes
        3. Compress with appropriate algorithm
        
        Args:
            tensor (torch.Tensor): Tensor to compress
            
        Returns:
            bytes: Compressed tensor data
        """
        if isinstance(tensor, tuple):
            return pickle.dumps([TensorCompressor.compress_tensor(t) for t in tensor])
        
        # Move to CPU and detach
        cpu_tensor = tensor.cpu().detach()
        
        # Prepare data package
        if cpu_tensor.is_sparse:
            # Handle sparse tensors differently
            data = {
                'is_sparse': True,
                'shape': cpu_tensor.shape,
                'indices': cpu_tensor.indices().numpy().tobytes(),
                'values': cpu_tensor.values().numpy().tobytes(),
                'dtype': str(cpu_tensor.dtype),
                'requires_grad': cpu_tensor.requires_grad
            }
        else:
            # Dense tensor
            data = {
                'is_sparse': False,
                'shape': cpu_tensor.shape,
                'data': cpu_tensor.numpy().tobytes(),
                'dtype': str(cpu_tensor.dtype),
                'requires_grad': cpu_tensor.requires_grad
            }
        
        # Serialize
        serialized = pickle.dumps(data)
        
        # Choose compression algorithm based on size
        size = len(serialized)
        
        if size < TensorCompressor.SMALL_TENSOR_THRESHOLD:
            # Small tensor: no compression overhead
            return b'\x00' + serialized  # Prefix with algorithm identifier
        elif size < TensorCompressor.MEDIUM_TENSOR_THRESHOLD:
            # Medium tensor: zlib (fast)
            compressed = zlib.compress(serialized, level=1)
            return b'\x01' + compressed  # zlib identifier
        else:
            # Large tensor: lzma (best compression)
            compressed = lzma.compress(serialized)
            return b'\x02' + compressed  # lzma identifier
    
    @staticmethod
    def decompress_tensor(data: bytes, device: torch.device) -> torch.Tensor:
        """
        Decompress a tensor from compressed data.
        
        Args:
            data (bytes): Compressed tensor data
            device (torch.device): Target device for the tensor
            
        Returns:
            torch.Tensor: Decompressed tensor on target device
        """
        # Read algorithm identifier
        algo = data[0]
        compressed_data = data[1:]
        
        # Decompress based on algorithm
        if algo == 0x00:
            serialized = compressed_data
        elif algo == 0x01:
            serialized = zlib.decompress(compressed_data)
        elif algo == 0x02:
            serialized = lzma.decompress(compressed_data)
        else:
            raise ValueError(f"Unknown compression algorithm: {algo}")
        
        # Deserialize
        tensor_data = pickle.loads(serialized)
        
        # Reconstruct tensor
        if tensor_data['is_sparse']:
            # Sparse tensor
            indices = torch.from_numpy(
                np.frombuffer(tensor_data['indices'], dtype=np.int64)
            ).reshape(2, -1)
            values = torch.from_numpy(
                np.frombuffer(tensor_data['values'], dtype=np.dtype(tensor_data['dtype'][6:]))
            )
            tensor = torch.sparse_coo_tensor(indices, values, tensor_data['shape'])
        else:
            # Dense tensor
            tensor = torch.from_numpy(
                np.frombuffer(tensor_data['data'], dtype=np.dtype(tensor_data['dtype'][6:]))
            ).reshape(tensor_data['shape'])
        
        # Set requires_grad
        tensor.requires_grad_(tensor_data['requires_grad'])
        
        # Move to target device
        return tensor.to(device)
    
    @staticmethod
    def benchmark_compression(tensor: torch.Tensor) -> dict:
        """
        Benchmark compression methods on a tensor.
        
        Args:
            tensor (torch.Tensor): Test tensor
            
        Returns:
            dict: Compression statistics for each method
        """
        import time
        
        results = {}
        methods = {
            'none': lambda d: b'\x00' + d,
            'zlib': lambda d: b'\x01' + zlib.compress(d, level=1),
            'lzma': lambda d: b'\x02' + lzma.compress(d)
        }
        
        serialized = pickle.dumps({
            'shape': tensor.shape,
            'data': tensor.numpy().tobytes(),
            'dtype': str(tensor.dtype)
        })
        
        for name, method in methods.items():
            start = time.time()
            compressed = method(serialized)
            comp_time = (time.time() - start) * 1000
            
            ratio = len(compressed) / len(serialized)
            
            results[name] = {
                'original_size': len(serialized),
                'compressed_size': len(compressed),
                'ratio': ratio,
                'time_ms': comp_time
            }
        
        return results