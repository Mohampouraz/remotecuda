"""
Tensor Bridge Module
====================
Bridges local tensors with remote GPU tensors seamlessly.
Manages the lifecycle of tensors across the network boundary,
handling serialization, transfer, and synchronization.
"""

import threading
import time
import weakref
from typing import Dict, Optional, Set, Tuple
from collections import defaultdict

import torch
import numpy as np

from ..protocol.compression import TensorCompressor


class TensorHandle:
    """
    Represents a tensor that lives on a remote GPU.
    
    Acts as a local proxy for a tensor stored on a remote GPU worker.
    Maintains metadata locally while the actual data resides on the GPU.
    
    Attributes:
        tensor_id (int): Unique identifier on the remote GPU
        shape (tuple): Tensor shape (maintained locally for quick access)
        dtype (torch.dtype): Tensor data type
        device_info (dict): Information about which GPU holds this tensor
        _local_cache (Optional[torch.Tensor]): Cached local copy when fetched
        _last_access (float): Timestamp of last operation
        _ref_count (int): Reference counting for memory management
    """
    
    __slots__ = [
        'tensor_id', 'shape', 'dtype', 'device_info',
        '_local_cache', '_last_access', '_ref_count', '_lock'
    ]
    
    def __init__(
        self,
        tensor_id: int,
        shape: tuple,
        dtype: torch.dtype,
        device_info: dict
    ):
        """
        Initialize a tensor handle for a remote GPU tensor.
        
        Args:
            tensor_id (int): Unique ID assigned by the GPU server
            shape (tuple): Tensor dimensions
            dtype (torch.dtype): Data type of the tensor
            device_info (dict): Server and GPU identification info
        """
        self.tensor_id = tensor_id
        self.shape = shape
        self.dtype = dtype
        self.device_info = device_info
        
        # Optional local cache for frequently accessed tensors
        self._local_cache: Optional[torch.Tensor] = None
        
        # Tracking
        self._last_access = time.time()
        self._ref_count = 1
        
        # Thread safety
        self._lock = threading.Lock()
    
    def touch(self):
        """
        Update the last access timestamp.
        Used for cache eviction decisions.
        """
        self._last_access = time.time()
    
    def add_ref(self):
        """
        Increment reference count.
        Prevents premature garbage collection of remote tensor.
        """
        with self._lock:
            self._ref_count += 1
    
    def release(self):
        """
        Decrement reference count.
        Returns True if this was the last reference (tensor should be freed).
        
        Returns:
            bool: True if reference count reached zero
        """
        with self._lock:
            self._ref_count -= 1
            return self._ref_count <= 0
    
    @property
    def is_cached_locally(self) -> bool:
        """
        Check if a local CPU copy is available.
        
        Returns:
            bool: True if local cache is populated
        """
        return self._local_cache is not None
    
    def __repr__(self) -> str:
        return (
            f"TensorHandle(id={self.tensor_id}, "
            f"shape={self.shape}, "
            f"dtype={self.dtype}, "
            f"refs={self._ref_count})"
        )


class TensorBridge:
    """
    Manages the complete lifecycle of tensors between local and remote GPU.
    
    Responsibilities:
        - Tensor allocation on remote GPUs
        - Serialization/deserialization for network transfer
        - Reference counting and garbage collection
        - Local caching for frequently accessed tensors
        - Memory pressure management
    
    Usage:
        bridge = TensorBridge(gpu_pool)
        
        # Allocate a tensor on remote GPU
        handle = bridge.allocate(local_tensor)
        
        # Fetch tensor back to local CPU
        local_tensor = bridge.fetch(handle)
        
        # Release when done
        bridge.release(handle)
    """
    
    # Maximum number of tensors to keep in local cache
    MAX_CACHE_SIZE = 100
    
    # Maximum memory (bytes) for local cache
    MAX_CACHE_MEMORY = 512 * 1024 * 1024  # 512 MB
    
    def __init__(self, gpu_pool):
        """
        Initialize the tensor bridge.
        
        Args:
            gpu_pool: Connected GPU pool for remote execution
        """
        self.gpu_pool = gpu_pool
        
        # All active tensor handles
        self._handles: Dict[int, TensorHandle] = {}
        
        # Reference tracking: which objects reference which tensors
        self._referrers: Dict[int, Set[int]] = defaultdict(set)
        
        # Local cache management
        self._cache_size = 0
        self._cache_memory = 0
        
        # Statistics
        self.stats = {
            'total_allocations': 0,
            'total_fetches': 0,
            'total_releases': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'bytes_transferred': 0,
            'bytes_saved_by_cache': 0
        }
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Background garbage collection
        self._gc_thread: Optional[threading.Thread] = None
        self._gc_running = False
    
    def allocate(self, tensor: torch.Tensor, preferred_gpu: Optional[str] = None) -> TensorHandle:
        """
        Allocate a tensor on a remote GPU.
        
        The tensor data is serialized, compressed, and sent to the
        GPU server. A local handle is returned for future operations.
        
        Args:
            tensor (torch.Tensor): Local tensor to transfer to GPU
            preferred_gpu (str, optional): Preferred GPU server ID
            
        Returns:
            TensorHandle: Handle for the remote tensor
            
        Raises:
            RuntimeError: If no GPU connections are available
        """
        # Get GPU connection
        if preferred_gpu:
            gpu = self.gpu_pool.connections.get(preferred_gpu)
            if gpu is None:
                raise ValueError(f"GPU {preferred_gpu} not found")
        else:
            gpu = self.gpu_pool.get_best_gpu()
            if gpu is None:
                raise RuntimeError("No GPU connections available")
        
        # Compress tensor for transfer
        compressed_data = TensorCompressor.compress_tensor(tensor)
        data_size = len(compressed_data)
        
        # Allocate on remote GPU
        result = gpu.execute({
            'type': 'allocate_tensor',
            'data': compressed_data,
            'shape': list(tensor.shape),
            'dtype': str(tensor.dtype)
        })
        
        if 'error' in result:
            raise RuntimeError(f"Tensor allocation failed: {result['error']}")
        
        # Create local handle
        handle = TensorHandle(
            tensor_id=result['tensor_id'],
            shape=tuple(result['shape']),
            dtype=getattr(torch, result['dtype'].split('.')[-1]),
            device_info={
                'server': gpu.host,
                'port': gpu.port,
                'gpu_name': gpu.gpu_name
            }
        )
        
        # Optionally cache locally
        if self._should_cache(tensor):
            handle._local_cache = tensor.cpu().detach()
            self._cache_memory += tensor.element_size() * tensor.nelement()
        
        # Register handle
        with self._lock:
            self._handles[handle.tensor_id] = handle
            self.stats['total_allocations'] += 1
            self.stats['bytes_transferred'] += data_size
        
        return handle
    
    def fetch(self, handle: TensorHandle, device: str = 'cpu') -> torch.Tensor:
        """
        Fetch a remote tensor back to local memory.
        
        If a local cache exists and is fresh, returns it immediately
        without network transfer.
        
        Args:
            handle (TensorHandle): Handle to the remote tensor
            device (str): Target local device ('cpu' or 'cuda:0', etc.)
            
        Returns:
            torch.Tensor: Local copy of the tensor
        """
        handle.touch()
        
        # Check local cache first
        if handle.is_cached_locally:
            with self._lock:
                self.stats['cache_hits'] += 1
                self.stats['bytes_saved_by_cache'] += (
                    handle._local_cache.element_size() * handle._local_cache.nelement()
                )
            
            tensor = handle._local_cache
            if device != 'cpu':
                tensor = tensor.to(device)
            return tensor
        
        # Cache miss - fetch from remote GPU
        with self._lock:
            self.stats['cache_misses'] += 1
        
        # Get the GPU connection
        gpu = self._get_gpu_for_handle(handle)
        
        # Fetch tensor data from remote GPU
        result = gpu.execute({
            'type': 'get_tensor_data',
            'tensor_id': handle.tensor_id
        })
        
        if 'error' in result:
            raise RuntimeError(f"Tensor fetch failed: {result['error']}")
        
        # Decompress and reconstruct
        tensor = TensorCompressor.decompress_tensor(
            result['data'],
            torch.device('cpu')
        )
        
        # Update cache if appropriate
        if self._should_cache(tensor):
            handle._local_cache = tensor.detach()
        
        # Move to requested device
        if device != 'cpu':
            tensor = tensor.to(device)
        
        with self._lock:
            self.stats['total_fetches'] += 1
            self.stats['bytes_transferred'] += result.get('data_size', 0)
        
        return tensor
    
    def release(self, handle: TensorHandle):
        """
        Release a tensor handle and free remote GPU memory.
        
        If reference count reaches zero, the tensor is freed on
        the remote GPU and the handle is removed.
        
        Args:
            handle (TensorHandle): Handle to release
        """
        should_free = handle.release()
        
        if should_free:
            try:
                gpu = self._get_gpu_for_handle(handle)
                gpu.execute({
                    'type': 'free_tensor',
                    'tensor_id': handle.tensor_id
                })
            except Exception as e:
                print(f"Warning: Failed to free remote tensor {handle.tensor_id}: {e}")
            
            with self._lock:
                self._handles.pop(handle.tensor_id, None)
                self.stats['total_releases'] += 1
                
                # Clean up local cache
                if handle.is_cached_locally:
                    memory_freed = (
                        handle._local_cache.element_size() *
                        handle._local_cache.nelement()
                    )
                    self._cache_memory -= memory_freed
                    handle._local_cache = None
    
    def release_all(self):
        """
        Release all tensor handles and free all remote GPU memory.
        
        Called during shutdown to clean up all resources.
        """
        with self._lock:
            handles = list(self._handles.values())
        
        for handle in handles:
            try:
                self.release(handle)
            except Exception:
                pass  # Best-effort cleanup
        
        with self._lock:
            self._handles.clear()
            self._cache_memory = 0
            self._cache_size = 0
    
    def _get_gpu_for_handle(self, handle: TensorHandle):
        """
        Get the GPU connection that owns this tensor handle.
        
        Args:
            handle (TensorHandle): Tensor handle
            
        Returns:
            GPUConnection: Connection to the owning GPU server
            
        Raises:
            RuntimeError: If the GPU connection is no longer available
        """
        server_key = f"{handle.device_info['server']}:{handle.device_info['port']}"
        gpu = self.gpu_pool.connections.get(server_key)
        
        if gpu is None or not gpu.is_connected:
            raise RuntimeError(
                f"GPU connection lost for tensor {handle.tensor_id}"
            )
        
        return gpu
    
    def _should_cache(self, tensor: torch.Tensor) -> bool:
        """
        Determine if a tensor should be cached locally.
        
        Decision based on:
        - Tensor size (don't cache very large tensors)
        - Available cache memory
        - Cache hit patterns
        
        Args:
            tensor (torch.Tensor): Tensor to evaluate
            
        Returns:
            bool: True if tensor should be cached
        """
        tensor_memory = tensor.element_size() * tensor.nelement()
        
        # Don't cache tensors larger than 25% of cache
        if tensor_memory > self.MAX_CACHE_MEMORY * 0.25:
            return False
        
        # Don't cache if cache is full
        if self._cache_memory + tensor_memory > self.MAX_CACHE_MEMORY:
            self._evict_cache(tensor_memory)
        
        return self._cache_memory + tensor_memory <= self.MAX_CACHE_MEMORY
    
    def _evict_cache(self, needed_memory: int):
        """
        Evict tensors from local cache to make room.
        
        Uses LRU (Least Recently Used) eviction policy.
        
        Args:
            needed_memory (int): Amount of memory to free (bytes)
        """
        with self._lock:
            # Sort handles by last access time
            cached_handles = [
                h for h in self._handles.values()
                if h.is_cached_locally
            ]
            cached_handles.sort(key=lambda h: h._last_access)
            
            freed_memory = 0
            
            for handle in cached_handles:
                if freed_memory >= needed_memory:
                    break
                
                memory = (
                    handle._local_cache.element_size() *
                    handle._local_cache.nelement()
                )
                
                handle._local_cache = None
                freed_memory += memory
                self._cache_memory -= memory
    
    def start_gc(self, interval: float = 30.0):
        """
        Start background garbage collection.
        
        Periodically checks for tensor handles with zero references
        and cleans them up.
        
        Args:
            interval (float): GC check interval in seconds
        """
        if self._gc_running:
            return
        
        self._gc_running = True
        self._gc_thread = threading.Thread(
            target=self._gc_loop,
            args=(interval,),
            daemon=True
        )
        self._gc_thread.start()
    
    def _gc_loop(self, interval: float):
        """
        Background GC loop.
        """
        while self._gc_running:
            time.sleep(interval)
            self._collect_garbage()
    
    def _collect_garbage(self):
        """
        Collect and free unreferenced tensors.
        """
        to_free = []
        
        with self._lock:
            for tensor_id, handle in self._handles.items():
                if handle._ref_count <= 0:
                    to_free.append(handle)
        
        for handle in to_free:
            try:
                self.release(handle)
            except Exception:
                pass
    
    def stop_gc(self):
        """
        Stop background garbage collection.
        """
        self._gc_running = False
        if self._gc_thread:
            self._gc_thread.join(timeout=5)
    
    def get_stats(self) -> dict:
        """
        Get tensor bridge statistics.
        
        Returns:
            dict: Detailed statistics about tensor operations
        """
        with self._lock:
            stats = self.stats.copy()
            stats.update({
                'active_handles': len(self._handles),
                'cache_size': self._cache_size,
                'cache_memory_mb': self._cache_memory / (1024 * 1024),
                'cache_hit_rate': (
                    stats['cache_hits'] / max(1, stats['cache_hits'] + stats['cache_misses'])
                )
            })
        return stats
    
    def __enter__(self):
        self.start_gc()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_gc()
        self.release_all()
        return False