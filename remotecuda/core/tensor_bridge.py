"""
Tensor Bridge Module — Client-Side Tensor Management
=====================================================
Provides unified tensor lifecycle management across the network boundary.

The TensorBridge manages:
    - Tensor allocation on remote GPU/CPU
    - Local caching of frequently accessed tensors
    - Reference counting for automatic cleanup
    - Conversion between Python lists and remote tensors
    - Shape and dtype tracking

Pure Python implementation — no PyTorch required.

Usage:
    from remotecuda.client.connection import ClientConnection
    from remotecuda.core.tensor_bridge import TensorBridge

    conn = ClientConnection('10.0.0.5', 55555)
    conn.connect()

    bridge = TensorBridge(conn)

    # Create a tensor on remote GPU
    tid = bridge.zeros((100, 100))

    # Get tensor data
    data = bridge.get(tid)

    # Release when done
    bridge.release(tid)

    conn.disconnect()
"""

import threading
import time
from typing import Dict, Optional, Any, List, Tuple


class TensorHandle:
    """
    Represents a tensor stored on a remote GPU/CPU server.

    Maintains metadata locally while the actual data resides
    on the server. Provides reference counting for memory management.

    Attributes:
        remote_id (int): Server-side tensor ID.
        shape (tuple): Tensor dimensions.
        dtype (str): Data type string.
        created_at (float): Timestamp of creation.
        last_accessed (float): Timestamp of last access.
        ref_count (int): Reference count.
        pinned (bool): If True, never auto-release.
    """

    __slots__ = [
        'remote_id', 'shape', 'dtype', 'created_at',
        'last_accessed', 'ref_count', 'pinned', '_local_cache',
    ]

    def __init__(self, remote_id: int, shape: tuple, dtype: str):
        self.remote_id = remote_id
        self.shape = shape
        self.dtype = dtype
        self.created_at = time.time()
        self.last_accessed = time.time()
        self.ref_count = 1
        self.pinned = False
        self._local_cache: Optional[dict] = None

    def touch(self):
        """Update last access timestamp."""
        self.last_accessed = time.time()

    @property
    def age_seconds(self) -> float:
        """Seconds since creation."""
        return time.time() - self.created_at

    @property
    def idle_seconds(self) -> float:
        """Seconds since last access."""
        return time.time() - self.last_accessed


class TensorBridge:
    """
    Manages tensor lifecycle between client and server.

    Features:
        - Tensor creation (zeros, ones, full, etc.)
        - Tensor retrieval as Python lists
        - Reference counting
        - Local caching (optional)
        - Automatic garbage collection
        - Statistics tracking

    Usage:
        bridge = TensorBridge(connection)
        tid = bridge.zeros((100, 100))
        data = bridge.get(tid)
        bridge.release(tid)
    """

    def __init__(self, connection, enable_cache: bool = True, max_cache_size: int = 100):
        """
        Initialize the tensor bridge.

        Args:
            connection: ClientConnection instance.
            enable_cache: Enable local tensor caching.
            max_cache_size: Maximum number of cached tensors.
        """
        self._conn = connection
        self._handles: Dict[int, TensorHandle] = {}
        self._lock = threading.RLock()
        self._enable_cache = enable_cache
        self._max_cache_size = max_cache_size
        self._cache_size = 0

        self.stats = {
            'created': 0,
            'retrieved': 0,
            'released': 0,
            'cache_hits': 0,
            'cache_misses': 0,
        }

    # ============================================================
    #  Tensor Creation
    # ============================================================

    def zeros(self, shape: Tuple[int, ...], dtype: str = 'float32') -> int:
        """Create zero-initialized tensor. Returns remote ID."""
        tid = self._conn.send_command('zeros', {
            'shape': list(shape),
            'dtype': dtype,
        })
        self._register(tid, shape, dtype)
        return tid

    def ones(self, shape: Tuple[int, ...], dtype: str = 'float32') -> int:
        """Create one-initialized tensor. Returns remote ID."""
        tid = self._conn.send_command('ones', {
            'shape': list(shape),
            'dtype': dtype,
        })
        self._register(tid, shape, dtype)
        return tid

    def full(self, shape: Tuple[int, ...], value: float, dtype: str = 'float32') -> int:
        """Create constant-filled tensor. Returns remote ID."""
        tid = self._conn.send_command('full', {
            'shape': list(shape),
            'value': value,
            'dtype': dtype,
        })
        self._register(tid, shape, dtype)
        return tid

    def send(self, data: list, dtype: str = 'float32') -> int:
        """Send Python list as tensor. Returns remote ID."""
        import struct
        import base64

        flat = self._flatten(data)
        shape = self._infer_shape(data)

        fmt_map = {
            'float32': 'f',
            'float64': 'd',
            'int32': 'i',
            'int64': 'q',
        }
        fmt = fmt_map.get(dtype, 'f')

        raw_bytes = b''
        for val in flat:
            if fmt in ('f', 'd'):
                raw_bytes += struct.pack(fmt, float(val))
            else:
                raw_bytes += struct.pack(fmt, int(val))

        encoded = base64.b64encode(raw_bytes).decode('ascii')

        tid = self._conn.send_command('send_tensor', {
            'data': encoded,
            'shape': shape,
            'dtype': dtype,
            'format': fmt,
            'size': len(raw_bytes),
        })

        self._register(tid, tuple(shape), dtype)
        return tid

    # ============================================================
    #  Tensor Retrieval
    # ============================================================

    def get(self, remote_id: int) -> dict:
        """
        Retrieve tensor data as a Python dictionary.

        Returns:
            dict: {'data': [...], 'shape': [...], 'dtype': '...'}

        Uses local cache if available and enabled.
        """
        with self._lock:
            handle = self._handles.get(remote_id)

        if handle:
            handle.touch()

            # Check cache
            if self._enable_cache and handle._local_cache is not None:
                self.stats['cache_hits'] += 1
                return handle._local_cache

        self.stats['cache_misses'] += 1
        self.stats['retrieved'] += 1

        response = self._conn.send_command('get_tensor', {'tensor_id': remote_id})

        encoded = response.get('data', '')
        dtype = response.get('dtype', 'float32')
        fmt_char = response.get('format', 'f')
        shape = response.get('shape', [])

        import struct
        import base64

        raw_bytes = base64.b64decode(encoded)
        element_size = struct.calcsize(fmt_char)
        num_elements = len(raw_bytes) // element_size

        data = []
        if fmt_char in ('f', 'd'):
            for i in range(num_elements):
                val = struct.unpack_from(fmt_char, raw_bytes, i * element_size)[0]
                data.append(float(val))
        else:
            for i in range(num_elements):
                val = struct.unpack_from(fmt_char, raw_bytes, i * element_size)[0]
                data.append(int(val))

        result = {
            'data': data,
            'shape': shape,
            'dtype': dtype,
            'tensor_id': remote_id,
        }

        # Update cache
        if self._enable_cache and handle:
            if self._cache_size < self._max_cache_size:
                handle._local_cache = result
                self._cache_size += 1
            else:
                self._evict_one()
                if self._cache_size < self._max_cache_size:
                    handle._local_cache = result
                    self._cache_size += 1

        return result

    # ============================================================
    #  Reference Counting
    # ============================================================

    def add_ref(self, remote_id: int):
        """Increment reference count for a tensor."""
        with self._lock:
            handle = self._handles.get(remote_id)
            if handle:
                handle.ref_count += 1

    def release(self, remote_id: int):
        """
        Decrement reference count for a tensor.

        Frees the tensor on the server if reference count reaches zero.
        """
        with self._lock:
            handle = self._handles.get(remote_id)

        if handle is None:
            return

        handle.ref_count -= 1

        if handle.ref_count <= 0 and not handle.pinned:
            try:
                self._conn.send_command('free_tensor', {'tensor_id': remote_id})
            except Exception:
                pass

            with self._lock:
                self._handles.pop(remote_id, None)
                if handle._local_cache is not None:
                    self._cache_size -= 1

            self.stats['released'] += 1

    def pin(self, remote_id: int):
        """Pin a tensor (never auto-release)."""
        with self._lock:
            handle = self._handles.get(remote_id)
            if handle:
                handle.pinned = True

    def unpin(self, remote_id: int):
        """Unpin a tensor."""
        with self._lock:
            handle = self._handles.get(remote_id)
            if handle:
                handle.pinned = False

    def release_all(self):
        """Release all managed tensors."""
        with self._lock:
            remote_ids = list(self._handles.keys())

        for rid in remote_ids:
            try:
                self._conn.send_command('free_tensor', {'tensor_id': rid})
            except Exception:
                pass

        with self._lock:
            self._handles.clear()
            self._cache_size = 0

    # ============================================================
    #  Internal Helpers
    # ============================================================

    def _register(self, remote_id: int, shape: tuple, dtype: str):
        """Register a new tensor handle."""
        with self._lock:
            self._handles[remote_id] = TensorHandle(remote_id, shape, dtype)
            self.stats['created'] += 1

    def _evict_one(self):
        """Evict one cached tensor (LRU policy)."""
        oldest_id = None
        oldest_time = float('inf')

        with self._lock:
            for rid, handle in self._handles.items():
                if handle._local_cache is not None:
                    if handle.last_accessed < oldest_time:
                        oldest_time = handle.last_accessed
                        oldest_id = rid

            if oldest_id is not None:
                self._handles[oldest_id]._local_cache = None
                self._cache_size -= 1

    @staticmethod
    def _flatten(nested) -> list:
        """Flatten nested lists."""
        if isinstance(nested, (list, tuple)):
            result = []
            for item in nested:
                result.extend(TensorBridge._flatten(item))
            return result
        return [nested]

    @staticmethod
    def _infer_shape(nested) -> list:
        """Infer shape from nested structure."""
        if isinstance(nested, (list, tuple)):
            if len(nested) > 0 and isinstance(nested[0], (list, tuple)):
                return [len(nested)] + TensorBridge._infer_shape(nested[0])
            return [len(nested)]
        return []

    def get_stats(self) -> dict:
        """Get bridge statistics."""
        with self._lock:
            return {
                **self.stats,
                'active_handles': len(self._handles),
                'cache_size': self._cache_size,
                'cache_hit_rate': (
                    self.stats['cache_hits'] / max(1, self.stats['cache_hits'] + self.stats['cache_misses'])
                ),
            }

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release_all()
        return False