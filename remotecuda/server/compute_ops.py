"""
Compute Operations Module — Pure PyTorch with Auto CPU/GPU Fallback
====================================================================
This module provides all tensor computation operations for the server.
It automatically detects whether CUDA is available and falls back
to CPU if no GPU is found.

Server Requirements:
    - PyTorch 1.10+ (CUDA version preferred, CPU-only is acceptable)
    - NumPy 1.20+
    - Python 3.8+

Architecture:
    All operations receive tensor IDs (integers) from the client,
    look up the actual PyTorch tensors in the tensor registry,
    perform the computation, store the result in the registry,
    and return a new tensor ID.

    The device selection is automatic:
    1. Check if CUDA is available → use cuda:0
    2. If not → use CPU
    3. All new tensors are created on the selected device

Tensor Registry:
    A dictionary mapping tensor_id → {
        'tensor': torch.Tensor,
        'shape': tuple,
        'dtype': str,
        'created_at': float,
        'ref_count': int,
    }

Supported Operations:
    Creation:   zeros, ones, full, arange, linspace, eye, rand, randn
    Arithmetic: add, subtract, multiply, divide, matmul, dot
    Activation: relu, sigmoid, tanh, softmax, log_softmax
    Reduction:  sum, mean, max, min, argmax, argmin
    Shape:      reshape, transpose, squeeze, unsqueeze, flatten
    Indexing:   slice, gather
    Comparison: eq, gt, lt, ge, le
    Math:       abs, sqrt, exp, log, pow, sin, cos, tan
"""

import time
import threading
from typing import Dict, Optional, Tuple, Any

import torch
import torch.nn.functional as F
import numpy as np


# ============================================================
#  Device Detection (Auto GPU/CPU Fallback)
# ============================================================

def _detect_device() -> torch.device:
    """
    Detect the best available compute device.
    
    Priority:
        1. CUDA GPU (if available)
        2. MPS (Metal Performance Shaders — Apple Silicon, if available)
        3. CPU (always available)

    Returns:
        torch.device: The selected compute device.

    Example:
        >>> device = _detect_device()
        >>> print(device)
        cuda:0  # or cpu
    """
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        # Warm up CUDA to avoid lazy initialization overhead
        try:
            _ = torch.zeros(1, device=device)
            torch.cuda.synchronize(device)
        except Exception:
            pass
        return device
    
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        try:
            device = torch.device('mps')
            _ = torch.zeros(1, device=device)
            return device
        except Exception:
            pass
    
    return torch.device('cpu')


# Global device — detected once at module load
DEVICE = _detect_device()
DEVICE_NAME = str(DEVICE)
DEVICE_IS_CUDA = DEVICE.type == 'cuda'
DEVICE_IS_CPU = DEVICE.type == 'cpu'


def get_device_info() -> dict:
    """
    Get detailed information about the current compute device.

    Returns:
        dict: Device information including name, memory, capability, etc.

    Example:
        >>> info = get_device_info()
        >>> print(info['device_name'])
        NVIDIA GeForce RTX 4090
    """
    info = {
        'device': DEVICE_NAME,
        'device_type': DEVICE.type,
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        'pytorch_version': torch.__version__,
        'numpy_version': np.__version__,
        'python_platform': __import__('platform').platform(),
    }

    if DEVICE_IS_CUDA:
        info['gpu_count'] = torch.cuda.device_count()
        if torch.cuda.device_count() > 0:
            props = torch.cuda.get_device_properties(DEVICE)
            info['gpu_name'] = props.name
            info['gpu_memory_total_mb'] = props.total_memory // (1024 * 1024)
            info['gpu_memory_free_mb'] = (
                torch.cuda.mem_get_info()[0] // (1024 * 1024)
            )
            info['compute_capability'] = f"{props.major}.{props.minor}"
            info['multi_processor_count'] = props.multi_processor_count
    else:
        info['gpu_count'] = 0
        info['gpu_name'] = None
        info['gpu_memory_total_mb'] = None
        info['gpu_memory_free_mb'] = None
        info['compute_capability'] = None

    return info


# ============================================================
#  Tensor Registry
# ============================================================

class TensorRegistry:
    """
    Thread-safe registry for managing tensors on the compute device.

    Each tensor is assigned a unique integer ID that is returned
    to the client. The client uses this ID for all subsequent operations.

    Features:
        - Thread-safe with reentrant lock
        - Automatic reference counting
        - Memory usage tracking
        - Garbage collection of zero-reference tensors
        - Statistics collection

    Attributes:
        _tensors (dict): Maps tensor_id → tensor_metadata
        _counter (int): Auto-incrementing tensor ID counter
        _lock (threading.RLock): Thread safety lock
        _total_allocated (int): Total bytes allocated
        _peak_allocated (int): Peak bytes allocated
        _stats (dict): Operation statistics
    """

    def __init__(self):
        """Initialize an empty tensor registry."""
        self._tensors: Dict[int, dict] = {}
        self._counter = 0
        self._lock = threading.RLock()
        self._total_allocated = 0
        self._peak_allocated = 0
        self._stats = {
            'tensors_created': 0,
            'tensors_freed': 0,
            'tensors_retrieved': 0,
            'memory_allocated_total': 0,
            'memory_freed_total': 0,
        }

    def register(self, tensor: torch.Tensor, pin: bool = False) -> int:
        """
        Register a tensor and return its unique ID.

        Args:
            tensor: PyTorch tensor to register.
            pin: If True, tensor is pinned (never auto-garbage-collected).

        Returns:
            int: Unique tensor ID for client reference.

        Example:
            >>> registry = TensorRegistry()
            >>> t = torch.zeros((100, 100), device=DEVICE)
            >>> tensor_id = registry.register(t)
            >>> print(tensor_id)
            1
        """
        with self._lock:
            self._counter += 1
            tensor_id = self._counter

            # Ensure tensor is on the correct device
            if tensor.device != DEVICE:
                tensor = tensor.to(DEVICE)

            # Calculate memory usage
            memory_used = tensor.element_size() * tensor.nelement()

            # Store tensor metadata
            self._tensors[tensor_id] = {
                'tensor': tensor,
                'shape': tuple(tensor.shape),
                'dtype': str(tensor.dtype).replace('torch.', ''),
                'created_at': time.time(),
                'last_accessed': time.time(),
                'ref_count': 1,
                'pinned': pin,
                'memory_used': memory_used,
            }

            # Update stats
            self._total_allocated += memory_used
            self._peak_allocated = max(self._peak_allocated, self._total_allocated)
            self._stats['tensors_created'] += 1
            self._stats['memory_allocated_total'] += memory_used

            return tensor_id

    def get(self, tensor_id: int) -> torch.Tensor:
        """
        Retrieve a tensor by its ID.

        Args:
            tensor_id: Tensor ID returned by register().

        Returns:
            torch.Tensor: The requested tensor.

        Raises:
            KeyError: If tensor_id is not found.

        Example:
            >>> tensor = registry.get(1)
            >>> print(tensor.shape)
            torch.Size([100, 100])
        """
        with self._lock:
            if tensor_id not in self._tensors:
                raise KeyError(f"Tensor {tensor_id} not found. It may have been freed.")

            entry = self._tensors[tensor_id]
            entry['last_accessed'] = time.time()
            self._stats['tensors_retrieved'] += 1

            return entry['tensor']

    def get_info(self, tensor_id: int) -> dict:
        """
        Get metadata about a registered tensor without returning the tensor.

        Args:
            tensor_id: Tensor ID.

        Returns:
            dict: Tensor metadata (shape, dtype, memory_used, etc.).

        Raises:
            KeyError: If tensor_id is not found.
        """
        with self._lock:
            if tensor_id not in self._tensors:
                raise KeyError(f"Tensor {tensor_id} not found.")

            entry = self._tensors[tensor_id]
            return {
                'tensor_id': tensor_id,
                'shape': entry['shape'],
                'dtype': entry['dtype'],
                'memory_used': entry['memory_used'],
                'ref_count': entry['ref_count'],
                'pinned': entry['pinned'],
                'created_at': entry['created_at'],
                'last_accessed': entry['last_accessed'],
                'age_seconds': time.time() - entry['created_at'],
                'idle_seconds': time.time() - entry['last_accessed'],
            }

    def add_ref(self, tensor_id: int):
        """
        Increment reference count for a tensor.

        Args:
            tensor_id: Tensor ID.

        Raises:
            KeyError: If tensor_id is not found.
        """
        with self._lock:
            if tensor_id not in self._tensors:
                raise KeyError(f"Tensor {tensor_id} not found.")
            self._tensors[tensor_id]['ref_count'] += 1

    def release(self, tensor_id: int):
        """
        Decrement reference count. Free tensor if count reaches zero.

        Args:
            tensor_id: Tensor ID.

        Raises:
            KeyError: If tensor_id is not found.
        """
        with self._lock:
            if tensor_id not in self._tensors:
                raise KeyError(f"Tensor {tensor_id} not found.")

            entry = self._tensors[tensor_id]
            entry['ref_count'] -= 1

            if entry['ref_count'] <= 0 and not entry['pinned']:
                self._free_tensor_internal(tensor_id)

    def free(self, tensor_id: int):
        """
        Force-free a tensor regardless of reference count.

        Args:
            tensor_id: Tensor ID.

        Raises:
            KeyError: If tensor_id is not found.
        """
        with self._lock:
            if tensor_id not in self._tensors:
                raise KeyError(f"Tensor {tensor_id} not found.")
            self._free_tensor_internal(tensor_id)

    def _free_tensor_internal(self, tensor_id: int):
        """
        Internal method to free a tensor (must be called with lock held).

        Args:
            tensor_id: Tensor ID to free.
        """
        entry = self._tensors.pop(tensor_id)
        memory_freed = entry['memory_used']

        # Explicitly delete the tensor to free GPU/CPU memory
        del entry['tensor']

        self._total_allocated -= memory_freed
        self._stats['tensors_freed'] += 1
        self._stats['memory_freed_total'] += memory_freed

    def free_all(self):
        """
        Free all registered tensors.

        Used during server shutdown to ensure clean memory release.
        """
        with self._lock:
            tensor_ids = list(self._tensors.keys())
            for tensor_id in tensor_ids:
                try:
                    self._free_tensor_internal(tensor_id)
                except Exception:
                    pass

    def garbage_collect(self, max_idle_seconds: float = 300.0):
        """
        Free unreferenced tensors that have been idle for too long.

        Args:
            max_idle_seconds: Maximum idle time before collection.
        """
        now = time.time()
        to_free = []

        with self._lock:
            for tensor_id, entry in self._tensors.items():
                if entry['pinned']:
                    continue
                if entry['ref_count'] > 0:
                    continue
                if now - entry['last_accessed'] > max_idle_seconds:
                    to_free.append(tensor_id)

            for tensor_id in to_free:
                try:
                    self._free_tensor_internal(tensor_id)
                except Exception:
                    pass

        return len(to_free)

    def get_stats(self) -> dict:
        """
        Get comprehensive registry statistics.

        Returns:
            dict: Statistics about tensor usage.
        """
        with self._lock:
            return {
                **self._stats,
                'active_tensors': len(self._tensors),
                'pinned_tensors': sum(
                    1 for e in self._tensors.values() if e['pinned']
                ),
                'total_allocated_mb': self._total_allocated / (1024 * 1024),
                'peak_allocated_mb': self._peak_allocated / (1024 * 1024),
            }

    def __len__(self) -> int:
        """Number of active tensors."""
        return len(self._tensors)

    def __contains__(self, tensor_id: int) -> bool:
        """Check if a tensor ID exists."""
        return tensor_id in self._tensors


# ============================================================
#  Global Tensor Registry Instance
# ============================================================

# Singleton registry — one per server process
_registry = TensorRegistry()


def get_registry() -> TensorRegistry:
    """Get the global tensor registry instance."""
    return _registry


# ============================================================
#  Data Encoding/Decoding for Client Communication
# ============================================================

def _encode_tensor_for_client(tensor: torch.Tensor) -> dict:
    """
    Encode a PyTorch tensor for transmission to the client.

    Converts tensor to CPU, then to raw bytes, then to base64.
    This format can be decoded by the pure-Python client.

    Args:
        tensor: PyTorch tensor to encode.

    Returns:
        dict: Encoded tensor data ready for JSON serialization.
        {
            'data': 'base64_encoded_string',
            'shape': [100, 100],
            'dtype': 'float32',
            'format': 'f',
            'size': 40000
        }
    """
    import base64
    import struct

    # Move to CPU for serialization
    cpu_tensor = tensor.detach().cpu()

    # Get raw bytes
    raw_bytes = cpu_tensor.numpy().tobytes()

    # Encode as base64
    encoded = base64.b64encode(raw_bytes).decode('ascii')

    # Determine struct format character
    dtype_str = str(tensor.dtype).replace('torch.', '')
    fmt_map = {
        'float32': 'f',
        'float64': 'd',
        'int32': 'i',
        'int64': 'q',
        'uint8': 'B',
        'int8': 'b',
        'int16': 'h',
    }
    fmt = fmt_map.get(dtype_str, 'f')

    return {
        'data': encoded,
        'shape': list(tensor.shape),
        'dtype': dtype_str,
        'format': fmt,
        'size': len(raw_bytes),
    }


def _decode_tensor_from_client(encoded_data: dict) -> torch.Tensor:
    """
    Decode tensor data received from the client.

    Args:
        encoded_data: Dictionary with 'data', 'shape', 'dtype', 'format'.

    Returns:
        torch.Tensor: Decoded tensor on the compute device.
    """
    import base64
    import struct

    encoded = encoded_data.get('data', '')
    shape = tuple(encoded_data.get('shape', []))
    dtype_str = encoded_data.get('dtype', 'float32')
    fmt = encoded_data.get('format', 'f')

    # Decode base64
    raw_bytes = base64.b64decode(encoded)

    # Convert to numpy first (handles conversion robustly)
    dtype_map = {
        'float32': np.float32,
        'float64': np.float64,
        'int32': np.int32,
        'int64': np.int64,
        'uint8': np.uint8,
        'int8': np.int8,
        'int16': np.int16,
    }
    np_dtype = dtype_map.get(dtype_str, np.float32)

    arr = np.frombuffer(raw_bytes, dtype=np_dtype).reshape(shape)

    # Convert to PyTorch tensor on device
    tensor = torch.from_numpy(arr.copy()).to(DEVICE)

    return tensor


# ============================================================
#  Tensor Creation Operations
# ============================================================

def tensor_zeros(shape: list, dtype: str = 'float32') -> int:
    """
    Create a zero-initialized tensor on the compute device.

    Args:
        shape: List of dimensions, e.g., [100, 100].
        dtype: Data type string ('float32', 'float64', 'int32', etc.).

    Returns:
        int: Tensor ID for the created tensor.

    Example:
        >>> tid = tensor_zeros([100, 100])
        >>> tid = tensor_zeros([32, 3, 224, 224], dtype='float32')
    """
    dtype_map = {
        'float32': torch.float32,
        'float64': torch.float64,
        'int32': torch.int32,
        'int64': torch.int64,
        'uint8': torch.uint8,
        'int8': torch.int8,
        'int16': torch.int16,
    }
    torch_dtype = dtype_map.get(dtype, torch.float32)
    tensor = torch.zeros(shape, dtype=torch_dtype, device=DEVICE)
    return _registry.register(tensor)


def tensor_ones(shape: list, dtype: str = 'float32') -> int:
    """
    Create a one-initialized tensor on the compute device.

    Args:
        shape: List of dimensions.
        dtype: Data type string.

    Returns:
        int: Tensor ID.
    """
    dtype_map = {
        'float32': torch.float32,
        'float64': torch.float64,
        'int32': torch.int32,
        'int64': torch.int64,
    }
    torch_dtype = dtype_map.get(dtype, torch.float32)
    tensor = torch.ones(shape, dtype=torch_dtype, device=DEVICE)
    return _registry.register(tensor)


def tensor_full(shape: list, value: float, dtype: str = 'float32') -> int:
    """
    Create a tensor filled with a constant value.

    Args:
        shape: List of dimensions.
        value: Fill value.
        dtype: Data type string.

    Returns:
        int: Tensor ID.
    """
    dtype_map = {
        'float32': torch.float32,
        'float64': torch.float64,
        'int32': torch.int32,
        'int64': torch.int64,
    }
    torch_dtype = dtype_map.get(dtype, torch.float32)
    tensor = torch.full(shape, value, dtype=torch_dtype, device=DEVICE)
    return _registry.register(tensor)


def tensor_arange(start: float, end: float, step: float = 1.0, dtype: str = 'float32') -> int:
    """
    Create a 1D tensor with evenly spaced values.

    Args:
        start: Start value.
        end: End value (exclusive).
        step: Step size.
        dtype: Data type string.

    Returns:
        int: Tensor ID.
    """
    dtype_map = {
        'float32': torch.float32,
        'float64': torch.float64,
        'int32': torch.int32,
        'int64': torch.int64,
    }
    torch_dtype = dtype_map.get(dtype, torch.float32)
    tensor = torch.arange(start, end, step, dtype=torch_dtype, device=DEVICE)
    return _registry.register(tensor)


def tensor_eye(n: int, m: int = None, dtype: str = 'float32') -> int:
    """
    Create a 2D identity matrix.

    Args:
        n: Number of rows.
        m: Number of columns (default: n).
        dtype: Data type string.

    Returns:
        int: Tensor ID.
    """
    if m is None:
        m = n
    dtype_map = {'float32': torch.float32, 'float64': torch.float64}
    torch_dtype = dtype_map.get(dtype, torch.float32)
    tensor = torch.eye(n, m, dtype=torch_dtype, device=DEVICE)
    return _registry.register(tensor)


def tensor_rand(shape: list, dtype: str = 'float32') -> int:
    """
    Create a tensor with random values from uniform [0, 1).

    Args:
        shape: List of dimensions.
        dtype: Data type string.

    Returns:
        int: Tensor ID.
    """
    dtype_map = {'float32': torch.float32, 'float64': torch.float64}
    torch_dtype = dtype_map.get(dtype, torch.float32)
    tensor = torch.rand(shape, dtype=torch_dtype, device=DEVICE)
    return _registry.register(tensor)


def tensor_randn(shape: list, dtype: str = 'float32') -> int:
    """
    Create a tensor with random values from standard normal distribution.

    Args:
        shape: List of dimensions.
        dtype: Data type string.

    Returns:
        int: Tensor ID.
    """
    dtype_map = {'float32': torch.float32, 'float64': torch.float64}
    torch_dtype = dtype_map.get(dtype, torch.float32)
    tensor = torch.randn(shape, dtype=torch_dtype, device=DEVICE)
    return _registry.register(tensor)


def tensor_from_client(encoded_data: dict) -> int:
    """
    Receive tensor data from the client and store it.

    Args:
        encoded_data: Encoded tensor data from client.

    Returns:
        int: Tensor ID.
    """
    tensor = _decode_tensor_from_client(encoded_data)
    return _registry.register(tensor)


def tensor_to_client(tensor_id: int) -> dict:
    """
    Encode a tensor for transmission to the client.

    Returns:
        dict: Encoded tensor data.
    """
    tensor = _registry.get(tensor_id)
    return _encode_tensor_for_client(tensor)


# ============================================================
#  Arithmetic Operations
# ============================================================

def tensor_add(a_id: int, b_id: int) -> int:
    """
    Element-wise addition: C = A + B.

    Args:
        a_id: First tensor ID.
        b_id: Second tensor ID.

    Returns:
        int: Result tensor ID.

    Raises:
        ValueError: If shapes are incompatible.
    """
    a = _registry.get(a_id)
    b = _registry.get(b_id)

    # Handle broadcasting
    try:
        result = a + b
    except RuntimeError as e:
        raise ValueError(f"Cannot add tensors with shapes {a.shape} and {b.shape}: {e}")

    return _registry.register(result)


def tensor_subtract(a_id: int, b_id: int) -> int:
    """Element-wise subtraction: C = A - B."""
    a = _registry.get(a_id)
    b = _registry.get(b_id)
    result = a - b
    return _registry.register(result)


def tensor_multiply(a_id: int, b_id: int) -> int:
    """Element-wise multiplication: C = A * B."""
    a = _registry.get(a_id)
    b = _registry.get(b_id)
    result = a * b
    return _registry.register(result)


def tensor_divide(a_id: int, b_id: int) -> int:
    """Element-wise division: C = A / B."""
    a = _registry.get(a_id)
    b = _registry.get(b_id)
    result = a / b
    return _registry.register(result)


def tensor_matmul(a_id: int, b_id: int) -> int:
    """
    Matrix multiplication: C = A @ B.

    Supports:
        - 2D @ 2D → 2D (standard matrix multiplication)
        - 3D @ 2D → 3D (batch matrix multiplication)
        - Broadcasting for higher dimensions

    Args:
        a_id: First tensor ID.
        b_id: Second tensor ID.

    Returns:
        int: Result tensor ID.

    Raises:
        ValueError: If shapes are incompatible for matrix multiplication.
    """
    a = _registry.get(a_id)
    b = _registry.get(b_id)

    try:
        result = torch.matmul(a, b)
    except RuntimeError as e:
        raise ValueError(
            f"Cannot matmul tensors with shapes {a.shape} and {b.shape}: {e}"
        )

    return _registry.register(result)


def tensor_dot(a_id: int, b_id: int) -> int:
    """
    Dot product of two 1D tensors.

    Args:
        a_id: First tensor ID (must be 1D).
        b_id: Second tensor ID (must be 1D).

    Returns:
        int: Scalar result tensor ID.

    Raises:
        ValueError: If tensors are not 1D.
    """
    a = _registry.get(a_id)
    b = _registry.get(b_id)

    if a.dim() != 1 or b.dim() != 1:
        raise ValueError(
            f"Dot product requires 1D tensors, got shapes {a.shape} and {b.shape}"
        )

    result = torch.dot(a, b)
    return _registry.register(result)


def tensor_scalar_multiply(tensor_id: int, scalar: float) -> int:
    """Multiply tensor by a scalar: C = A * scalar."""
    a = _registry.get(tensor_id)
    result = a * scalar
    return _registry.register(result)


def tensor_scalar_add(tensor_id: int, scalar: float) -> int:
    """Add scalar to tensor: C = A + scalar."""
    a = _registry.get(tensor_id)
    result = a + scalar
    return _registry.register(result)


def tensor_negate(tensor_id: int) -> int:
    """Negate tensor: C = -A."""
    a = _registry.get(tensor_id)
    result = -a
    return _registry.register(result)


# ============================================================
#  Activation Functions
# ============================================================

def tensor_relu(tensor_id: int) -> int:
    """ReLU activation: max(0, x)."""
    a = _registry.get(tensor_id)
    result = F.relu(a)
    return _registry.register(result)


def tensor_sigmoid(tensor_id: int) -> int:
    """Sigmoid activation: 1 / (1 + exp(-x))."""
    a = _registry.get(tensor_id)
    result = torch.sigmoid(a)
    return _registry.register(result)


def tensor_tanh(tensor_id: int) -> int:
    """Tanh activation."""
    a = _registry.get(tensor_id)
    result = torch.tanh(a)
    return _registry.register(result)


def tensor_softmax(tensor_id: int, dim: int = -1) -> int:
    """Softmax activation along a dimension."""
    a = _registry.get(tensor_id)
    result = F.softmax(a, dim=dim)
    return _registry.register(result)


def tensor_log_softmax(tensor_id: int, dim: int = -1) -> int:
    """Log-Softmax activation."""
    a = _registry.get(tensor_id)
    result = F.log_softmax(a, dim=dim)
    return _registry.register(result)


def tensor_gelu(tensor_id: int) -> int:
    """GELU activation (Gaussian Error Linear Unit)."""
    a = _registry.get(tensor_id)
    result = F.gelu(a)
    return _registry.register(result)


def tensor_leaky_relu(tensor_id: int, negative_slope: float = 0.01) -> int:
    """Leaky ReLU activation."""
    a = _registry.get(tensor_id)
    result = F.leaky_relu(a, negative_slope=negative_slope)
    return _registry.register(result)


def tensor_elu(tensor_id: int, alpha: float = 1.0) -> int:
    """ELU activation."""
    a = _registry.get(tensor_id)
    result = F.elu(a, alpha=alpha)
    return _registry.register(result)


def tensor_selu(tensor_id: int) -> int:
    """SELU activation."""
    a = _registry.get(tensor_id)
    result = F.selu(a)
    return _registry.register(result)


# ============================================================
#  Reduction Operations
# ============================================================

def tensor_sum(tensor_id: int, dim: int = None, keepdim: bool = False) -> int:
    """
    Sum of tensor elements.

    Args:
        tensor_id: Tensor ID.
        dim: Dimension to reduce (None for all elements).
        keepdim: Keep reduced dimensions as size 1.

    Returns:
        int: Result tensor ID.
    """
    a = _registry.get(tensor_id)

    if dim is None:
        result = a.sum()
    else:
        result = a.sum(dim=dim, keepdim=keepdim)

    return _registry.register(result)


def tensor_mean(tensor_id: int, dim: int = None, keepdim: bool = False) -> int:
    """Mean of tensor elements."""
    a = _registry.get(tensor_id)

    if dim is None:
        result = a.mean()
    else:
        result = a.mean(dim=dim, keepdim=keepdim)

    return _registry.register(result)


def tensor_max(tensor_id: int, dim: int = None, keepdim: bool = False) -> int:
    """Maximum of tensor elements."""
    a = _registry.get(tensor_id)

    if dim is None:
        result = a.max()
    else:
        result, _ = a.max(dim=dim, keepdim=keepdim)

    return _registry.register(result)


def tensor_min(tensor_id: int, dim: int = None, keepdim: bool = False) -> int:
    """Minimum of tensor elements."""
    a = _registry.get(tensor_id)

    if dim is None:
        result = a.min()
    else:
        result, _ = a.min(dim=dim, keepdim=keepdim)

    return _registry.register(result)


def tensor_argmax(tensor_id: int, dim: int = None) -> int:
    """Indices of maximum values."""
    a = _registry.get(tensor_id)
    result = a.argmax(dim=dim)
    return _registry.register(result)


def tensor_argmin(tensor_id: int, dim: int = None) -> int:
    """Indices of minimum values."""
    a = _registry.get(tensor_id)
    result = a.argmin(dim=dim)
    return _registry.register(result)


def tensor_std(tensor_id: int, dim: int = None, keepdim: bool = False) -> int:
    """Standard deviation."""
    a = _registry.get(tensor_id)

    if dim is None:
        result = a.std()
    else:
        result = a.std(dim=dim, keepdim=keepdim)

    return _registry.register(result)


def tensor_var(tensor_id: int, dim: int = None, keepdim: bool = False) -> int:
    """Variance."""
    a = _registry.get(tensor_id)

    if dim is None:
        result = a.var()
    else:
        result = a.var(dim=dim, keepdim=keepdim)

    return _registry.register(result)


def tensor_norm(tensor_id: int, p: float = 2.0, dim: int = None) -> int:
    """Vector/matrix norm."""
    a = _registry.get(tensor_id)

    if dim is None:
        result = a.norm(p=p)
    else:
        result = a.norm(p=p, dim=dim)

    return _registry.register(result)


# ============================================================
#  Shape Operations
# ============================================================

def tensor_reshape(tensor_id: int, shape: list) -> int:
    """
    Reshape tensor to new shape.

    One dimension can be -1 to infer from other dimensions.

    Args:
        tensor_id: Tensor ID.
        shape: New shape list.

    Returns:
        int: Result tensor ID (same data, new shape).
    """
    a = _registry.get(tensor_id)
    result = a.reshape(shape)
    return _registry.register(result)


def tensor_transpose(tensor_id: int, dim0: int = 0, dim1: int = 1) -> int:
    """Transpose by swapping two dimensions."""
    a = _registry.get(tensor_id)
    result = a.transpose(dim0, dim1)
    return _registry.register(result)


def tensor_permute(tensor_id: int, dims: list) -> int:
    """Permute dimensions."""
    a = _registry.get(tensor_id)
    result = a.permute(*dims)
    return _registry.register(result)


def tensor_squeeze(tensor_id: int, dim: int = None) -> int:
    """Remove dimensions of size 1."""
    a = _registry.get(tensor_id)

    if dim is None:
        result = a.squeeze()
    else:
        result = a.squeeze(dim=dim)

    return _registry.register(result)


def tensor_unsqueeze(tensor_id: int, dim: int) -> int:
    """Add a dimension of size 1 at position dim."""
    a = _registry.get(tensor_id)
    result = a.unsqueeze(dim=dim)
    return _registry.register(result)


def tensor_flatten(tensor_id: int, start_dim: int = 0, end_dim: int = -1) -> int:
    """Flatten a range of dimensions."""
    a = _registry.get(tensor_id)
    result = torch.flatten(a, start_dim=start_dim, end_dim=end_dim)
    return _registry.register(result)


def tensor_view(tensor_id: int, shape: list) -> int:
    """
    View tensor with new shape (must be contiguous).

    Args:
        tensor_id: Tensor ID.
        shape: New shape.

    Returns:
        int: Result tensor ID.

    Raises:
        ValueError: If tensor is not contiguous.
    """
    a = _registry.get(tensor_id)

    if not a.is_contiguous():
        a = a.contiguous()

    result = a.view(shape)
    return _registry.register(result)


def tensor_expand(tensor_id: int, shape: list) -> int:
    """Expand tensor to larger shape (broadcasting)."""
    a = _registry.get(tensor_id)
    result = a.expand(shape)
    return _registry.register(result)


def tensor_repeat(tensor_id: int, repeats: list) -> int:
    """Repeat tensor along dimensions."""
    a = _registry.get(tensor_id)
    result = a.repeat(*repeats)
    return _registry.register(result)


def tensor_cat(tensor_ids: list, dim: int = 0) -> int:
    """
    Concatenate multiple tensors along a dimension.

    Args:
        tensor_ids: List of tensor IDs.
        dim: Dimension to concatenate along.

    Returns:
        int: Result tensor ID.
    """
    tensors = [_registry.get(tid) for tid in tensor_ids]
    result = torch.cat(tensors, dim=dim)
    return _registry.register(result)


def tensor_stack(tensor_ids: list, dim: int = 0) -> int:
    """Stack tensors along a new dimension."""
    tensors = [_registry.get(tid) for tid in tensor_ids]
    result = torch.stack(tensors, dim=dim)
    return _registry.register(result)


def tensor_split(tensor_id: int, split_size: int, dim: int = 0) -> list:
    """Split tensor into chunks. Returns list of tensor IDs."""
    a = _registry.get(tensor_id)
    chunks = torch.split(a, split_size, dim=dim)
    return [_registry.register(chunk) for chunk in chunks]


# ============================================================
#  Comparison Operations
# ============================================================

def tensor_eq(a_id: int, b_id: int) -> int:
    """Element-wise equality: A == B."""
    a = _registry.get(a_id)
    b = _registry.get(b_id)
    result = (a == b).to(torch.float32)
    return _registry.register(result)


def tensor_gt(a_id: int, b_id: int) -> int:
    """Element-wise greater than: A > B."""
    a = _registry.get(a_id)
    b = _registry.get(b_id)
    result = (a > b).to(torch.float32)
    return _registry.register(result)


def tensor_lt(a_id: int, b_id: int) -> int:
    """Element-wise less than: A < B."""
    a = _registry.get(a_id)
    b = _registry.get(b_id)
    result = (a < b).to(torch.float32)
    return _registry.register(result)


def tensor_ge(a_id: int, b_id: int) -> int:
    """Element-wise greater than or equal: A >= B."""
    a = _registry.get(a_id)
    b = _registry.get(b_id)
    result = (a >= b).to(torch.float32)
    return _registry.register(result)


def tensor_le(a_id: int, b_id: int) -> int:
    """Element-wise less than or equal: A <= B."""
    a = _registry.get(a_id)
    b = _registry.get(b_id)
    result = (a <= b).to(torch.float32)
    return _registry.register(result)


# ============================================================
#  Math Operations
# ============================================================

def tensor_abs(tensor_id: int) -> int:
    """Absolute value."""
    a = _registry.get(tensor_id)
    result = torch.abs(a)
    return _registry.register(result)


def tensor_sqrt(tensor_id: int) -> int:
    """Square root."""
    a = _registry.get(tensor_id)
    result = torch.sqrt(a)
    return _registry.register(result)


def tensor_exp(tensor_id: int) -> int:
    """Exponential."""
    a = _registry.get(tensor_id)
    result = torch.exp(a)
    return _registry.register(result)


def tensor_log(tensor_id: int) -> int:
    """Natural logarithm."""
    a = _registry.get(tensor_id)
    result = torch.log(a)
    return _registry.register(result)


def tensor_log2(tensor_id: int) -> int:
    """Base-2 logarithm."""
    a = _registry.get(tensor_id)
    result = torch.log2(a)
    return _registry.register(result)


def tensor_log10(tensor_id: int) -> int:
    """Base-10 logarithm."""
    a = _registry.get(tensor_id)
    result = torch.log10(a)
    return _registry.register(result)


def tensor_pow(tensor_id: int, exponent: float) -> int:
    """Power: A ^ exponent."""
    a = _registry.get(tensor_id)
    result = torch.pow(a, exponent)
    return _registry.register(result)


def tensor_sin(tensor_id: int) -> int:
    """Sine."""
    a = _registry.get(tensor_id)
    result = torch.sin(a)
    return _registry.register(result)


def tensor_cos(tensor_id: int) -> int:
    """Cosine."""
    a = _registry.get(tensor_id)
    result = torch.cos(a)
    return _registry.register(result)


def tensor_tan(tensor_id: int) -> int:
    """Tangent."""
    a = _registry.get(tensor_id)
    result = torch.tan(a)
    return _registry.register(result)


def tensor_clamp(tensor_id: int, min_val: float = None, max_val: float = None) -> int:
    """Clamp values to range [min, max]."""
    a = _registry.get(tensor_id)
    result = torch.clamp(a, min=min_val, max=max_val)
    return _registry.register(result)


def tensor_round(tensor_id: int, decimals: int = 0) -> int:
    """Round to decimal places."""
    a = _registry.get(tensor_id)
    result = torch.round(a, decimals=decimals)
    return _registry.register(result)


def tensor_sign(tensor_id: int) -> int:
    """Sign of each element (-1, 0, 1)."""
    a = _registry.get(tensor_id)
    result = torch.sign(a)
    return _registry.register(result)


# ============================================================
#  Operation Dispatcher
# ============================================================

_COMMAND_MAP = {
    # Creation
    'zeros':            lambda p: tensor_zeros(p.get('shape', [1]), p.get('dtype', 'float32')),
    'ones':             lambda p: tensor_ones(p.get('shape', [1]), p.get('dtype', 'float32')),
    'full':             lambda p: tensor_full(p.get('shape', [1]), p.get('value', 0.0), p.get('dtype', 'float32')),
    'arange':           lambda p: tensor_arange(p.get('start', 0.0), p.get('end', 1.0), p.get('step', 1.0), p.get('dtype', 'float32')),
    'eye':              lambda p: tensor_eye(p.get('n', 1), p.get('m'), p.get('dtype', 'float32')),
    'rand':             lambda p: tensor_rand(p.get('shape', [1]), p.get('dtype', 'float32')),
    'randn':            lambda p: tensor_randn(p.get('shape', [1]), p.get('dtype', 'float32')),
    'send_tensor':      lambda p: tensor_from_client(p),
    'get_tensor':       lambda p: tensor_to_client(p.get('tensor_id')),
    'free_tensor':      lambda p: _registry.free(p.get('tensor_id')),
    'tensor_info':      lambda p: _registry.get_info(p.get('tensor_id')),
    # Arithmetic
    'add':              lambda p: tensor_add(p['a_id'], p['b_id']),
    'subtract':         lambda p: tensor_subtract(p['a_id'], p['b_id']),
    'multiply':         lambda p: tensor_multiply(p['a_id'], p['b_id']),
    'divide':           lambda p: tensor_divide(p['a_id'], p['b_id']),
    'matmul':           lambda p: tensor_matmul(p['a_id'], p['b_id']),
    'dot':              lambda p: tensor_dot(p['a_id'], p['b_id']),
    'scalar_multiply':  lambda p: tensor_scalar_multiply(p['tensor_id'], p['scalar']),
    'scalar_add':       lambda p: tensor_scalar_add(p['tensor_id'], p['scalar']),
    'negate':           lambda p: tensor_negate(p['tensor_id']),
    # Activation
    'relu':             lambda p: tensor_relu(p['tensor_id']),
    'sigmoid':          lambda p: tensor_sigmoid(p['tensor_id']),
    'tanh':             lambda p: tensor_tanh(p['tensor_id']),
    'softmax':          lambda p: tensor_softmax(p['tensor_id'], p.get('dim', -1)),
    'log_softmax':      lambda p: tensor_log_softmax(p['tensor_id'], p.get('dim', -1)),
    'gelu':             lambda p: tensor_gelu(p['tensor_id']),
    'leaky_relu':       lambda p: tensor_leaky_relu(p['tensor_id'], p.get('negative_slope', 0.01)),
    'elu':              lambda p: tensor_elu(p['tensor_id'], p.get('alpha', 1.0)),
    'selu':             lambda p: tensor_selu(p['tensor_id']),
    # Reduction
    'sum':              lambda p: tensor_sum(p['tensor_id'], p.get('dim'), p.get('keepdim', False)),
    'mean':             lambda p: tensor_mean(p['tensor_id'], p.get('dim'), p.get('keepdim', False)),
    'max':              lambda p: tensor_max(p['tensor_id'], p.get('dim'), p.get('keepdim', False)),
    'min':              lambda p: tensor_min(p['tensor_id'], p.get('dim'), p.get('keepdim', False)),
    'argmax':           lambda p: tensor_argmax(p['tensor_id'], p.get('dim')),
    'argmin':           lambda p: tensor_argmin(p['tensor_id'], p.get('dim')),
    'std':              lambda p: tensor_std(p['tensor_id'], p.get('dim'), p.get('keepdim', False)),
    'var':              lambda p: tensor_var(p['tensor_id'], p.get('dim'), p.get('keepdim', False)),
    'norm':             lambda p: tensor_norm(p['tensor_id'], p.get('p', 2.0), p.get('dim')),
    # Shape
    'reshape':          lambda p: tensor_reshape(p['tensor_id'], p['shape']),
    'transpose':        lambda p: tensor_transpose(p['tensor_id'], p.get('dim0', 0), p.get('dim1', 1)),
    'permute':          lambda p: tensor_permute(p['tensor_id'], p['dims']),
    'squeeze':          lambda p: tensor_squeeze(p['tensor_id'], p.get('dim')),
    'unsqueeze':        lambda p: tensor_unsqueeze(p['tensor_id'], p['dim']),
    'flatten':          lambda p: tensor_flatten(p['tensor_id'], p.get('start_dim', 0), p.get('end_dim', -1)),
    'view':             lambda p: tensor_view(p['tensor_id'], p['shape']),
    'expand':           lambda p: tensor_expand(p['tensor_id'], p['shape']),
    'repeat':           lambda p: tensor_repeat(p['tensor_id'], p['repeats']),
    'cat':              lambda p: tensor_cat(p['tensor_ids'], p.get('dim', 0)),
    'stack':            lambda p: tensor_stack(p['tensor_ids'], p.get('dim', 0)),
    'split':            lambda p: tensor_split(p['tensor_id'], p['split_size'], p.get('dim', 0)),
    # Comparison
    'eq':               lambda p: tensor_eq(p['a_id'], p['b_id']),
    'gt':               lambda p: tensor_gt(p['a_id'], p['b_id']),
    'lt':               lambda p: tensor_lt(p['a_id'], p['b_id']),
    'ge':               lambda p: tensor_ge(p['a_id'], p['b_id']),
    'le':               lambda p: tensor_le(p['a_id'], p['b_id']),
    # Math
    'abs':              lambda p: tensor_abs(p['tensor_id']),
    'sqrt':             lambda p: tensor_sqrt(p['tensor_id']),
    'exp':              lambda p: tensor_exp(p['tensor_id']),
    'log':              lambda p: tensor_log(p['tensor_id']),
    'log2':             lambda p: tensor_log2(p['tensor_id']),
    'log10':            lambda p: tensor_log10(p['tensor_id']),
    'pow':              lambda p: tensor_pow(p['tensor_id'], p.get('exponent', 2.0)),
    'sin':              lambda p: tensor_sin(p['tensor_id']),
    'cos':              lambda p: tensor_cos(p['tensor_id']),
    'tan':              lambda p: tensor_tan(p['tensor_id']),
    'clamp':            lambda p: tensor_clamp(p['tensor_id'], p.get('min'), p.get('max')),
    'round':            lambda p: tensor_round(p['tensor_id'], p.get('decimals', 0)),
    'sign':             lambda p: tensor_sign(p['tensor_id']),
    # System
    'info':             lambda p: get_device_info(),
    'ping':             lambda p: {'status': 'ok', 'device': DEVICE_NAME, 'timestamp': time.time()},
    'registry_stats':   lambda p: _registry.get_stats(),
    'gc':               lambda p: {'freed': _registry.garbage_collect(p.get('max_idle', 300.0))},
}


def execute_command(command: str, params: dict) -> Any:
    """
    Execute a command received from the client.

    This is the main entry point for the server's command processing.
    All client requests are routed through this function.

    Args:
        command: Command name (e.g., 'zeros', 'add', 'info').
        params: Command parameters dictionary.

    Returns:
        Any: Command result (int, dict, list, etc.)

    Raises:
        ValueError: If command is unknown.
        KeyError: If required parameters are missing.
        Exception: Various exceptions from tensor operations.

    Example:
        >>> result = execute_command('zeros', {'shape': [100, 100]})
        >>> print(result)
        1
        >>> result = execute_command('info', {})
        >>> print(result['device'])
        cuda:0
    """
    handler = _COMMAND_MAP.get(command)

    if handler is None:
        raise ValueError(
            f"Unknown command: '{command}'. "
            f"Available commands: {list(_COMMAND_MAP.keys())}"
        )

    return handler(params)