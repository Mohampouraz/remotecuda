"""
Tensor Adapter Module — Pure Python ↔ PyTorch Bridge
======================================================
Provides conversion between Pure Python tensor representations
and PyTorch tensors for optional client-side usage.

This module is OPTIONAL. It requires PyTorch to be installed
on the client side. Without PyTorch, use the pure Python API
in remotecuda/__init__.py directly.

Features:
    - Convert Python lists to remote tensors
    - Convert remote tensors to PyTorch tensors
    - Automatic dtype handling
    - Shape validation
    - Thread-safe operations

Usage:
    from remotecuda.client.tensor_adapter import TensorAdapter
    from remotecuda.client.connection import ClientConnection

    conn = ClientConnection('10.0.0.5', 55555)
    conn.connect()

    adapter = TensorAdapter(conn)

    # Python list → Remote GPU tensor
    data = [[1.0, 2.0], [3.0, 4.0]]
    remote_id = adapter.send_python_list(data)

    # Remote GPU tensor → PyTorch tensor
    torch_tensor = adapter.get_as_torch(remote_id)

    conn.disconnect()
"""

import threading
import struct
import base64
from typing import Dict, Optional, List, Tuple, Any


class TensorAdapter:
    """
    Bridges Pure Python tensors with optional PyTorch tensors.

    Manages the mapping between local tensor IDs (used by the
    client) and remote tensor IDs (used by the server).

    All pure Python operations are available without PyTorch.
    PyTorch conversion methods require PyTorch to be installed.

    Attributes:
        _conn: ClientConnection to the server.
        _registry: Dict mapping local_id → remote_info.
    """

    def __init__(self, connection):
        """
        Initialize the tensor adapter.

        Args:
            connection: ClientConnection instance.
        """
        self._conn = connection
        self._registry: Dict[int, dict] = {}
        self._lock = threading.Lock()
        self._counter = 0

    # ============================================================
    #  Pure Python Operations (No PyTorch Needed)
    # ============================================================

    def send_python_list(
        self,
        data: list,
        dtype: str = 'float32',
        shape: list = None,
    ) -> int:
        """
        Send a Python list to the remote GPU/CPU as a tensor.

        Pure Python — no NumPy or PyTorch needed.

        Args:
            data: Nested Python list (e.g., [[1,2],[3,4]]) or flat list.
            dtype: Data type string ('float32', 'float64', 'int32', 'int64').
            shape: Shape of the tensor. If None, inferred from nesting.

        Returns:
            int: Remote tensor ID.

        Raises:
            ValueError: If data cannot be converted.

        Example:
            >>> adapter = TensorAdapter(conn)
            >>> tid = adapter.send_python_list([[1.0, 2.0], [3.0, 4.0]])
            >>> tid = adapter.send_python_list([1,2,3,4,5,6], shape=[2,3])
        """
        # Flatten nested lists
        flat = self._flatten(data)

        # Infer shape
        if shape is None:
            shape = self._infer_shape(data)

        # Determine struct format
        fmt_map = {
            'float32': 'f',
            'float64': 'd',
            'int32': 'i',
            'int64': 'q',
            'uint8': 'B',
        }
        fmt = fmt_map.get(dtype, 'f')

        # Pack to bytes
        raw_bytes = b''
        for val in flat:
            if fmt in ('f', 'd'):
                raw_bytes += struct.pack(fmt, float(val))
            else:
                raw_bytes += struct.pack(fmt, int(val))

        # Encode
        encoded = base64.b64encode(raw_bytes).decode('ascii')

        # Send to server
        remote_id = self._conn.send_command('send_tensor', {
            'data': encoded,
            'shape': shape,
            'dtype': dtype,
            'format': fmt,
            'size': len(raw_bytes),
        })

        # Register locally
        with self._lock:
            self._counter += 1
            local_id = self._counter
            self._registry[local_id] = {
                'remote_id': remote_id,
                'shape': shape,
                'dtype': dtype,
            }

        return remote_id

    def get_as_python(self, remote_id: int) -> dict:
        """
        Retrieve a remote tensor as a Python list.

        Pure Python — no dependencies.

        Args:
            remote_id: Remote tensor ID.

        Returns:
            dict: {'data': [...], 'shape': [...], 'dtype': '...'}

        Example:
            >>> result = adapter.get_as_python(remote_id)
            >>> print(result['data'][:5])  # First 5 values
        """
        response = self._conn.send_command('get_tensor', {'tensor_id': remote_id})

        encoded = response.get('data', '')
        dtype = response.get('dtype', 'float32')
        fmt = response.get('format', 'f')
        shape = response.get('shape', [])

        # Decode
        raw_bytes = base64.b64decode(encoded)
        element_size = struct.calcsize(fmt)
        num_elements = len(raw_bytes) // element_size

        data = []
        if fmt in ('f', 'd'):
            for i in range(num_elements):
                val = struct.unpack_from(fmt, raw_bytes, i * element_size)[0]
                data.append(float(val))
        else:
            for i in range(num_elements):
                val = struct.unpack_from(fmt, raw_bytes, i * element_size)[0]
                data.append(int(val))

        return {
            'data': data,
            'shape': shape,
            'dtype': dtype,
            'tensor_id': remote_id,
        }

    # ============================================================
    #  PyTorch Operations (Requires PyTorch)
    # ============================================================

    def get_as_torch(self, remote_id: int):
        """
        Retrieve a remote tensor as a PyTorch tensor.

        Requires PyTorch to be installed on the client.

        Args:
            remote_id: Remote tensor ID.

        Returns:
            torch.Tensor: PyTorch tensor on CPU.

        Raises:
            ImportError: If PyTorch is not installed.

        Example:
            >>> torch_tensor = adapter.get_as_torch(remote_id)
            >>> print(torch_tensor.shape)
        """
        try:
            import torch
            import numpy as np
        except ImportError:
            raise ImportError(
                "PyTorch is required for this method. "
                "Use get_as_python() for pure Python access, or "
                "install PyTorch: pip install torch"
            )

        response = self._conn.send_command('get_tensor', {'tensor_id': remote_id})

        encoded = response.get('data', '')
        shape = response.get('shape', [])
        dtype_str = response.get('dtype', 'float32')

        raw_bytes = base64.b64decode(encoded)

        dtype_map = {
            'float32': np.float32,
            'float64': np.float64,
            'int32': np.int32,
            'int64': np.int64,
            'uint8': np.uint8,
        }
        np_dtype = dtype_map.get(dtype_str, np.float32)

        arr = np.frombuffer(raw_bytes, dtype=np_dtype).reshape(shape)
        return torch.from_numpy(arr.copy())

    def send_torch_tensor(self, tensor) -> int:
        """
        Send a PyTorch tensor to the remote GPU/CPU.

        Requires PyTorch to be installed on the client.

        Args:
            tensor: PyTorch tensor.

        Returns:
            int: Remote tensor ID.

        Raises:
            ImportError: If PyTorch is not installed.
        """
        try:
            import numpy as np
        except ImportError:
            raise ImportError(
                "NumPy is required for this method. "
                "Install: pip install numpy"
            )

        cpu_tensor = tensor.detach().cpu()
        raw_bytes = cpu_tensor.numpy().tobytes()
        encoded = base64.b64encode(raw_bytes).decode('ascii')

        dtype_map = {
            'torch.float32': 'float32',
            'torch.float64': 'float64',
            'torch.int32': 'int32',
            'torch.int64': 'int64',
            'torch.uint8': 'uint8',
        }
        dtype_str = dtype_map.get(str(tensor.dtype), 'float32')

        fmt_map = {
            'float32': 'f',
            'float64': 'd',
            'int32': 'i',
            'int64': 'q',
            'uint8': 'B',
        }
        fmt = fmt_map.get(dtype_str, 'f')

        remote_id = self._conn.send_command('send_tensor', {
            'data': encoded,
            'shape': list(tensor.shape),
            'dtype': dtype_str,
            'format': fmt,
            'size': len(raw_bytes),
        })

        return remote_id

    # ============================================================
    #  Helpers
    # ============================================================

    @staticmethod
    def _flatten(nested) -> list:
        """Flatten arbitrarily nested lists."""
        if isinstance(nested, (list, tuple)):
            result = []
            for item in nested:
                result.extend(TensorAdapter._flatten(item))
            return result
        return [nested]

    @staticmethod
    def _infer_shape(nested) -> list:
        """Infer shape from nested list structure."""
        if isinstance(nested, (list, tuple)):
            if len(nested) > 0 and isinstance(nested[0], (list, tuple)):
                return [len(nested)] + TensorAdapter._infer_shape(nested[0])
            return [len(nested)]
        return []

    def free(self, remote_id: int):
        """Free a remote tensor."""
        try:
            self._conn.send_command('free_tensor', {'tensor_id': remote_id})
        except Exception:
            pass

        with self._lock:
            to_remove = []
            for local_id, info in self._registry.items():
                if info['remote_id'] == remote_id:
                    to_remove.append(local_id)
            for local_id in to_remove:
                del self._registry[local_id]

    def clear(self):
        """Free all remote tensors managed by this adapter."""
        with self._lock:
            for info in self._registry.values():
                try:
                    self._conn.send_command(
                        'free_tensor',
                        {'tensor_id': info['remote_id']}
                    )
                except Exception:
                    pass
            self._registry.clear()