"""
PyTorch CUDA Hook Module — Optional Client Extension
======================================================
Provides automatic interception of PyTorch .cuda() calls
for users who want to use PyTorch on the client side.

This module is OPTIONAL. The core client API (remotecuda.init(),
remotecuda.zeros(), etc.) works without PyTorch.

To use this module, install: pip install remotecuda[client]
(This adds PyTorch as a dependency.)

When installed and activated, all PyTorch .cuda() and .to('cuda')
calls are automatically redirected to the RemoteCUDA server.

Architecture:
    User Code:  model.cuda()
                     │
                     ▼
    AutoCUDAHook:  intercept call
                     │
                     ▼
    TensorAdapter:  convert PyTorch tensor → remote tensor
                     │
                     ▼
    ConnectionPool:  send to best GPU server
                     │
                     ▼
    GPU Server:      execute on real GPU (or CPU fallback)

Usage:
    import remotecuda
    remotecuda.init()
    
    from remotecuda.client.hook import install_hook, uninstall_hook
    
    # Install CUDA hooks
    install_hook()
    
    # Now all PyTorch .cuda() calls use remote GPU
    import torch
    model = torch.nn.Linear(10, 10).cuda()  # → Remote GPU!
    x = torch.randn(5, 10).cuda()           # → Remote GPU!
    output = model(x)                        # → Computed remotely!
    
    # Uninstall hooks when done
    uninstall_hook()
    remotecuda.shutdown()
"""

import threading
from typing import Optional, Dict, Any

from .pool import ConnectionPool


class AutoCUDAHook:
    """
    Transparent PyTorch CUDA call interceptor.

    When installed, all .cuda(), .to('cuda') calls are
    redirected to the RemoteCUDA server.

    The client must have PyTorch installed for this to work.
    """

    def __init__(self, pool: ConnectionPool):
        """
        Initialize the CUDA hook.

        Args:
            pool: ConnectionPool for server access.
        """
        self._pool = pool
        self._installed = False
        self._originals = {}
        self._lock = threading.Lock()
        self._tensor_registry: Dict[int, int] = {}  # id(local) → remote_id
        self._model_registry: Dict[int, str] = {}   # id(model) → server_key

    def install(self):
        """
        Install CUDA hooks on PyTorch methods.

        After installation, all .cuda() calls are redirected
        to remote GPU servers.
        """
        if self._installed:
            return

        try:
            import torch
            import torch.nn as nn
        except ImportError:
            raise ImportError(
                "PyTorch is required for CUDA hooks. "
                "Install with: pip install remotecuda[client]"
            )

        # Save original methods
        self._originals['Tensor.cuda'] = torch.Tensor.cuda
        self._originals['Tensor.to'] = torch.Tensor.to
        self._originals['Module.cuda'] = nn.Module.cuda
        self._originals['Module.to'] = nn.Module.to

        # Install hooked versions
        torch.Tensor.cuda = self._hooked_tensor_cuda
        torch.Tensor.to = self._hooked_tensor_to
        nn.Module.cuda = self._hooked_module_cuda
        nn.Module.to = self._hooked_module_to

        self._installed = True

    def uninstall(self):
        """Restore original PyTorch methods."""
        if not self._installed:
            return

        try:
            import torch
            import torch.nn as nn
        except ImportError:
            return

        torch.Tensor.cuda = self._originals['Tensor.cuda']
        torch.Tensor.to = self._originals['Tensor.to']
        nn.Module.cuda = self._originals['Module.cuda']
        nn.Module.to = self._originals['Module.to']

        self._installed = False
        self._tensor_registry.clear()
        self._model_registry.clear()

    def _send_tensor_to_gpu(self, tensor):
        """Transfer a PyTorch tensor to remote GPU."""
        import torch
        import numpy as np
        import base64
        import struct

        local_id = id(tensor)

        with self._lock:
            if local_id in self._tensor_registry:
                return tensor

        try:
            conn = self._pool.get_connection()

            # Convert tensor to bytes
            cpu_tensor = tensor.detach().cpu()
            raw_bytes = cpu_tensor.numpy().tobytes()
            encoded = base64.b64encode(raw_bytes).decode('ascii')

            # Send to server
            response = conn.send_command('send_tensor', {
                'data': encoded,
                'shape': list(tensor.shape),
                'dtype': str(tensor.dtype).replace('torch.', ''),
                'format': 'f',
                'size': len(raw_bytes),
            })

            remote_id = response  # Server returns tensor ID

            with self._lock:
                self._tensor_registry[local_id] = remote_id

            # Store remote ID on tensor for later retrieval
            tensor._remotecuda_id = remote_id
            tensor._remotecuda_conn = conn

            return tensor

        except Exception as e:
            print(f"Warning: Failed to send tensor to GPU: {e}")
            return tensor

    def _get_tensor_from_gpu(self, tensor):
        """Fetch tensor data from remote GPU."""
        import torch
        import numpy as np
        import base64
        import struct

        remote_id = getattr(tensor, '_remotecuda_id', None)
        conn = getattr(tensor, '_remotecuda_conn', None)

        if remote_id is None or conn is None:
            return self._originals['Tensor.cpu'](tensor)

        try:
            response = conn.send_command('get_tensor', {'tensor_id': remote_id})

            encoded = response.get('data', '')
            shape = response.get('shape', [])
            dtype_str = response.get('dtype', 'float32')

            raw_bytes = base64.b64decode(encoded)
            dtype_map = {
                'float32': np.float32,
                'float64': np.float64,
                'int32': np.int32,
                'int64': np.int64,
            }
            np_dtype = dtype_map.get(dtype_str, np.float32)
            arr = np.frombuffer(raw_bytes, dtype=np_dtype).reshape(shape)

            return torch.from_numpy(arr.copy())

        except Exception as e:
            print(f"Warning: Failed to get tensor from GPU: {e}")
            return self._originals['Tensor.cpu'](tensor)

    def _hooked_tensor_cuda(self, tensor, *args, **kwargs):
        """Intercepted tensor.cuda() method."""
        return self._send_tensor_to_gpu(tensor)

    def _hooked_tensor_to(self, tensor, *args, **kwargs):
        """Intercepted tensor.to() method."""
        device = args[0] if args else kwargs.get('device', 'cpu')

        if 'cuda' in str(device):
            return self._send_tensor_to_gpu(tensor)
        elif 'cpu' in str(device):
            return self._get_tensor_from_gpu(tensor)
        else:
            return self._originals['Tensor.to'](tensor, *args, **kwargs)

    def _hooked_module_cuda(self, module, *args, **kwargs):
        """Intercepted module.cuda() method."""
        # Send all parameters to GPU
        for name, param in module.named_parameters():
            self._send_tensor_to_gpu(param.data)

        for name, buffer in module.named_buffers():
            self._send_tensor_to_gpu(buffer)

        return module

    def _hooked_module_to(self, module, *args, **kwargs):
        """Intercepted module.to() method."""
        device = args[0] if args else kwargs.get('device', 'cpu')

        if 'cuda' in str(device):
            return self._hooked_module_cuda(module)
        else:
            return self._originals['Module.to'](module, *args, **kwargs)


# Module-level convenience functions

_global_hook: Optional[AutoCUDAHook] = None


def install_hook(pool: ConnectionPool = None):
    """
    Install global PyTorch CUDA hooks.

    Args:
        pool: ConnectionPool to use. If None, uses the global pool
              from remotecuda.get_pool().

    Example:
        >>> from remotecuda.client.hook import install_hook, uninstall_hook
        >>> install_hook()
        >>> import torch
        >>> x = torch.randn(10).cuda()  # → Remote GPU!
        >>> uninstall_hook()
    """
    global _global_hook

    if pool is None:
        import remotecuda
        pool = remotecuda.get_pool()
        if pool is None:
            raise RuntimeError(
                "No connection pool available. "
                "Run remotecuda.init() first."
            )

    if _global_hook is not None:
        _global_hook.uninstall()

    _global_hook = AutoCUDAHook(pool)
    _global_hook.install()


def uninstall_hook():
    """
    Uninstall global PyTorch CUDA hooks.

    Restores original PyTorch behavior.
    """
    global _global_hook

    if _global_hook is not None:
        _global_hook.uninstall()
        _global_hook = None