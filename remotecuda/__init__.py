"""
RemoteCUDA v3.0 — Ultimate Zero-Dependency Architecture
========================================================

CLIENT: Pure Python standard library only.
        No NumPy, no PyTorch, no CUDA, no external dependencies.
        Communicates via JSON over TCP sockets.

SERVER: PyTorch required (install with: pip install remotecuda[server])
        Auto-detects GPU availability:
        - CUDA available → GPU acceleration
        - CUDA unavailable → CPU computation (automatic fallback)

Protocol:
    Client (Python stdlib) ←→ JSON over TCP ←→ Server (PyTorch)

    All tensor data is transmitted as base64-encoded raw bytes
    with shape and dtype metadata. No pickle, no numpy serialization.

Quick Start:
    # Server (GPU machine):
    pip install remotecuda[server]
    remotecuda start

    # Client (ANY machine, even a toaster with Python):
    pip install remotecuda
    python
    >>> import remotecuda
    >>> remotecuda.init()
    >>> # Pure Python API — no PyTorch needed on client!
    >>> t = remotecuda.zeros((100, 100))
    >>> r = remotecuda.add(t, t)
    >>> data = remotecuda.get(r)
    >>> remotecuda.shutdown()

Architecture Diagram:
    ┌─────────────────────────┐              ┌──────────────────────────┐
    │     CLIENT (Any OS)     │              │    SERVER (GPU Machine)  │
    │                         │              │                          │
    │  Python 3.8+ stdlib     │    JSON      │  PyTorch 1.10+           │
    │  No NumPy               │◄────────────►│  NumPy 1.20+             │
    │  No PyTorch             │    TCP/IP    │  NVIDIA Driver (optional) │
    │  No CUDA                │              │                          │
    │  No anything            │              │  GPU ──→ CPU fallback    │
    │                         │              │  (automatic)             │
    └─────────────────────────┘              └──────────────────────────┘
"""

__version__ = "3.0.0"
__author__ = "RemoteCUDA Team"
__license__ = "MIT"
__description__ = "Remote GPU — pure Python client, zero dependencies, auto CPU fallback"

# ============================================================
#  Client-Side Global State (Pure Python — no imports needed)
# ============================================================

_global_connection = None
_global_auto_init = False

# ============================================================
#  Pure Python Client API
# ============================================================

def init(server: str = 'auto', port: int = 55555, timeout: float = 30.0) -> bool:
    """
    Initialize RemoteCUDA client.

    Pure Python implementation — zero dependencies required.

    Args:
        server: Server address. 'auto' for UDP auto-discovery,
                or specific IP/hostname.
        port: Server port number.
        timeout: Connection timeout in seconds.

    Returns:
        bool: True if connected successfully.

    Raises:
        ConnectionError: If server cannot be reached.
        RuntimeError: If already initialized.

    Example:
        >>> import remotecuda
        >>> remotecuda.init()                    # Auto-discover
        >>> remotecuda.init(server='10.0.0.5')   # Specific server
    """
    global _global_connection, _global_auto_init

    if _global_connection is not None:
        raise RuntimeError(
            "RemoteCUDA already initialized. Call shutdown() first."
        )

    # Lazy import to keep client pure Python
    from .client.connection import ClientConnection

    if server == 'auto':
        # Try auto-discovery
        from .client.discovery import discover_server
        discovered = discover_server(timeout=3.0)
        if discovered:
            server, port = discovered
        else:
            raise ConnectionError(
                "No GPU server found. Specify server IP manually, "
                "or run 'remotecuda start' on the GPU machine."
            )

    # Connect
    conn = ClientConnection(server, port, timeout=timeout)
    conn.connect()
    _global_connection = conn
    _global_auto_init = True

    return True


def shutdown():
    """
    Gracefully disconnect from the GPU server.

    Safe to call multiple times.
    """
    global _global_connection, _global_auto_init

    if _global_connection is not None:
        try:
            _global_connection.disconnect()
        except Exception:
            pass
        _global_connection = None

    _global_auto_init = False


def _get_connection():
    """Get or auto-initialize connection."""
    global _global_connection, _global_auto_init

    if _global_connection is None:
        if _global_auto_init:
            raise RuntimeError("RemoteCUDA not initialized. Call remotecuda.init() first.")
        init()

    return _global_connection


# ============================================================
#  Tensor Operations (Pure Python Client API)
# ============================================================

def zeros(shape, dtype: str = 'float32'):
    """
    Create a zero-initialized tensor on the remote GPU/CPU.

    Pure Python API — no NumPy or PyTorch needed on client.

    Args:
        shape: Tuple/list of dimensions, e.g., (100, 100) or [32, 3, 224, 224].
        dtype: Data type string ('float32', 'float64', 'int32', 'int64').

    Returns:
        int: Remote tensor ID (opaque handle).

    Example:
        >>> t = remotecuda.zeros((100, 100))
        >>> t = remotecuda.zeros([32, 3, 224, 224], dtype='float32')
    """
    if isinstance(shape, int):
        shape = (shape,)

    conn = _get_connection()
    return conn.send_command('zeros', {
        'shape': list(shape),
        'dtype': dtype,
    })


def ones(shape, dtype: str = 'float32'):
    """
    Create a one-initialized tensor on the remote GPU/CPU.

    Args:
        shape: Tuple/list of dimensions.
        dtype: Data type string.

    Returns:
        int: Remote tensor ID.

    Example:
        >>> t = remotecuda.ones((50, 50))
    """
    if isinstance(shape, int):
        shape = (shape,)

    conn = _get_connection()
    return conn.send_command('ones', {
        'shape': list(shape),
        'dtype': dtype,
    })


def full(shape, value: float, dtype: str = 'float32'):
    """
    Create a tensor filled with a constant value on remote GPU/CPU.

    Args:
        shape: Tuple/list of dimensions.
        value: Fill value (float).
        dtype: Data type string.

    Returns:
        int: Remote tensor ID.

    Example:
        >>> t = remotecuda.full((100, 100), 3.14)
    """
    if isinstance(shape, int):
        shape = (shape,)

    conn = _get_connection()
    return conn.send_command('full', {
        'shape': list(shape),
        'value': float(value),
        'dtype': dtype,
    })


def send(data, dtype: str = 'float32'):
    """
    Send a Python list/array to the remote GPU/CPU as a tensor.

    Pure Python — no NumPy needed. Supports nested lists.

    Args:
        data: Python list, nested list, or flat list.
        dtype: Data type string.

    Returns:
        int: Remote tensor ID.

    Example:
        >>> data = [[1.0, 2.0], [3.0, 4.0]]
        >>> t = remotecuda.send(data)
        >>> # Flat list with inferred shape
        >>> t2 = remotecuda.send([1,2,3,4,5,6], shape=(2,3))
    """
    import json
    import base64
    import struct

    # Flatten the data
    def flatten(nested):
        if isinstance(nested, (list, tuple)):
            result = []
            for item in nested:
                result.extend(flatten(item))
            return result
        return [float(nested)]

    flat = flatten(data)

    # Determine shape
    def get_shape(nested):
        if isinstance(nested, (list, tuple)) and len(nested) > 0:
            if isinstance(nested[0], (list, tuple)):
                return [len(nested)] + get_shape(nested[0])
            return [len(nested)]
        return []

    shape = get_shape(data)
    if not shape:
        shape = [len(flat)]

    # Determine format string based on dtype
    dtype_map = {
        'float32': 'f',
        'float64': 'd',
        'int32': 'i',
        'int64': 'q',
        'uint8': 'B',
        'int8': 'b',
    }

    fmt = dtype_map.get(dtype, 'f')
    element_size = struct.calcsize(fmt)

    # Pack data to bytes
    raw_bytes = b''
    if fmt in ('f', 'd'):
        for val in flat:
            raw_bytes += struct.pack(fmt, float(val))
    elif fmt in ('i', 'q', 'b'):
        for val in flat:
            raw_bytes += struct.pack(fmt, int(val))
    elif fmt == 'B':
        for val in flat:
            raw_bytes += struct.pack(fmt, int(val) & 0xFF)

    # Encode as base64
    encoded = base64.b64encode(raw_bytes).decode('ascii')

    conn = _get_connection()
    return conn.send_command('send_tensor', {
        'data': encoded,
        'shape': shape,
        'dtype': dtype,
        'encoding': 'base64',
        'format': fmt,
        'size': len(raw_bytes),
    })


def get(tensor_id: int):
    """
    Retrieve tensor data from the remote GPU/CPU.

    Returns data as a Python list (flattened). Use shape info
    to reconstruct if needed.

    Args:
        tensor_id: Remote tensor ID.

    Returns:
        dict: {'data': [...], 'shape': [...], 'dtype': '...'}

    Example:
        >>> result = remotecuda.get(tensor_id)
        >>> print(result['data'])   # Flat list
        >>> print(result['shape'])  # Original shape
    """
    import base64
    import struct

    conn = _get_connection()
    response = conn.send_command('get_tensor', {'tensor_id': tensor_id})

    # Decode base64 data
    encoded = response.get('data', '')
    dtype = response.get('dtype', 'float32')
    fmt = response.get('format', 'f')
    raw_bytes = base64.b64decode(encoded)

    # Unpack to Python list
    element_size = struct.calcsize(fmt)
    num_elements = len(raw_bytes) // element_size

    data = []
    if fmt in ('f', 'd'):
        for i in range(num_elements):
            val = struct.unpack_from(fmt, raw_bytes, i * element_size)[0]
            data.append(float(val))
    elif fmt in ('i', 'q', 'b', 'B'):
        for i in range(num_elements):
            val = struct.unpack_from(fmt, raw_bytes, i * element_size)[0]
            data.append(int(val))

    return {
        'data': data,
        'shape': response.get('shape', []),
        'dtype': dtype,
        'tensor_id': tensor_id,
    }


def add(a_id: int, b_id: int):
    """
    Element-wise addition on remote GPU/CPU: C = A + B.

    Args:
        a_id: First tensor ID.
        b_id: Second tensor ID.

    Returns:
        int: Result tensor ID.

    Example:
        >>> a = remotecuda.ones((100, 100))
        >>> b = remotecuda.ones((100, 100))
        >>> c = remotecuda.add(a, b)  # c = all 2.0
    """
    conn = _get_connection()
    return conn.send_command('add', {
        'a_id': a_id,
        'b_id': b_id,
    })


def subtract(a_id: int, b_id: int):
    """
    Element-wise subtraction on remote GPU/CPU: C = A - B.

    Args:
        a_id: First tensor ID.
        b_id: Second tensor ID.

    Returns:
        int: Result tensor ID.
    """
    conn = _get_connection()
    return conn.send_command('subtract', {
        'a_id': a_id,
        'b_id': b_id,
    })


def multiply(a_id: int, b_id: int):
    """
    Element-wise multiplication on remote GPU/CPU: C = A * B.

    Args:
        a_id: First tensor ID.
        b_id: Second tensor ID.

    Returns:
        int: Result tensor ID.
    """
    conn = _get_connection()
    return conn.send_command('multiply', {
        'a_id': a_id,
        'b_id': b_id,
    })


def divide(a_id: int, b_id: int):
    """
    Element-wise division on remote GPU/CPU: C = A / B.

    Args:
        a_id: First tensor ID.
        b_id: Second tensor ID.

    Returns:
        int: Result tensor ID.
    """
    conn = _get_connection()
    return conn.send_command('divide', {
        'a_id': a_id,
        'b_id': b_id,
    })


def matmul(a_id: int, b_id: int):
    """
    Matrix multiplication on remote GPU/CPU: C = A @ B.

    Args:
        a_id: First tensor ID (2D).
        b_id: Second tensor ID (2D).

    Returns:
        int: Result tensor ID.
    """
    conn = _get_connection()
    return conn.send_command('matmul', {
        'a_id': a_id,
        'b_id': b_id,
    })


def relu(tensor_id: int):
    """
    ReLU activation on remote GPU/CPU: max(0, x).

    Args:
        tensor_id: Input tensor ID.

    Returns:
        int: Result tensor ID.
    """
    conn = _get_connection()
    return conn.send_command('relu', {'tensor_id': tensor_id})


def sigmoid(tensor_id: int):
    """
    Sigmoid activation on remote GPU/CPU.

    Args:
        tensor_id: Input tensor ID.

    Returns:
        int: Result tensor ID.
    """
    conn = _get_connection()
    return conn.send_command('sigmoid', {'tensor_id': tensor_id})


def tanh(tensor_id: int):
    """
    Tanh activation on remote GPU/CPU.

    Args:
        tensor_id: Input tensor ID.

    Returns:
        int: Result tensor ID.
    """
    conn = _get_connection()
    return conn.send_command('tanh', {'tensor_id': tensor_id})


def softmax(tensor_id: int, dim: int = -1):
    """
    Softmax on remote GPU/CPU.

    Args:
        tensor_id: Input tensor ID.
        dim: Dimension for softmax.

    Returns:
        int: Result tensor ID.
    """
    conn = _get_connection()
    return conn.send_command('softmax', {
        'tensor_id': tensor_id,
        'dim': dim,
    })


def sum(tensor_id: int, dim: int = None):
    """
    Sum of tensor elements on remote GPU/CPU.

    Args:
        tensor_id: Input tensor ID.
        dim: Dimension to sum over (None = all elements).

    Returns:
        int: Result tensor ID (scalar or reduced).
    """
    conn = _get_connection()
    return conn.send_command('sum', {
        'tensor_id': tensor_id,
        'dim': dim,
    })


def mean(tensor_id: int, dim: int = None):
    """
    Mean of tensor elements on remote GPU/CPU.

    Args:
        tensor_id: Input tensor ID.
        dim: Dimension for mean (None = all elements).

    Returns:
        int: Result tensor ID.
    """
    conn = _get_connection()
    return conn.send_command('mean', {
        'tensor_id': tensor_id,
        'dim': dim,
    })


def reshape(tensor_id: int, shape):
    """
    Reshape tensor on remote GPU/CPU.

    Args:
        tensor_id: Input tensor ID.
        shape: New shape tuple/list.

    Returns:
        int: Result tensor ID (same data, new shape).
    """
    if isinstance(shape, int):
        shape = (shape,)

    conn = _get_connection()
    return conn.send_command('reshape', {
        'tensor_id': tensor_id,
        'shape': list(shape),
    })


def transpose(tensor_id: int, dim0: int = 0, dim1: int = 1):
    """
    Transpose tensor on remote GPU/CPU.

    Args:
        tensor_id: Input tensor ID.
        dim0: First dimension to swap.
        dim1: Second dimension to swap.

    Returns:
        int: Result tensor ID.
    """
    conn = _get_connection()
    return conn.send_command('transpose', {
        'tensor_id': tensor_id,
        'dim0': dim0,
        'dim1': dim1,
    })


def free(tensor_id: int):
    """
    Free a tensor from the remote GPU/CPU memory.

    Args:
        tensor_id: Tensor ID to free.

    Example:
        >>> remotecuda.free(tensor_id)
    """
    conn = _get_connection()
    conn.send_command('free_tensor', {'tensor_id': tensor_id})


def info():
    """
    Get information about the remote GPU/CPU server.

    Returns:
        dict: Server information including device type, memory, etc.

    Example:
        >>> info = remotecuda.info()
        >>> print(info['device'])  # 'cuda:0' or 'cpu'
    """
    conn = _get_connection()
    return conn.send_command('info', {})


def status():
    """
    Print current RemoteCUDA status.
    """
    conn = _get_connection()
    server_info = conn.send_command('info', {})

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║              RemoteCUDA v3.0 — Status                       ║
╠══════════════════════════════════════════════════════════════╣
║  Server:     {server_info.get('host', '?'):<42}║
║  Device:     {server_info.get('device', '?')[:42]:<42}║
║  GPU Count:  {server_info.get('gpu_count', 0):<42}║
║  Tensors:    {server_info.get('active_tensors', 0):<42}║
║  Memory:     {server_info.get('memory_used_mb', 0):.0f}/{server_info.get('memory_total_mb', 0):.0f} MB{' ' * (30 - len(f'{server_info.get("memory_used_mb", 0):.0f}/{server_info.get("memory_total_mb", 0):.0f}'))}║
╚══════════════════════════════════════════════════════════════╝
    """)


# For backward compatibility with PyTorch client code
def _get_hook_module():
    """Get the optional PyTorch hook module (only if PyTorch is installed)."""
    try:
        from .client.hook import AutoCUDAHook
        from .client.pool import GPUPool
        return AutoCUDAHook, GPUPool
    except ImportError:
        return None, None