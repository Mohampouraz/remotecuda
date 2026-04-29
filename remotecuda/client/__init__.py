"""
RemoteCUDA Client Package — Pure Python
========================================
Zero-dependency client for RemoteCUDA v3.0.

All modules use ONLY Python standard library:
    - connection.py: TCP client with JSON protocol
    - discovery.py:  UDP multicast server discovery
"""

from .connection import (
    ClientConnection,
    RemoteCUDAError,
    ConnectionLostError,
    ServerError,
    MessageTooLargeError,
    encode_tensor_data,
    decode_tensor_data,
    get_dtype_info,
)

from .discovery import (
    discover_server,
    discover_all_servers,
    discover_with_info,
    NetworkScanner,
)

__all__ = [
    # Connection
    'ClientConnection',
    'RemoteCUDAError',
    'ConnectionLostError',
    'ServerError',
    'MessageTooLargeError',
    'encode_tensor_data',
    'decode_tensor_data',
    'get_dtype_info',
    # Discovery
    'discover_server',
    'discover_all_servers',
    'discover_with_info',
    'NetworkScanner',
]