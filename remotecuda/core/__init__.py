"""
RemoteCUDA Core Module
=======================
Core components for advanced client-side functionality.

These modules provide:
    - TensorBridge: Unified tensor lifecycle management
    - StreamManager: Async operation pipeline management
    - Cache: Intelligent tensor caching to reduce network transfers

All modules are optional and work with the pure Python client.
No external dependencies required for basic usage.
"""

from .tensor_bridge import TensorBridge
from .stream_manager import StreamManager
from .cache import TensorCache

__all__ = ['TensorBridge', 'StreamManager', 'TensorCache']