"""
RemoteCUDA Client Package
=========================
Client-side components for remote GPU access.
Provides automatic GPU pool management and transparent CUDA hooks.
"""

from .pool import GPUPool
from .connection import GPUConnection
from .hook import AutoCUDAHook

__all__ = ['GPUPool', 'GPUConnection', 'AutoCUDAHook']