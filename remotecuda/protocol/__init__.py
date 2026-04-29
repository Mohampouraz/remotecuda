"""
RemoteCUDA Server Package
=========================
Computation server with auto GPU/CPU fallback.

Requirements:
    pip install remotecuda[server]
    (installs PyTorch + NumPy)

Start with:
    remotecuda start
    remotecuda start --port 55555

The server automatically:
    1. Detects CUDA availability
    2. Falls back to CPU if no GPU
    3. Broadcasts presence on the network
    4. Accepts TCP connections from clients
    5. Executes tensor operations on the best available device
"""

from .gpu_service import GPUService
from .discovery import NetworkDiscovery
from .compute_ops import (
    get_device_info,
    get_registry,
    execute_command,
    DEVICE,
    DEVICE_NAME,
    DEVICE_IS_CUDA,
    DEVICE_IS_CPU,
)

__all__ = [
    'GPUService',
    'NetworkDiscovery',
    'get_device_info',
    'get_registry',
    'execute_command',
    'DEVICE',
    'DEVICE_NAME',
    'DEVICE_IS_CUDA',
    'DEVICE_IS_CPU',
]