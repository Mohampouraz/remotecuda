"""
RemoteCUDA Server Package
=========================
Background GPU service that runs on GPU-equipped machines.

Quick start:
    remotecuda start
    
No configuration needed - the server automatically announces
its presence on the network and clients discover it.
"""

from .gpu_service import GPUService, GPUWorker
from .discovery import NetworkDiscovery

__all__ = ['GPUService', 'GPUWorker', 'NetworkDiscovery']