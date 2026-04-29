"""
Memory Manager Module — GPU/CPU Memory Tracking
================================================
Tracks memory allocations, provides statistics, and performs
automatic garbage collection for the tensor registry.

Works with both GPU (CUDA) and CPU devices transparently.
"""

import time
import threading
from typing import Dict, Optional


class MemoryStats:
    """
    Tracks memory usage statistics for the compute device.

    Provides real-time and peak memory usage information.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._total_allocated = 0
        self._peak_allocated = 0
        self._allocation_count = 0
        self._free_count = 0
        self._last_gc_time = time.time()

    def record_allocation(self, size_bytes: int):
        """Record a memory allocation."""
        with self._lock:
            self._total_allocated += size_bytes
            self._peak_allocated = max(self._peak_allocated, self._total_allocated)
            self._allocation_count += 1

    def record_free(self, size_bytes: int):
        """Record a memory deallocation."""
        with self._lock:
            self._total_allocated = max(0, self._total_allocated - size_bytes)
            self._free_count += 1

    def record_gc(self):
        """Record a garbage collection event."""
        with self._lock:
            self._last_gc_time = time.time()

    @property
    def total_allocated(self) -> int:
        return self._total_allocated

    @property
    def peak_allocated(self) -> int:
        return self._peak_allocated

    @property
    def allocation_count(self) -> int:
        return self._allocation_count

    @property
    def free_count(self) -> int:
        return self._free_count

    def get_stats(self) -> dict:
        """Get memory statistics as a dictionary."""
        with self._lock:
            return {
                'total_allocated_bytes': self._total_allocated,
                'total_allocated_mb': round(self._total_allocated / (1024 * 1024), 2),
                'peak_allocated_bytes': self._peak_allocated,
                'peak_allocated_mb': round(self._peak_allocated / (1024 * 1024), 2),
                'allocations': self._allocation_count,
                'frees': self._free_count,
                'last_gc_seconds_ago': round(time.time() - self._last_gc_time, 1),
            }


# Global memory stats instance
_memory_stats = MemoryStats()


def get_memory_stats() -> MemoryStats:
    """Get the global memory statistics tracker."""
    return _memory_stats