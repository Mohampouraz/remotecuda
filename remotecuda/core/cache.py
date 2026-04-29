"""
Tensor Cache Module — Local Tensor Data Caching
=================================================
Intelligent local caching for remote tensor data.

Reduces network transfers by keeping frequently accessed
tensor data in local memory. Supports LRU and LFU eviction
policies.

Pure Python implementation — no external dependencies.

Architecture:
    ┌─────────────────────────────────────────┐
    │            Tensor Cache                  │
    │                                          │
    │  ┌──────────┐  ┌──────────┐  ┌────────┐│
    │  │ Entry 1  │  │ Entry 2  │  │ Ent N  ││
    │  │ (LRU)    │  │ (Hot)    │  │ (Cold) ││
    │  └──────────┘  └──────────┘  └────────┘│
    │                                          │
    │  Max Size: 512 MB                        │
    │  Eviction: LRU / LFU / FIFO             │
    │  Thread-Safe: Yes                       │
    └─────────────────────────────────────────┘

Usage:
    from remotecuda.core.cache import TensorCache

    cache = TensorCache(max_size_mb=256)

    # Store tensor data
    cache.put('my_key', tensor_data)

    # Retrieve
    data = cache.get('my_key')

    # Check existence
    if 'my_key' in cache:
        print("Cache hit!")
"""

import threading
import time
from typing import Dict, Optional, Any, List
from collections import OrderedDict


class CacheEntry:
    """
    A single entry in the tensor cache.

    Tracks access metadata for eviction decisions.

    Attributes:
        key: Cache key (string).
        data: Stored tensor data (dict with 'data', 'shape', 'dtype').
        size_bytes: Approximate size of stored data in bytes.
        created_at: Timestamp of creation.
        last_access: Timestamp of last access.
        access_count: Number of times accessed.
        pinned: If True, never evict.
    """

    __slots__ = [
        'key', 'data', 'size_bytes', 'created_at',
        'last_access', 'access_count', 'pinned',
    ]

    def __init__(self, key: str, data: dict, size_bytes: int = 0):
        self.key = key
        self.data = data
        self.size_bytes = size_bytes or self._estimate_size(data)
        self.created_at = time.time()
        self.last_access = time.time()
        self.access_count = 0
        self.pinned = False

    def touch(self):
        """Update access metadata."""
        self.last_access = time.time()
        self.access_count += 1

    @property
    def age_seconds(self) -> float:
        """Seconds since creation."""
        return time.time() - self.created_at

    @property
    def idle_seconds(self) -> float:
        """Seconds since last access."""
        return time.time() - self.last_access

    @staticmethod
    def _estimate_size(data: dict) -> int:
        """Estimate size of tensor data in bytes."""
        if 'data' in data and isinstance(data['data'], list):
            return len(data['data']) * 8  # Rough estimate
        return 1024  # Default estimate


class TensorCache:
    """
    LRU/LFU tensor data cache with configurable size limit.

    Reduces network transfers by storing frequently accessed
    tensor data locally.

    Features:
        - Configurable maximum size (MB)
        - LRU (Least Recently Used) eviction
        - LFU (Least Frequently Used) eviction
        - Pinned entries (never evicted)
        - Thread-safe operations
        - Hit/miss statistics
        - Pattern-based invalidation

    Usage:
        cache = TensorCache(max_size_mb=256, policy='lru')

        # Store
        cache.put('model_weights', data_dict)

        # Retrieve
        data = cache.get('model_weights')
        if data:
            print("Cache hit!")

        # Pin important entries
        cache.pin('model_weights')

        # Get statistics
        stats = cache.get_stats()
    """

    POLICY_LRU = 'lru'
    POLICY_LFU = 'lfu'

    def __init__(
        self,
        max_size_mb: float = 256.0,
        policy: str = 'lru',
        max_entry_size_mb: float = 64.0,
    ):
        """
        Initialize the tensor cache.

        Args:
            max_size_mb: Maximum cache size in megabytes.
            policy: Eviction policy ('lru' or 'lfu').
            max_entry_size_mb: Maximum size for a single entry.
        """
        self._max_size = int(max_size_mb * 1024 * 1024)
        self._max_entry_size = int(max_entry_size_mb * 1024 * 1024)
        self._policy = policy
        self._entries: Dict[str, CacheEntry] = OrderedDict()
        self._current_size = 0
        self._lock = threading.RLock()

        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'insertions': 0,
            'invalidations': 0,
        }

    def get(self, key: str) -> Optional[dict]:
        """
        Retrieve data from the cache.

        Args:
            key: Cache key.

        Returns:
            Optional[dict]: Cached data, or None if not found.

        Example:
            >>> data = cache.get('my_tensor')
            >>> if data:
            ...     print(f"Shape: {data['shape']}")
        """
        with self._lock:
            entry = self._entries.get(key)

            if entry is not None:
                entry.touch()
                self.stats['hits'] += 1

                # Move to end for LRU ordering
                if self._policy == self.POLICY_LRU:
                    self._entries.move_to_end(key)

                return entry.data
            else:
                self.stats['misses'] += 1
                return None

    def put(self, key: str, data: dict, pin: bool = False):
        """
        Store data in the cache.

        Args:
            key: Cache key.
            data: Tensor data dictionary.
            pin: If True, this entry will never be evicted.

        Raises:
            ValueError: If entry size exceeds maximum.

        Example:
            >>> cache.put('result_123', {'data': [...], 'shape': [100, 100]})
            >>> cache.put('important', data, pin=True)
        """
        entry = CacheEntry(key, data)

        if entry.size_bytes > self._max_entry_size:
            raise ValueError(
                f"Entry size ({entry.size_bytes / 1024 / 1024:.1f} MB) "
                f"exceeds maximum ({self._max_entry_size / 1024 / 1024:.1f} MB)"
            )

        with self._lock:
            # Remove old entry if exists
            if key in self._entries:
                old = self._entries.pop(key)
                self._current_size -= old.size_bytes

            # Evict if needed
            while (
                self._current_size + entry.size_bytes > self._max_size
                and len(self._entries) > 0
            ):
                victim_key = self._select_victim()
                if victim_key is None:
                    break

                victim = self._entries.pop(victim_key)
                self._current_size -= victim.size_bytes
                self.stats['evictions'] += 1

            # Store
            entry.pinned = pin
            self._entries[key] = entry
            self._current_size += entry.size_bytes
            self.stats['insertions'] += 1

    def pin(self, key: str):
        """
        Pin an entry to prevent eviction.

        Args:
            key: Cache key.

        Raises:
            KeyError: If key not found.
        """
        with self._lock:
            if key not in self._entries:
                raise KeyError(f"Cache key not found: {key}")
            self._entries[key].pinned = True

    def unpin(self, key: str):
        """Unpin an entry, allowing eviction."""
        with self._lock:
            if key in self._entries:
                self._entries[key].pinned = False

    def invalidate(self, key: str):
        """
        Remove a specific entry from the cache.

        Args:
            key: Cache key to remove.
        """
        with self._lock:
            if key in self._entries:
                entry = self._entries.pop(key)
                self._current_size -= entry.size_bytes
                self.stats['invalidations'] += 1

    def invalidate_pattern(self, pattern: str):
        """
        Remove all entries matching a key pattern.

        Args:
            pattern: Substring to match in keys.

        Example:
            >>> cache.invalidate_pattern('temp_')
            >>> cache.invalidate_pattern('batch_123')
        """
        with self._lock:
            keys_to_remove = [
                k for k in self._entries if pattern in k
            ]
            for key in keys_to_remove:
                entry = self._entries.pop(key)
                self._current_size -= entry.size_bytes
                self.stats['invalidations'] += 1

    def clear(self):
        """Remove all entries."""
        with self._lock:
            self._entries.clear()
            self._current_size = 0

    def _select_victim(self) -> Optional[str]:
        """
        Select an entry for eviction.

        Uses the configured policy (LRU or LFU).
        Never selects pinned entries.

        Returns:
            Optional[str]: Key to evict, or None.
        """
        if self._policy == self.POLICY_LRU:
            return self._select_lru_victim()
        elif self._policy == self.POLICY_LFU:
            return self._select_lfu_victim()
        return self._select_lru_victim()

    def _select_lru_victim(self) -> Optional[str]:
        """Select least recently used entry."""
        oldest_key = None
        oldest_time = float('inf')

        for key, entry in self._entries.items():
            if entry.pinned:
                continue
            if entry.last_access < oldest_time:
                oldest_time = entry.last_access
                oldest_key = key

        return oldest_key

    def _select_lfu_victim(self) -> Optional[str]:
        """Select least frequently used entry."""
        victim_key = None
        lowest_count = float('inf')
        oldest_time = float('inf')

        for key, entry in self._entries.items():
            if entry.pinned:
                continue
            if entry.access_count < lowest_count:
                lowest_count = entry.access_count
                oldest_time = entry.last_access
                victim_key = key
            elif entry.access_count == lowest_count:
                if entry.last_access < oldest_time:
                    oldest_time = entry.last_access
                    victim_key = key

        return victim_key

    def get_stats(self) -> dict:
        """Get cache statistics."""
        with self._lock:
            total = self.stats['hits'] + self.stats['misses']
            return {
                **self.stats,
                'entries': len(self._entries),
                'pinned_entries': sum(1 for e in self._entries.values() if e.pinned),
                'current_size_mb': self._current_size / (1024 * 1024),
                'max_size_mb': self._max_size / (1024 * 1024),
                'utilization': self._current_size / self._max_size if self._max_size > 0 else 0,
                'hit_rate': self.stats['hits'] / max(1, total),
                'policy': self._policy,
            }

    def contains(self, key: str) -> bool:
        """Check if key exists in cache."""
        with self._lock:
            return key in self._entries

    def __contains__(self, key: str) -> bool:
        """Support for 'in' operator."""
        return self.contains(key)

    def __getitem__(self, key: str) -> dict:
        """Support for [] access."""
        data = self.get(key)
        if data is None:
            raise KeyError(f"Cache key not found: {key}")
        return data

    def __setitem__(self, key: str, data: dict):
        """Support for [] assignment."""
        self.put(key, data)

    def __len__(self) -> int:
        """Number of entries."""
        return len(self._entries)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.clear()
        return False