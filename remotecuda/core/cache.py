"""
Tensor Cache Module
===================
Intelligent caching system for remote GPU tensors.
Reduces network transfers by keeping frequently accessed tensors locally.
Implements various caching strategies for different workload patterns.
"""

import threading
import time
import heapq
from typing import Dict, Optional, Any, Callable
from collections import OrderedDict
from dataclasses import dataclass, field

import torch


@dataclass
class CacheEntry:
    """
    A single entry in the tensor cache.
    
    Tracks metadata for cache management decisions.
    """
    key: str
    tensor: torch.Tensor
    size_bytes: int
    created_at: float = field(default_factory=time.time)
    last_access: float = field(default_factory=time.time)
    access_count: int = 0
    pinned: bool = False  # Pinned entries are never evicted
    
    def touch(self):
        """
        Update access metadata.
        """
        self.last_access = time.time()
        self.access_count += 1
    
    @property
    def age_seconds(self) -> float:
        """
        Time since creation in seconds.
        """
        return time.time() - self.created_at
    
    @property
    def idle_seconds(self) -> float:
        """
        Time since last access in seconds.
        """
        return time.time() - self.last_access


class CachePolicy:
    """
    Base class for cache eviction policies.
    """
    
    def select_victim(self, entries: Dict[str, CacheEntry]) -> Optional[str]:
        """
        Select which entry to evict.
        
        Args:
            entries (dict): All cache entries
            
        Returns:
            Optional[str]: Key of entry to evict, or None
        """
        raise NotImplementedError


class LRUPolicy(CachePolicy):
    """
    Least Recently Used eviction policy.
    
    Evicts the entry that hasn't been accessed for the longest time.
    Ignores pinned entries.
    """
    
    def select_victim(self, entries: Dict[str, CacheEntry]) -> Optional[str]:
        """
        Select LRU entry for eviction.
        """
        oldest_key = None
        oldest_time = float('inf')
        
        for key, entry in entries.items():
            if entry.pinned:
                continue
            if entry.last_access < oldest_time:
                oldest_time = entry.last_access
                oldest_key = key
        
        return oldest_key


class LFUPolicy(CachePolicy):
    """
    Least Frequently Used eviction policy.
    
    Evicts the entry with the lowest access count.
    Ties broken by LRU.
    """
    
    def select_victim(self, entries: Dict[str, CacheEntry]) -> Optional[str]:
        """
        Select LFU entry for eviction.
        """
        victim_key = None
        lowest_count = float('inf')
        oldest_time = float('inf')
        
        for key, entry in entries.items():
            if entry.pinned:
                continue
            
            if entry.access_count < lowest_count:
                lowest_count = entry.access_count
                oldest_time = entry.last_access
                victim_key = key
            elif entry.access_count == lowest_count:
                # Tie-break with LRU
                if entry.last_access < oldest_time:
                    oldest_time = entry.last_access
                    victim_key = key
        
        return victim_key


class SizeAwarePolicy(CachePolicy):
    """
    Size-aware eviction policy.
    
    Evicts the largest entries first to free up space quickly.
    Ties broken by LRU.
    """
    
    def select_victim(self, entries: Dict[str, CacheEntry]) -> Optional[str]:
        """
        Select largest entry for eviction.
        """
        victim_key = None
        largest_size = 0
        oldest_time = float('inf')
        
        for key, entry in entries.items():
            if entry.pinned:
                continue
            
            if entry.size_bytes > largest_size:
                largest_size = entry.size_bytes
                oldest_time = entry.last_access
                victim_key = key
            elif entry.size_bytes == largest_size:
                if entry.last_access < oldest_time:
                    oldest_time = entry.last_access
                    victim_key = key
        
        return victim_key


class TensorCache:
    """
    Intelligent tensor cache for remote GPU operations.
    
    Caches tensors locally to avoid repeated network transfers.
    Supports multiple eviction policies and provides detailed statistics.
    
    Features:
        - Configurable cache size
        - Multiple eviction policies (LRU, LFU, Size-aware)
        - Pinned entries that never get evicted
        - Cache statistics and hit rate tracking
        - Thread-safe operations
        - Automatic memory management
    
    Usage:
        cache = TensorCache(max_size_mb=1024, policy='lru')
        
        # Store a tensor
        cache.put('my_tensor', tensor_data)
        
        # Retrieve a tensor
        tensor = cache.get('my_tensor')
        
        # Pin important tensors
        cache.pin('model_weights')
    """
    
    def __init__(
        self,
        max_size_mb: float = 512.0,
        policy: str = 'lru',
        max_entry_size_mb: float = 128.0
    ):
        """
        Initialize the tensor cache.
        
        Args:
            max_size_mb (float): Maximum cache size in megabytes
            policy (str): Eviction policy: 'lru', 'lfu', or 'size_aware'
            max_entry_size_mb (float): Maximum size for a single cache entry
        """
        self.max_size = int(max_size_mb * 1024 * 1024)  # Convert to bytes
        self.max_entry_size = int(max_entry_size_mb * 1024 * 1024)
        
        # Select eviction policy
        self.policy_name = policy
        if policy == 'lru':
            self.policy = LRUPolicy()
        elif policy == 'lfu':
            self.policy = LFUPolicy()
        elif policy == 'size_aware':
            self.policy = SizeAwarePolicy()
        else:
            raise ValueError(f"Unknown cache policy: {policy}")
        
        # Storage
        self._entries: Dict[str, CacheEntry] = OrderedDict()
        self._current_size = 0
        
        # Statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'insertions': 0,
            'bytes_saved': 0,  # Estimated bandwidth saved
            'total_lookup_time': 0.0
        }
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Background maintenance
        self._maintenance_thread: Optional[threading.Thread] = None
        self._maintenance_running = False
    
    def get(self, key: str) -> Optional[torch.Tensor]:
        """
        Retrieve a tensor from the cache.
        
        Args:
            key (str): Cache key
            
        Returns:
            Optional[torch.Tensor]: Cached tensor, or None if not found
        """
        start_time = time.time()
        
        with self._lock:
            entry = self._entries.get(key)
            
            if entry is not None:
                entry.touch()
                self.stats['hits'] += 1
                self.stats['bytes_saved'] += entry.size_bytes
                
                # Move to end for LRU ordering
                self._entries.move_to_end(key)
                
                result = entry.tensor
            else:
                self.stats['misses'] += 1
                result = None
        
        self.stats['total_lookup_time'] += time.time() - start_time
        return result
    
    def put(self, key: str, tensor: torch.Tensor, pin: bool = False):
        """
        Store a tensor in the cache.
        
        Args:
            key (str): Cache key
            tensor (torch.Tensor): Tensor to cache
            pin (bool): If True, this entry will never be evicted
            
        Raises:
            ValueError: If tensor is too large for cache
        """
        # Detach and move to CPU for cache storage
        cached_tensor = tensor.detach().cpu()
        size_bytes = cached_tensor.element_size() * cached_tensor.nelement()
        
        if size_bytes > self.max_entry_size:
            raise ValueError(
                f"Tensor size ({size_bytes / 1024 / 1024:.1f} MB) exceeds "
                f"maximum entry size ({self.max_entry_size / 1024 / 1024:.1f} MB)"
            )
        
        with self._lock:
            # If already exists, remove old entry first
            if key in self._entries:
                old_entry = self._entries.pop(key)
                self._current_size -= old_entry.size_bytes
            
            # Make space if needed
            while self._current_size + size_bytes > self.max_size and len(self._entries) > 0:
                victim_key = self.policy.select_victim(self._entries)
                if victim_key is None:
                    break  # All entries are pinned
                
                victim = self._entries.pop(victim_key)
                self._current_size -= victim.size_bytes
                self.stats['evictions'] += 1
            
            # Create and store entry
            entry = CacheEntry(
                key=key,
                tensor=cached_tensor,
                size_bytes=size_bytes,
                pinned=pin
            )
            
            self._entries[key] = entry
            self._current_size += size_bytes
            self.stats['insertions'] += 1
    
    def pin(self, key: str):
        """
        Pin a cache entry to prevent eviction.
        
        Args:
            key (str): Cache key to pin
            
        Raises:
            KeyError: If key not found
        """
        with self._lock:
            if key not in self._entries:
                raise KeyError(f"Cache key not found: {key}")
            self._entries[key].pinned = True
    
    def unpin(self, key: str):
        """
        Unpin a cache entry, allowing eviction.
        
        Args:
            key (str): Cache key to unpin
        """
        with self._lock:
            if key in self._entries:
                self._entries[key].pinned = False
    
    def invalidate(self, key: str):
        """
        Remove a specific entry from the cache.
        
        Args:
            key (str): Cache key to remove
        """
        with self._lock:
            if key in self._entries:
                entry = self._entries.pop(key)
                self._current_size -= entry.size_bytes
    
    def invalidate_pattern(self, pattern: str):
        """
        Remove all entries matching a key pattern.
        
        Args:
            pattern (str): String pattern to match in keys
        """
        with self._lock:
            keys_to_remove = [
                key for key in self._entries.keys()
                if pattern in key
            ]
            
            for key in keys_to_remove:
                entry = self._entries.pop(key)
                self._current_size -= entry.size_bytes
    
    def clear(self):
        """
        Clear all entries from the cache.
        """
        with self._lock:
            self._entries.clear()
            self._current_size = 0
    
    def contains(self, key: str) -> bool:
        """
        Check if a key exists in the cache.
        
        Args:
            key (str): Cache key
            
        Returns:
            bool: True if key exists
        """
        with self._lock:
            return key in self._entries
    
    def get_stats(self) -> dict:
        """
        Get cache statistics.
        
        Returns:
            dict: Detailed cache statistics
        """
        with self._lock:
            total_lookups = self.stats['hits'] + self.stats['misses']
            hit_rate = (
                self.stats['hits'] / total_lookups
                if total_lookups > 0 else 0.0
            )
            
            return {
                **self.stats,
                'current_entries': len(self._entries),
                'pinned_entries': sum(1 for e in self._entries.values() if e.pinned),
                'current_size_mb': self._current_size / (1024 * 1024),
                'max_size_mb': self.max_size / (1024 * 1024),
                'utilization': self._current_size / self.max_size if self.max_size > 0 else 0,
                'hit_rate': hit_rate,
                'total_lookups': total_lookups,
                'avg_lookup_time_ms': (
                    (self.stats['total_lookup_time'] / total_lookups * 1000)
                    if total_lookups > 0 else 0.0
                ),
                'estimated_bandwidth_saved_mb': self.stats['bytes_saved'] / (1024 * 1024)
            }
    
    def start_maintenance(self, interval: float = 60.0):
        """
        Start background cache maintenance.
        
        Periodically cleans up old entries and optimizes memory.
        
        Args:
            interval (float): Maintenance interval in seconds
        """
        if self._maintenance_running:
            return
        
        self._maintenance_running = True
        self._maintenance_thread = threading.Thread(
            target=self._maintenance_loop,
            args=(interval,),
            daemon=True
        )
        self._maintenance_thread.start()
    
    def _maintenance_loop(self, interval: float):
        """
        Background maintenance loop.
        """
        while self._maintenance_running:
            time.sleep(interval)
            self._perform_maintenance()
    
    def _perform_maintenance(self):
        """
        Perform cache maintenance tasks.
        
        - Remove entries idle for too long
        - Compact memory
        - Update statistics
        """
        with self._lock:
            # Remove entries idle for more than 30 minutes
            max_idle = 1800  # 30 minutes
            keys_to_remove = []
            
            for key, entry in self._entries.items():
                if entry.pinned:
                    continue
                if entry.idle_seconds > max_idle:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                entry = self._entries.pop(key)
                self._current_size -= entry.size_bytes
                self.stats['evictions'] += 1
    
    def stop_maintenance(self):
        """
        Stop background maintenance.
        """
        self._maintenance_running = False
        if self._maintenance_thread:
            self._maintenance_thread.join(timeout=5)
    
    def __len__(self) -> int:
        """
        Number of entries in the cache.
        """
        return len(self._entries)
    
    def __contains__(self, key: str) -> bool:
        """
        Check if key exists (for 'in' operator).
        """
        return self.contains(key)
    
    def __getitem__(self, key: str) -> torch.Tensor:
        """
        Get tensor by key (for [] operator).
        """
        result = self.get(key)
        if result is None:
            raise KeyError(f"Cache key not found: {key}")
        return result
    
    def __setitem__(self, key: str, tensor: torch.Tensor):
        """
        Set tensor by key (for [] operator).
        """
        self.put(key, tensor)
    
    def __enter__(self):
        self.start_maintenance()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_maintenance()
        self.clear()
        return False