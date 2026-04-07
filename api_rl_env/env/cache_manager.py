"""
Cache Manager
=============
LRU cache with TTL expiry for task results.
Agent can use CACHE_USE action to skip an API call if result is cached.
"""

from collections import OrderedDict
from typing import Optional


class CacheManager:
    """
    Simple LRU cache keyed by task_id.
    Each entry expires after `ttl` timesteps.

    Methods
    -------
    put(task_id)  : Store result in cache
    has(task_id)  : Check if cache entry is valid
    tick()        : Advance time — expire old entries
    reset()       : Clear all entries
    """

    def __init__(self, capacity: int = 20, ttl: int = 20):
        self.capacity = capacity
        self.ttl      = ttl
        self._store: OrderedDict[int, int] = OrderedDict()  # task_id → expires_at
        self._clock  = 0

    def reset(self):
        self._store.clear()
        self._clock = 0

    def tick(self):
        self._clock += 1
        # Expire stale entries
        expired = [k for k, exp in self._store.items() if self._clock >= exp]
        for k in expired:
            del self._store[k]

    def put(self, task_id: int):
        """Add / refresh a cache entry."""
        if task_id in self._store:
            self._store.move_to_end(task_id)
        elif len(self._store) >= self.capacity:
            self._store.popitem(last=False)   # evict LRU
        self._store[task_id] = self._clock + self.ttl

    def has(self, task_id: int) -> bool:
        """Return True if task_id has a valid, non-expired cache entry."""
        if task_id not in self._store:
            return False
        if self._clock >= self._store[task_id]:
            del self._store[task_id]
            return False
        return True

    def invalidate(self, task_id: int):
        self._store.pop(task_id, None)

    def __len__(self):
        return len(self._store)

    def __repr__(self):
        return f"CacheManager(size={len(self)}/{self.capacity}, clock={self._clock})"
