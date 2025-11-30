from typing import Optional, Any
import json
import hashlib
from datetime import timedelta
import asyncio
from collections import OrderedDict
import time

class LRUCache:
    """Simple in-memory LRU cache with TTL support."""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        self.cache = OrderedDict()
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.expiry = {}
        self.lock = asyncio.Lock()
    
    def _hash_key(self, key: Any) -> str:
        """Generate hash for cache key."""
        if isinstance(key, str):
            return hashlib.md5(key.encode()).hexdigest()
        return hashlib.md5(json.dumps(key, sort_keys=True).encode()).hexdigest()
    
    async def get(self, key: Any) -> Optional[Any]:
        """Get value from cache."""
        async with self.lock:
            cache_key = self._hash_key(key)
            
            if cache_key not in self.cache:
                return None
            
            # Check expiry
            if cache_key in self.expiry:
                if time.time() > self.expiry[cache_key]:
                    del self.cache[cache_key]
                    del self.expiry[cache_key]
                    return None
            
            # Move to end (most recently used)
            self.cache.move_to_end(cache_key)
            return self.cache[cache_key]
    
    async def set(self, key: Any, value: Any, ttl: Optional[int] = None):
        """Set value in cache with TTL."""
        async with self.lock:
            cache_key = self._hash_key(key)
            
            # Remove oldest if at capacity
            if len(self.cache) >= self.max_size and cache_key not in self.cache:
                oldest = next(iter(self.cache))
                del self.cache[oldest]
                if oldest in self.expiry:
                    del self.expiry[oldest]
            
            self.cache[cache_key] = value
            
            # Set expiry
            ttl_value = ttl if ttl is not None else self.default_ttl
            self.expiry[cache_key] = time.time() + ttl_value
    
    async def delete(self, key: Any):
        """Delete value from cache."""
        async with self.lock:
            cache_key = self._hash_key(key)
            if cache_key in self.cache:
                del self.cache[cache_key]
            if cache_key in self.expiry:
                del self.expiry[cache_key]
    
    async def clear(self):
        """Clear entire cache."""
        async with self.lock:
            self.cache.clear()
            self.expiry.clear()
    
    async def cleanup_expired(self):
        """Remove expired entries."""
        async with self.lock:
            now = time.time()
            expired = [k for k, exp_time in self.expiry.items() if now > exp_time]
            
            for key in expired:
                if key in self.cache:
                    del self.cache[key]
                del self.expiry[key]
    
    def stats(self) -> dict:
        """Get cache statistics."""
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hit_rate": getattr(self, '_hits', 0) / max(getattr(self, '_requests', 1), 1)
        }

# Global cache instances
analysis_cache = LRUCache(max_size=1000, default_ttl=3600)  # 1 hour
rag_cache = LRUCache(max_size=500, default_ttl=7200)  # 2 hours