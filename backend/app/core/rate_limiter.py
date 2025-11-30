from fastapi import HTTPException, Request
from datetime import datetime, timedelta
from collections import defaultdict
import asyncio
from typing import Dict, Tuple

class RateLimiter:
    """In-memory rate limiter with sliding window."""
    
    def __init__(self):
        self.requests: Dict[str, list] = defaultdict(list)
        self.locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
    
    async def check_rate_limit(
        self,
        identifier: str,
        limit: int = 100,
        window_seconds: int = 3600
    ) -> Tuple[bool, dict]:
        """
        Check if request is within rate limit.
        
        Args:
            identifier: User identifier (user_id, IP, etc.)
            limit: Maximum requests allowed
            window_seconds: Time window in seconds
        
        Returns:
            (is_allowed, stats_dict)
        """
        async with self.locks[identifier]:
            now = datetime.utcnow()
            cutoff = now - timedelta(seconds=window_seconds)
            
            # Clean old requests
            self.requests[identifier] = [
                ts for ts in self.requests[identifier]
                if ts > cutoff
            ]
            
            current_count = len(self.requests[identifier])
            
            if current_count >= limit:
                reset_time = self.requests[identifier][0] + timedelta(seconds=window_seconds)
                return False, {
                    "limit": limit,
                    "remaining": 0,
                    "reset": reset_time.isoformat(),
                    "retry_after": int((reset_time - now).total_seconds())
                }
            
            # Add current request
            self.requests[identifier].append(now)
            
            return True, {
                "limit": limit,
                "remaining": limit - current_count - 1,
                "reset": (now + timedelta(seconds=window_seconds)).isoformat()
            }
    
    async def cleanup_old_entries(self, max_age_hours: int = 24):
        """Periodically clean up old entries to prevent memory bloat."""
        cutoff = datetime.utcnow() - timedelta(hours=max_age_hours)
        
        for identifier in list(self.requests.keys()):
            async with self.locks[identifier]:
                self.requests[identifier] = [
                    ts for ts in self.requests[identifier]
                    if ts > cutoff
                ]
                
                # Remove empty entries
                if not self.requests[identifier]:
                    del self.requests[identifier]
                    del self.locks[identifier]

# Global rate limiter instance
rate_limiter = RateLimiter()

async def rate_limit_middleware(
    request: Request,
    user_info: dict,
    limit: int = 100,
    window: int = 3600
):
    """Middleware to enforce rate limits."""
    identifier = user_info.get("user_id", request.client.host)
    user_limit = user_info.get("rate_limit", limit)
    
    is_allowed, stats = await rate_limiter.check_rate_limit(
        identifier,
        user_limit,
        window
    )
    
    # Add rate limit headers to response
    request.state.rate_limit_stats = stats
    
    if not is_allowed:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Retry after {stats['retry_after']} seconds.",
            headers={
                "X-RateLimit-Limit": str(stats["limit"]),
                "X-RateLimit-Remaining": str(stats["remaining"]),
                "X-RateLimit-Reset": stats["reset"],
                "Retry-After": str(stats["retry_after"])
            }
        )