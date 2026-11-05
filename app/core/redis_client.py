"""
Redis client for caching and performance optimization
Provides fast lookups for fingerprints, file hashes, and embedding results
"""
import os
import json
import redis
import structlog
from typing import Optional, List, Dict, Any
from functools import wraps
import hashlib

logger = structlog.get_logger()

class RedisCache:
    """Redis client wrapper for PoC caching operations"""
    
    def __init__(self):
        self.client = None
        self.enabled = False
        
        redis_url = os.getenv("REDIS_URL")
        if not redis_url:
            logger.warning("REDIS_URL not configured - caching disabled")
            return
        
        try:
            # Initialize Redis client with connection pooling
            self.client = redis.from_url(
                redis_url,
                decode_responses=True,  # Auto-decode bytes to strings
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30,
            )
            
            # Test connection
            self.client.ping()
            self.enabled = True
            
            logger.info("Redis cache initialized", 
                       redis_url=redis_url.split('@')[0] + '@***')  # Hide credentials
            
        except redis.ConnectionError as e:
            logger.error("Redis connection failed", error=str(e))
            logger.warning("Continuing without Redis cache")
        except Exception as e:
            logger.error("Redis initialization failed", error=str(e))
            logger.warning("Continuing without Redis cache")
    
    # === Audio Fingerprint Redis helpers (informational cache only; similarity uses PostgreSQL blobs) ===

    def cache_fingerprint(self, fp_hash: str, media_id: str, offset: float = 0.0, ttl: int = 86400):
        """Legacy SET helper — oracle similarity does not read this for authoritative matching."""
        if not self.enabled:
            return False
        
        try:
            # Use Redis Set to store all media_ids for a fingerprint hash
            key = f"fp:{fp_hash}"
            value = json.dumps({"media_id": media_id, "offset": offset})
            
            # Add to set (allows multiple matches per fingerprint)
            self.client.sadd(key, value)
            self.client.expire(key, ttl)
            
            logger.debug("Fingerprint cached", fp_hash=fp_hash, media_id=media_id)
            return True
            
        except Exception as e:
            logger.warning("Failed to cache fingerprint", error=str(e))
            return False
    
    def get_fingerprint_matches(self, fp_hash: str) -> List[tuple]:
        """Legacy read — prefer PostgreSQL search_fingerprint_rows for similarity candidates."""
        if not self.enabled:
            return None
        
        try:
            key = f"fp:{fp_hash}"
            cached_data = self.client.smembers(key)
            
            if not cached_data:
                return None
            
            # Parse JSON values back to tuples
            matches = []
            for item in cached_data:
                data = json.loads(item)
                matches.append((data["media_id"], data["offset"]))
            
            logger.debug("Fingerprint cache hit", fp_hash=fp_hash, matches=len(matches))
            return matches
            
        except Exception as e:
            logger.warning("Failed to get cached fingerprint", error=str(e))
            return None
    
    # === File Hash Deduplication ===
    
    def cache_file_hash(self, file_hash: str, media_id: str, ttl: int = 2592000):
        """Cache file hash for deduplication (30 days TTL)"""
        if not self.enabled:
            return False
        
        try:
            # Store in sorted set with timestamp as score (for TTL management)
            import time
            key = "file_hashes"
            self.client.zadd(key, {f"{file_hash}:{media_id}": time.time()})
            
            logger.debug("File hash cached", file_hash=file_hash[:16], media_id=media_id)
            return True
            
        except Exception as e:
            logger.warning("Failed to cache file hash", error=str(e))
            return False
    
    def check_file_hash_exists(self, file_hash: str) -> Optional[str]:
        """Check if file hash exists (fast duplicate detection)"""
        if not self.enabled:
            return None
        
        try:
            key = "file_hashes"
            # Search for any entry starting with this hash
            all_hashes = self.client.zrange(key, 0, -1)
            
            for entry in all_hashes:
                if entry.startswith(file_hash + ":"):
                    media_id = entry.split(":", 1)[1]
                    logger.debug("File hash cache hit", file_hash=file_hash[:16], media_id=media_id)
                    return media_id
            
            return None
            
        except Exception as e:
            logger.warning("Failed to check cached file hash", error=str(e))
            return None
    
    # === Embedding Search Result Caching ===
    
    def cache_embedding_results(self, file_hash: str, media_type: str, results: List[Dict], ttl: int = 3600):
        """Cache embedding search results (1 hour TTL)"""
        if not self.enabled:
            return False
        
        try:
            key = f"embedding:{media_type}:{file_hash}"
            value = json.dumps(results)
            self.client.setex(key, ttl, value)
            
            logger.debug("Embedding results cached", file_hash=file_hash[:16], results_count=len(results))
            return True
            
        except Exception as e:
            logger.warning("Failed to cache embedding results", error=str(e))
            return False
    
    def get_cached_embedding_results(self, file_hash: str, media_type: str) -> Optional[List[Dict]]:
        """Get cached embedding search results"""
        if not self.enabled:
            return None
        
        try:
            key = f"embedding:{media_type}:{file_hash}"
            cached_data = self.client.get(key)
            
            if not cached_data:
                return None
            
            results = json.loads(cached_data)
            logger.debug("Embedding cache hit", file_hash=file_hash[:16], results_count=len(results))
            return results
            
        except Exception as e:
            logger.warning("Failed to get cached embedding results", error=str(e))
            return None
    
    # === Rate Limiting ===
    
    def check_rate_limit(self, identifier: str, limit: int = 100, window: int = 3600) -> tuple[bool, int]:
        """
        Check rate limit for an identifier (IP, user_id, etc.)
        
        Args:
            identifier: Unique identifier (IP address, user_id, etc.)
            limit: Maximum requests allowed in window
            window: Time window in seconds (default: 1 hour)
        
        Returns:
            (allowed: bool, current_count: int)
        """
        if not self.enabled:
            return True, 0  # No rate limiting if Redis disabled
        
        try:
            key = f"ratelimit:{identifier}"
            
            # Increment counter
            current = self.client.incr(key)
            
            # Set expiry on first request
            if current == 1:
                self.client.expire(key, window)
            
            allowed = current <= limit
            
            if not allowed:
                logger.warning("Rate limit exceeded", 
                             identifier=identifier, 
                             current=current, 
                             limit=limit)
            
            return allowed, current
            
        except Exception as e:
            logger.warning("Failed to check rate limit", error=str(e))
            return True, 0  # Allow on error
    
    def reset_rate_limit(self, identifier: str):
        """Reset rate limit for an identifier (admin action)"""
        if not self.enabled:
            return False
        
        try:
            key = f"ratelimit:{identifier}"
            self.client.delete(key)
            logger.info("Rate limit reset", identifier=identifier)
            return True
        except Exception as e:
            logger.warning("Failed to reset rate limit", error=str(e))
            return False
    
    # === General Caching ===
    
    def get(self, key: str) -> Optional[str]:
        """Generic get operation"""
        if not self.enabled:
            return None
        try:
            return self.client.get(key)
        except Exception as e:
            logger.warning("Redis GET failed", key=key, error=str(e))
            return None
    
    def set(self, key: str, value: str, ttl: Optional[int] = None) -> bool:
        """Generic set operation"""
        if not self.enabled:
            return False
        try:
            if ttl:
                self.client.setex(key, ttl, value)
            else:
                self.client.set(key, value)
            return True
        except Exception as e:
            logger.warning("Redis SET failed", key=key, error=str(e))
            return False
    
    def delete(self, key: str) -> bool:
        """Delete a key"""
        if not self.enabled:
            return False
        try:
            self.client.delete(key)
            return True
        except Exception as e:
            logger.warning("Redis DELETE failed", key=key, error=str(e))
            return False
    
    # === Health Check ===
    
    def health_check(self) -> Dict[str, Any]:
        """Check Redis health"""
        if not self.enabled:
            return {"available": False, "error": "not_configured"}
        
        try:
            # Ping test
            latency_start = __import__('time').time()
            self.client.ping()
            latency = (__import__('time').time() - latency_start) * 1000  # ms
            
            # Get stats
            info = self.client.info()
            
            return {
                "available": True,
                "latency_ms": round(latency, 2),
                "connected_clients": info.get("connected_clients", 0),
                "used_memory_human": info.get("used_memory_human", "unknown"),
                "uptime_seconds": info.get("uptime_in_seconds", 0),
            }
            
        except Exception as e:
            return {"available": False, "error": str(e)}
    
    def clear_all_cache(self):
        """Clear all cached data (admin operation)"""
        if not self.enabled:
            return False
        
        try:
            # Only delete our specific patterns
            patterns = ["fp:*", "embedding:*", "file_hashes", "ratelimit:*"]
            for pattern in patterns:
                keys = self.client.keys(pattern)
                if keys:
                    self.client.delete(*keys)
            
            logger.info("Cache cleared", patterns=patterns)
            return True
            
        except Exception as e:
            logger.error("Failed to clear cache", error=str(e))
            return False


# Global Redis cache instance
_redis_cache: Optional[RedisCache] = None

def get_redis() -> RedisCache:
    """Get the global Redis cache instance"""
    global _redis_cache
    if _redis_cache is None:
        _redis_cache = RedisCache()
    return _redis_cache

