"""Caching mechanism for RAG responses.

This module provides caching functionality to improve response times
for repeated queries and reduce LLM API costs.
"""

import hashlib
import json
import logging
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class ResponseCache:
    """In-memory cache for RAG responses."""

    def __init__(
        self,
        max_size: int = 1000,
        ttl_seconds: int = 3600,  # 1 hour default
        enable_persistence: bool = False,
        cache_file: Optional[str] = None,
    ):
        """Initialize the response cache.

        Args:
            max_size: Maximum number of cached entries
            ttl_seconds: Time-to-live for cache entries in seconds
            enable_persistence: Whether to persist cache to disk
            cache_file: Path to cache file for persistence
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.enable_persistence = enable_persistence
        self.cache_file = cache_file or "data/cache/response_cache.json"

        self._cache: Dict[str, Dict] = {}
        self._access_times: Dict[str, float] = {}

        # Load persisted cache if enabled
        if self.enable_persistence:
            self._load_cache()

    def get(
        self, question: str, context_hash: str, user_role: str = "public"
    ) -> Optional[Dict]:
        """Get cached response for a query.

        Args:
            question: User's question
            context_hash: Hash of the context documents used
            user_role: User's role

        Returns:
            Cached response or None if not found/expired
        """
        cache_key = self._generate_cache_key(question, context_hash, user_role)

        if cache_key not in self._cache:
            return None

        # Check if entry is expired
        entry = self._cache[cache_key]
        if time.time() - entry["timestamp"] > self.ttl_seconds:
            del self._cache[cache_key]
            del self._access_times[cache_key]
            logger.debug(f"Cache entry expired for key: {cache_key[:16]}...")
            return None

        # Update access time
        self._access_times[cache_key] = time.time()

        logger.debug(f"Cache hit for key: {cache_key[:16]}...")
        return entry["response"]

    def set(
        self,
        question: str,
        context_hash: str,
        response: Dict,
        user_role: str = "public",
    ):
        """Cache a response.

        Args:
            question: User's question
            context_hash: Hash of the context documents
            response: Response to cache
            user_role: User's role
        """
        # Check if cache is full
        if len(self._cache) >= self.max_size:
            self._evict_lru()

        cache_key = self._generate_cache_key(question, context_hash, user_role)

        # Store response
        self._cache[cache_key] = {
            "response": response,
            "timestamp": time.time(),
            "question": question,
            "context_hash": context_hash,
            "user_role": user_role,
        }
        self._access_times[cache_key] = time.time()

        # Persist if enabled
        if self.enable_persistence:
            self._save_cache()

        logger.debug(f"Cached response for key: {cache_key[:16]}...")

    def _generate_cache_key(
        self, question: str, context_hash: str, user_role: str
    ) -> str:
        """Generate a cache key for the query.

        Args:
            question: User's question
            context_hash: Hash of context documents
            user_role: User's role

        Returns:
            Cache key string
        """
        # Normalize question
        normalized_question = question.lower().strip()

        # Create combined string
        combined = f"{normalized_question}|{context_hash}|{user_role}"

        # Generate hash
        return hashlib.sha256(combined.encode()).hexdigest()

    def _evict_lru(self):
        """Evict least recently used entry from cache."""
        if not self._access_times:
            return

        # Find LRU key
        lru_key = min(self._access_times.items(), key=lambda x: x[1])[0]

        # Remove from cache
        del self._cache[lru_key]
        del self._access_times[lru_key]

        logger.debug(f"Evicted LRU cache entry: {lru_key[:16]}...")

    def clear(self):
        """Clear all cached entries."""
        self._cache.clear()
        self._access_times.clear()
        logger.info("Cache cleared")

    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        # Count expired entries
        current_time = time.time()
        expired_count = sum(
            1
            for entry in self._cache.values()
            if current_time - entry["timestamp"] > self.ttl_seconds
        )

        return {
            "total_entries": len(self._cache),
            "max_size": self.max_size,
            "ttl_seconds": self.ttl_seconds,
            "expired_entries": expired_count,
            "utilization": len(self._cache) / self.max_size,
            "persistence_enabled": self.enable_persistence,
        }

    def _save_cache(self):
        """Save cache to disk."""
        try:
            import os

            # Ensure directory exists
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)

            # Prepare data for serialization
            cache_data = {
                "cache": self._cache,
                "access_times": self._access_times,
                "saved_at": time.time(),
            }

            # Write to file
            with open(self.cache_file, "w") as f:
                json.dump(cache_data, f, indent=2)

            logger.debug(f"Cache saved to {self.cache_file}")

        except Exception as e:
            logger.error(f"Failed to save cache: {e}")

    def _load_cache(self):
        """Load cache from disk."""
        try:
            with open(self.cache_file, "r") as f:
                cache_data = json.load(f)

            self._cache = cache_data.get("cache", {})
            self._access_times = cache_data.get("access_times", {})

            # Clean expired entries
            self._clean_expired()

            logger.debug(f"Cache loaded from {self.cache_file}")

        except FileNotFoundError:
            logger.debug("Cache file not found, starting with empty cache")
        except Exception as e:
            logger.error(f"Failed to load cache: {e}")

    def _clean_expired(self):
        """Remove expired entries from cache."""
        current_time = time.time()
        expired_keys = []

        for key, entry in self._cache.items():
            if current_time - entry["timestamp"] > self.ttl_seconds:
                expired_keys.append(key)

        for key in expired_keys:
            del self._cache[key]
            if key in self._access_times:
                del self._access_times[key]

        if expired_keys:
            logger.debug(f"Cleaned {len(expired_keys)} expired cache entries")

    def generate_context_hash(self, context_docs: list) -> str:
        """Generate hash for context documents.

        Args:
            context_docs: List of context documents

        Returns:
            Hash string
        """
        # Extract relevant info from documents
        context_info = []
        for doc in context_docs:
            context_info.append(
                {
                    "chunk_id": doc.get("chunk_id"),
                    "metadata": doc.get("metadata", {}),
                    # Don't include full content to avoid unnecessary cache misses
                    "content_preview": doc.get("content", "")[:100],
                }
            )

        # Create JSON and hash
        context_json = json.dumps(context_info, sort_keys=True)
        return hashlib.sha256(context_json.encode()).hexdigest()[:16]


class RedisCache(ResponseCache):
    """Redis-based cache for distributed deployments."""

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        key_prefix: str = "vyasa:",
        ttl_seconds: int = 3600,
    ):
        """Initialize Redis cache.

        Args:
            redis_url: Redis connection URL
            key_prefix: Prefix for cache keys
            ttl_seconds: Time-to-live for entries
        """
        super().__init__(max_size=0, ttl_seconds=ttl_seconds)  # Size managed by Redis

        try:
            import redis

            self.redis_client = redis.from_url(redis_url)
            self.key_prefix = key_prefix

            # Test connection
            self.redis_client.ping()
            logger.info("Connected to Redis cache")

        except ImportError:
            logger.error("redis package not installed. Install with: pip install redis")
            raise
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    def get(
        self, question: str, context_hash: str, user_role: str = "public"
    ) -> Optional[Dict]:
        """Get cached response from Redis."""
        cache_key = self.key_prefix + self._generate_cache_key(
            question, context_hash, user_role
        )

        try:
            data = self.redis_client.get(cache_key)
            if data:
                entry = json.loads(data)
                logger.debug(f"Redis cache hit for key: {cache_key[-16:]}...")
                return entry["response"]
        except Exception as e:
            logger.error(f"Redis get error: {e}")

        return None

    def set(
        self,
        question: str,
        context_hash: str,
        response: Dict,
        user_role: str = "public",
    ):
        """Cache response in Redis."""
        cache_key = self.key_prefix + self._generate_cache_key(
            question, context_hash, user_role
        )

        entry = {
            "response": response,
            "timestamp": time.time(),
            "question": question,
            "context_hash": context_hash,
            "user_role": user_role,
        }

        try:
            self.redis_client.setex(cache_key, self.ttl_seconds, json.dumps(entry))
            logger.debug(f"Cached response in Redis: {cache_key[-16:]}...")
        except Exception as e:
            logger.error(f"Redis set error: {e}")
