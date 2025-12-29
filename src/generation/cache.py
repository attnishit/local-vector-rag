"""
Answer Caching

Author: Nishit Attrey

This module provides LRU (Least Recently Used) caching for generated answers.
Caching speeds up repeated queries and reduces LLM API calls.

Key Features:
- LRU cache with configurable size
- TTL (Time-To-Live) support for cache expiration
- Cache key generation from query + parameters
- Cache statistics (hits, misses, hit rate)
- Thread-safe operations

Example:
    >>> from src.generation.cache import AnswerCache
    >>>
    >>> cache = AnswerCache(max_size=100, ttl=3600)
    >>> cache.set("What is HNSW?", answer_data)
    >>> cached = cache.get("What is HNSW?")
    >>> print(cache.get_stats())
"""

import logging
import hashlib
import json
import time
from typing import Dict, Any, Optional, Tuple
from collections import OrderedDict
from threading import RLock

logger = logging.getLogger(__name__)


class AnswerCache:
    """
    LRU cache for generated answers with TTL support.

    Caches answers to avoid redundant LLM generation for repeated queries.
    Automatically evicts least recently used entries when cache is full.

    Example:
        >>> cache = AnswerCache(max_size=100, ttl=3600)
        >>> cache.set("What is vector search?", result_data)
        >>> cached_result = cache.get("What is vector search?")
        >>> if cached_result:
        ...     print("Cache hit!")
    """

    def __init__(self, max_size: int = 100, ttl: Optional[int] = 3600):
        """
        Initialize answer cache.

        Args:
            max_size: Maximum number of entries to cache
            ttl: Time-to-live in seconds (None = no expiration)

        Example:
            >>> cache = AnswerCache(max_size=50, ttl=1800)  # 30 minutes
        """
        self.max_size = max_size
        self.ttl = ttl

        # OrderedDict for LRU behavior
        self._cache: OrderedDict[str, Tuple[Any, float]] = OrderedDict()

        # Thread safety
        self._lock = RLock()

        # Statistics
        self._hits = 0
        self._misses = 0

        logger.debug(f"Initialized AnswerCache: max_size={max_size}, ttl={ttl}")

    def _generate_key(
        self,
        query: str,
        collection: str = "",
        k: int = 5,
        template: str = "qa",
        model: Optional[str] = None,
        temperature: float = 0.7,
    ) -> str:
        """
        Generate cache key from query and parameters.

        Creates a unique hash based on all parameters that affect the answer.

        Args:
            query: User query
            collection: Collection name
            k: Number of results
            template: Prompt template
            model: LLM model
            temperature: Generation temperature

        Returns:
            Cache key (SHA256 hash)
        """
        # Create canonical representation
        key_data = {
            "query": query.strip().lower(),  # Normalize query
            "collection": collection,
            "k": k,
            "template": template,
            "model": model or "default",
            "temperature": round(temperature, 2),  # Round to avoid float precision issues
        }

        # Generate hash
        key_string = json.dumps(key_data, sort_keys=True)
        key_hash = hashlib.sha256(key_string.encode()).hexdigest()

        return key_hash

    def get(
        self,
        query: str,
        collection: str = "",
        k: int = 5,
        template: str = "qa",
        model: Optional[str] = None,
        temperature: float = 0.7,
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached answer if available and not expired.

        Args:
            query: User query
            collection: Collection name
            k: Number of results
            template: Prompt template
            model: LLM model
            temperature: Generation temperature

        Returns:
            Cached answer data or None if not found/expired

        Example:
            >>> result = cache.get("What is HNSW?", collection="docs", k=5)
            >>> if result:
            ...     print("Using cached answer")
        """
        key = self._generate_key(query, collection, k, template, model, temperature)

        with self._lock:
            if key not in self._cache:
                self._misses += 1
                logger.debug(f"Cache miss for query: {query[:50]}...")
                return None

            # Get entry
            value, timestamp = self._cache[key]

            # Check TTL
            if self.ttl is not None:
                age = time.time() - timestamp
                if age > self.ttl:
                    # Expired - remove from cache
                    del self._cache[key]
                    self._misses += 1
                    logger.debug(
                        f"Cache expired for query: {query[:50]}... (age: {age:.0f}s)"
                    )
                    return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)

            self._hits += 1
            logger.debug(f"Cache hit for query: {query[:50]}...")
            return value

    def set(
        self,
        query: str,
        value: Dict[str, Any],
        collection: str = "",
        k: int = 5,
        template: str = "qa",
        model: Optional[str] = None,
        temperature: float = 0.7,
    ) -> None:
        """
        Store answer in cache.

        Args:
            query: User query
            value: Answer data to cache
            collection: Collection name
            k: Number of results
            template: Prompt template
            model: LLM model
            temperature: Generation temperature

        Example:
            >>> cache.set("What is HNSW?", answer_result, collection="docs")
        """
        key = self._generate_key(query, collection, k, template, model, temperature)

        with self._lock:
            # Add/update entry with current timestamp
            self._cache[key] = (value, time.time())

            # Move to end (most recently used)
            self._cache.move_to_end(key)

            # Evict oldest if over size limit
            while len(self._cache) > self.max_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                logger.debug(f"Evicted oldest cache entry: {oldest_key[:16]}...")

        logger.debug(f"Cached answer for query: {query[:50]}... (total: {len(self._cache)})")

    def clear(self) -> None:
        """
        Clear all cached entries.

        Example:
            >>> cache.clear()
            >>> assert cache.size() == 0
        """
        with self._lock:
            self._cache.clear()
            logger.info("Cleared answer cache")

    def size(self) -> int:
        """
        Get current cache size.

        Returns:
            Number of cached entries

        Example:
            >>> print(f"Cache contains {cache.size()} entries")
        """
        with self._lock:
            return len(self._cache)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache stats (hits, misses, size, hit_rate)

        Example:
            >>> stats = cache.get_stats()
            >>> print(f"Hit rate: {stats['hit_rate']:.2%}")
        """
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0.0

            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "total_requests": total_requests,
                "hit_rate": hit_rate,
                "ttl": self.ttl,
            }

    def reset_stats(self) -> None:
        """
        Reset cache statistics (hits/misses).

        Example:
            >>> cache.reset_stats()
        """
        with self._lock:
            self._hits = 0
            self._misses = 0
            logger.debug("Reset cache statistics")

    def invalidate(
        self,
        query: str,
        collection: str = "",
        k: int = 5,
        template: str = "qa",
        model: Optional[str] = None,
        temperature: float = 0.7,
    ) -> bool:
        """
        Invalidate specific cache entry.

        Args:
            query: User query
            collection: Collection name
            k: Number of results
            template: Prompt template
            model: LLM model
            temperature: Generation temperature

        Returns:
            True if entry was removed, False if not found

        Example:
            >>> cache.invalidate("What is HNSW?", collection="docs")
        """
        key = self._generate_key(query, collection, k, template, model, temperature)

        with self._lock:
            if key in self._cache:
                del self._cache[key]
                logger.debug(f"Invalidated cache entry for query: {query[:50]}...")
                return True
            return False

    def invalidate_collection(self, collection: str) -> int:
        """
        Invalidate all entries for a specific collection.

        Useful when a collection is updated and cached answers may be stale.

        Args:
            collection: Collection name to invalidate

        Returns:
            Number of entries removed

        Example:
            >>> removed = cache.invalidate_collection("my_docs")
            >>> print(f"Removed {removed} cached entries")
        """
        with self._lock:
            # Find all keys for this collection
            # This requires iterating through all entries since keys are hashed
            # For better performance, could maintain a collection -> keys index
            count = 0
            keys_to_remove = []

            for key, (value, _) in self._cache.items():
                # Check if this entry is for the target collection
                # This is approximate - relies on value structure
                if isinstance(value, dict) and value.get("metadata", {}).get("collection") == collection:
                    keys_to_remove.append(key)

            for key in keys_to_remove:
                del self._cache[key]
                count += 1

            logger.info(f"Invalidated {count} cache entries for collection: {collection}")
            return count

    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"AnswerCache(size={stats['size']}/{stats['max_size']}, "
            f"hit_rate={stats['hit_rate']:.2%}, ttl={self.ttl})"
        )


# Global cache instance (can be configured at module level)
_global_cache: Optional[AnswerCache] = None


def get_global_cache(
    max_size: int = 100,
    ttl: Optional[int] = 3600,
    create_if_missing: bool = True,
) -> Optional[AnswerCache]:
    """
    Get or create global answer cache instance.

    Args:
        max_size: Maximum cache size
        ttl: Time-to-live in seconds
        create_if_missing: Create cache if it doesn't exist

    Returns:
        Global AnswerCache instance or None

    Example:
        >>> cache = get_global_cache()
        >>> cache.set("query", result)
    """
    global _global_cache

    if _global_cache is None and create_if_missing:
        _global_cache = AnswerCache(max_size=max_size, ttl=ttl)
        logger.info("Created global answer cache")

    return _global_cache


def clear_global_cache() -> None:
    """
    Clear the global cache instance.

    Example:
        >>> clear_global_cache()
    """
    global _global_cache

    if _global_cache is not None:
        _global_cache.clear()
        logger.info("Cleared global answer cache")
