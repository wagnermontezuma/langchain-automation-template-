import time
from typing import Any, Dict, Optional, Tuple
from collections import OrderedDict

from core.logger import get_logger

logger = get_logger(__name__)

class RoutingCache:
    """
    A simple in-memory cache with Time-To-Live (TTL) and max size for storing routing decisions
    or task classifications to speed up processing for repeated requests.
    """

    def __init__(self, default_ttl_seconds: int = 3600, max_size: int = 1000):
        """
        Initializes the RoutingCache.

        Args:
            default_ttl_seconds (int): Default time-to-live for cache entries in seconds.
            max_size (int): Maximum number of entries to store in the cache.
                            Uses OrderedDict to implement LRU eviction when full.
        """
        self.cache: OrderedDict[str, Tuple[Any, float]] = OrderedDict() # key -> (value, expiry_timestamp)
        self.default_ttl_seconds = default_ttl_seconds
        self.max_size = max_size
        logger.info(f"RoutingCache initialized with TTL: {default_ttl_seconds}s, Max size: {max_size}")

    def _is_expired(self, expiry_timestamp: float) -> bool:
        """Checks if a cache entry has expired."""
        return time.monotonic() > expiry_timestamp

    def get(self, key: str) -> Optional[Any]:
        """
        Retrieves an item from the cache if it exists and has not expired.
        Moves the accessed item to the end to mark it as recently used (for LRU).
        """
        if key not in self.cache:
            logger.debug(f"Cache miss for key: {key}")
            return None

        value, expiry_timestamp = self.cache[key]

        if self._is_expired(expiry_timestamp):
            logger.debug(f"Cache hit but expired for key: {key}. Removing.")
            del self.cache[key]
            return None

        # Move to end to mark as recently used for LRU
        self.cache.move_to_end(key)
        logger.debug(f"Cache hit for key: {key}")
        return value

    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None):
        """
        Adds an item to the cache with a specified or default TTL.
        If the cache is full, it removes the least recently used item.
        """
        if self.max_size <= 0: # Cache disabled
            return

        if len(self.cache) >= self.max_size and key not in self.cache: # Only evict if new key and cache is full
            # Evict the least recently used item (first item in OrderedDict)
            lru_key, _ = self.cache.popitem(last=False)
            logger.debug(f"Cache full. Evicted LRU item with key: {lru_key}")

        effective_ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl_seconds
        expiry_timestamp = time.monotonic() + effective_ttl

        self.cache[key] = (value, expiry_timestamp)
        # If the key already existed, move_to_end marks it as recently used.
        # If it's a new key, it's added at the end by default.
        self.cache.move_to_end(key)
        logger.debug(f"Cached item with key: {key}. TTL: {effective_ttl}s")


    def clear(self):
        """Clears all items from the cache."""
        self.cache.clear()
        logger.info("RoutingCache cleared.")

    def remove(self, key: str):
        """Removes a specific item from the cache if it exists."""
        if key in self.cache:
            del self.cache[key]
            logger.debug(f"Removed item with key: {key} from cache.")

    def __len__(self) -> int:
        """Returns the current number of items in the cache (including expired ones not yet cleaned)."""
        return len(self.cache)

    def clean_expired_entries(self):
        """
        Removes all expired entries from the cache.
        This can be called periodically if needed, though get() also handles individual expirations.
        """
        # Create a list of keys to delete to avoid modifying dict during iteration
        expired_keys = [key for key, (_, expiry_timestamp) in list(self.cache.items()) if self._is_expired(expiry_timestamp)]
        for key in expired_keys:
            try:
                del self.cache[key]
                logger.debug(f"Cleaned expired cache entry for key: {key}")
            except KeyError:
                 # Item might have been removed by another operation between collecting keys and deleting
                pass
        if expired_keys:
            logger.info(f"Cleaned {len(expired_keys)} expired entries from cache.")


if __name__ == '__main__':
    cache = RoutingCache(default_ttl_seconds=2, max_size=3)

    print("--- Cache Test ---")
    # Set items
    cache.set("key1", "value1") # k1
    cache.set("key2", "value2", ttl_seconds=5) # k1, k2
    print(f"Cache state: {list(cache.cache.keys())}")
    print(f"Cache size: {len(cache)}")

    # Get items
    print(f"Get key1: {cache.get('key1')}") # k2, k1
    print(f"Cache state: {list(cache.cache.keys())}")
    print(f"Get key2: {cache.get('key2')}") # k1, k2
    print(f"Cache state: {list(cache.cache.keys())}")
    print(f"Get key3 (miss): {cache.get('key3')}")

    # Test max_size eviction (LRU)
    cache.set("key3", "value3") # k1, k2, k3
    print(f"Cache state after adding key3: {list(cache.cache.keys())}")
    print(f"Cache size: {len(cache)}")

    # Current state: k1, k2, k3 (k1 is LRU based on last access due to set, k2 was next accessed by set)
    # No, get() also moves to end.
    # Trace:
    # set(k1) -> cache: [k1]
    # set(k2) -> cache: [k1, k2]
    # get(k1) -> cache: [k2, k1] (k1 moved to end)
    # get(k2) -> cache: [k1, k2] (k2 moved to end)
    # set(k3) -> cache: [k1, k2, k3] (k3 added to end)
    # At this point, k1 is LRU.
    cache.set("key4", "value4") # k1 evicted. cache: [k2, k3, k4]
    print(f"Cache state after adding key4 (eviction): {list(cache.cache.keys())}")
    print(f"Cache size: {len(cache)}")
    print(f"Get key1 after eviction: {cache.get('key1')}") # Expected: None
    print(f"Get key2: {cache.get('key2')}") # k3, k4, k2
    print(f"Cache state: {list(cache.cache.keys())}")
    print(f"Get key3: {cache.get('key3')}") # k4, k2, k3
    print(f"Cache state: {list(cache.cache.keys())}")
    print(f"Get key4: {cache.get('key4')}") # k2, k3, k4
    print(f"Cache state: {list(cache.cache.keys())}")


    print("\n--- TTL Test ---")
    cache.set("ttl_key", "ttl_value", ttl_seconds=1) # k3, k4, k2 (if k2 was LRU from above), ttl_key. k3 is evicted.
    print(f"Cache state: {list(cache.cache.keys())}") # Should be [k4, k2, ttl_key]
    print(f"Get ttl_key (before expiry): {cache.get('ttl_key')}") # k4, k2, ttl_key (ttl_key moved to end)
    time.sleep(1.5)
    print(f"Get ttl_key (after expiry): {cache.get('ttl_key')}") # Expected: None. ttl_key removed. Cache: [k4, k2]

    print("\n--- Clean Expired Test ---")
    cache.set("exp1", "val_exp1", 1) # k4, k2, exp1
    cache.set("exp2", "val_exp2", 1) # k2, exp1, exp2 (k4 evicted)
    cache.set("valid1", "val_valid1", 10) # exp1, exp2, valid1 (k2 evicted)
    print(f"Cache state before sleep: {list(cache.cache.keys())}")
    print(f"Cache size before sleep: {len(cache)}")
    time.sleep(1.5) # exp1, exp2 are now expired
    cache.clean_expired_entries() # Should remove exp1, exp2
    print(f"Cache state after clean: {list(cache.cache.keys())}") # Expected: [valid1]
    print(f"Cache size after clean_expired_entries: {len(cache)}")
    print(f"Get exp1: {cache.get('exp1')}")
    print(f"Get valid1: {cache.get('valid1')}")

    cache.clear()
    print(f"Cache size after clear: {len(cache)}")
