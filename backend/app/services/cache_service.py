"""
MediGenius — services/cache_service.py
Process-local TTL cache for low-risk deterministic node outputs.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import threading
import time
from typing import Any

from app.core.config import CACHE_ENABLED, CACHE_TTL_SECONDS


class CacheService:
    """Very small in-memory TTL cache with JSON-stable keys."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._items: dict[str, tuple[float, Any]] = {}

    def make_key(self, namespace: str, payload: dict[str, Any]) -> str:
        serialized = json.dumps(payload or {}, ensure_ascii=False, sort_keys=True, default=str)
        digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        return f"{namespace}:{digest}"

    def get(self, key: str) -> Any | None:
        if not CACHE_ENABLED:
            return None

        with self._lock:
            record = self._items.get(key)
            if not record:
                return None
            expires_at, value = record
            if expires_at < time.time():
                self._items.pop(key, None)
                return None
            return deepcopy(value)

    def set(self, key: str, value: Any, ttl_seconds: int = CACHE_TTL_SECONDS) -> None:
        if not CACHE_ENABLED:
            return
        expires_at = time.time() + max(1, ttl_seconds)
        with self._lock:
            self._items[key] = (expires_at, deepcopy(value))

    def clear(self) -> None:
        with self._lock:
            self._items.clear()


def record_cache_result(state: dict[str, Any], node_name: str, hit: bool) -> None:
    cache_stats = state.setdefault("cache_stats", {"hits": 0, "misses": 0})
    metric = state.setdefault("node_metrics", {}).setdefault(node_name, {})
    metric["cache_hit"] = bool(hit)
    if hit:
        cache_stats["hits"] = int(cache_stats.get("hits", 0)) + 1
    else:
        cache_stats["misses"] = int(cache_stats.get("misses", 0)) + 1


cache_service = CacheService()
