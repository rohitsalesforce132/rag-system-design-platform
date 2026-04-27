"""Performance & Scaling — ANN, Index Management, Latency, Caching, Cost (Ch6)."""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# ANNSearcher
# ---------------------------------------------------------------------------

class ANNSearcher:
    """Approximate nearest neighbor search (simulated HNSW)."""

    def __init__(self) -> None:
        self._index: Dict[str, List[float]] = {}
        self._ef_search: int = 50

    def build_index(self, embeddings: Dict[str, List[float]], method: str = "hnsw") -> int:
        self._index = dict(embeddings)
        return len(self._index)

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Tuple[str, float]]:
        """Brute-force cosine search simulating ANN."""
        scored = []
        for did, emb in self._index.items():
            dot = sum(a * b for a, b in zip(query_embedding, emb))
            na = math.sqrt(sum(a * a for a in query_embedding)) or 1e-9
            nb = math.sqrt(sum(b * b for b in emb)) or 1e-9
            scored.append((did, dot / (na * nb)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def set_ef_search(self, ef: int) -> None:
        self._ef_search = ef


# ---------------------------------------------------------------------------
# IndexManager
# ---------------------------------------------------------------------------

@dataclass
class _IndexInfo:
    name: str
    config: Dict[str, Any]
    doc_count: int = 0
    created_at: float = field(default_factory=time.time)


class IndexManager:
    """Manage multiple search indices."""

    def __init__(self) -> None:
        self._indices: Dict[str, _IndexInfo] = {}

    def create_index(self, name: str, config: Optional[Dict[str, Any]] = None) -> bool:
        if name in self._indices:
            return False
        self._indices[name] = _IndexInfo(name=name, config=config or {})
        return True

    def rebuild_index(self, name: str) -> bool:
        info = self._indices.get(name)
        if info is None:
            return False
        self._indices[name] = _IndexInfo(name=name, config=info.config)
        return True

    def get_index_stats(self, name: str) -> Optional[Dict[str, Any]]:
        info = self._indices.get(name)
        if info is None:
            return None
        return {"name": info.name, "doc_count": info.doc_count, "config": info.config}

    def optimize_index(self, name: str) -> bool:
        info = self._indices.get(name)
        if info is None:
            return False
        info.config["optimized"] = True
        return True


# ---------------------------------------------------------------------------
# LatencyOptimizer
# ---------------------------------------------------------------------------

@dataclass
class _StepProfile:
    name: str
    duration_ms: float


class LatencyOptimizer:
    """Profile and optimize RAG pipeline latency."""

    def __init__(self) -> None:
        self._profiles: Dict[str, List[_StepProfile]] = {}
        self._current: Optional[str] = None

    def profile_pipeline(self, pipeline: Dict[str, float]) -> str:
        """Accept {step_name: duration_ms} mapping."""
        pid = hashlib.md5(str(sorted(pipeline.items())).encode()).hexdigest()[:8]
        self._profiles[pid] = [_StepProfile(name=k, duration_ms=v) for k, v in pipeline.items()]
        self._current = pid
        return pid

    def get_bottlenecks(self) -> List[Tuple[str, float]]:
        if self._current is None:
            return []
        steps = self._profiles.get(self._current, [])
        return sorted([(s.name, s.duration_ms) for s in steps], key=lambda x: x[1], reverse=True)

    def suggest_optimizations(self) -> List[str]:
        bottlenecks = self.get_bottlenecks()
        if not bottlenecks:
            return ["No bottlenecks detected."]
        suggestions = []
        for name, dur in bottlenecks[:3]:
            if "embedding" in name.lower():
                suggestions.append(f"Cache embeddings for '{name}' to reduce {dur:.0f}ms")
            elif "retrieval" in name.lower():
                suggestions.append(f"Use ANN index for '{name}' to reduce {dur:.0f}ms")
            else:
                suggestions.append(f"Consider batching for '{name}' ({dur:.0f}ms)")
        return suggestions

    def estimate_p99_latency(self) -> float:
        if self._current is None:
            return 0.0
        steps = self._profiles.get(self._current, [])
        total = sum(s.duration_ms for s in steps)
        # p99 ≈ 1.5x mean for simulation
        return total * 1.5


# ---------------------------------------------------------------------------
# EmbeddingCache
# ---------------------------------------------------------------------------

@dataclass
class _EmbeddingCacheEntry:
    embedding: List[float]
    expires_at: float


class EmbeddingCache:
    """Cache embeddings to reduce recomputation cost."""

    def __init__(self) -> None:
        self._cache: Dict[str, _EmbeddingCacheEntry] = {}
        self._hits: int = 0
        self._misses: int = 0

    def get(self, text_hash: str) -> Optional[List[float]]:
        entry = self._cache.get(text_hash)
        if entry is None:
            self._misses += 1
            return None
        if time.time() > entry.expires_at:
            del self._cache[text_hash]
            self._misses += 1
            return None
        self._hits += 1
        return entry.embedding

    def put(self, text_hash: str, embedding: List[float], ttl: float = 3600.0) -> None:
        self._cache[text_hash] = _EmbeddingCacheEntry(embedding=embedding, expires_at=time.time() + ttl)

    def get_hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# CostOptimizer
# ---------------------------------------------------------------------------

class CostOptimizer:
    """Estimate and reduce embedding/search costs."""

    def __init__(self) -> None:
        self._queries: int = 0
        self._tokens: int = 0
        self._cost_per_1k_tokens: float = 0.0001

    def estimate_query_cost(self, query: str) -> float:
        tokens = len(query.split())
        return tokens * self._cost_per_1k_tokens / 1000.0

    def suggest_batching(self, queries: List[str]) -> List[List[str]]:
        """Group queries into batches of ~10 for efficiency."""
        batches: List[List[str]] = []
        for i in range(0, len(queries), 10):
            batches.append(queries[i:i + 10])
        return batches if batches else [[]]

    def get_cost_report(self) -> Dict[str, Any]:
        return {
            "total_queries": self._queries,
            "total_tokens": self._tokens,
            "estimated_cost": self._tokens * self._cost_per_1k_tokens / 1000.0,
            "cost_per_1k_tokens": self._cost_per_1k_tokens,
        }
