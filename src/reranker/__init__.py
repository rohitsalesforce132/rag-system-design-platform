"""Re-ranking Pipeline — Cross-encoder, LLM reranking, chained stages (Ch2)."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# ScoreNormalizer
# ---------------------------------------------------------------------------

class ScoreNormalizer:
    """Normalize scores across rerankers."""

    @staticmethod
    def min_max_normalize(scores: List[float]) -> List[float]:
        if not scores:
            return []
        lo, hi = min(scores), max(scores)
        rng = hi - lo
        if rng == 0:
            return [1.0] * len(scores)
        return [(s - lo) / rng for s in scores]

    @staticmethod
    def z_score_normalize(scores: List[float]) -> List[float]:
        if not scores:
            return []
        mean = sum(scores) / len(scores)
        var = sum((s - mean) ** 2 for s in scores) / len(scores)
        std = math.sqrt(var) if var > 0 else 1.0
        return [(s - mean) / std for s in scores]

    @staticmethod
    def percentile_normalize(scores: List[float]) -> List[float]:
        if not scores:
            return []
        sorted_s = sorted(scores)
        n = len(sorted_s)
        return [sorted_s.index(s) / max(n - 1, 1) for s in scores]


# ---------------------------------------------------------------------------
# CrossEncoderReranker
# ---------------------------------------------------------------------------

class CrossEncoderReranker:
    """Simulates cross-encoder scoring via deterministic hash."""

    @staticmethod
    def score_pair(query: str, doc: str) -> float:
        combined = f"{query}|||{doc}"
        h = hashlib.sha256(combined.encode()).hexdigest()
        return (int(h[:8], 16) % 1000) / 1000.0

    def rerank(self, query: str, documents: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
        scored = [(doc, self.score_pair(query, doc)) for doc in documents]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


# ---------------------------------------------------------------------------
# LLMReranker
# ---------------------------------------------------------------------------

class LLMReranker:
    """Simulates LLM-based re-ranking with relevance scoring."""

    @staticmethod
    def score_relevance(query: str, doc: str) -> float:
        """Score based on word overlap (deterministic proxy for LLM judgment)."""
        q_words = set(query.lower().split())
        d_words = set(doc.lower().split())
        if not q_words:
            return 0.0
        overlap = len(q_words & d_words)
        return overlap / len(q_words)

    def rerank(self, query: str, documents: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
        scored = [(doc, self.score_relevance(query, doc)) for doc in documents]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


# ---------------------------------------------------------------------------
# RerankingPipeline
# ---------------------------------------------------------------------------

@dataclass
class _PipelineStage:
    name: str
    reranker: Any
    top_k: int


class RerankingPipeline:
    """Chain multiple rerankers in sequence."""

    def __init__(self) -> None:
        self._stages: List[_PipelineStage] = []

    def add_stage(self, name: str, reranker: Any, top_k: int = 5) -> None:
        self._stages.append(_PipelineStage(name=name, reranker=reranker, top_k=top_k))

    def execute(self, query: str, documents: List[str]) -> List[Tuple[str, float]]:
        current_docs = list(documents)
        current_scores: Dict[str, float] = {}

        for stage in self._stages:
            results = stage.reranker.rerank(query, current_docs, top_k=stage.top_k)
            current_docs = [doc for doc, _ in results]
            current_scores = {doc: score for doc, score in results}

        return [(doc, current_scores.get(doc, 0.0)) for doc in current_docs]
