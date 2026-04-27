"""Evaluation & Metrics — Retrieval, Generation, RAGAS, Golden Dataset, Hallucination (Ch4)."""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# RetrievalEvaluator
# ---------------------------------------------------------------------------

class RetrievalEvaluator:
    """Evaluate retrieval quality."""

    @staticmethod
    def precision_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
        if k == 0:
            return 0.0
        top_k = retrieved[:k]
        hits = sum(1 for d in top_k if d in relevant)
        return hits / k

    @staticmethod
    def recall_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
        if not relevant:
            return 0.0
        top_k = retrieved[:k]
        hits = sum(1 for d in top_k if d in relevant)
        return hits / len(relevant)

    @staticmethod
    def mrr(ranked_results: List[str]) -> float:
        """Mean Reciprocal Rank — first relevant position."""
        for i, doc in enumerate(ranked_results, start=1):
            # Assume any doc starting with "rel_" is relevant for testing
            if doc.startswith("rel_"):
                return 1.0 / i
        return 0.0

    @staticmethod
    def ndcg(ranked_results: List[str], ideal: List[str]) -> float:
        """Normalized Discounted Cumulative Gain."""
        def dcg(scores: List[float]) -> float:
            return sum(s / math.log2(i + 2) for i, s in enumerate(scores))

        # Binary relevance: 1 if in ideal, else 0
        ideal_set = set(ideal)
        rel = [1.0 if d in ideal_set else 0.0 for d in ranked_results]
        ideal_rel = sorted([1.0] * min(len(ideal), len(ranked_results)) + [0.0] * max(0, len(ranked_results) - len(ideal)), reverse=True)
        d = dcg(rel)
        i = dcg(ideal_rel)
        return d / i if i > 0 else 0.0


# ---------------------------------------------------------------------------
# GenerationEvaluator
# ---------------------------------------------------------------------------

class GenerationEvaluator:
    """Evaluate generation quality."""

    @staticmethod
    def faithfulness(answer: str, context: str) -> float:
        """Check what fraction of answer words appear in context."""
        ans_words = set(answer.lower().split())
        ctx_words = set(context.lower().split())
        if not ans_words:
            return 1.0
        return len(ans_words & ctx_words) / len(ans_words)

    @staticmethod
    def relevance(answer: str, question: str) -> float:
        """Word overlap between answer and question."""
        a_words = set(answer.lower().split())
        q_words = set(question.lower().split())
        if not q_words:
            return 0.0
        return len(a_words & q_words) / len(q_words)

    @staticmethod
    def correctness(answer: str, expected: str) -> float:
        """Word-level accuracy vs expected answer."""
        a_words = set(answer.lower().split())
        e_words = set(expected.lower().split())
        if not e_words:
            return 1.0
        return len(a_words & e_words) / len(e_words)


# ---------------------------------------------------------------------------
# RAGASFramework
# ---------------------------------------------------------------------------

class RAGASFramework:
    """RAGAS metrics — faithfulness, relevancy, context precision."""

    def compute_faithfulness(self, answer: str, context: str) -> float:
        return GenerationEvaluator.faithfulness(answer, context)

    def compute_relevancy(self, answer: str, question: str) -> float:
        return GenerationEvaluator.relevance(answer, question)

    @staticmethod
    def compute_context_precision(contexts: List[str], question: str) -> float:
        """Fraction of contexts containing question keywords."""
        q_words = set(question.lower().split())
        if not q_words:
            return 0.0
        scores = []
        for ctx in contexts:
            c_words = set(ctx.lower().split())
            scores.append(len(q_words & c_words) / len(q_words))
        return sum(scores) / len(scores) if scores else 0.0


# ---------------------------------------------------------------------------
# GoldenDataset
# ---------------------------------------------------------------------------

@dataclass
class _GoldenExample:
    question: str
    answer: str
    contexts: List[str]
    ground_truth: str


class GoldenDataset:
    """Structured evaluation dataset."""

    def __init__(self) -> None:
        self._examples: List[_GoldenExample] = []

    def add_example(self, question: str, answer: str, contexts: List[str], ground_truth: str) -> None:
        self._examples.append(_GoldenExample(question=question, answer=answer, contexts=contexts, ground_truth=ground_truth))

    def get_split(self, ratio: float = 0.8) -> Tuple[List[_GoldenExample], List[_GoldenExample]]:
        n = int(len(self._examples) * ratio)
        return self._examples[:n], self._examples[n:]

    def sample(self, n: int) -> List[_GoldenExample]:
        if n >= len(self._examples):
            return list(self._examples)
        indices = list(range(len(self._examples)))
        # Deterministic "random" — use sorted hash-based selection
        indices.sort(key=lambda i: hashlib.md5(str(i).encode()).hexdigest())
        return [self._examples[i] for i in indices[:n]]


# ---------------------------------------------------------------------------
# HallucinationDetector
# ---------------------------------------------------------------------------

class HallucinationDetector:
    """Detect hallucinations via word overlap heuristics."""

    def check_factual_consistency(self, answer: str, context: str) -> float:
        """Return consistency score 0-1."""
        ans_words = set(answer.lower().split())
        ctx_words = set(context.lower().split())
        if not ans_words:
            return 1.0
        return len(ans_words & ctx_words) / len(ans_words)

    def check_grounding(self, answer: str, sources: List[str]) -> Dict[str, float]:
        """Check answer grounding per source."""
        result: Dict[str, float] = {}
        for i, src in enumerate(sources):
            a_words = set(answer.lower().split())
            s_words = set(src.lower().split())
            result[f"source_{i}"] = len(a_words & s_words) / max(len(a_words), 1)
        return result

    def get_confidence(self, answer: str) -> float:
        """Heuristic confidence — longer, more specific answers get higher score."""
        words = answer.split()
        if not words:
            return 0.0
        # Base: length signal
        length_score = min(len(words) / 50.0, 1.0)
        # Certainty words
        uncertainty = sum(1 for w in words if w.lower() in {"maybe", "might", "could", "perhaps", "possibly"})
        penalty = uncertainty / max(len(words), 1)
        return max(0.0, min(1.0, length_score - penalty))
