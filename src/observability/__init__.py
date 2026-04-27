"""Observability & Monitoring — Tracing, Drift, Hallucination Monitor, Feedback, Metrics (Ch11)."""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# RAGTracer
# ---------------------------------------------------------------------------

@dataclass
class _TraceStep:
    step: str
    duration: float
    timestamp: float


@dataclass
class _Trace:
    trace_id: str
    query: str
    steps: List[_TraceStep] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0


class RAGTracer:
    """Distributed tracing for RAG pipelines."""

    def __init__(self) -> None:
        self._traces: Dict[str, _Trace] = {}

    def start_trace(self, query: str) -> str:
        tid = hashlib.md5(f"{query}{time.time()}".encode()).hexdigest()[:12]
        self._traces[tid] = _Trace(trace_id=tid, query=query, start_time=time.time())
        return tid

    def log_step(self, trace_id: str, step: str, duration: float) -> None:
        trace = self._traces.get(trace_id)
        if trace:
            trace.steps.append(_TraceStep(step=step, duration=duration, timestamp=time.time()))

    def end_trace(self, trace_id: str) -> None:
        trace = self._traces.get(trace_id)
        if trace:
            trace.end_time = time.time()

    def get_trace(self, trace_id: str) -> Optional[Dict[str, Any]]:
        trace = self._traces.get(trace_id)
        if trace is None:
            return None
        return {
            "trace_id": trace.trace_id,
            "query": trace.query,
            "steps": [{"step": s.step, "duration": s.duration} for s in trace.steps],
            "total_duration": trace.end_time - trace.start_time if trace.end_time else 0,
        }


# ---------------------------------------------------------------------------
# DriftDetector
# ---------------------------------------------------------------------------

class DriftDetector:
    """Detect data/model drift via embedding statistics."""

    def __init__(self) -> None:
        self._baseline_mean: float = 0.0
        self._baseline_std: float = 0.0

    def baseline(self, embeddings: Dict[str, List[float]]) -> Dict[str, float]:
        all_vals = [v for emb in embeddings.values() for v in emb]
        if not all_vals:
            self._baseline_mean = 0.0
            self._baseline_std = 0.0
        else:
            self._baseline_mean = sum(all_vals) / len(all_vals)
            var = sum((v - self._baseline_mean) ** 2 for v in all_vals) / len(all_vals)
            self._baseline_std = math.sqrt(var)
        return {"mean": self._baseline_mean, "std": self._baseline_std}

    def check_drift(self, current: Dict[str, List[float]]) -> Tuple[bool, float]:
        all_vals = [v for emb in current.values() for v in emb]
        if not all_vals:
            return False, 0.0
        current_mean = sum(all_vals) / len(all_vals)
        if self._baseline_std == 0:
            return False, 0.0
        z_score = abs(current_mean - self._baseline_mean) / self._baseline_std
        return z_score > 2.0, z_score

    def get_drift_report(self) -> Dict[str, Any]:
        return {"baseline_mean": self._baseline_mean, "baseline_std": self._baseline_std}


# ---------------------------------------------------------------------------
# HallucinationMonitor
# ---------------------------------------------------------------------------

@dataclass
class _HCheck:
    answer: str
    context: str
    score: float
    timestamp: float


class HallucinationMonitor:
    """Monitor hallucination rates in production."""

    def __init__(self) -> None:
        self._checks: List[_HCheck] = []
        self._threshold: float = 0.1

    def check_answer(self, answer: str, context: str) -> float:
        """Return hallucination score (lower = less hallucination)."""
        a_words = set(answer.lower().split())
        c_words = set(context.lower().split())
        if not a_words:
            return 0.0
        score = 1.0 - len(a_words & c_words) / len(a_words)
        self._checks.append(_HCheck(answer=answer, context=context, score=score, timestamp=time.time()))
        return score

    def get_hallucination_rate(self) -> float:
        if not self._checks:
            return 0.0
        return sum(c.score for c in self._checks) / len(self._checks)

    def alert_if_above(self, threshold: float) -> bool:
        rate = self.get_hallucination_rate()
        return rate > threshold


# ---------------------------------------------------------------------------
# FeedbackCollector
# ---------------------------------------------------------------------------

@dataclass
class _Feedback:
    query: str
    answer: str
    rating: int
    comment: str
    timestamp: float


class FeedbackCollector:
    """Collect and analyze user feedback."""

    def __init__(self) -> None:
        self._feedbacks: List[_Feedback] = []

    def collect(self, query: str, answer: str, rating: int, comment: str = "") -> None:
        self._feedbacks.append(_Feedback(query=query, answer=answer, rating=rating, comment=comment, timestamp=time.time()))

    def get_feedback_stats(self) -> Dict[str, Any]:
        if not self._feedbacks:
            return {"count": 0, "avg_rating": 0.0}
        ratings = [f.rating for f in self._feedbacks]
        return {"count": len(ratings), "avg_rating": sum(ratings) / len(ratings)}

    def get_low_rated(self, threshold: int = 3) -> List[Dict[str, Any]]:
        return [
            {"query": f.query, "answer": f.answer, "rating": f.rating}
            for f in self._feedbacks if f.rating <= threshold
        ]


# ---------------------------------------------------------------------------
# MetricsExporter
# ---------------------------------------------------------------------------

class MetricsExporter:
    """Export metrics in various formats."""

    def __init__(self) -> None:
        self._metrics: Dict[str, float] = {}

    def record(self, name: str, value: float) -> None:
        self._metrics[name] = value

    def export_prometheus(self) -> str:
        lines = []
        for name, value in sorted(self._metrics.items()):
            lines.append(f"{name} {value}")
        return "\n".join(lines)

    def export_json(self) -> Dict[str, float]:
        return dict(self._metrics)

    def get_summary(self) -> Dict[str, Any]:
        if not self._metrics:
            return {"count": 0}
        vals = list(self._metrics.values())
        return {
            "count": len(vals),
            "min": min(vals),
            "max": max(vals),
            "avg": sum(vals) / len(vals),
        }
