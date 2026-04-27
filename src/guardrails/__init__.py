"""Hallucination & Reliability — Citations, Confidence, Grounding, Refusal (Ch5)."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# CitationEngine
# ---------------------------------------------------------------------------

@dataclass
class _Claim:
    text: str
    source_idx: Optional[int] = None
    verified: bool = False


class CitationEngine:
    """Enforce citation-backed answers."""

    def extract_claims(self, answer: str) -> List[str]:
        """Split answer into sentences (claims)."""
        sentences = re.split(r'(?<=[.!?])\s+', answer.strip())
        return [s for s in sentences if s]

    def match_claims_to_sources(self, claims: List[str], sources: List[str]) -> List[_Claim]:
        """Match each claim to best source by word overlap."""
        matched: List[_Claim] = []
        for claim in claims:
            c_words = set(claim.lower().split())
            best_idx: Optional[int] = None
            best_overlap = 0
            for i, src in enumerate(sources):
                s_words = set(src.lower().split())
                overlap = len(c_words & s_words)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_idx = i
            matched.append(_Claim(text=claim, source_idx=best_idx, verified=best_overlap > 0))
        return matched

    def generate_citations(self, answer: str, sources: List[str]) -> str:
        """Annotate answer with [Source N] citations."""
        claims = self.extract_claims(answer)
        matched = self.match_claims_to_sources(claims, sources)
        cited: List[str] = []
        for claim in matched:
            if claim.source_idx is not None:
                cited.append(f"{claim.text} [Source {claim.source_idx + 1}]")
            else:
                cited.append(claim.text)
        return " ".join(cited)

    def verify_citations(self, answer: str) -> List[bool]:
        """Check which sentences have citations."""
        sentences = re.split(r'(?<=[.!?])\s+', answer.strip())
        return [bool(re.search(r'\[Source \d+\]', s)) for s in sentences if s]


# ---------------------------------------------------------------------------
# ConfidenceScorer
# ---------------------------------------------------------------------------

class ConfidenceScorer:
    """Score answer confidence and decide on refusals."""

    def __init__(self) -> None:
        self._refusal_threshold: float = 0.3

    def score(self, answer: str, context: str) -> float:
        """Confidence = fraction of answer words backed by context."""
        a_words = set(answer.lower().split())
        c_words = set(context.lower().split())
        if not a_words:
            return 0.0
        return len(a_words & c_words) / len(a_words)

    def get_low_confidence_spots(self, answer: str) -> List[str]:
        """Return sentences with uncertainty markers."""
        sentences = re.split(r'(?<=[.!?])\s+', answer)
        uncertainty = {"maybe", "might", "could", "perhaps", "possibly", "i think", "not sure", "uncertain"}
        return [s for s in sentences if any(w in s.lower() for w in uncertainty)]

    def set_refusal_threshold(self, threshold: float) -> None:
        self._refusal_threshold = threshold

    def should_refuse(self, confidence_score: float) -> bool:
        return confidence_score < self._refusal_threshold


# ---------------------------------------------------------------------------
# GroundingChecker
# ---------------------------------------------------------------------------

class GroundingChecker:
    """Ensure generated answers are grounded in context."""

    @staticmethod
    def check_grounded(answer: str, context: str) -> Tuple[bool, float]:
        """Return (is_grounded, grounding_score)."""
        a_words = set(answer.lower().split())
        c_words = set(context.lower().split())
        if not a_words:
            return True, 1.0
        score = len(a_words & c_words) / len(a_words)
        return score >= 0.5, score

    @staticmethod
    def get_ungrounded_claims(answer: str) -> List[str]:
        """Find sentences with no factual basis markers."""
        sentences = re.split(r'(?<=[.!?])\s+', answer)
        factual_markers = {"data shows", "research indicates", "according to", "study", "evidence", "reported"}
        return [s for s in sentences if not any(m in s.lower() for m in factual_markers)]

    @staticmethod
    def suggest_corrections(answer: str, context: str) -> List[str]:
        """Suggest replacing ungrounded claims with context-backed text."""
        a_words = set(answer.lower().split())
        c_words = set(context.lower().split())
        ungrounded = a_words - c_words
        if not ungrounded:
            return ["Answer appears well-grounded."]
        return [f"Consider revising terms not found in context: {', '.join(list(ungrounded)[:5])}"]


# ---------------------------------------------------------------------------
# RefusalEngine
# ---------------------------------------------------------------------------

@dataclass
class _RefusalLog:
    query: str
    reason: str
    timestamp: float


class RefusalEngine:
    """Intelligent answer refusal."""

    def __init__(self) -> None:
        self._log: List[_RefusalLog] = []
        self._refusal_keywords = {"harmful", "illegal", "dangerous", "weapon", "exploit"}

    def should_refuse(self, query: str, confidence: float) -> bool:
        q_lower = query.lower()
        if any(kw in q_lower for kw in self._refusal_keywords):
            return True
        if confidence < 0.2:
            return True
        return False

    def generate_refusal(self, query: str) -> str:
        q_lower = query.lower()
        if any(kw in q_lower for kw in self._refusal_keywords):
            return "I cannot assist with that request as it may involve harmful or illegal content."
        return "I don't have enough reliable information to answer that question confidently."

    def log_refusal(self, query: str, reason: str) -> None:
        import time
        self._log.append(_RefusalLog(query=query, reason=reason, timestamp=time.time()))
