"""Hybrid Retrieval Engine — Dense, Sparse, Hybrid search with RRF fusion (Ch1-2)."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simple_hash(text: str) -> int:
    return int(hashlib.md5(text.encode()).hexdigest(), 16)


def _tokenize(text: str) -> List[str]:
    """Lowercase whitespace + punctuation tokenizer."""
    return re.findall(r"[a-z0-9]+", text.lower())


# ---------------------------------------------------------------------------
# EmbeddingEngine
# ---------------------------------------------------------------------------

class EmbeddingEngine:
    """Deterministic embedding simulation using character n-gram hashing."""

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    def embed(self, text: str) -> List[float]:
        """Generate a deterministic pseudo-embedding for *text*."""
        h = hashlib.sha512(text.encode()).digest()
        vec = []
        for i in range(self.dim):
            byte_pair = h[(i * 2) % len(h): (i * 2) % len(h) + 2]
            if len(byte_pair) < 2:
                byte_pair = h[:2]
            val = int.from_bytes(byte_pair, "little") / 65536.0 - 0.5
            vec.append(val)
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def batch_embed(self, texts: List[str]) -> List[List[float]]:
        return [self.embed(t) for t in texts]

    @staticmethod
    def cosine_similarity(a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a)) or 1e-9
        nb = math.sqrt(sum(x * x for x in b)) or 1e-9
        return dot / (na * nb)

    def similarity(self, a: List[float], b: List[float]) -> float:
        return self.cosine_similarity(a, b)

    @staticmethod
    def dot_product(a: List[float], b: List[float]) -> float:
        return sum(x * y for x, y in zip(a, b))


# ---------------------------------------------------------------------------
# DenseRetriever
# ---------------------------------------------------------------------------

class DenseRetriever:
    """Simulates dense vector search using cosine similarity."""

    def __init__(self, engine: Optional[EmbeddingEngine] = None) -> None:
        self.engine = engine or EmbeddingEngine()
        self._docs: Dict[str, List[float]] = {}

    def index(self, doc_id: str, embedding: List[float]) -> None:
        self._docs[doc_id] = embedding

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Tuple[str, float]]:
        scored = [
            (did, EmbeddingEngine.cosine_similarity(query_embedding, emb))
            for did, emb in self._docs.items()
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def delete(self, doc_id: str) -> bool:
        return self._docs.pop(doc_id, None) is not None


# ---------------------------------------------------------------------------
# SparseRetriever — BM25-style
# ---------------------------------------------------------------------------

@dataclass
class _SparseEntry:
    doc_id: str
    tokens: List[str]
    tf: Dict[str, int] = field(default_factory=dict)


class SparseRetriever:
    """BM25-style sparse retrieval over tokenized documents."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._docs: Dict[str, _SparseEntry] = {}
        self._df: Dict[str, int] = {}
        self._avgdl: float = 0.0

    def index(self, doc_id: str, tokens: List[str]) -> None:
        tf: Dict[str, int] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        entry = _SparseEntry(doc_id=doc_id, tokens=tokens, tf=tf)
        # Update df
        for t in set(tokens):
            self._df[t] = self._df.get(t, 0) + 1
        self._docs[doc_id] = entry
        total = sum(len(e.tokens) for e in self._docs.values())
        self._avgdl = total / len(self._docs) if self._docs else 1.0

    def search(self, query_tokens: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
        n = len(self._docs)
        if n == 0:
            return []
        scores: Dict[str, float] = {}
        for qt in query_tokens:
            df = self._df.get(qt, 0)
            idf = math.log((n - df + 0.5) / (df + 0.5) + 1.0)
            for did, entry in self._docs.items():
                tf = entry.tf.get(qt, 0)
                dl = len(entry.tokens)
                score = idf * (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * dl / self._avgdl))
                if score > 0:
                    scores[did] = scores.get(did, 0.0) + score
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def get_term_frequency(self, term: str) -> Dict[str, int]:
        """Return per-doc TF for *term*."""
        result: Dict[str, int] = {}
        for did, entry in self._docs.items():
            if term in entry.tf:
                result[did] = entry.tf[term]
        return result


# ---------------------------------------------------------------------------
# ReciprocalRankFusion
# ---------------------------------------------------------------------------

class ReciprocalRankFusion:
    """Merge multiple ranked lists using RRF."""

    def __init__(self, k: int = 60) -> None:
        self.k = k
        self._scores: Dict[str, float] = {}

    def fuse(self, ranked_lists: List[List[Tuple[str, float]]], k: int = 60) -> List[Tuple[str, float]]:
        self._scores: Dict[str, float] = {}
        for rlist in ranked_lists:
            for rank, (doc_id, _score) in enumerate(rlist, start=1):
                self._scores[doc_id] = self._scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        result = sorted(self._scores.items(), key=lambda x: x[1], reverse=True)
        return result

    def get_scores(self) -> Dict[str, float]:
        return dict(self._scores)


# ---------------------------------------------------------------------------
# HybridRetriever
# ---------------------------------------------------------------------------

@dataclass
class _RetrieverEntry:
    name: str
    retriever: Any
    weight: float = 1.0


class HybridRetriever:
    """Combine dense + sparse retrievers via RRF."""

    def __init__(self) -> None:
        self._retrievers: List[_RetrieverEntry] = []
        self._rrf_k: int = 60

    def add_retriever(self, name: str, weight: float = 1.0, *, retriever: Any = None) -> None:
        self._retrievers.append(_RetrieverEntry(name=name, retriever=retriever, weight=weight))

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        fusion = ReciprocalRankFusion(k=self._rrf_k)
        all_ranked: List[List[Tuple[str, float]]] = []
        for entry in self._retrievers:
            results = entry.retriever.search(query, top_k=top_k * 2)  # type: ignore[attr-defined]
            all_ranked.append(results)
        return fusion.fuse(all_ranked, k=self._rrf_k)[:top_k]

    def set_rrf_k(self, k: int) -> None:
        self._rrf_k = k


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------

class Chunker:
    """Document chunking strategies."""

    @staticmethod
    def chunk_fixed(text: str, size: int = 200, overlap: int = 50) -> List[str]:
        if size <= 0:
            return [text]
        chunks: List[str] = []
        start = 0
        while start < len(text):
            end = start + size
            chunks.append(text[start:end])
            start += size - overlap
            if start >= len(text):
                break
        return chunks if chunks else [text]

    @staticmethod
    def chunk_recursive(text: str, max_size: int = 500) -> List[str]:
        """Split on paragraphs, then sentences, then characters."""
        if len(text) <= max_size:
            return [text]
        # Split on double newline first
        paras = text.split("\n\n")
        chunks: List[str] = []
        current = ""
        for para in paras:
            if len(current) + len(para) + 2 <= max_size:
                current = current + "\n\n" + para if current else para
            else:
                if current:
                    chunks.append(current)
                if len(para) <= max_size:
                    current = para
                else:
                    # Split on sentences
                    sentences = re.split(r"(?<=[.!?])\s+", para)
                    current = ""
                    for s in sentences:
                        if len(current) + len(s) + 1 <= max_size:
                            current = current + " " + s if current else s
                        else:
                            if current:
                                chunks.append(current)
                            current = s
        if current:
            chunks.append(current)
        return chunks if chunks else [text]

    @staticmethod
    def chunk_semantic(text: str, similarity_threshold: float = 0.5) -> List[str]:
        """Simulate semantic chunking by splitting on topic-change heuristics."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        if not sentences:
            return [text]
        engine = EmbeddingEngine(dim=32)
        chunks: List[str] = []
        current_chunk = [sentences[0]]
        prev_emb = engine.embed(sentences[0])

        for s in sentences[1:]:
            emb = engine.embed(s)
            sim = EmbeddingEngine.cosine_similarity(prev_emb, emb)
            if sim < similarity_threshold and len(current_chunk) > 0:
                chunks.append(" ".join(current_chunk))
                current_chunk = [s]
            else:
                current_chunk.append(s)
            prev_emb = emb

        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return chunks if chunks else [text]
