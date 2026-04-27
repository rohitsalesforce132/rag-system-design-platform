"""End-to-End RAG Architecture — Pipeline, Agentic RAG, Caching, State, Tools (Ch3)."""

from __future__ import annotations

import hashlib
import re
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# CacheManager
# ---------------------------------------------------------------------------

@dataclass
class _CacheEntry:
    result: Any
    expires_at: float


class CacheManager:
    """Simple TTL cache for RAG results."""

    def __init__(self) -> None:
        self._cache: Dict[str, _CacheEntry] = {}
        self._hits: int = 0
        self._misses: int = 0

    def get(self, query_hash: str) -> Optional[Any]:
        entry = self._cache.get(query_hash)
        if entry is None:
            self._misses += 1
            return None
        if time.time() > entry.expires_at:
            del self._cache[query_hash]
            self._misses += 1
            return None
        self._hits += 1
        return entry.result

    def put(self, query_hash: str, result: Any, ttl: float = 3600.0) -> None:
        self._cache[query_hash] = _CacheEntry(result=result, expires_at=time.time() + ttl)

    def invalidate(self, pattern: str) -> int:
        regex = re.compile(pattern.replace("*", ".*"))
        to_remove = [k for k in self._cache if regex.match(k)]
        for k in to_remove:
            del self._cache[k]
        return len(to_remove)

    def get_stats(self) -> Dict[str, int]:
        return {"hits": self._hits, "misses": self._misses, "size": len(self._cache)}


# ---------------------------------------------------------------------------
# StateManager
# ---------------------------------------------------------------------------

@dataclass
class _Session:
    session_id: str
    context: List[Dict[str, str]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)


class StateManager:
    """Stateful RAG session management."""

    def __init__(self) -> None:
        self._sessions: Dict[str, _Session] = {}

    def create_session(self) -> str:
        sid = str(uuid.uuid4())[:8]
        self._sessions[sid] = _Session(session_id=sid)
        return sid

    def get_context(self, session_id: str) -> List[Dict[str, str]]:
        s = self._sessions.get(session_id)
        if s is None:
            return []
        return list(s.context)

    def update_context(self, session_id: str, turn: Dict[str, str]) -> None:
        s = self._sessions.get(session_id)
        if s is not None:
            s.context.append(turn)

    def clear_session(self, session_id: str) -> bool:
        return self._sessions.pop(session_id, None) is not None


# ---------------------------------------------------------------------------
# ToolCaller
# ---------------------------------------------------------------------------

class ToolCaller:
    """Register and call tools in the RAG pipeline."""

    def __init__(self) -> None:
        self._tools: Dict[str, Callable] = {}

    def register_tool(self, name: str, handler: Callable) -> None:
        self._tools[name] = handler

    def call_tool(self, name: str, params: Dict[str, Any]) -> Any:
        handler = self._tools.get(name)
        if handler is None:
            raise KeyError(f"Tool not found: {name}")
        return handler(**params)

    def list_tools(self) -> List[str]:
        return list(self._tools.keys())


# ---------------------------------------------------------------------------
# RAGPipeline
# ---------------------------------------------------------------------------

class RAGPipeline:
    """Full RAG pipeline — ingest, query, streaming simulation."""

    def __init__(self) -> None:
        self._docs: Dict[str, str] = {}
        self._cache = CacheManager()
        self._counter: int = 0

    def ingest(self, documents: Dict[str, str]) -> int:
        """Ingest documents {id: text}. Returns count ingested."""
        self._docs.update(documents)
        return len(documents)

    def query(self, question: str) -> Dict[str, Any]:
        qhash = hashlib.md5(question.encode()).hexdigest()
        cached = self._cache.get(qhash)
        if cached is not None:
            return {**cached, "cached": True}

        # Simulate retrieval + generation
        scored_docs: List[Tuple[str, float]] = []
        q_tokens = set(question.lower().split())
        for did, text in self._docs.items():
            t_tokens = set(text.lower().split())
            overlap = len(q_tokens & t_tokens)
            scored_docs.append((did, overlap / max(len(q_tokens), 1)))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        top = scored_docs[:3]
        context = "\n".join(self._docs[did] for did, _ in top)
        answer = f"Based on retrieved context: {context[:200]}"
        result = {"question": question, "answer": answer, "sources": [did for did, _ in top], "cached": False}
        self._cache.put(qhash, result)
        return result

    def query_streaming(self, question: str) -> List[str]:
        """Simulate streaming by yielding tokens."""
        result = self.query(question)
        answer = result["answer"]
        words = answer.split()
        return words


# ---------------------------------------------------------------------------
# PlannerExecutorVerifier — Agentic RAG
# ---------------------------------------------------------------------------

class PlannerExecutorVerifier:
    """Agentic RAG pattern: Plan → Execute → Verify."""

    def __init__(self) -> None:
        self._pipeline = RAGPipeline()

    def plan(self, query: str) -> List[str]:
        """Break query into sub-queries."""
        words = query.split()
        if len(words) <= 3:
            return [query]
        mid = len(words) // 2
        return [" ".join(words[:mid]), " ".join(words[mid:])]

    def execute(self, plan: List[str]) -> List[Dict[str, Any]]:
        return [self._pipeline.query(q) for q in plan]

    @staticmethod
    def verify(results: List[Dict[str, Any]]) -> Dict[str, Any]:
        sources = set()
        for r in results:
            for s in r.get("sources", []):
                sources.add(s)
        return {"num_results": len(results), "unique_sources": len(sources), "verified": len(results) > 0}

    def run(self, query: str) -> Dict[str, Any]:
        plan = self.plan(query)
        results = self.execute(plan)
        verification = self.verify(results)
        return {"plan": plan, "results": results, "verification": verification}
