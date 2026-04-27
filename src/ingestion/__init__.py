"""Data Pipeline & Ingestion — Parsing, Incremental Updates, Dedup, Versioning (Ch7)."""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# DocumentParser
# ---------------------------------------------------------------------------

class DocumentParser:
    """Parse various document formats."""

    @staticmethod
    def parse_text(content: str) -> str:
        return content.strip()

    @staticmethod
    def parse_html(content: str) -> str:
        """Strip HTML tags."""
        return re.sub(r"<[^>]+>", "", content).strip()

    @staticmethod
    def parse_markdown(content: str) -> str:
        """Strip markdown formatting."""
        text = re.sub(r"#+\s", "", content)  # headers
        text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)  # bold
        text = re.sub(r"\*(.+?)\*", r"\1", text)  # italic
        text = re.sub(r"`(.+?)`", r"\1", text)  # code
        text = re.sub(r"\[(.+?)\]\(.+?\)", r"\1", text)  # links
        return text.strip()

    @staticmethod
    def extract_metadata(content: str) -> Dict[str, Any]:
        """Extract basic metadata from content."""
        words = content.split()
        return {
            "char_count": len(content),
            "word_count": len(words),
            "line_count": len(content.splitlines()),
            "has_code": "```" in content,
            "has_links": bool(re.search(r"https?://", content)),
        }


# ---------------------------------------------------------------------------
# IncrementalUpdater
# ---------------------------------------------------------------------------

class IncrementalUpdater:
    """Handle incremental document updates."""

    def __init__(self) -> None:
        self._source_hashes: Dict[str, str] = {}
        self._last_sync: float = 0.0

    def detect_changes(self, source: Dict[str, str]) -> List[str]:
        """Return list of changed document IDs."""
        changed: List[str] = []
        for doc_id, content in source.items():
            h = hashlib.md5(content.encode()).hexdigest()
            if self._source_hashes.get(doc_id) != h:
                changed.append(doc_id)
        return changed

    def apply_updates(self, changes: Dict[str, str]) -> int:
        for doc_id, content in changes.items():
            self._source_hashes[doc_id] = hashlib.md5(content.encode()).hexdigest()
        self._last_sync = time.time()
        return len(changes)

    def get_last_sync(self) -> float:
        return self._last_sync

    def get_sync_status(self) -> Dict[str, Any]:
        return {"last_sync": self._last_sync, "tracked_docs": len(self._source_hashes)}


# ---------------------------------------------------------------------------
# Deduplicator
# ---------------------------------------------------------------------------

class Deduplicator:
    """Remove duplicate documents."""

    @staticmethod
    def compute_fingerprint(doc: str) -> str:
        return hashlib.sha256(doc.encode()).hexdigest()

    def check_duplicate(self, doc: str, existing: List[str]) -> bool:
        fp = self.compute_fingerprint(doc)
        return fp in {self.compute_fingerprint(e) for e in existing}

    @staticmethod
    def merge_duplicates(docs: List[str]) -> List[str]:
        """Return unique documents, preserving order."""
        seen: set = set()
        result: List[str] = []
        for doc in docs:
            h = hashlib.sha256(doc.encode()).hexdigest()
            if h not in seen:
                seen.add(h)
                result.append(doc)
        return result


# ---------------------------------------------------------------------------
# VersionManager
# ---------------------------------------------------------------------------

@dataclass
class _Version:
    id: str
    embeddings: Dict[str, List[float]]
    metadata: Dict[str, Any]
    created_at: float = field(default_factory=time.time)


class VersionManager:
    """Version embeddings for rollback and comparison."""

    def __init__(self) -> None:
        self._versions: Dict[str, _Version] = {}
        self._counter: int = 0

    def create_version(self, embeddings: Dict[str, List[float]], metadata: Optional[Dict[str, Any]] = None) -> str:
        self._counter += 1
        vid = f"v{self._counter}"
        self._versions[vid] = _Version(id=vid, embeddings=embeddings, metadata=metadata or {})
        return vid

    def get_version(self, id: str) -> Optional[_Version]:
        return self._versions.get(id)

    def get_latest(self) -> Optional[_Version]:
        if not self._versions:
            return None
        return max(self._versions.values(), key=lambda v: v.created_at)

    def diff_versions(self, v1: str, v2: str) -> Dict[str, Any]:
        ver1 = self._versions.get(v1)
        ver2 = self._versions.get(v2)
        if ver1 is None or ver2 is None:
            return {"error": "Version not found"}
        keys1 = set(ver1.embeddings.keys())
        keys2 = set(ver2.embeddings.keys())
        return {
            "added": list(keys2 - keys1),
            "removed": list(keys1 - keys2),
            "common": list(keys1 & keys2),
        }


# ---------------------------------------------------------------------------
# DataFreshnessTracker
# ---------------------------------------------------------------------------

class DataFreshnessTracker:
    """Track how fresh documents are."""

    def __init__(self) -> None:
        self._timestamps: Dict[str, float] = {}

    def mark_fresh(self, doc_id: str) -> None:
        self._timestamps[doc_id] = time.time()

    def get_stale(self, threshold_hours: float = 24.0) -> List[str]:
        cutoff = time.time() - threshold_hours * 3600
        return [did for did, ts in self._timestamps.items() if ts < cutoff]

    def get_freshness_score(self, doc_id: str) -> float:
        """Return 0-1 score, 1 = just updated."""
        ts = self._timestamps.get(doc_id)
        if ts is None:
            return 0.0
        age_hours = (time.time() - ts) / 3600.0
        return max(0.0, 1.0 - age_hours / 168.0)  # decays over a week
