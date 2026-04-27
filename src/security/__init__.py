"""Security & Enterprise RAG — RBAC, Multi-tenant, PII, Audit (Ch8)."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set


class AccessLevel(Enum):
    NONE = "none"
    READ = "read"
    WRITE = "write"
    ADMIN = "admin"


# ---------------------------------------------------------------------------
# AccessControl
# ---------------------------------------------------------------------------

@dataclass
class _ACLEntry:
    user: str
    document: str
    level: AccessLevel


class AccessControl:
    """RBAC for RAG documents."""

    def __init__(self) -> None:
        self._acls: Dict[str, Dict[str, AccessLevel]] = {}  # user -> {doc -> level}

    def check_access(self, user: str, document: str) -> AccessLevel:
        return self._acls.get(user, {}).get(document, AccessLevel.NONE)

    def grant_access(self, user: str, document: str, level: AccessLevel) -> None:
        if user not in self._acls:
            self._acls[user] = {}
        self._acls[user][document] = level

    def revoke_access(self, user: str, document: str) -> bool:
        if user in self._acls and document in self._acls[user]:
            del self._acls[user][document]
            return True
        return False


# ---------------------------------------------------------------------------
# TenantIsolator
# ---------------------------------------------------------------------------

class TenantIsolator:
    """Multi-tenant data isolation."""

    def __init__(self) -> None:
        self._tenants: Dict[str, Dict[str, str]] = {}

    def create_tenant(self, tenant_id: str) -> bool:
        if tenant_id in self._tenants:
            return False
        self._tenants[tenant_id] = {}
        return True

    def get_tenant_data(self, tenant_id: str) -> Dict[str, str]:
        return dict(self._tenants.get(tenant_id, {}))

    def isolate_query(self, query: str, tenant_id: str) -> Dict[str, Any]:
        """Return query scoped to tenant data."""
        data = self._tenants.get(tenant_id, {})
        return {"query": query, "tenant_id": tenant_id, "scoped_docs": len(data)}


# ---------------------------------------------------------------------------
# PIIFilter
# ---------------------------------------------------------------------------

_PII_PATTERNS = {
    "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
    "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    "credit_card": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
}


class PIIFilter:
    """Detect and redact PII."""

    def detect_pii(self, text: str) -> List[Dict[str, Any]]:
        findings: List[Dict[str, Any]] = []
        for pii_type, pattern in _PII_PATTERNS.items():
            for match in re.finditer(pattern, text):
                findings.append({"type": pii_type, "value": match.group(), "start": match.start(), "end": match.end()})
        return findings

    def redact_pii(self, text: str, replacement: str = "[REDACTED]") -> str:
        for pattern in _PII_PATTERNS.values():
            text = re.sub(pattern, replacement, text)
        return text

    def get_pii_types(self, text: str) -> List[str]:
        found: List[str] = []
        for pii_type, pattern in _PII_PATTERNS.items():
            if re.search(pattern, text):
                found.append(pii_type)
        return found


# ---------------------------------------------------------------------------
# AuditLogger
# ---------------------------------------------------------------------------

@dataclass
class _AuditEntry:
    user: str
    action: str
    target: str
    details: str
    timestamp: float


class AuditLogger:
    """Audit RAG operations."""

    def __init__(self) -> None:
        self._entries: List[_AuditEntry] = []

    def log_query(self, user: str, query: str, results: List[str]) -> None:
        self._entries.append(_AuditEntry(
            user=user, action="query", target=query,
            details=f"Returned {len(results)} results", timestamp=time.time()
        ))

    def log_access(self, user: str, document: str, action: str) -> None:
        self._entries.append(_AuditEntry(
            user=user, action=action, target=document,
            details="", timestamp=time.time()
        ))

    def get_audit_trail(self, user: str, start_date: float = 0.0, end_date: float = 0.0) -> List[Dict[str, Any]]:
        if end_date == 0.0:
            end_date = time.time() + 1
        return [
            {"user": e.user, "action": e.action, "target": e.target, "timestamp": e.timestamp}
            for e in self._entries
            if e.user == user and start_date <= e.timestamp <= end_date
        ]
