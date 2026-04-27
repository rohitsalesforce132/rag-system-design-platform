"""Deployment & LLMOps — Model Versioning, A/B Testing, Canary, KV Cache, Inference (Ch12-13)."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class DeploymentStage(Enum):
    STAGING = "staging"
    PRODUCTION = "production"
    ROLLED_BACK = "rolled_back"


# ---------------------------------------------------------------------------
# ModelVersionManager
# ---------------------------------------------------------------------------

@dataclass
class _ModelVersion:
    model: str
    version: str
    stage: DeploymentStage
    timestamp: float = field(default_factory=time.time)


class ModelVersionManager:
    """Model versioning and promotion."""

    def __init__(self) -> None:
        self._versions: Dict[str, List[_ModelVersion]] = {}

    def register_version(self, model: str, version: str) -> None:
        if model not in self._versions:
            self._versions[model] = []
        self._versions[model].append(_ModelVersion(model=model, version=version, stage=DeploymentStage.STAGING))

    def promote(self, model: str, version: str, stage: DeploymentStage) -> bool:
        vers = self._versions.get(model, [])
        for v in vers:
            if v.version == version:
                v.stage = stage
                return True
        return False

    def rollback(self, model: str) -> bool:
        vers = self._versions.get(model, [])
        prod = [v for v in vers if v.stage == DeploymentStage.PRODUCTION]
        if not prod:
            return False
        latest_prod = prod[-1]
        latest_prod.stage = DeploymentStage.ROLLED_BACK
        # Promote previous staging
        staging = [v for v in vers if v.stage == DeploymentStage.STAGING]
        if staging:
            staging[-1].stage = DeploymentStage.PRODUCTION
        return True

    def get_active(self, model: str) -> Optional[str]:
        vers = self._versions.get(model, [])
        prod = [v for v in vers if v.stage == DeploymentStage.PRODUCTION]
        return prod[-1].version if prod else None


# ---------------------------------------------------------------------------
# ABTester
# ---------------------------------------------------------------------------

@dataclass
class _Experiment:
    name: str
    variants: List[str]
    assignments: Dict[str, str] = field(default_factory=dict)
    results: Dict[str, List[float]] = field(default_factory=dict)


class ABTester:
    """A/B testing for model variants."""

    def __init__(self) -> None:
        self._experiments: Dict[str, _Experiment] = {}

    def create_experiment(self, name: str, variants: List[str]) -> None:
        exp = _Experiment(name=name, variants=variants)
        for v in variants:
            exp.results[v] = []
        self._experiments[name] = exp

    def assign_user(self, user_id: str, experiment: str = "") -> str:
        # Hash user_id for deterministic assignment
        for exp_name, exp in self._experiments.items():
            if experiment and exp_name != experiment:
                continue
            idx = int(hashlib.md5(user_id.encode()).hexdigest(), 16) % len(exp.variants)
            variant = exp.variants[idx]
            exp.assignments[user_id] = variant
            return variant
        return ""

    def record_result(self, user_id: str, variant: str, metric: float) -> None:
        for exp in self._experiments.values():
            if variant in exp.results:
                exp.results[variant].append(metric)
                return

    def get_results(self, experiment: str) -> Dict[str, Any]:
        exp = self._experiments.get(experiment)
        if exp is None:
            return {}
        summary: Dict[str, Any] = {}
        for v, vals in exp.results.items():
            summary[v] = {"count": len(vals), "avg": sum(vals) / len(vals) if vals else 0.0}
        return summary


# ---------------------------------------------------------------------------
# CanaryDeployer
# ---------------------------------------------------------------------------

class CanaryDeployer:
    """Canary deployment for models."""

    def __init__(self) -> None:
        self._canary_version: Optional[str] = None
        self._traffic_pct: float = 0.0
        self._metrics: Dict[str, List[float]] = {"canary": [], "stable": []}
        self._active: bool = False

    def start_canary(self, version: str, traffic_pct: float) -> bool:
        self._canary_version = version
        self._traffic_pct = traffic_pct
        self._active = True
        return True

    def get_canary_metrics(self) -> Dict[str, Any]:
        canary = self._metrics["canary"]
        stable = self._metrics["stable"]
        return {
            "canary_avg": sum(canary) / len(canary) if canary else 0.0,
            "stable_avg": sum(stable) / len(stable) if stable else 0.0,
            "traffic_pct": self._traffic_pct,
        }

    def promote_or_rollback(self) -> str:
        canary = self._metrics["canary"]
        stable = self._metrics["stable"]
        if not canary or not stable:
            return "insufficient_data"
        c_avg = sum(canary) / len(canary)
        s_avg = sum(stable) / len(stable)
        if c_avg >= s_avg * 0.95:
            self._active = False
            return "promoted"
        self._active = False
        return "rolled_back"

    def get_status(self) -> Dict[str, Any]:
        return {
            "active": self._active,
            "canary_version": self._canary_version,
            "traffic_pct": self._traffic_pct,
        }


# ---------------------------------------------------------------------------
# KVCacheManager
# ---------------------------------------------------------------------------

@dataclass
class _KVEntry:
    session_id: str
    tokens: int
    allocated_at: float = field(default_factory=time.time)


class KVCacheManager:
    """KV cache for LLM inference sessions."""

    def __init__(self, max_total_tokens: int = 100000) -> None:
        self._entries: Dict[str, _KVEntry] = {}
        self._max_tokens = max_total_tokens

    def allocate(self, session_id: str, tokens: int) -> bool:
        current = sum(e.tokens for e in self._entries.values())
        if current + tokens > self._max_tokens:
            return False
        self._entries[session_id] = _KVEntry(session_id=session_id, tokens=tokens)
        return True

    def get(self, session_id: str) -> Optional[_KVEntry]:
        return self._entries.get(session_id)

    def evict(self, session_id: str) -> bool:
        return self._entries.pop(session_id, None) is not None

    def get_utilization(self) -> float:
        current = sum(e.tokens for e in self._entries.values())
        return current / self._max_tokens if self._max_tokens > 0 else 0.0


# ---------------------------------------------------------------------------
# InferenceOptimizer
# ---------------------------------------------------------------------------

class InferenceOptimizer:
    """Optimize LLM inference throughput."""

    def __init__(self) -> None:
        self._continuous_batching: bool = False
        self._max_tokens: int = 4096
        self._gpu_utilization: float = 0.0

    def enable_continuous_batching(self, config: Dict[str, Any]) -> None:
        self._continuous_batching = True
        self._max_tokens = config.get("max_tokens", self._max_tokens)

    def set_max_tokens(self, limit: int) -> None:
        self._max_tokens = limit

    def estimate_throughput(self) -> float:
        """Estimate tokens/sec based on config."""
        base = 100.0
        if self._continuous_batching:
            base *= 2.5
        return base

    def get_gpu_utilization(self) -> float:
        return self._gpu_utilization
