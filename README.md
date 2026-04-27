# RAG System Design Platform

A complete RAG (Retrieval-Augmented Generation) system design platform based on *The AI Engineer's System Design Interview Guide*. Every subsystem maps to real production RAG concerns.

**Pure Python stdlib. Zero external dependencies. 138 tests. All passing.**

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    RAG System Design Platform            │
├──────────┬──────────┬───────────┬──────────┬────────────┤
│Retrieval │ Reranker │ RAG       │Evaluation│ Guardrails │
│(Ch1-2)   │ (Ch2)    │Pipeline   │(Ch4)     │ (Ch5)      │
│          │          │(Ch3)      │          │            │
├──────────┼──────────┼───────────┼──────────┼────────────┤
│Perform.  │Ingestion │ Security  │Observab. │Deployment  │
│(Ch6)     │ (Ch7)    │ (Ch8)     │(Ch11)    │(Ch12-13)   │
└──────────┴──────────┴───────────┴──────────┴────────────┘
```

## 10 Subsystems

| # | Subsystem | Module | Key Classes |
|---|-----------|--------|-------------|
| 1 | **Hybrid Retrieval** | `src/retrieval/` | DenseRetriever, SparseRetriever, HybridRetriever, Chunker, EmbeddingEngine, ReciprocalRankFusion |
| 2 | **Re-ranking Pipeline** | `src/reranker/` | CrossEncoderReranker, LLMReranker, RerankingPipeline, ScoreNormalizer |
| 3 | **End-to-End RAG** | `src/rag_pipeline/` | RAGPipeline, PlannerExecutorVerifier, CacheManager, StateManager, ToolCaller |
| 4 | **Evaluation & Metrics** | `src/evaluation/` | RetrievalEvaluator, GenerationEvaluator, RAGASFramework, GoldenDataset, HallucinationDetector |
| 5 | **Hallucination & Reliability** | `src/guardrails/` | CitationEngine, ConfidenceScorer, GroundingChecker, RefusalEngine |
| 6 | **Performance & Scaling** | `src/performance/` | ANNSearcher, IndexManager, LatencyOptimizer, EmbeddingCache, CostOptimizer |
| 7 | **Data Pipeline & Ingestion** | `src/ingestion/` | DocumentParser, IncrementalUpdater, Deduplicator, VersionManager, DataFreshnessTracker |
| 8 | **Security & Enterprise** | `src/security/` | AccessControl, TenantIsolator, PIIFilter, AuditLogger |
| 9 | **Observability & Monitoring** | `src/observability/` | RAGTracer, DriftDetector, HallucinationMonitor, FeedbackCollector, MetricsExporter |
| 10 | **Deployment & LLMOps** | `src/deployment/` | ModelVersionManager, ABTester, CanaryDeployer, KVCacheManager, InferenceOptimizer |

## Quick Start

```bash
# Run all tests
pytest tests/test_all.py -v

# Or with the Linuxbrew pytest
/home/rohit/.linuxbrew/bin/pytest tests/test_all.py -v
```

## Design Principles

- **Pure Python stdlib** — zero dependencies, runs anywhere
- **Deterministic** — all tests are reproducible
- **Type hints** on every public method
- **Docstrings** on all classes and methods
- **Dataclasses** for structured data
- **Enums** for fixed vocabularies
- **Production patterns** — every class mirrors real production RAG concerns

## Interview Use

Each subsystem directly maps to system design interview topics:
- Dense/Sparse/Hybrid retrieval → **Vector databases, BM25, RRF fusion**
- Cross-encoder reranking → **Two-stage retrieval architecture**
- Agentic RAG (Plan-Execute-Verify) → **Multi-agent RAG systems**
- RAGAS evaluation → **Production quality metrics**
- Citation/grounding → **Hallucination prevention**
- ANN search, KV cache → **Latency optimization**
- PII filtering, RBAC → **Enterprise security**
- A/B testing, canary → **Safe deployment patterns**

## License

MIT
