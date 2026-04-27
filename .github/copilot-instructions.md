# GitHub Copilot Instructions

## Project Overview
RAG System Design Platform — a complete production-grade RAG (Retrieval-Augmented Generation) system covering all 13 chapters of "The AI Engineer's System Design Interview Guide".

## Architecture
```
src/
├── retrieval/       # Hybrid retrieval (Dense + Sparse + RRF), chunking, embeddings
├── reranker/        # Cross-encoder, LLM reranker, score normalization
├── rag_pipeline/    # End-to-end RAG, agentic planner-executor-verifier, caching
├── evaluation/      # Precision/Recall/MRR/NDCG, RAGAS, golden datasets, hallucination detection
├── guardrails/      # Citation enforcement, confidence scoring, grounding, refusal
├── performance/     # ANN search, index management, latency optimization, cost tracking
├── ingestion/       # Document parsing, incremental updates, dedup, versioning
├── security/        # RBAC access control, multi-tenant isolation, PII filtering, audit
├── observability/   # Distributed tracing, drift detection, hallucination monitoring, feedback
└── deployment/      # Model versioning, A/B testing, canary deploys, KV cache, inference optimization
```

## Conventions
- Pure Python stdlib only — zero external dependencies
- Type hints on all public methods
- Dataclasses for structured data, Enums for vocabularies
- Tests in tests/test_all.py using pytest

## Running
```bash
pytest tests/test_all.py -v  # Run all 138 tests
```
