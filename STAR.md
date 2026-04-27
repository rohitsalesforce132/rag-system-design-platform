# STAR.md — RAG System Design Platform

## 30-Second Pitch

I built a production-grade RAG system design platform with 10 subsystems and 138 tests using pure Python — covering everything from hybrid retrieval and reranking to observability and canary deployments. It's the complete RAG architecture from The AI Engineer's System Design Interview Guide, with every class mapping to real production concerns.

---

## Situation

As an Azure DevOps engineer transitioning to AI/ML, I needed a deep understanding of how production RAG systems work end-to-end — not just the happy path, but the full lifecycle: retrieval, reranking, evaluation, guardrails, scaling, security, observability, and deployment.

Most learning resources cover individual components in isolation. I wanted to build something that connects all the pieces into a coherent system design that I could both learn from and demonstrate in interviews.

## Task

Design and implement a complete RAG system design platform that:
- Covers all 13 chapters of the system design interview guide
- Implements 10 production subsystems with 40+ classes
- Runs with zero external dependencies (pure Python stdlib)
- Has 130+ deterministic tests, all passing
- Demonstrates production patterns: RBAC, canary deployments, A/B testing, PII filtering, drift detection, KV caching

## Action

1. **Designed 10 subsystems** mapping each interview topic to real production code:
   - **Retrieval**: Dense (cosine similarity), Sparse (BM25), Hybrid (RRF fusion), document chunking (fixed/recursive/semantic)
   - **Reranking**: Cross-encoder simulation, LLM-based reranking, chained pipeline with score normalization
   - **RAG Pipeline**: End-to-end pipeline with caching, session management, agentic Plan-Execute-Verify pattern
   - **Evaluation**: Precision/Recall/MRR/NDCG, RAGAS metrics, hallucination detection, golden datasets
   - **Guardrails**: Citation enforcement, confidence scoring, grounding checks, refusal engine
   - **Performance**: ANN search simulation, index management, latency profiling, embedding caching, cost optimization
   - **Ingestion**: Multi-format parsing, incremental updates, deduplication, versioning, freshness tracking
   - **Security**: RBAC access control, multi-tenant isolation, PII detection/redaction, audit logging
   - **Observability**: Distributed tracing, drift detection, hallucination monitoring, feedback collection, metrics export
   - **Deployment**: Model versioning, A/B testing, canary deployments, KV cache management, inference optimization

2. **Built deterministic test suite** — 138 tests covering edge cases, all reproducible

3. **Zero dependencies** — everything uses Python stdlib: hashlib for embeddings, math for similarity, re for parsing

## Result

- **138 tests, 100% passing**, 0 failures
- **40+ production classes** across 10 subsystems
- **Zero external dependencies** — runs on any Python 3.8+
- Deep understanding of every RAG system design component
- Interview-ready with concrete implementation examples for every topic

## Follow-Up Questions

**Q: How would you scale this to production?**
A: Replace simulated components with real services — FAISS/HNSW for ANN, Redis for caching, PostgreSQL for metadata, Prometheus for metrics. The architecture stays the same.

**Q: Why RRF for hybrid search?**
A: RRF is rank-based, not score-based, so it naturally handles different score distributions from dense and sparse retrievers without needing score normalization.

**Q: How does the canary deployment work?**
A: Route a small percentage of traffic to the new model version, compare metrics against the stable version, and automatically promote or rollback based on performance threshold.

**Q: What's the hardest part of production RAG?**
A: Evaluation and guardrails — it's easy to build retrieval, but ensuring answers are faithful, grounded, and citation-backed at scale requires careful monitoring and automated checks.

## Key Skills

- System Design (RAG architecture, retrieval, reranking)
- Production ML (A/B testing, canary deployments, model versioning)
- Software Engineering (clean architecture, testing, type hints)
- Security (RBAC, PII filtering, audit logging, multi-tenancy)
- Observability (tracing, drift detection, metrics export)
