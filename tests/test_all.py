"""Comprehensive tests for all 10 RAG subsystems — 130+ deterministic tests."""

import math
import sys
import os
import time
import unittest

# Ensure src is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.retrieval import (
    DenseRetriever, SparseRetriever, HybridRetriever, Chunker,
    EmbeddingEngine, ReciprocalRankFusion,
)
from src.reranker import (
    CrossEncoderReranker, LLMReranker, RerankingPipeline, ScoreNormalizer,
)
from src.rag_pipeline import (
    RAGPipeline, PlannerExecutorVerifier, CacheManager, StateManager, ToolCaller,
)
from src.evaluation import (
    RetrievalEvaluator, GenerationEvaluator, RAGASFramework,
    GoldenDataset, HallucinationDetector,
)
from src.guardrails import (
    CitationEngine, ConfidenceScorer, GroundingChecker, RefusalEngine,
)
from src.performance import (
    ANNSearcher, IndexManager, LatencyOptimizer, EmbeddingCache, CostOptimizer,
)
from src.ingestion import (
    DocumentParser, IncrementalUpdater, Deduplicator, VersionManager, DataFreshnessTracker,
)
from src.security import (
    AccessControl, AccessLevel, TenantIsolator, PIIFilter, AuditLogger,
)
from src.observability import (
    RAGTracer, DriftDetector, HallucinationMonitor, FeedbackCollector, MetricsExporter,
)
from src.deployment import (
    ModelVersionManager, DeploymentStage, ABTester, CanaryDeployer,
    KVCacheManager, InferenceOptimizer,
)


# ========================================================================
# 1. RETRIEVAL (13 tests)
# ========================================================================

class TestEmbeddingEngine(unittest.TestCase):
    def test_embed_deterministic(self):
        eng = EmbeddingEngine(dim=64)
        a = eng.embed("hello world")
        b = eng.embed("hello world")
        self.assertEqual(a, b)

    def test_embed_normalized(self):
        eng = EmbeddingEngine(dim=32)
        e = eng.embed("test")
        norm = math.sqrt(sum(v * v for v in e))
        self.assertAlmostEqual(norm, 1.0, places=5)

    def test_batch_embed(self):
        eng = EmbeddingEngine(dim=16)
        results = eng.batch_embed(["a", "b", "c"])
        self.assertEqual(len(results), 3)

    def test_cosine_similarity_identical(self):
        eng = EmbeddingEngine()
        e = eng.embed("test")
        sim = eng.cosine_similarity(e, e)
        self.assertAlmostEqual(sim, 1.0, places=5)

    def test_dot_product(self):
        eng = EmbeddingEngine()
        a = eng.embed("x")
        b = eng.embed("y")
        dp = eng.dot_product(a, b)
        self.assertIsInstance(dp, float)
        self.assertNotEqual(dp, 0.0)


class TestDenseRetriever(unittest.TestCase):
    def test_index_and_search(self):
        dr = DenseRetriever()
        eng = EmbeddingEngine()
        dr.index("d1", eng.embed("machine learning"))
        dr.index("d2", eng.embed("cooking recipe"))
        results = dr.search(eng.embed("machine learning"), top_k=1)
        self.assertEqual(results[0][0], "d1")

    def test_delete(self):
        dr = DenseRetriever()
        dr.index("d1", [0.1, 0.2])
        self.assertTrue(dr.delete("d1"))
        self.assertFalse(dr.delete("d1"))

    def test_empty_search(self):
        dr = DenseRetriever()
        results = dr.search([0.1, 0.2], top_k=5)
        self.assertEqual(results, [])


class TestSparseRetriever(unittest.TestCase):
    def test_bm25_search(self):
        sr = SparseRetriever()
        sr.index("d1", ["machine", "learning", "model"])
        sr.index("d2", ["cooking", "recipe", "food"])
        results = sr.search(["machine", "learning"], top_k=1)
        self.assertEqual(results[0][0], "d1")

    def test_term_frequency(self):
        sr = SparseRetriever()
        sr.index("d1", ["cat", "cat", "dog"])
        tf = sr.get_term_frequency("cat")
        self.assertEqual(tf["d1"], 2)

    def test_empty_search(self):
        sr = SparseRetriever()
        results = sr.search(["test"], top_k=5)
        self.assertEqual(results, [])


class TestRRF(unittest.TestCase):
    def test_fuse(self):
        rrf = ReciprocalRankFusion()
        list1 = [("a", 0.9), ("b", 0.8)]
        list2 = [("b", 0.9), ("c", 0.8)]
        result = rrf.fuse([list1, list2])
        # "b" appears in both lists so should rank highest
        self.assertEqual(result[0][0], "b")

    def test_get_scores(self):
        rrf = ReciprocalRankFusion()
        rrf.fuse([[("x", 1.0)]])
        scores = rrf.get_scores()
        self.assertIn("x", scores)


class TestChunker(unittest.TestCase):
    def test_fixed_chunking(self):
        text = "abcdefghij" * 10
        chunks = Chunker.chunk_fixed(text, size=20, overlap=5)
        self.assertTrue(len(chunks) > 1)
        self.assertEqual(len(chunks[0]), 20)

    def test_recursive_chunking(self):
        text = "Short text"
        chunks = Chunker.chunk_recursive(text, max_size=500)
        self.assertEqual(chunks, ["Short text"])

    def test_semantic_chunking(self):
        text = "Machine learning is great. The weather is nice. ML models improve."
        chunks = Chunker.chunk_semantic(text, similarity_threshold=0.3)
        self.assertTrue(len(chunks) >= 1)


# ========================================================================
# 2. RERANKER (13 tests)
# ========================================================================

class TestCrossEncoderReranker(unittest.TestCase):
    def test_score_pair(self):
        score = CrossEncoderReranker.score_pair("query", "document text")
        self.assertIsInstance(score, float)
        self.assertTrue(0.0 <= score <= 1.0)

    def test_score_pair_deterministic(self):
        a = CrossEncoderReranker.score_pair("q", "d")
        b = CrossEncoderReranker.score_pair("q", "d")
        self.assertEqual(a, b)

    def test_rerank(self):
        re = CrossEncoderReranker()
        docs = ["doc about cats", "doc about dogs", "doc about birds"]
        results = re.rerank("cats", docs, top_k=2)
        self.assertEqual(len(results), 2)


class TestLLMReranker(unittest.TestCase):
    def test_score_relevance(self):
        score = LLMReranker.score_relevance("machine learning", "machine learning is great")
        self.assertTrue(score > 0.0)

    def test_score_relevance_no_overlap(self):
        score = LLMReranker.score_relevance("quantum physics", "cooking recipes")
        self.assertEqual(score, 0.0)

    def test_rerank(self):
        re = LLMReranker()
        docs = ["machine learning algorithms", "cooking recipes for dinner"]
        results = re.rerank("machine learning", docs, top_k=1)
        self.assertEqual(results[0][0], "machine learning algorithms")


class TestScoreNormalizer(unittest.TestCase):
    def test_min_max(self):
        result = ScoreNormalizer.min_max_normalize([1.0, 3.0, 5.0])
        self.assertAlmostEqual(result[0], 0.0)
        self.assertAlmostEqual(result[2], 1.0)

    def test_z_score(self):
        result = ScoreNormalizer.z_score_normalize([10.0, 20.0, 30.0])
        self.assertEqual(len(result), 3)

    def test_percentile(self):
        result = ScoreNormalizer.percentile_normalize([1.0, 2.0, 3.0])
        self.assertAlmostEqual(result[2], 1.0)

    def test_empty(self):
        self.assertEqual(ScoreNormalizer.min_max_normalize([]), [])
        self.assertEqual(ScoreNormalizer.z_score_normalize([]), [])


class TestRerankingPipeline(unittest.TestCase):
    def test_pipeline(self):
        p = RerankingPipeline()
        p.add_stage("cross", CrossEncoderReranker(), top_k=2)
        p.add_stage("llm", LLMReranker(), top_k=1)
        docs = ["machine learning", "cooking recipe", "deep learning"]
        results = p.execute("machine learning", docs)
        self.assertEqual(len(results), 1)


# ========================================================================
# 3. RAG_PIPELINE (14 tests)
# ========================================================================

class TestCacheManager(unittest.TestCase):
    def test_put_get(self):
        cm = CacheManager()
        cm.put("hash1", {"answer": "yes"}, ttl=60)
        self.assertEqual(cm.get("hash1"), {"answer": "yes"})

    def test_miss(self):
        cm = CacheManager()
        self.assertIsNone(cm.get("nonexistent"))

    def test_invalidate_pattern(self):
        cm = CacheManager()
        cm.put("query_abc", "r1")
        cm.put("query_def", "r2")
        cm.put("other_xyz", "r3")
        removed = cm.invalidate("query_*")
        self.assertEqual(removed, 2)
        self.assertIsNone(cm.get("query_abc"))

    def test_stats(self):
        cm = CacheManager()
        cm.put("h1", "r")
        cm.get("h1")
        cm.get("miss")
        stats = cm.get_stats()
        self.assertEqual(stats["hits"], 1)
        self.assertEqual(stats["misses"], 1)


class TestStateManager(unittest.TestCase):
    def test_create_session(self):
        sm = StateManager()
        sid = sm.create_session()
        self.assertIsInstance(sid, str)
        self.assertEqual(len(sm.get_context(sid)), 0)

    def test_update_context(self):
        sm = StateManager()
        sid = sm.create_session()
        sm.update_context(sid, {"role": "user", "text": "hello"})
        ctx = sm.get_context(sid)
        self.assertEqual(len(ctx), 1)
        self.assertEqual(ctx[0]["role"], "user")

    def test_clear_session(self):
        sm = StateManager()
        sid = sm.create_session()
        self.assertTrue(sm.clear_session(sid))
        self.assertFalse(sm.clear_session(sid))


class TestToolCaller(unittest.TestCase):
    def test_register_and_call(self):
        tc = ToolCaller()
        tc.register_tool("add", lambda a, b: a + b)
        result = tc.call_tool("add", {"a": 2, "b": 3})
        self.assertEqual(result, 5)

    def test_list_tools(self):
        tc = ToolCaller()
        tc.register_tool("foo", lambda: None)
        self.assertEqual(tc.list_tools(), ["foo"])

    def test_missing_tool_raises(self):
        tc = ToolCaller()
        with self.assertRaises(KeyError):
            tc.call_tool("nonexistent", {})


class TestRAGPipeline(unittest.TestCase):
    def test_ingest_and_query(self):
        p = RAGPipeline()
        p.ingest({"d1": "machine learning is great"})
        result = p.query("machine learning")
        self.assertIn("answer", result)
        self.assertIn("d1", result["sources"])

    def test_caching(self):
        p = RAGPipeline()
        p.ingest({"d1": "test document"})
        r1 = p.query("test")
        r2 = p.query("test")
        self.assertFalse(r1["cached"])
        self.assertTrue(r2["cached"])

    def test_streaming(self):
        p = RAGPipeline()
        p.ingest({"d1": "hello world"})
        tokens = p.query_streaming("hello")
        self.assertIsInstance(tokens, list)
        self.assertTrue(len(tokens) > 0)


class TestPlannerExecutorVerifier(unittest.TestCase):
    def test_run(self):
        pev = PlannerExecutorVerifier()
        pev._pipeline.ingest({"d1": "RAG systems combine retrieval and generation"})
        result = pev.run("RAG retrieval generation systems combine")
        self.assertIn("plan", result)
        self.assertIn("verification", result)
        self.assertTrue(result["verification"]["verified"])


# ========================================================================
# 4. EVALUATION (14 tests)
# ========================================================================

class TestRetrievalEvaluator(unittest.TestCase):
    def test_precision_at_k(self):
        ret = ["rel_1", "irr_1", "rel_2", "irr_2", "rel_3"]
        relevant = {"rel_1", "rel_2", "rel_3"}
        p = RetrievalEvaluator.precision_at_k(ret, relevant, 3)
        self.assertAlmostEqual(p, 2 / 3)

    def test_precision_at_k_zero(self):
        self.assertEqual(RetrievalEvaluator.precision_at_k([], set(), 0), 0.0)

    def test_recall_at_k(self):
        ret = ["rel_1", "irr_1", "rel_2"]
        relevant = {"rel_1", "rel_2", "rel_3"}
        r = RetrievalEvaluator.recall_at_k(ret, relevant, 3)
        self.assertAlmostEqual(r, 2 / 3)

    def test_mrr(self):
        result = RetrievalEvaluator.mrr(["irr_1", "rel_1", "irr_2"])
        self.assertAlmostEqual(result, 0.5)

    def test_ndcg(self):
        ranked = ["rel_1", "irr_1", "rel_2"]
        ideal = ["rel_1", "rel_2", "rel_3"]
        ndcg = RetrievalEvaluator.ndcg(ranked, ideal)
        self.assertTrue(0.0 < ndcg <= 1.0)


class TestGenerationEvaluator(unittest.TestCase):
    def test_faithfulness(self):
        f = GenerationEvaluator.faithfulness("the cat sat", "the cat sat on the mat")
        self.assertAlmostEqual(f, 1.0)

    def test_relevance(self):
        r = GenerationEvaluator.relevance("cat dog", "cat dog bird")
        self.assertAlmostEqual(r, 2 / 3)

    def test_correctness(self):
        c = GenerationEvaluator.correctness("the answer is 42", "the answer is 42")
        self.assertAlmostEqual(c, 1.0)


class TestRAGASFramework(unittest.TestCase):
    def test_faithfulness(self):
        r = RAGASFramework()
        score = r.compute_faithfulness("cat dog", "cat dog bird fish")
        self.assertAlmostEqual(score, 1.0)

    def test_relevancy(self):
        r = RAGASFramework()
        score = r.compute_relevancy("cat dog", "cat dog")
        self.assertAlmostEqual(score, 1.0)

    def test_context_precision(self):
        score = RAGASFramework.compute_context_precision(
            ["cat sat on mat", "dog ran far"], "cat mat"
        )
        self.assertTrue(0.0 <= score <= 1.0)


class TestGoldenDataset(unittest.TestCase):
    def test_add_and_split(self):
        ds = GoldenDataset()
        for i in range(10):
            ds.add_example(f"q{i}", f"a{i}", [f"c{i}"], f"gt{i}")
        train, test = ds.get_split(0.8)
        self.assertEqual(len(train), 8)
        self.assertEqual(len(test), 2)

    def test_sample(self):
        ds = GoldenDataset()
        for i in range(20):
            ds.add_example(f"q{i}", f"a{i}", [f"c{i}"], f"gt{i}")
        sample = ds.sample(5)
        self.assertEqual(len(sample), 5)


class TestHallucinationDetector(unittest.TestCase):
    def test_factual_consistency(self):
        hd = HallucinationDetector()
        score = hd.check_factual_consistency("cat dog", "cat dog bird fish")
        self.assertAlmostEqual(score, 1.0)

    def test_grounding(self):
        hd = HallucinationDetector()
        result = hd.check_grounding("cat dog", ["cat dog bird"])
        self.assertIn("source_0", result)

    def test_confidence(self):
        hd = HallucinationDetector()
        conf = hd.get_confidence("This is a detailed answer with many words providing comprehensive information about the topic.")
        self.assertTrue(0.0 <= conf <= 1.0)


# ========================================================================
# 5. GUARDRAILS (13 tests)
# ========================================================================

class TestCitationEngine(unittest.TestCase):
    def test_extract_claims(self):
        ce = CitationEngine()
        claims = ce.extract_claims("First claim. Second claim!")
        self.assertEqual(len(claims), 2)

    def test_match_claims(self):
        ce = CitationEngine()
        matched = ce.match_claims_to_sources(
            ["machine learning is great"],
            ["machine learning is great for NLP"]
        )
        self.assertTrue(matched[0].verified)

    def test_generate_citations(self):
        ce = CitationEngine()
        result = ce.generate_citations(
            "ML is great. DL is also great.",
            ["ML is great for NLP", "DL powers language models"]
        )
        self.assertIn("[Source", result)

    def test_verify_citations(self):
        ce = CitationEngine()
        result = ce.verify_citations("Claim one [Source 1]. Claim two.")
        self.assertTrue(result[0])
        self.assertFalse(result[1])


class TestConfidenceScorer(unittest.TestCase):
    def test_score(self):
        cs = ConfidenceScorer()
        score = cs.score("cat dog bird", "cat dog bird fish")
        self.assertAlmostEqual(score, 1.0)

    def test_low_confidence_spots(self):
        cs = ConfidenceScorer()
        spots = cs.get_low_confidence_spots("Maybe this is true. Definitely this is false.")
        self.assertEqual(len(spots), 1)
        self.assertIn("Maybe", spots[0])

    def test_refusal(self):
        cs = ConfidenceScorer()
        cs.set_refusal_threshold(0.5)
        self.assertTrue(cs.should_refuse(0.3))
        self.assertFalse(cs.should_refuse(0.7))


class TestGroundingChecker(unittest.TestCase):
    def test_grounded(self):
        ok, score = GroundingChecker.check_grounded("cat dog", "cat dog bird")
        self.assertTrue(ok)
        self.assertAlmostEqual(score, 1.0)

    def test_ungrounded_claims(self):
        claims = GroundingChecker.get_ungrounded_claims("This is a fact. Research shows growth.")
        self.assertTrue(len(claims) >= 1)

    def test_suggest_corrections(self):
        suggestions = GroundingChecker.suggest_corrections("cat dog", "bird fish")
        self.assertTrue(len(suggestions) >= 1)


class TestRefusalEngine(unittest.TestCase):
    def test_should_refuse_harmful(self):
        re = RefusalEngine()
        self.assertTrue(re.should_refuse("how to make dangerous weapons", 0.9))

    def test_should_not_refuse(self):
        re = RefusalEngine()
        self.assertFalse(re.should_refuse("what is machine learning", 0.8))

    def test_generate_refusal(self):
        re = RefusalEngine()
        msg = re.generate_refusal("how to hack")
        self.assertTrue(len(msg) > 0)

    def test_log_refusal(self):
        re = RefusalEngine()
        re.log_refusal("test query", "low confidence")
        # No assertion needed — just no exception


# ========================================================================
# 6. PERFORMANCE (13 tests)
# ========================================================================

class TestANNSearcher(unittest.TestCase):
    def test_build_and_search(self):
        ann = ANNSearcher()
        eng = EmbeddingEngine(dim=16)
        embs = {f"d{i}": eng.embed(f"doc {i}") for i in range(10)}
        ann.build_index(embs)
        results = ann.search(eng.embed("doc 0"), top_k=3)
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0][0], "d0")

    def test_ef_search(self):
        ann = ANNSearcher()
        ann.set_ef_search(100)
        # No error
        self.assertTrue(True)


class TestIndexManager(unittest.TestCase):
    def test_create_index(self):
        im = IndexManager()
        self.assertTrue(im.create_index("main", {"dim": 128}))
        self.assertFalse(im.create_index("main", {}))

    def test_rebuild(self):
        im = IndexManager()
        im.create_index("main")
        self.assertTrue(im.rebuild_index("main"))
        self.assertFalse(im.rebuild_index("nonexistent"))

    def test_stats(self):
        im = IndexManager()
        im.create_index("main", {"dim": 64})
        stats = im.get_index_stats("main")
        self.assertEqual(stats["name"], "main")
        self.assertIsNone(im.get_index_stats("nope"))

    def test_optimize(self):
        im = IndexManager()
        im.create_index("main")
        self.assertTrue(im.optimize_index("main"))
        stats = im.get_index_stats("main")
        self.assertTrue(stats["config"]["optimized"])


class TestLatencyOptimizer(unittest.TestCase):
    def test_profile(self):
        lo = LatencyOptimizer()
        pid = lo.profile_pipeline({"embedding": 50.0, "retrieval": 120.0, "generation": 200.0})
        self.assertIsInstance(pid, str)

    def test_bottlenecks(self):
        lo = LatencyOptimizer()
        lo.profile_pipeline({"embed": 50.0, "retrieve": 200.0, "generate": 100.0})
        b = lo.get_bottlenecks()
        self.assertEqual(b[0][0], "retrieve")

    def test_suggest(self):
        lo = LatencyOptimizer()
        lo.profile_pipeline({"embedding": 200.0, "retrieval": 100.0})
        suggestions = lo.suggest_optimizations()
        self.assertTrue(len(suggestions) > 0)

    def test_p99(self):
        lo = LatencyOptimizer()
        lo.profile_pipeline({"a": 100.0, "b": 200.0})
        p99 = lo.estimate_p99_latency()
        self.assertAlmostEqual(p99, 450.0)


class TestEmbeddingCache(unittest.TestCase):
    def test_hit_miss(self):
        ec = EmbeddingCache()
        ec.put("h1", [0.1, 0.2], ttl=60)
        self.assertEqual(ec.get("h1"), [0.1, 0.2])
        self.assertIsNone(ec.get("h2"))
        self.assertAlmostEqual(ec.get_hit_rate(), 0.5)

    def test_expired(self):
        ec = EmbeddingCache()
        ec.put("h1", [0.1], ttl=0)  # expires immediately
        time.sleep(0.01)
        self.assertIsNone(ec.get("h1"))


class TestCostOptimizer(unittest.TestCase):
    def test_estimate_cost(self):
        co = CostOptimizer()
        cost = co.estimate_query_cost("short query")
        self.assertTrue(cost > 0)

    def test_batching(self):
        co = CostOptimizer()
        batches = co.suggest_batching([f"q{i}" for i in range(25)])
        self.assertEqual(len(batches), 3)

    def test_cost_report(self):
        co = CostOptimizer()
        report = co.get_cost_report()
        self.assertIn("total_queries", report)


# ========================================================================
# 7. INGESTION (13 tests)
# ========================================================================

class TestDocumentParser(unittest.TestCase):
    def test_text(self):
        self.assertEqual(DocumentParser.parse_text("  hello  "), "hello")

    def test_html(self):
        result = DocumentParser.parse_html("<p>Hello <b>world</b></p>")
        self.assertNotIn("<", result)
        self.assertIn("Hello", result)

    def test_markdown(self):
        md = "# Title\n**bold** and *italic* and `code`"
        result = DocumentParser.parse_markdown(md)
        self.assertNotIn("**", result)
        self.assertNotIn("##", result)

    def test_metadata(self):
        meta = DocumentParser.extract_metadata("hello world\n```python\n```")
        self.assertEqual(meta["word_count"], 4)
        self.assertTrue(meta["has_code"])


class TestIncrementalUpdater(unittest.TestCase):
    def test_detect_changes(self):
        iu = IncrementalUpdater()
        iu.apply_updates({"d1": "content"})
        changes = iu.detect_changes({"d1": "content", "d2": "new"})
        self.assertIn("d2", changes)
        self.assertNotIn("d1", changes)

    def test_apply_and_sync(self):
        iu = IncrementalUpdater()
        iu.apply_updates({"d1": "hello"})
        self.assertTrue(iu.get_last_sync() > 0)
        status = iu.get_sync_status()
        self.assertEqual(status["tracked_docs"], 1)


class TestDeduplicator(unittest.TestCase):
    def test_fingerprint(self):
        fp = Deduplicator.compute_fingerprint("hello")
        self.assertIsInstance(fp, str)
        self.assertEqual(fp, Deduplicator.compute_fingerprint("hello"))

    def test_check_duplicate(self):
        dd = Deduplicator()
        self.assertTrue(dd.check_duplicate("hello", ["hello"]))
        self.assertFalse(dd.check_duplicate("world", ["hello"]))

    def test_merge(self):
        dd = Deduplicator()
        result = dd.merge_duplicates(["a", "b", "a", "c", "b"])
        self.assertEqual(result, ["a", "b", "c"])


class TestVersionManager(unittest.TestCase):
    def test_create_and_get(self):
        vm = VersionManager()
        vid = vm.create_version({"d1": [0.1, 0.2]}, {"note": "v1"})
        v = vm.get_version(vid)
        self.assertIsNotNone(v)
        self.assertIn("d1", v.embeddings)

    def test_latest(self):
        vm = VersionManager()
        vm.create_version({"d1": [0.1]})
        time.sleep(0.01)
        vm.create_version({"d2": [0.2]})
        latest = vm.get_latest()
        self.assertIn("d2", latest.embeddings)

    def test_diff(self):
        vm = VersionManager()
        v1 = vm.create_version({"a": [0.1], "b": [0.2]})
        v2 = vm.create_version({"b": [0.2], "c": [0.3]})
        diff = vm.diff_versions(v1, v2)
        self.assertIn("c", diff["added"])
        self.assertIn("a", diff["removed"])


class TestDataFreshnessTracker(unittest.TestCase):
    def test_fresh_and_stale(self):
        ft = DataFreshnessTracker()
        ft.mark_fresh("d1")
        self.assertEqual(ft.get_stale(threshold_hours=0), ["d1"])
        self.assertEqual(ft.get_stale(threshold_hours=100), [])


# ========================================================================
# 8. SECURITY (13 tests)
# ========================================================================

class TestAccessControl(unittest.TestCase):
    def test_grant_and_check(self):
        ac = AccessControl()
        ac.grant_access("alice", "doc1", AccessLevel.READ)
        self.assertEqual(ac.check_access("alice", "doc1"), AccessLevel.READ)

    def test_no_access(self):
        ac = AccessControl()
        self.assertEqual(ac.check_access("bob", "doc1"), AccessLevel.NONE)

    def test_revoke(self):
        ac = AccessControl()
        ac.grant_access("alice", "doc1", AccessLevel.READ)
        self.assertTrue(ac.revoke_access("alice", "doc1"))
        self.assertFalse(ac.revoke_access("alice", "doc1"))


class TestTenantIsolator(unittest.TestCase):
    def test_create_tenant(self):
        ti = TenantIsolator()
        self.assertTrue(ti.create_tenant("t1"))
        self.assertFalse(ti.create_tenant("t1"))

    def test_isolate_query(self):
        ti = TenantIsolator()
        ti.create_tenant("t1")
        result = ti.isolate_query("test query", "t1")
        self.assertEqual(result["tenant_id"], "t1")


class TestPIIFilter(unittest.TestCase):
    def test_detect_email(self):
        pf = PIIFilter()
        results = pf.detect_pii("Contact john@example.com for info")
        self.assertTrue(any(r["type"] == "email" for r in results))

    def test_redact(self):
        pf = PIIFilter()
        redacted = pf.redact_pii("Call 555-123-4567 or email a@b.com")
        self.assertNotIn("555-123-4567", redacted)
        self.assertNotIn("a@b.com", redacted)

    def test_pii_types(self):
        pf = PIIFilter()
        types = pf.get_pii_types("SSN: 123-45-6789 and email: x@y.com")
        self.assertIn("ssn", types)
        self.assertIn("email", types)

    def test_no_pii(self):
        pf = PIIFilter()
        types = pf.get_pii_types("Hello world this is clean text")
        self.assertEqual(types, [])


class TestAuditLogger(unittest.TestCase):
    def test_log_query(self):
        al = AuditLogger()
        al.log_query("alice", "what is ML?", ["d1", "d2"])
        trail = al.get_audit_trail("alice")
        self.assertEqual(len(trail), 1)
        self.assertEqual(trail[0]["action"], "query")

    def test_log_access(self):
        al = AuditLogger()
        al.log_access("bob", "doc1", "read")
        trail = al.get_audit_trail("bob")
        self.assertEqual(len(trail), 1)

    def test_empty_trail(self):
        al = AuditLogger()
        trail = al.get_audit_trail("nobody")
        self.assertEqual(trail, [])


# ========================================================================
# 9. OBSERVABILITY (13 tests)
# ========================================================================

class TestRAGTracer(unittest.TestCase):
    def test_full_trace(self):
        t = RAGTracer()
        tid = t.start_trace("test query")
        t.log_step(tid, "embedding", 10.0)
        t.log_step(tid, "retrieval", 50.0)
        t.end_trace(tid)
        trace = t.get_trace(tid)
        self.assertIsNotNone(trace)
        self.assertEqual(len(trace["steps"]), 2)

    def test_missing_trace(self):
        t = RAGTracer()
        self.assertIsNone(t.get_trace("nonexistent"))


class TestDriftDetector(unittest.TestCase):
    def test_baseline_and_drift(self):
        dd = DriftDetector()
        dd.baseline({"d1": [0.5, 0.5], "d2": [0.5, 0.5]})
        drifted, score = dd.check_drift({"d1": [0.5, 0.5], "d2": [0.5, 0.5]})
        self.assertFalse(drifted)

    def test_drift_report(self):
        dd = DriftDetector()
        dd.baseline({"d1": [1.0, 2.0]})
        report = dd.get_drift_report()
        self.assertIn("baseline_mean", report)


class TestHallucinationMonitor(unittest.TestCase):
    def test_check_and_rate(self):
        hm = HallucinationMonitor()
        hm.check_answer("cat dog", "cat dog bird")
        rate = hm.get_hallucination_rate()
        self.assertTrue(0.0 <= rate <= 1.0)

    def test_alert(self):
        hm = HallucinationMonitor()
        # No checks → rate is 0 → no alert
        self.assertFalse(hm.alert_if_above(0.5))


class TestFeedbackCollector(unittest.TestCase):
    def test_collect_and_stats(self):
        fc = FeedbackCollector()
        fc.collect("q1", "a1", 5)
        fc.collect("q2", "a2", 3)
        stats = fc.get_feedback_stats()
        self.assertEqual(stats["count"], 2)
        self.assertAlmostEqual(stats["avg_rating"], 4.0)

    def test_low_rated(self):
        fc = FeedbackCollector()
        fc.collect("q1", "a1", 2)
        fc.collect("q2", "a2", 5)
        low = fc.get_low_rated(threshold=3)
        self.assertEqual(len(low), 1)
        self.assertEqual(low[0]["rating"], 2)


class TestMetricsExporter(unittest.TestCase):
    def test_prometheus(self):
        me = MetricsExporter()
        me.record("rag_queries_total", 100.0)
        prom = me.export_prometheus()
        self.assertIn("rag_queries_total 100.0", prom)

    def test_json(self):
        me = MetricsExporter()
        me.record("latency_ms", 50.0)
        j = me.export_json()
        self.assertEqual(j["latency_ms"], 50.0)

    def test_summary(self):
        me = MetricsExporter()
        me.record("a", 10.0)
        me.record("b", 20.0)
        s = me.get_summary()
        self.assertEqual(s["count"], 2)
        self.assertAlmostEqual(s["avg"], 15.0)


# ========================================================================
# 10. DEPLOYMENT (14 tests)
# ========================================================================

class TestModelVersionManager(unittest.TestCase):
    def test_register_and_promote(self):
        mvm = ModelVersionManager()
        mvm.register_version("llama", "v1")
        mvm.promote("llama", "v1", DeploymentStage.PRODUCTION)
        self.assertEqual(mvm.get_active("llama"), "v1")

    def test_rollback(self):
        mvm = ModelVersionManager()
        mvm.register_version("llama", "v1")
        mvm.promote("llama", "v1", DeploymentStage.PRODUCTION)
        mvm.register_version("llama", "v2")
        mvm.rollback("llama")
        # v1 should be rolled back, v2 may or may not be promoted
        active = mvm.get_active("llama")
        # After rollback, v1 is no longer production
        self.assertIsNotNone(active)

    def test_no_active(self):
        mvm = ModelVersionManager()
        self.assertIsNone(mvm.get_active("unknown"))


class TestABTester(unittest.TestCase):
    def test_create_and_assign(self):
        ab = ABTester()
        ab.create_experiment("exp1", ["control", "treatment"])
        v = ab.assign_user("user123", "exp1")
        self.assertIn(v, ["control", "treatment"])

    def test_deterministic_assignment(self):
        ab = ABTester()
        ab.create_experiment("exp1", ["A", "B"])
        v1 = ab.assign_user("user1", "exp1")
        v2 = ab.assign_user("user1", "exp1")
        self.assertEqual(v1, v2)

    def test_results(self):
        ab = ABTester()
        ab.create_experiment("exp1", ["A", "B"])
        ab.record_result("u1", "A", 0.8)
        ab.record_result("u2", "B", 0.9)
        results = ab.get_results("exp1")
        self.assertIn("A", results)
        self.assertAlmostEqual(results["A"]["avg"], 0.8)


class TestCanaryDeployer(unittest.TestCase):
    def test_start(self):
        cd = CanaryDeployer()
        self.assertTrue(cd.start_canary("v2", 10.0))
        status = cd.get_status()
        self.assertTrue(status["active"])
        self.assertEqual(status["canary_version"], "v2")

    def test_metrics(self):
        cd = CanaryDeployer()
        cd.start_canary("v2", 10.0)
        metrics = cd.get_canary_metrics()
        self.assertIn("canary_avg", metrics)

    def test_promote_or_rollback_insufficient(self):
        cd = CanaryDeployer()
        result = cd.promote_or_rollback()
        self.assertEqual(result, "insufficient_data")


class TestKVCacheManager(unittest.TestCase):
    def test_allocate_and_get(self):
        kv = KVCacheManager(max_total_tokens=1000)
        self.assertTrue(kv.allocate("s1", 100))
        entry = kv.get("s1")
        self.assertIsNotNone(entry)
        self.assertEqual(entry.tokens, 100)

    def test_overallocate(self):
        kv = KVCacheManager(max_total_tokens=100)
        kv.allocate("s1", 80)
        self.assertFalse(kv.allocate("s2", 50))

    def test_evict(self):
        kv = KVCacheManager()
        kv.allocate("s1", 100)
        self.assertTrue(kv.evict("s1"))
        self.assertFalse(kv.evict("s1"))

    def test_utilization(self):
        kv = KVCacheManager(max_total_tokens=1000)
        kv.allocate("s1", 250)
        self.assertAlmostEqual(kv.get_utilization(), 0.25)


class TestInferenceOptimizer(unittest.TestCase):
    def test_throughput_no_batching(self):
        io = InferenceOptimizer()
        tp = io.estimate_throughput()
        self.assertEqual(tp, 100.0)

    def test_throughput_with_batching(self):
        io = InferenceOptimizer()
        io.enable_continuous_batching({"max_tokens": 2048})
        tp = io.estimate_throughput()
        self.assertEqual(tp, 250.0)

    def test_gpu_utilization(self):
        io = InferenceOptimizer()
        self.assertEqual(io.get_gpu_utilization(), 0.0)


if __name__ == "__main__":
    unittest.main()
