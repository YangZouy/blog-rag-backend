from types import SimpleNamespace

from api.rag_graph import run_rag_with_trace, stream_rag
from core.planning import RetrievalPlan
from data.parse_hexo import DocumentChunk


def chunk(score: float) -> DocumentChunk:
    return DocumentChunk(title="Title", url="/post", content="content", slug="post", score=score)


def settings():
    return SimpleNamespace(
        RETRIEVAL_CANDIDATE_K=8,
        RETRIEVAL_REMEDY_CANDIDATE_K=16,
        MAX_RETRIEVAL_ROUNDS=2,
        EVIDENCE_RELEVANCE_THRESHOLD=0.3,
        RERANK_RELEVANCE_THRESHOLD=0.3,
        GENERATION_CONTEXT_K=5,
        CITATION_MIN_SCORE=0.5,
    )


def test_trace_records_remedy_and_two_evidence_rounds(monkeypatch):
    plan = RetrievalPlan("complex", "比较架构和评测", ("架构", "评测"))

    def fake_route(_plan, _top_k, candidate_k=None):
        score = 0.1 if candidate_k == 8 else 0.8
        docs = [chunk(score)]
        return "rag", docs, {"架构": docs, "评测": docs}, "rag"

    class FakeLLM:
        def invoke(self, _prompt):
            return type("Response", (), {"content": "回答"})()

    monkeypatch.setattr("api.rag_graph.get_settings", settings)
    monkeypatch.setattr("api.rag_graph.prepare_query", lambda query, history: (query, True))
    monkeypatch.setattr("api.rag_graph.build_retrieval_plan", lambda query: plan)
    monkeypatch.setattr("api.rag_graph._route_and_retrieve", fake_route)
    monkeypatch.setattr("api.rag_graph.get_gen_llm", lambda: FakeLLM())
    response, _ = run_rag_with_trace("比较架构和评测")

    assert response.trace is not None
    assert response.trace.question_type == "complex"
    assert response.trace.rewritten is True
    assert response.trace.sub_query_count == 2
    assert response.trace.retrieval_rounds == 2
    assert response.trace.evidence_statuses == ["insufficient", "sufficient"]
    assert response.trace.remedy_action == "expand_candidates"
    assert response.trace.final_decision == "answer"


def test_stream_emits_trace_before_done(monkeypatch):
    plan = RetrievalPlan("simple", "问题", ("问题",))
    docs = [chunk(0.8)]

    class FakeLLM:
        def stream(self, _prompt):
            return [type("Chunk", (), {"content": "回答"})()]

    monkeypatch.setattr("api.rag_graph.get_settings", settings)
    monkeypatch.setattr("api.rag_graph.prepare_query", lambda query, history: (query, False))
    monkeypatch.setattr("api.rag_graph.build_retrieval_plan", lambda query: plan)
    monkeypatch.setattr("api.rag_graph._route_and_retrieve", lambda *args: ("rag", docs, {"问题": docs}, "rag"))
    monkeypatch.setattr("api.rag_graph.get_gen_llm", lambda: FakeLLM())
    events = list(stream_rag("问题"))
    names = [name for name, _ in events]
    assert names.index("trace") < names.index("done")
    trace = next(data for name, data in events if name == "trace")
    assert trace["final_decision"] == "answer"
