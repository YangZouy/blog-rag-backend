from api.rag_graph import _insufficient_evidence_response, run_rag_with_trace
from core.evidence import EvidenceAssessment, EvidenceStatus, assess_evidence
from core.planning import RetrievalPlan
from data.parse_hexo import DocumentChunk


def chunk(score: float, slug: str = "post") -> DocumentChunk:
    return DocumentChunk(title="Title", url="/post", content="content", slug=slug, score=score)


def test_simple_question_with_reliable_top_hit_is_sufficient():
    plan = RetrievalPlan("simple", "什么是混合检索？", ("什么是混合检索？",))
    evidence = assess_evidence(plan, [chunk(0.7)], {plan.queries[0]: [chunk(0.7)]}, 0.3)
    assert evidence.status is EvidenceStatus.SUFFICIENT


def test_low_top_score_is_insufficient_even_when_documents_exist():
    plan = RetrievalPlan("simple", "不存在的项目", ("不存在的项目",))
    evidence = assess_evidence(plan, [chunk(0.29)], {plan.queries[0]: [chunk(0.29)]}, 0.3)
    assert evidence.status is EvidenceStatus.INSUFFICIENT


def test_complex_question_requires_each_sub_query_to_have_evidence():
    plan = RetrievalPlan("complex", "比较架构与评测", ("架构", "评测"))
    evidence = assess_evidence(
        plan, [chunk(0.8)], {"架构": [chunk(0.8)], "评测": [chunk(0.1)]}, 0.3,
    )
    assert evidence.status is EvidenceStatus.PARTIAL
    assert evidence.missing_aspects == ("评测",)


def test_insufficient_response_refuses_without_citations():
    response = _insufficient_evidence_response(EvidenceAssessment(EvidenceStatus.INSUFFICIENT, ("稳定性",)))
    assert response.mode == "not_found"
    assert response.citations == []
    assert response.evidence_status == "insufficient"
    assert "稳定性" in response.answer


def test_pipeline_does_not_generate_when_evidence_is_insufficient(monkeypatch):
    plan = RetrievalPlan("simple", "不存在的项目", ("不存在的项目",))
    monkeypatch.setattr("api.rag_graph.prepare_query", lambda query, history: (query, False))
    monkeypatch.setattr("api.rag_graph.build_retrieval_plan", lambda query: plan)
    monkeypatch.setattr("api.rag_graph._route_and_retrieve", lambda plan, top_k, candidate_k=None: ("rag", [chunk(0.1)], {plan.queries[0]: [chunk(0.1)]}, "rag"))
    monkeypatch.setattr("api.rag_graph.get_gen_llm", lambda: (_ for _ in ()).throw(AssertionError("must not generate")))
    response, docs = run_rag_with_trace("不存在的项目")
    assert response.evidence_status == "insufficient"
    assert response.mode == "not_found"
    assert docs == []


def test_pipeline_reports_partial_complex_evidence(monkeypatch):
    plan = RetrievalPlan("complex", "比较架构和评测", ("架构", "评测"))
    reliable = chunk(0.8)
    monkeypatch.setattr("api.rag_graph.prepare_query", lambda query, history: (query, False))
    monkeypatch.setattr("api.rag_graph.build_retrieval_plan", lambda query: plan)
    monkeypatch.setattr("api.rag_graph._route_and_retrieve", lambda plan, top_k, candidate_k=None: ("rag", [reliable], {"架构": [reliable], "评测": [chunk(0.1)]}, "rag"))

    class FakeLLM:
        def invoke(self, prompt):
            assert "站内资料未直接覆盖以下方面：评测" in prompt
            return type("Response", (), {"content": "已确认架构部分；评测资料不足。"})()

    monkeypatch.setattr("api.rag_graph.get_gen_llm", lambda: FakeLLM())
    response, _ = run_rag_with_trace("比较架构和评测")
    assert response.evidence_status == "partial"
    assert response.missing_aspects == ["评测"]
