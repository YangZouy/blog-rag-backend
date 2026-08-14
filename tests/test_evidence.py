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


def test_quantified_claim_requires_direct_numeric_evidence():
    plan = RetrievalPlan("simple", "平台目前服务了多少付费企业客户？", ("平台目前服务了多少付费企业客户？",))
    related = chunk(0.8)
    related.content = "平台提供模板化批量渲染能力，并持续优化客户体验。"
    evidence = assess_evidence(plan, [related], {plan.queries[0]: [related]}, 0.3)
    # 相关文档已检索到（top 分达标），仅缺具体数字：应判 PARTIAL（部分作答并标注缺口），
    # 而非整句拒答 INSUFFICIENT。生成侧 _build_prompt 会注入“只回答有证据支持的部分”前缀防编造。
    assert evidence.status is EvidenceStatus.PARTIAL
    assert evidence.missing_aspects == ("平台目前服务了多少付费企业客户？",)


def test_quantified_claim_accepts_direct_numeric_evidence():
    plan = RetrievalPlan("simple", "自动化成功率是多少？", ("自动化成功率是多少？",))
    supported = chunk(0.8)
    supported.content = "评测共运行 100 次，自动化成功率为 92%。"
    evidence = assess_evidence(plan, [supported], {plan.queries[0]: [supported]}, 0.3)
    assert evidence.status is EvidenceStatus.SUFFICIENT


def test_complex_quantified_claim_can_be_partial():
    plan = RetrievalPlan(
        "complex", "说明架构和线上 SLA", ("系统架构", "线上 SLA 是多少"),
    )
    architecture = chunk(0.8, "architecture")
    architecture.content = "系统采用分层架构。"
    sla_related = chunk(0.8, "sla")
    sla_related.content = "文章讨论了线上服务，但没有给出服务等级数据。"
    evidence = assess_evidence(
        plan,
        [architecture, sla_related],
        {"系统架构": [architecture], "线上 SLA 是多少": [sla_related]},
        0.3,
    )
    assert evidence.status is EvidenceStatus.PARTIAL
    assert evidence.missing_aspects == ("线上 SLA 是多少",)


def test_standard_and_quantified_claims_use_different_thresholds():
    ordinary = RetrievalPlan("simple", "防抖和节流有什么区别？", ("防抖和节流有什么区别？",))
    quantified = RetrievalPlan("simple", "自动化成功率是多少？", ("自动化成功率是多少？",))
    low_but_related = chunk(0.2)
    low_but_related.content = "自动化成功率为 92%。"
    ordinary_result = assess_evidence(
        ordinary, [low_but_related], {ordinary.queries[0]: [low_but_related]}, 0.15, 0.3,
    )
    quantified_result = assess_evidence(
        quantified, [low_but_related], {quantified.queries[0]: [low_but_related]}, 0.15, 0.3,
    )
    assert ordinary_result.status is EvidenceStatus.SUFFICIENT
    assert quantified_result.status is EvidenceStatus.INSUFFICIENT
    assert quantified_result.aspect_details[0].reason == "low_score"


def test_evidence_records_missing_quantified_support_reason():
    plan = RetrievalPlan("simple", "线上 SLA 是多少？", ("线上 SLA 是多少？",))
    related = chunk(0.8)
    related.content = "文章讨论线上服务，但没有披露指标。"
    result = assess_evidence(plan, [related], {plan.queries[0]: [related]}, 0.15, 0.3)
    detail = result.aspect_details[0]
    assert detail.top_score == 0.8
    assert detail.threshold == 0.3
    assert detail.reason == "missing_quantified_evidence"
