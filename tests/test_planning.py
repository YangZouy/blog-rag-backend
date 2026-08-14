from api.rag_graph import _ensure_subquery_coverage, _retrieve_plan
from core.planning import MAX_SUB_QUERIES, RetrievalPlan, build_retrieval_plan, is_obviously_complex
from data.parse_hexo import DocumentChunk


def test_simple_question_skips_planning_model(monkeypatch):
    monkeypatch.setattr("core.planning.get_planning_llm", lambda: (_ for _ in ()).throw(AssertionError("unexpected LLM call")))
    plan = build_retrieval_plan("什么是混合检索？")
    assert plan.query_type == "simple"
    assert plan.original_query == "什么是混合检索？"
    assert plan.queries == ("什么是混合检索？",)


def test_complex_question_is_detected():
    assert is_obviously_complex("对比内容分发工具和博客 RAG 的架构、稳定性设计与评测方式")
    assert is_obviously_complex("介绍平台方案、项目问题和商业化收入")


def test_all_evaluation_partial_answer_cases_trigger_planning():
    import json
    from pathlib import Path

    dataset = json.loads(
        (Path(__file__).resolve().parents[1] / "eval" / "agentic_eval_queries.json").read_text(encoding="utf-8")
    )["queries"]
    rows = [row for row in dataset if row["type"] == "partial_answer"]
    assert rows and all(is_obviously_complex(row["query"]) for row in rows)


def test_all_evaluation_complex_cases_trigger_planning():
    import json
    from pathlib import Path

    dataset = json.loads(
        (Path(__file__).resolve().parents[1] / "eval" / "agentic_eval_queries.json").read_text(encoding="utf-8")
    )["queries"]
    rows = [row for row in dataset if row["type"] == "complex"]
    assert rows and all(is_obviously_complex(row["query"]) for row in rows)


def test_planning_drops_summary_and_prevents_new_metric_constraints(monkeypatch):
    class FakeLLM:
        def invoke(self, _messages):
            return type("Response", (), {"content": '''{"sub_queries":[
                "如何搭建 RAG 应用？",
                "如何使用 LangSmith 观测？",
                "如何观测性能，包括 QPS、SLA、收入和成功率？",
                "RAG 与 LangSmith 有何异同？"
            ]}'''})()

    monkeypatch.setattr("core.planning.get_planning_llm", lambda: FakeLLM())
    plan = build_retrieval_plan("结合 RAG 和 LangSmith，说明如何搭建并观测检索应用。")
    assert plan.queries == ("如何搭建 RAG 应用？", "如何使用 LangSmith 观测？")


def test_planning_cleans_one_invented_metric_without_losing_aspect(monkeypatch):
    class FakeLLM:
        def invoke(self, _messages):
            return type("Response", (), {"content": '{"sub_queries":["手工基线是多少？","自动化效果 QPS 是多少？","线上故障率是多少？"]}'})()

    monkeypatch.setattr("core.planning.get_planning_llm", lambda: FakeLLM())
    plan = build_retrieval_plan("手工基线、自动化效果和线上故障率是多少？")
    assert len(plan.queries) == 3
    assert all("QPS" not in query for query in plan.queries)


def test_complex_question_is_bounded_and_deduplicated(monkeypatch):
    class FakeLLM:
        def invoke(self, _messages):
            return type("Response", (), {"content": '{"sub_queries":["比较两者架构","比较两者稳定性设计","比较两者评测方式","比较两者评测方式","额外问题","不应保留"]}'})()

    monkeypatch.setattr("core.planning.get_planning_llm", lambda: FakeLLM())
    plan = build_retrieval_plan("对比内容分发工具和博客 RAG 的架构、稳定性设计与评测方式")
    assert plan.query_type == "complex"
    assert len(plan.queries) == MAX_SUB_QUERIES
    assert len(set(plan.queries)) == MAX_SUB_QUERIES


def test_invalid_planning_response_falls_back(monkeypatch):
    class FakeLLM:
        def invoke(self, _messages):
            return type("Response", (), {"content": "不是 JSON"})()

    monkeypatch.setattr("core.planning.get_planning_llm", lambda: FakeLLM())
    query = "对比内容分发工具和博客 RAG 的架构、稳定性设计与评测方式"
    assert build_retrieval_plan(query).queries == (query,)


def test_complex_retrieval_merges_deduplicates_and_reranks(monkeypatch):
    first = DocumentChunk(title="A", url="/a", content="first", slug="a", chunk_index=0)
    duplicate = DocumentChunk(title="A", url="/a", content="duplicate", slug="a", chunk_index=0)
    second = DocumentChunk(title="B", url="/b", content="second", slug="b", chunk_index=0)
    first.score, duplicate.score, second.score = 0.2, 0.8, 0.6

    def fake_retrieve(query):
        return [first] if query == "架构" else [duplicate, second]

    seen = {}
    def fake_rerank(query, chunks, limit):
        seen["query"] = query
        seen["chunks"] = chunks
        return chunks[:limit]

    monkeypatch.setattr("api.rag_graph._retrieve_docs", fake_retrieve)
    monkeypatch.setattr("api.rag_graph.rerank", fake_rerank)
    plan = RetrievalPlan(query_type="complex", original_query="原始复杂问题", queries=("架构", "评测"))
    result, chunks_by_query = _retrieve_plan(plan)
    assert seen["query"] == "原始复杂问题"
    assert len(seen["chunks"]) == 2
    assert {chunk.slug for chunk in result} == {"a", "b"}
    assert set(chunks_by_query) == {"架构", "评测"}
    assert chunks_by_query["架构"][0] is not first


def test_generation_context_reserves_one_candidate_per_sub_query():
    a = DocumentChunk(title="A", url="/a", content="a", slug="a", chunk_index=0)
    b = DocumentChunk(title="B", url="/b", content="b", slug="b", chunk_index=0)
    c = DocumentChunk(title="C", url="/c", content="c", slug="c", chunk_index=0)
    result = _ensure_subquery_coverage(
        [a, b, c], {"first": [a], "second": [c]}, coverage_limit=2, total_limit=3,
    )
    assert {_chunk_key.slug for _chunk_key in result[:2]} == {"a", "c"}
