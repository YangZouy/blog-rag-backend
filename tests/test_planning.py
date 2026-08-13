from api.rag_graph import _retrieve_plan
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
