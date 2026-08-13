from types import SimpleNamespace

from api.rag_graph import _retrieve_with_evidence
from core.planning import RetrievalPlan
from core.remediation import RemedyAction, choose_remedy
from data.parse_hexo import DocumentChunk


def chunk(score: float) -> DocumentChunk:
    return DocumentChunk(title="Title", url="/post", content="content", slug="post", score=score)


def controlled_settings():
    return SimpleNamespace(
        RETRIEVAL_CANDIDATE_K=8,
        RETRIEVAL_REMEDY_CANDIDATE_K=16,
        MAX_RETRIEVAL_ROUNDS=2,
        EVIDENCE_RELEVANCE_THRESHOLD=0.3,
    )


def test_remedy_policy_only_allows_candidate_expansion():
    decision = choose_remedy(8, 16)
    assert decision is not None
    assert decision.action is RemedyAction.EXPAND_CANDIDATES
    assert decision.candidate_k == 16
    assert choose_remedy(16, 16) is None


def test_insufficient_first_round_retries_exactly_once(monkeypatch):
    plan = RetrievalPlan("simple", "问题", ("问题",))
    calls = []

    def fake_route(_plan, _top_k, candidate_k=None):
        calls.append(candidate_k)
        score = 0.1 if candidate_k == 8 else 0.8
        docs = [chunk(score)]
        return "rag", docs, {"问题": docs}, "rag"

    monkeypatch.setattr("api.rag_graph.get_settings", controlled_settings)
    monkeypatch.setattr("api.rag_graph._route_and_retrieve", fake_route)
    intent, docs, _, assessment, remedy = _retrieve_with_evidence(plan, 5)
    assert intent == "rag"
    assert calls == [8, 16]
    assert docs[0].score == 0.8
    assert assessment.status.value == "sufficient"
    assert remedy is not None and remedy.action is RemedyAction.EXPAND_CANDIDATES


def test_second_insufficient_round_terminates_without_third_call(monkeypatch):
    plan = RetrievalPlan("simple", "问题", ("问题",))
    calls = []

    def fake_route(_plan, _top_k, candidate_k=None):
        calls.append(candidate_k)
        docs = [chunk(0.1)]
        return "rag", docs, {"问题": docs}, "rag"

    monkeypatch.setattr("api.rag_graph.get_settings", controlled_settings)
    monkeypatch.setattr("api.rag_graph._route_and_retrieve", fake_route)
    _, _, _, assessment, remedy = _retrieve_with_evidence(plan, 5)
    assert calls == [8, 16]
    assert assessment.status.value == "insufficient"
    assert remedy is not None


def test_sufficient_first_round_does_not_remediate(monkeypatch):
    plan = RetrievalPlan("simple", "问题", ("问题",))
    calls = []

    def fake_route(_plan, _top_k, candidate_k=None):
        calls.append(candidate_k)
        docs = [chunk(0.8)]
        return "rag", docs, {"问题": docs}, "rag"

    monkeypatch.setattr("api.rag_graph.get_settings", controlled_settings)
    monkeypatch.setattr("api.rag_graph._route_and_retrieve", fake_route)
    _, _, _, assessment, remedy = _retrieve_with_evidence(plan, 5)
    assert calls == [8]
    assert assessment.status.value == "sufficient"
    assert remedy is None
