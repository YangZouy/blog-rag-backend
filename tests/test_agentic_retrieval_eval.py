from argparse import Namespace

from eval.eval_agentic_retrieval import (
    build_per_query_rows,
    should_hide_per_query,
)


def test_build_per_query_rows_deduplicates_ranked_slugs_and_scores_hits():
    rows = [
        {
            "id": "cx-01",
            "query": "比较架构与评测",
            "type": "complex",
            "category": "cross_document",
            "mode": "rag",
            "expected_action": "answer",
            "trace": {"retrieval_rounds": 2},
            "expected_slugs": ["good/path"],
            "context_sources": [
                {"slug": "bad/path"},
                {"slug": "good\\path"},
                {"slug": "good/path"},
            ],
        }
    ]
    result = build_per_query_rows(rows)
    retrieval = result[0]["retrieval"]
    assert retrieval["retrieved_slugs"] == ["bad/path", "good/path"]
    assert retrieval["first_hit_rank"] == 2
    assert retrieval["hit"] is True
    assert retrieval["hit@1"] is False
    assert retrieval["hit@3"] is True


def test_should_hide_per_query_only_for_final_split_without_override():
    assert should_hide_per_query(Namespace(split="final", reveal_final_per_query=False)) is True
    assert should_hide_per_query(Namespace(split="final", reveal_final_per_query=True)) is False
    assert should_hide_per_query(Namespace(split="dev", reveal_final_per_query=False)) is False
    assert should_hide_per_query(Namespace(split=None, reveal_final_per_query=False)) is False
