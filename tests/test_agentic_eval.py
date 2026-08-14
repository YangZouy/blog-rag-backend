import json
from pathlib import Path

from eval.eval_ragas import score_agentic, score_pipeline_retrieval
from eval.compare_agentic_runs import compare
from eval.dataset_splits import load_agentic_split


ROOT = Path(__file__).resolve().parents[1]


def test_agentic_dataset_has_required_behavior_contracts():
    data = json.loads((ROOT / "eval" / "agentic_eval_queries.json").read_text(encoding="utf-8"))
    rows = data["queries"]
    types = {row["type"] for row in rows}
    assert {"multi_turn", "complex", "partial_answer", "no_answer", "live_data", "prompt_injection", "authorization_boundary"} <= types
    assert all("expected_action" in row and "should_refuse" in row for row in rows)


def test_all_single_hop_cases_have_human_review_confirmation():
    rows = json.loads((ROOT / "eval" / "agentic_eval_queries.json").read_text(encoding="utf-8"))["queries"]
    review = json.loads((ROOT / "eval" / "agentic_single_hop_review.json").read_text(encoding="utf-8"))
    actual_ids = {row["id"] for row in rows if row["type"] == "single_hop"}
    assert review["result"]["confirmed"] == len(actual_ids)
    assert set(review["confirmed_ids"]) == actual_ids


def test_agentic_split_is_frozen_disjoint_and_exhaustive():
    dev, dev_metadata = load_agentic_split("dev")
    final, final_metadata = load_agentic_split("final")
    assert len(dev) == 56
    assert len(final) == 24
    assert {row["id"] for row in dev}.isdisjoint({row["id"] for row in final})
    assert dev_metadata["frozen"] is True
    assert final_metadata["source_sha256"] == dev_metadata["source_sha256"]


def test_agentic_summary_scores_trace_contracts():
    rows = [
        {
            "history": [{"role": "user", "content": "first"}], "expected_sub_queries": ["架构"],
            "expected_action": "answer", "should_refuse": False, "expected_slugs": ["a"],
            "context_sources": [{"slug": "a"}], "citations": [], "type": "multi_turn",
            "latency_sec": 1.2, "estimated_token_cost": 10,
            "trace": {"rewritten": True, "sub_queries": ["架构设计"], "final_decision": "answer", "retrieval_rounds": 2},
        },
        {
            "history": [], "expected_sub_queries": [], "expected_action": "refuse", "should_refuse": True,
            "expected_slugs": [], "context_sources": [], "latency_sec": 2.4, "estimated_token_cost": 20,
            "answer": "站内资料不足，无法可靠回答。", "citations": [], "type": "no_answer",
            "trace": {"rewritten": False, "sub_queries": [], "final_decision": "refuse", "retrieval_rounds": 1},
        },
    ]
    result = score_agentic(rows)
    assert result["multi_turn_rewrite_accuracy"] == 1.0
    assert result["sub_query_coverage"] == 1.0
    assert result["expected_action_accuracy"] == 1.0
    assert result["no_answer_refusal_accuracy"] == 1.0
    assert result["citation_support_rate"] == 1.0
    assert result["citation_coverage_rate"] == 1.0
    assert result["refusal_behavior_accuracy"] == 1.0
    assert result["macro_expected_action_accuracy"] == 1.0
    assert result["planning_constraint_violation_rate"] == 0.0
    assert result["average_retrieval_rounds"] == 1.5


def test_agentic_summary_detects_invented_planning_constraint():
    rows = [{
        "query": "如何搭建并观测 RAG 应用？",
        "expected_sub_queries": ["RAG", "观测"],
        "expected_action": "answer",
        "type": "complex",
        "trace": {
            "sub_queries": ["如何搭建 RAG？", "生产 QPS 是多少？"],
            "final_decision": "answer",
            "retrieval_rounds": 1,
        },
    }]
    assert score_agentic(rows)["planning_constraint_violation_rate"] == 1.0


def test_pipeline_retrieval_scores_final_context_rank_and_deduplicates_slugs():
    rows = [
        {"expected_slugs": ["relevant"], "context_sources": [{"slug": "wrong"}, {"slug": "relevant"}, {"slug": "relevant"}]},
        {"expected_slugs": ["missing"], "context_sources": [{"slug": "wrong"}]},
    ]
    result = score_pipeline_retrieval(rows)
    assert result["evaluated_rows"] == 2
    assert result["recall@1"] == 0.0
    assert result["recall@3"] == 0.5
    assert result["MRR"] == 0.25
    assert result["slug_hit_rate"] == 0.5


def test_pipeline_retrieval_distinguishes_any_hit_from_document_coverage():
    rows = [{
        "expected_slugs": ["a", "b", "c"],
        "context_sources": [{"slug": "a"}, {"slug": "wrong"}, {"slug": "b"}],
    }]
    result = score_pipeline_retrieval(rows)
    assert result["hit@1"] == 1.0
    assert result["recall@1"] == 1.0
    assert result["coverage@1"] == 0.3333
    assert result["coverage@3"] == 0.6667
    assert result["all_hit@3"] == 0.0


def test_compare_agentic_runs_reports_numeric_delta():
    baseline = {"summary": {"agentic": {"expected_action_accuracy": 0.5}, "pipeline_retrieval": {"MRR": 0.5}}}
    candidate = {"summary": {"agentic": {"expected_action_accuracy": 0.75}, "pipeline_retrieval": {"MRR": 0.75}}}
    rows = compare(baseline, candidate)
    row = next(item for item in rows if item["metric"] == "expected_action_accuracy")
    assert row["delta"] == 0.25
    pipeline_row = next(item for item in rows if item["metric"] == "pipeline_retrieval.MRR")
    assert pipeline_row["delta"] == 0.25
