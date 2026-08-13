import json
from pathlib import Path

from eval.eval_ragas import score_agentic
from eval.compare_agentic_runs import compare


ROOT = Path(__file__).resolve().parents[1]


def test_agentic_dataset_has_required_behavior_contracts():
    data = json.loads((ROOT / "eval" / "agentic_eval_queries.json").read_text(encoding="utf-8"))
    rows = data["queries"]
    types = {row["type"] for row in rows}
    assert {"multi_turn", "complex", "partial_answer", "no_answer", "live_data", "prompt_injection", "authorization_boundary"} <= types
    assert all("expected_action" in row and "should_refuse" in row for row in rows)


def test_agentic_summary_scores_trace_contracts():
    rows = [
        {
            "history": [{"role": "user", "content": "first"}], "expected_sub_queries": ["架构"],
            "expected_action": "answer", "should_refuse": False, "expected_slugs": ["a"],
            "context_sources": [{"slug": "a"}], "latency_sec": 1.2, "estimated_token_cost": 10,
            "trace": {"rewritten": True, "sub_queries": ["架构设计"], "final_decision": "answer", "retrieval_rounds": 2},
        },
        {
            "history": [], "expected_sub_queries": [], "expected_action": "refuse", "should_refuse": True,
            "expected_slugs": [], "context_sources": [], "latency_sec": 2.4, "estimated_token_cost": 20,
            "trace": {"rewritten": False, "sub_queries": [], "final_decision": "refuse", "retrieval_rounds": 1},
        },
    ]
    result = score_agentic(rows)
    assert result["multi_turn_rewrite_accuracy"] == 1.0
    assert result["sub_query_coverage"] == 1.0
    assert result["expected_action_accuracy"] == 1.0
    assert result["no_answer_refusal_accuracy"] == 1.0
    assert result["citation_support_rate"] == 1.0
    assert result["average_retrieval_rounds"] == 1.5


def test_compare_agentic_runs_reports_numeric_delta():
    baseline = {"summary": {"agentic": {"expected_action_accuracy": 0.5}}}
    candidate = {"summary": {"agentic": {"expected_action_accuracy": 0.75}}}
    rows = compare(baseline, candidate)
    row = next(item for item in rows if item["metric"] == "expected_action_accuracy")
    assert row["delta"] == 0.25
