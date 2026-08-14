import json
from pathlib import Path

import pytest

import eval.eval_recall as retrieval_regression
from eval.eval_recall import (
    RETRIEVAL_REGRESSION_COUNT,
    aggregate,
    eval_one,
    latest_previous_result,
    load_regression_dataset,
    metric_diff,
)


ROOT = Path(__file__).resolve().parents[1]


def test_historical_retrieval_dataset_is_frozen_and_valid():
    queries, metadata = load_regression_dataset()
    assert len(queries) == RETRIEVAL_REGRESSION_COUNT == 49
    assert metadata["frozen"] is True
    assert 25 not in {query["id"] for query in queries}


def test_dataset_change_requires_an_explicit_escape_hatch(monkeypatch):
    dataset = ROOT / "eval" / "eval_queries.json"
    monkeypatch.setattr(retrieval_regression, "RETRIEVAL_REGRESSION_COUNT", 50)
    with pytest.raises(ValueError, match="dataset changed"):
        load_regression_dataset(dataset)
    queries, metadata = load_regression_dataset(dataset, enforce_fingerprint=False)
    assert len(queries) == 49
    assert metadata["frozen"] is False


def test_recall_and_mrr_use_first_matching_slug_rank():
    query = {
        "id": 1, "query": "q", "type": "concept", "category": "test",
        "expected_slugs": ["relevant", "also-relevant"],
    }

    def retrieve(_query, _limit):
        return [("wrong", 0.9), ("relevant", 0.8), ("also-relevant", 0.7)]

    row = eval_one(query, retrieve)
    assert row["first_rank"] == 2
    assert row["rr"] == 0.5
    assert row["recall"][3] == 1.0


def test_aggregate_and_diff_report_regression_direction():
    row = {
        "type": "concept", "category": "test", "rr": 0.5, "delta": 0.1,
        "diagnosis": "B_rerank_candidate", "recall": {3: 0, 5: 1, 10: 1, 12: 1, 50: 1},
    }
    summary = aggregate([row])
    assert summary["overall"]["recall@3"] == 0.0
    assert summary["overall"]["MRR"] == 0.5

    previous = {"overall": {"recall@3": 1.0, "MRR": 0.75}}
    diff = metric_diff(previous, summary)
    assert diff["recall@3"]["delta"] == -1.0
    assert diff["MRR"]["delta"] == -0.25


def test_committed_baselines_exist_for_every_retrieval_stage():
    for mode in ("raw", "hybrid", "rerank"):
        baseline = latest_previous_result(mode, ROOT / "does-not-exist.json")
        assert baseline is not None
        data = json.loads(baseline.read_text(encoding="utf-8"))
        assert data["config"]["retrieval_mode"] == mode
        assert data["dataset"]["record_count"] == 49
