from evaluation import scoring
from evaluation.scoring import evaluate_rag, mrr_at_k, recall_at_k


def test_recall_all_hits():
    recall, ratio = recall_at_k(expected=[1, 2], observed=[1, 2, 3])
    assert recall == 1.0
    assert ratio == "2/3"


def test_recall_partial_hit():
    recall, _ = recall_at_k(expected=[1, 2, 3, 4], observed=[1, 9, 3])
    assert recall == 0.5


def test_recall_no_hits():
    recall, ratio = recall_at_k(expected=[1], observed=[7, 8])
    assert recall == 0.0
    assert ratio == "0/2"


def test_mrr_first_position():
    assert mrr_at_k(expected=[5], observed=[5, 1, 2]) == 1.0


def test_mrr_third_position():
    assert mrr_at_k(expected=[2], observed=[9, 8, 2]) == 1 / 3


def test_mrr_no_hit():
    assert mrr_at_k(expected=[1], observed=[2, 3]) == 0.0


def test_evaluate_rag(monkeypatch, config):
    test_set = [
        {"query": "q1", "expected_chunk_ids": [10]},
        {"query": "q2", "expected_chunk_ids": [99]},
    ]

    def fake_search(**kwargs):
        # q1 retrieves the relevant chunk at rank 1; q2 misses entirely.
        return [
            [{"entity": {"chunk_id": 10}}, {"entity": {"chunk_id": 11}}],
            [{"entity": {"chunk_id": 1}}, {"entity": {"chunk_id": 2}}],
        ]

    monkeypatch.setattr(scoring, "search", fake_search)

    recall, mrr = evaluate_rag(config, iter(test_set))
    assert recall == [1.0, 0.0]
    assert mrr == [1.0, 0.0]
