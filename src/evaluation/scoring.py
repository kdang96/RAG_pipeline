from typing import Generator
from src.vector_store.milvus import search
from src.config.config import Config


"""
Offline retrieval evaluation.

This module intentionally excludes any LLM usage to ensure
deterministic, reproducible measurement of retrieval quality
(Recall@k) independent of generation effects.
"""


# ---------------------------------------------------------------- #
# Metrics - Recall@k --------------------------------------------- #
# ---------------------------------------------------------------- #

def recall_at_k(expected: list[int], observed: list[int]):
    # 1.  Convert expected list to a set (hash‑based, O(1) look‑up)
    expected_set = set(expected)

    # 2.  Count how many of the top‑k predictions are relevant
    hits = sum(1 for item in observed if item in expected_set)

    # 3.  The “recall@k” value
    recall_at_k = hits / len(expected)

    hit_ratio = f"{hits}/{len(observed)}"

    return recall_at_k, hit_ratio,


def mrr_at_k(expected: list[int], observed: list[int]) -> float:
    """
    Compute the reciprocal rank for a single query.
    Returns 1/(rank) where rank is 1‑based index of the first relevant item.
    If no relevant item is found, returns 0.0.
    """
    for idx, item in enumerate(observed):
        if item in expected:
            return 1.0 / (idx + 1)   # 1‑based rank
    return 0.0

# ---------------------------------------------------------------- #
# Evaluate RAG --------------------------------------------------- #
# ---------------------------------------------------------------- #
def evaluate_rag(config: Config, test_set: Generator) -> tuple[list[int], list[int]]:
    # Convert the one‑shot generator to a concrete list once.
    # After this point you can iterate over `items` as many times as you like.
    items = list(test_set)          # <-- materialise here

    # 1  Build the query list and the expected‑chunk‑id list.
    queries = [x["query"] for x in items]
    expected_chunk_ids = [x["expected_chunk_ids"] for x in items]

    # 2  Perform a single search for all queries.
    all_obs = search(
        config=config,
        user_queries=queries,
        search_col="combined_vector",
        output_fields=[
            "doc_title", "chunk_id", "heading_2",
            "heading_3", "heading_4", "chunk"
        ],
        search_radius=0.7,
        k_limit=10
    )

    # 3  Iterate over the materialised items again for recall@k.
    all_recall = []
    all_mrr = []
    for i, chunk in enumerate(items):
        obs_chunk_ids = [x["entity"]["chunk_id"] for x in all_obs[i]]
        recall, _ = recall_at_k(expected_chunk_ids[i], obs_chunk_ids)
        all_recall.append(recall)

        mrr = mrr_at_k(expected_chunk_ids[i], obs_chunk_ids)
        all_mrr.append(mrr)

    return all_recall, all_mrr




    