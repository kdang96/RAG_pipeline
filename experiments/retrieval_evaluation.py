from pathlib import Path
from typing import Generator
import numpy as np
import typer
from src.utils.general_util import read_jsonl
from src.vector_store.milvus import Config, search

def recall_at_k(expected: list[int], observed: list[int]):
    # 1.  Convert expected list to a set (hash‑based, O(1) look‑up)
    expected_set = set(expected)

    # 2.  Count how many of the top‑k predictions are relevant
    hits = sum(1 for item in observed if item in expected_set)

    # 3.  The “recall@k” value
    recall_at_k = hits / len(expected)

    hit_ratio = f"{hits}/{len(observed)}"

    return recall_at_k, hit_ratio,


# ---------------------------------------------------------------- #
# Evaluate RAG --------------------------------------------------- #
# ---------------------------------------------------------------- #
def evaluate_rag(config: Config, test_set: Generator) -> None:
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
    for i, chunk in enumerate(items):
        obs_chunk_ids = [x["entity"]["chunk_id"] for x in all_obs[i]]
        recall, hit_ratio = recall_at_k(expected_chunk_ids[i], obs_chunk_ids)
        all_recall.append(recall)

    # 4  Report mean recall across all queries.
    print(f"Mean Recall@k: {np.array(all_recall).mean()}")

    

# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #


def main(
    data_dir: Path = typer.Option("data/eval/queries.jsonl", help="Root directory containing the JSONL files"),
    db_path: str = typer.Option("rag_demo.db", help="Target Milvus database file"),
    collection: str = typer.Option("demo_collection", help="Milvus collection name"),
    model_name: str = typer.Option(
        "intfloat/e5-large-v2", help="SentenceTransformer model name or path"
    ),
    device: str = typer.Option("cuda", help="Device to run the embedding model on"),
    trust_remote_code: bool = typer.Option(True, help="Whether to trust remote code when loading the model")
) -> None:
    config = Config(
        data_dir=data_dir,
        db_path=db_path,
        collection=collection,
        model_name=model_name,
        device=device,
        trust_remote_code=trust_remote_code,
    )
    test_set = read_jsonl(data_dir)

    # logger.info("Running with config: %s", config)

    evaluate_rag(config, test_set)


if __name__ == "__main__":
    typer.run(main)


    