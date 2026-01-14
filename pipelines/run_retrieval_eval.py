import logging
from pathlib import Path

import numpy as np
import typer
from evaluation.scoring import evaluate_rag
from config.config import Config
from utils.general_util import read_jsonl
from config.logging_config import setup_logging

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #
def main(
    data_dir: Path = typer.Option("data/eval/queries.jsonl", help="Root directory containing the JSONL files"),
    db_path: str = typer.Option("data/output/rag_demo.db", help="Target Milvus database file"),
    collection: str = typer.Option("demo_collection", help="Milvus collection name"),
    model_name: str = typer.Option(
        "intfloat/e5-large-v2", help="SentenceTransformer model name or path"
    ),
    device: str = typer.Option("cuda", help="Device to run the embedding model on"),
    trust_remote_code: bool = typer.Option(True, help="Whether to trust remote code when loading the model")
) -> None:
    logger = logging.getLogger(__name__)

    config = Config(
        data_dir=data_dir,
        db_path=db_path,
        collection=collection,
        model_name=model_name,
        device=device,
        trust_remote_code=trust_remote_code,
    )
    test_set = read_jsonl(data_dir)

    logger.info("Running with config: %s", config)

    all_recall, all_mrr = evaluate_rag(config, test_set)

    # 4  Report mean recall and MRR across all queries.
    print(f"Mean Recall@k: {np.array(all_recall).mean()}")
    print(f"Mean MRR: {np.array(all_mrr).mean()}")


if __name__ == "__main__":
    setup_logging()
    typer.run(main)
