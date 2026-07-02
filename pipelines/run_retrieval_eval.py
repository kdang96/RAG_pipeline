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
    db_path: str | None = typer.Option(None, help="Target Milvus database file (env: RAG_DB_PATH)"),
    collection: str | None = typer.Option(None, help="Milvus collection name (env: RAG_COLLECTION)"),
    model_name: str | None = typer.Option(
        None, help="SentenceTransformer model name or path (env: RAG_MODEL_NAME)"
    ),
    device: str | None = typer.Option(None, help="Device to run the embedding model on (env: RAG_DEVICE)"),
    trust_remote_code: bool | None = typer.Option(
        None, help="Whether to trust remote code when loading the model (env: RAG_TRUST_REMOTE_CODE)"
    ),
) -> None:
    logger = logging.getLogger(__name__)

    overrides = {
        "data_dir": data_dir,
        "db_path": db_path,
        "collection": collection,
        "model_name": model_name,
        "device": device,
        "trust_remote_code": trust_remote_code,
    }
    config = Config(**{k: v for k, v in overrides.items() if v is not None})
    test_set = read_jsonl(data_dir)

    logger.info("Running with config: %s", config)

    all_recall, all_mrr = evaluate_rag(config, test_set)

    # 4  Report mean recall and MRR across all queries.
    print(f"Mean Recall@k: {np.array(all_recall).mean()}")
    print(f"Mean MRR: {np.array(all_mrr).mean()}")


if __name__ == "__main__":
    setup_logging()
    typer.run(main)
