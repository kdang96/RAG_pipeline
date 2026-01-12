"""
This module orchestrates the end‑to‑end pipeline for a Retrieval‑Augmented Generation (RAG) demo:

1. Extracts and splits all DOCX documents in a given directory into text chunks.
2. Loads those chunks into a Milvus vector database for efficient similarity search.
3. Configures the database path, collection name, embedding model, and device settings.

Typical usage:
    python pipelines/run_extraction_n_import.py
"""


import typer
from pathlib import Path
from src.ingest.docx_extract_n_chunk import get_all_chunks
from src.data_import.milvus_import_data import process_pipeline
from src.config.config import Config
from src.config.logging_config import setup_logging


def run_pipline(docx_dir: str | Path):
    # --------------------------------------------------------------------------- #
    # Convert DOCX to chunks
    # --------------------------------------------------------------------------- #
    chunks = get_all_chunks(docx_dir)

    # --------------------------------------------------------------------------- #
    # Import in Milvus vector database
    # --------------------------------------------------------------------------- #
    process_pipeline(
        Config(
            data_dir=Path("data/processed/chunks.jsonl"),
            db_path="data/output/rag_demo.db",
            collection="demo_collection",
            model_name="intfloat/e5-large-v2",
            device="cuda",
            trust_remote_code=True,
        ),
        chunks
        )

# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #
def main(
    docx_dir: Path = typer.Option("data/input", help="Directory containing the *.docx files"),
) -> None:
    run_pipline(docx_dir)


if __name__ == "__main__":
    setup_logging()
    typer.run(main)

