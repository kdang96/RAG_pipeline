#!/usr/bin/env python3

"""
Data‑import pipeline for RAG chunk.

The script:

*   Reads all files under a supplied directory.
*   Validates and normalises the data to a stable schema.
*   Generates embeddings with a SentenceTransformer model.
*   Inserts the enriched records into a Milvus‑style database.

All I/O is wrapped in robust error handling and the whole
process is fully typed and easily testable.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Generator, List

import numpy as np
import typer
from tqdm import tqdm

from utils.general_util import read_jsonl
from src.vector_store.milvus  import insert_data, VECTOR_DB_ROW, connect_to_db, create_index, Embedder, Config, does_collection_exist
from pymilvus import DataType


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)


def read_jsonl(path: Path) -> Generator[Any, Any, Any]:
    """Load JSON and raise a descriptive exception on failure."""
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)                  
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}") from exc
    except OSError as exc:
        raise RuntimeError(f"Cannot read {path}") from exc


# --------------------------------------------------------------------------- #
# DB schema setup
# --------------------------------------------------------------------------- #


def add_soi_collection(db_name:str, name: str) -> None:
    client, _ = connect_to_db(db_name)
    exists = does_collection_exist(db_name, name)

    if not exists:
        schema = client.create_schema(auto_id=True, enable_dynamic_field=True)

        schema.add_field(
            field_name="id",
            datatype=DataType.INT64,
            is_primary=True,
            auto_id=True,
            description="The primary key.",
        )
        schema.add_field(
            field_name="doc_title",
            datatype=DataType.VARCHAR,
            max_length=50,
            description="The name of the document.",
        )
        schema.add_field(
            field_name="chunk",
            datatype=DataType.VARCHAR,
            max_length=65535,
            description="The chunk of text to be embedded.",
        )
        schema.add_field(
            field_name="chunk_vector",
            datatype=DataType.FLOAT_VECTOR,
            dim=1024,
            description="A vector embedding of the 'chunk' field.",
        )

        client.create_collection(collection_name=name, schema=schema)

        logger.info(f"Collection {name} added.")
    else:
        pass # Collection already exists



# --------------------------------------------------------------------------- #
# Data transformation
# --------------------------------------------------------------------------- #
def chunk_to_db_row(chunks: dict[str, str]) -> VECTOR_DB_ROW:
    """Transform a validated ``SOI`` into the DB‑row schema."""
    return VECTOR_DB_ROW(
        doc_title=chunks["doc_title"],
        chunk=chunks["chunk"],
        chunk_vector=np.empty((0,)),  # placeholder, will be replaced later
    )


# --------------------------------------------------------------------------- #
# Main pipeline
# --------------------------------------------------------------------------- #


def process_pipeline(config: Config, chunks: Generator[dict]| list[dict]) -> None:
    """Run the entire pipeline end‑to‑end."""
    if not config.data_dir:
        logger.warning("%s not found.", config.data_dir)
        return

    db_rows: List[VECTOR_DB_ROW] = []

    for chunk in tqdm(chunks, desc="Processing chunks"):
        db_rows.append(chunk_to_db_row(chunk))

    # -------------------------------------------------------------------- #
    # 1. Create embeddings for *all* summaries in one shot
    # -------------------------------------------------------------------- #
    summaries = [row["chunk"] for row in db_rows]
    embeddings: list = Embedder.embed(summaries, config)

    # Attach embeddings to the rows
    for row, vector in zip(db_rows, embeddings):
        row["chunk_vector"] = vector

    # -------------------------------------------------------------------- #
    # 2. Insert into the database
    # -------------------------------------------------------------------- #
    add_soi_collection(config.db_path, config.collection)

    # -------------------------------------------------------------------- #
    # 3. Insert into the database
    # -------------------------------------------------------------------- #
    insert_data(
        db_path=config.db_path,
        collection_name=config.collection,
        data=db_rows,
    )

    # -------------------------------------------------------------------- #
    # 4. Index vector field
    # -------------------------------------------------------------------- #
    create_index(
        db_path=config.db_path,
        collection_name=config.collection,
        vector_col_name="chunk_vector",
        index_name="chunk_vector_index",
    )


# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #


def main(
    data_dir: Path = typer.Option("src/demo_docx/chunks.jsonl", help="Root directory containing the JSONL files"),
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
    chunks = read_jsonl(data_dir)
    logger.info("Running with config: %s", config)

    process_pipeline(config, chunks)


if __name__ == "__main__":
    typer.run(main)
