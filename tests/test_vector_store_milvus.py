"""End-to-end checks against a real Milvus-Lite database (a local file).

No embedding model is loaded: `patch_embeddings` provides deterministic
vectors, so identical text produces identical vectors and cosine
similarity is exactly 1.0 for an exact match.
"""

from data_import.milvus_import_data import add_soi_collection
from vector_store.milvus import (
    create_index,
    delete_collection,
    does_collection_exist,
    get_collection_fields,
    insert_data,
    search,
)

CHUNK_TEXTS = {
    0: "rules on media services",
    1: "due diligence obligations",
    2: "single market transparency",
}


def _rows(embedder):
    rows = []
    for chunk_id, text in CHUNK_TEXTS.items():
        rows.append(
            {
                "chunk_id": chunk_id,
                "doc_title": "Doc A",
                "heading_2": "na",
                "heading_3": "na",
                "heading_4": "na",
                "chunk": text,
                "combined_vector": embedder.encode([text])[0],
            }
        )
    return rows


def _populate(config, embedder):
    add_soi_collection(config.db_path, config.collection)
    insert_data(config.db_path, config.collection, _rows(embedder))
    create_index(config.db_path, config.collection, "combined_vector", "vec_idx")


def test_collection_lifecycle(config, patch_embeddings):
    assert does_collection_exist(config.db_path, config.collection) is False
    add_soi_collection(config.db_path, config.collection)
    assert does_collection_exist(config.db_path, config.collection) is True

    field_names = {f["field"] for f in get_collection_fields(config.db_path, config.collection)}
    assert {"chunk_id", "chunk", "combined_vector"} <= field_names

    delete_collection(config.db_path, config.collection)
    assert does_collection_exist(config.db_path, config.collection) is False


def test_search_returns_exact_match_first(config, patch_embeddings):
    _populate(config, patch_embeddings)

    results = search(
        config=config,
        user_queries=["due diligence obligations"],
        search_col="combined_vector",
        output_fields=["chunk_id", "chunk"],
        k_limit=3,
        search_radius=-1.0,
    )

    top_hit = results[0][0]
    assert top_hit["entity"]["chunk_id"] == 1
