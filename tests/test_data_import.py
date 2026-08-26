from data_import.milvus_import_data import chunk_to_db_row, process_pipeline
from vector_store.milvus import does_collection_exist, search


def test_chunk_to_db_row_maps_fields(sample_chunks):
    row = chunk_to_db_row(sample_chunks[1])
    assert row["chunk_id"] == 1
    assert row["doc_title"] == "Doc A"
    assert row["heading_3"] == "Due diligence"
    assert row["combined_vector"].size == 0  # placeholder until embedded


def test_process_pipeline_populates_collection(config, patch_embeddings, sample_chunks):
    process_pipeline(config, sample_chunks)

    assert does_collection_exist(config.db_path, config.collection)

    results = search(
        config=config,
        user_queries=["This treaty applies to media services."],
        search_col="combined_vector",
        output_fields=["chunk_id", "chunk"],
        k_limit=2,
        search_radius=-1.0,
    )
    retrieved_ids = {hit["entity"]["chunk_id"] for hit in results[0]}
    assert retrieved_ids == {0, 1}
