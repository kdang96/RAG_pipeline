import logging
from typing import TypedDict

import numpy as np
from pymilvus import MilvusClient
from config.config import Config
from vector_store.embedding import embed_texts
from config.logging_config import setup_logging

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Typed dictionaries
# --------------------------------------------------------------------------- #

class VECTOR_DB_ROW(TypedDict):
    """Schema that will be inserted into the database."""

    doc_title: str
    chunk_id: int
    heading_2: str
    heading_3: str
    heading_4: str
    chunk: str
    combined_vector: np.ndarray



def connect_to_db(db_path: str) -> tuple[MilvusClient, list[str]]:
    client = MilvusClient(db_path)
    collections = client.list_collections()
    return client, collections

def does_collection_exist(db_path: str, name: str) -> bool:
    client, _ = connect_to_db(db_path)
    collections = client.list_collections()

    if name in collections:
        logger.info(f"Collection {name} already EXISTS.")
        return True
    else:
        logger.info(f"Collection {name} does NOT exist.")
        return False


def get_collection_fields(db_path: str, collection_name: str) -> list[str]:

    client, _ = connect_to_db(db_path)
    schema = client.describe_collection(collection_name)
    client.close()

    fields_n_descr = [
        {"field": f["name"], "description": f["description"]} for f in schema["fields"]
    ]

    return fields_n_descr



def delete_collection(db_path: str, collection_name: str):
    client, _ = connect_to_db(db_path)
    client.drop_collection(collection_name)
    logger.info(f"Collection {collection_name} in {db_path} successfully deleted.")



def insert_data(db_path: str, collection_name: str, data: list[dict] | list[VECTOR_DB_ROW]):
    client, _ = connect_to_db(db_path)

    logger.info(
        "Would insert %d records into %s/%s",
        len(data),
        db_path,
        collection_name,
    )

    try:
        client.insert(collection_name=collection_name, data=data, timeout=20)
        logger.info("Data successfully inserted.")

    except Exception as e:
        # client.flush(collection_name)
        client.close()

        logger.info(
            "If it gets stuck and throws a similar error, it is probably a data "
            "validation issue.\n"
            "Make sure your data matches the schema of the collection.\n"
            "Example error:\n"
            "failed to connect to all addresses; last error: UNKNOWN: "
            "unix:/tmp/tmptkai20wr_milvus_demo.db.sock: connect: Connection refused (111)\n"
        )
        
        raise


def search(
    config: Config,
    user_queries: list[str],
    search_col: str,
    output_fields: list[str],
    k_limit: int = 10,
    search_radius: float = 0.75,
) -> list[list[dict]]:
    client, _ = connect_to_db(config.db_path)
    client.load_collection(config.collection)
    client.refresh_load(config.collection)

    query_vectors: np.ndarray = embed_texts(user_queries, config)

    results = client.search(
        collection_name=config.collection,
        anns_field=search_col,
        data=list(query_vectors),
        limit=k_limit,
        output_fields=output_fields,
        search_params={"metric_type": "COSINE", "radius": search_radius},
    )

    client.close()

    return results



def create_index(
    db_path: str, collection_name: str, vector_col_name: str, index_name: str
):
    client, _ = connect_to_db(db_path)

    # Prepare index parameters
    index_params = client.prepare_index_params()

    # Add an index on the vector field
    index_params.add_index(
        field_name=vector_col_name,
        metric_type="COSINE",
        index_type="FLAT",
        index_name=index_name,
    )

    # Create the index
    client.create_index(collection_name=collection_name, index_params=index_params)

    client.flush(collection_name)

    client.close()

    logger.info(f"Index {index_name} in {collection_name} created.")


if __name__ == "__main__":
    setup_logging()
    client, collections = connect_to_db("data/output/rag_demo.db")
    logger.info(f"Existing collections: {collections}")
    client.close()

    delete_collection("data/output/rag_demo.db", collections[0])

    # create_index(
    #     db_path="data/output/rag_demo.db",
    #     collection_name="demo_collection",
    #     vector_col_name="combined_vector",
    #     index_name="combined_vector_index",
    # )

  

