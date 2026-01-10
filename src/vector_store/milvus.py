import json
from pathlib import Path
from pprint import pprint
import logging
import sys
from typing import Iterable, TypedDict, List, Dict, Any

from attr import dataclass
import numpy as np
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)


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



# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Config:
    """Immutable configuration container."""

    db_path: str
    collection: str
    data_dir: Path = Path("")
    model_name: str = "intfloat/e5-large-v2"
    device: str = "cuda"
    trust_remote_code: bool = True

    def __post_init__(self):
        if self.device not in {"cpu", "cuda"}:
            raise ValueError(
                f"Invalid device '{self.device}'. Expected 'cpu' or 'cuda'."
            )

        if not self.data_dir.exists():
            raise ValueError(f"data_dir does not exist: {self.data_dir}")
    
class Embedder:
    """Lazy‑loaded SentenceTransformer wrapper."""

    _model: SentenceTransformer | None = None

    @classmethod
    def model(cls, config: Config) -> SentenceTransformer:
        if cls._model is None:
            logger.info("Loading SentenceTransformer (%s) …", config.model_name)
            cls._model = SentenceTransformer(
                model_name_or_path=config.model_name,
                device=config.device,
                trust_remote_code=config.trust_remote_code,
            )
        return cls._model

    @classmethod
    def embed(cls, texts: list[str], config: Config) -> List[np.ndarray]:
        model = cls.model(config)
        return model.encode(list(texts), convert_to_numpy=True)


def connect_to_db(db_name: str) -> tuple[MilvusClient, list[str]]:
    client = MilvusClient(db_name)
    collections = client.list_collections()
    return client, collections

def does_collection_exist(db_name: str, name: str) -> bool:
    client = MilvusClient(db_name)
    collections = client.list_collections()

    if name in collections:
        logger.info(f"Collection {name} already EXISTS.")
        return True
    else:
        logger.info(f"Collection {name} does NOT exist.")
        return False


def get_collection_fields(db_name: str, collection_name: str) -> list[str]:
    client, _ = connect_to_db(db_name)
    schema = client.describe_collection(collection_name)
    client.close()

    # field_names = [field_json["name"] for field_json in schema["fields"]]
    # field_descr = [field_json["description"] for field_json in schema["fields"]]
    # return field_names, field_descr

    # fields_n_descr = {f["name"]: f["description"] for f in schema["fields"]}
    fields_n_descr = [
        {"field": f["name"], "description": f["description"]} for f in schema["fields"]
    ]

    return fields_n_descr


def _list_collections(db_name: str):
    client, collections = connect_to_db(db_name)
    print("Collections:", collections)
    client.close()


def list_collections_schema(db_name: str):
    client, _ = connect_to_db(db_name)
    collections = client.list_collections()
    for coll_name in collections:
        schema = client.describe_collection(coll_name)
        stats = client.get_collection_stats(coll_name)
        pprint(schema)
        pprint(stats)
        # print("Collection Schema:", schema)
        # print("Collection Stats:", stats)
        # print("Number of entities:", coll.num_entities)

    client.close()


def delete_collection(collection_name: str):
    client.drop_collection(collection_name)


def _describe_collection(name: str):
    print(json.dumps(client.describe_collection(collection_name=name), indent=2))


def list_all_entities(db_name: str, collection_name: str, output_fields: list[str]):
    client, _ = connect_to_db(db_name)

    res = client.query(
        collection_name=collection_name,
        filter="",  # Empty filter returns all entities
        output_fields=output_fields,
        limit=50,
    )

    return res


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
        print("Data successfully inserted.")



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




def update_data(db_name: str, collection_name: str, data: list[dict]):
    client, _ = connect_to_db(db_name)

    try:
        res = client.upsert(collection_name=collection_name, data=data, timeout=50)
    except Exception as e:
        print("Current error:\n", e)
        print("""If it gets stuck and it throws a a similar error, it is probably a data validaiton error. \n
              Make sure your data is matching the schema of the collection.\n
              Example error: \n
              failed to connect to all addresses; last error: UNKNOWN: unix:/tmp/tmptkai20wr_milvus_demo.db.sock: connect: Connection refused (111))>, <Time:{'RPC start': '2025-09-23 11:29:14.321350', 'RPC error': '2025-09-23 11:29:18.261007'}> (decorators.py:140)
              """)
    client.flush(collection_name)
    print(res)
    print(client.get_collection_stats(collection_name))
    print("Data successfully updated.")
    client.close()


def query(
    db_name: str,
    collection_name: str,
    filter: str = "",
    output_fields: list[str] = ["*"],
) -> list[dict]:
    client, _ = connect_to_db(db_name)
    client.load_collection(collection_name)
    client.refresh_load(collection_name)

    results = client.query(
        collection_name=collection_name,
        filter=filter,
        limit=100,
        # output_fields = ["*"]
        # output_fields = ['wo_number', 'project_title', 'wo_release_date', 'scope', 'project_id', 'technical_authority', 'originator', 'ip_sensitivity', 'project_manager']
        output_fields=output_fields,
    )

    return results


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

    query_vector = Embedder.embed(user_queries, config)

    results = client.search(
        collection_name=config.collection,
        anns_field=search_col,
        data=query_vector,
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
    index_params = MilvusClient.prepare_index_params()

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

    print(f"Index {index_name} in {collection_name} created.")


def drop_index(db_name: str):
    client, _ = connect_to_db(db_name)
    client.drop_index(collection_name="DISs", index_name="vector_index")

    print("Index dropped.")


if __name__ == "__main__":
    client = MilvusClient(r"./rag_demo.db")
    collections = client.list_collections()
    print(collections)

    # delete_all_embeddings(collections[0])
    delete_collection(collections[0])
    # add_dis_collection("DISs_e5_large_v2")
    # add_soi_collection("SOIs_e5_large_v2")
    # _list_collections("milvus_demo.db")
    # result = list_all_entities(db_name="milvus_demo.db", collection_name="SOIs_e5_large_v2", output_fields=["document_name"])
    # print(list(result))
    # drop_index("milvus_demo.db")
    # print(get_collection_fields("milvus_demo.db", "DISs_e5_large_v2"))
    # list_collections_schema("milvus_demo.db")
    # _describe_collection("SOIs_e5_large_v2")
    # list_all_entities("milvus_demo.db", "DISs_e5_large_v2", ["wo_number"])

    # create_index(
    #     db_name="./milvus_demo.db",
    #     in_collection="SOIs_e5_large_v2",
    #     vector_col_name="doc_summary_vector",
    #     index_name="doc_summary_vector",
    # )

    # create_json_index(
    #     db_name="./milvus_demo.db",
    #     in_collection="SOIs_e5_large_v2",
    #     json_col_name="content",
    #     main_key="content",
    #     index_name="content_index"
    # )

    client.close()
