from pathlib import Path
from src.ingest.docx_extract import get_all_chunks
from src.data_import.milvus_import_data import process_pipeline
from src.vector_store.milvus import Config


def run_pipline(docx_dir: str):
    # --------------------------------------------------------------------------- #
    # Convert DOCX to chunks
    # --------------------------------------------------------------------------- #
    chunks = get_all_chunks(docx_dir)

    # --------------------------------------------------------------------------- #
    # Import in Milvus vector database
    # --------------------------------------------------------------------------- #
    process_pipeline(
        Config(
            data_dir=Path("src/demo_docx/chunks.jsonl"),
            db_path="rag_demo.db",
            collection="demo_collection",
            model_name="intfloat/e5-large-v2",
            device="cuda",
            trust_remote_code=True,
        ),
        chunks
        )



if __name__ == "__main__":
    run_pipline(r"./data/demo_docx")